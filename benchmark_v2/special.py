"""Train-only, identity-aware components for the three specialist baselines."""
import copy
import numpy as np
import torch
import torch.nn.functional as F


def bind_eac_ids(model, sensor_ids):
    """Retain rows by sensor ID, including departed sensors for later return."""
    bank=copy.deepcopy(getattr(model,'sensor_bank',{}))
    for i,sid in enumerate(getattr(model,'sensor_ids',[])):
        bank[int(sid)]=model.U[i].detach().clone()
    ids=list(map(int,sensor_ids))
    rows=torch.empty((len(ids),model.rank),device=model.U.device,dtype=model.U.dtype).uniform_(-.1,.1)
    for i,sid in enumerate(ids):
        if sid in bank:rows[i]=bank[sid].to(rows.device)
    model.U=torch.nn.Parameter(rows)
    model.num_nodes=len(ids);model.sensor_ids=ids;model.sensor_bank=bank


def daily_clusters(data, requested):
    """Original daily-mean/uncentered-correlation clustering on observed train days."""
    from Bio.Cluster import kcluster
    days=int(data['bounds'][1]//288)
    if not days:raise ValueError('STKEC requires at least one full training day')
    N=len(data['sensor_ids']);v=data['y'][:days*288].reshape(days,288,N)
    m=data['observed'][:days*288].reshape(days,288,N)
    count=m.sum(1);total=(v*m).sum(1)
    mean=np.divide(total,count,out=np.zeros_like(total,dtype=float),where=count>0).T
    available=(count.T>0);valid=np.flatnonzero(available.any(1))
    if not len(valid):raise ValueError('No observed training-day profiles for clustering')
    K=min(int(requested),len(valid))
    if K<1:raise ValueError('Invalid cluster count')
    # A deterministic initial assignment avoids the extension's hidden RNG.
    initial=np.arange(len(valid),dtype=np.int32)%K
    labels,_,_=kcluster(mean[valid],nclusters=K,mask=available[valid].astype(np.int32),
                       dist='u',initialid=initial,npass=1)
    all_labels=np.full(N,-1,dtype=np.int64);all_labels[valid]=labels
    return all_labels,K


def cluster_loss(scores, labels, target_mask):
    """Scores are already probabilities; retain gradients and proper class indices."""
    B,N,_=target_mask.shape
    labels=torch.as_tensor(labels,device=scores.device).repeat(B)
    valid=(labels>=0)&target_mask.any(-1).reshape(-1)
    if not valid.any():return scores.sum()*0
    return F.nll_loss(scores[valid].clamp_min(1e-12).log(),labels[valid])


def train_features(model,data,adj):
    from .runner import tensor_batch
    L=data['metadata']['input_length'];end=int(data['bounds'][1])
    start=max(0,end-7*288);origins=np.arange(start+L-1,end,L)
    values=[];valid=[];was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            for k in range(0,len(origins),32):
                x,_,_,b=tensor_batch(data,origins[k:k+32],adj.device)
                feat=model.feature(x,adj).reshape(-1,len(data['sensor_ids']),model.args.gcn['out_channel'])
                values.append(feat.cpu().numpy())
                valid.append(b['x_observed'].any(1))
    finally:model.train(was_training)
    return np.concatenate(values),np.concatenate(valid)


def _js_distance(x,y):
    def hist(v):
        span=float(v.max()-v.min());v=(v-v.min())/span if span>0 else np.zeros_like(v)
        h=np.histogram(v,bins=10,range=(0,1))[0].astype(float)
        return h/h.sum()
    p,q=hist(x),hist(y);mid=(p+q)/2
    def kl(a):
        valid=a>0
        return float((a[valid]*np.log(a[valid]/mid[valid])).sum())
    return max(0.,.5*(kl(p)+kl(q)))**.5


def select_nodes(model,previous,current,previous_adj,current_adj,changes,method,cfg,rng):
    old,om=train_features(model,previous,previous_adj)
    new,nm=train_features(model,current,current_adj)
    scored=[]
    for i,j in zip(changes['retained_previous'],changes['retained_current']):
        a,b=old[om[:,i],i],new[nm[:,j],j]
        if not len(a) or not len(b):continue
        score=sum(_js_distance(a[:,f],b[:,f]) for f in range(a.shape[1]))
        scored.append((int(j),float(score)))
    count=int(.05*len(current['sensor_ids'])) if method=='TrafficStream' else int(cfg['infl_topk'])
    count=min(count,len(scored))
    ranked=sorted(scored,key=lambda x:(x[1],int(current['sensor_ids'][x[0]])))
    selected=set(map(int,changes['entered_current'])) if cfg.get('increase',True) else set()
    if count and cfg.get('detect',False):selected.update(i for i,_ in ranked[-count:])
    if count and cfg.get('replay',False):
        if cfg['replay_strategy']=='inforeplay':selected.update(i for i,_ in ranked[:count])
        elif cfg['replay_strategy']=='random':selected.update(rng.choice([i for i,_ in ranked],size=count,replace=True).tolist())
        else:raise ValueError('Unsupported replay policy')
    if method=='STKEC':
        deficit=int(.1*len(current['sensor_ids']))-len(selected)
        if deficit>0:
            remaining=sorted(set(range(len(current['sensor_ids'])))-selected)
            selected.update(rng.choice(remaining,size=deficit,replace=False).tolist())
    return np.array(sorted(selected),dtype=np.int64),{
        'scope':'last 7 days within each train split; feature histogram JS distance; stable-ID alignment',
        'scored_sensor_ids':[int(current['sensor_ids'][i]) for i,_ in scored],
        'scores':[s for _,s in scored],'topk':count,
        'previous_max_timestamp':int(previous['timestamps'][previous['bounds'][1]-1]),
        'current_max_timestamp':int(current['timestamps'][current['bounds'][1]-1])}


def fisher_state(model,data,adj,batch_size,loss_kind,max_batches=None):
    """Match legacy current-train Fisher estimation, now with explicit label masks."""
    from .runner import tensor_batch,masked_loss,split_origins
    params=dict(model.named_parameters());means={k:v.detach().clone() for k,v in params.items()}
    fisher={k:torch.zeros_like(v) for k,v in params.items()}
    origins=split_origins(data,'train')
    was_training=model.training;model.eval();used=0
    try:
        for j in range(0,len(origins),batch_size):
            if max_batches is not None and used>=max_batches:break
            x,y,m,_=tensor_batch(data,origins[j:j+batch_size],adj.device)
            if not m.any():continue
            output=model(x,adj);pred=output[0] if isinstance(output,tuple) else output
            loss=masked_loss(pred.reshape(y.shape),y,m,loss_kind)
            gradients=torch.autograd.grad(loss,tuple(params.values()),allow_unused=True)
            for (k,_),g in zip(params.items(),gradients):
                if g is not None:
                    if not torch.isfinite(g).all():raise ValueError('Nonfinite Fisher gradient')
                    fisher[k]+=g.detach().square()
            used+=1
    finally:model.train(was_training)
    if not used:raise ValueError('No observed training labels for Fisher estimation')
    return means,fisher,used


def ewc_penalty(model,means,fisher,weight,kind):
    terms=[]
    for name,p in model.named_parameters():
        factor=1e-5 if kind=='l2' else fisher[name]
        terms.append((factor*(p-means[name]).square()).sum())
    return (weight/2)*sum(terms)
