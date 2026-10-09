"""Observation-aware runner for existing TSAS baselines.

Invoked via eac/main.py --benchmark_manifest FILE. Old caches/checkpoints are
never accepted here. Unsupported continual strategies fail explicitly.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_year, split_origins, window_batch
from .identity import transition
from .metrics import MetricAccumulator
from .online import LabelDelayQueue
from .calibration import FRPlusModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eac/src'))
from model.model import (TrafficStream_Model, DCRNN_Model, ASTGNN_Model,
                         TGCN_Model, PECPM_Model, RAP_Model, STTTC_Model, EAC_Model, STKEC_Model)
from .special import bind_eac_ids, daily_clusters, cluster_loss, select_nodes, fisher_state, ewc_penalty

MODELS = {'Retrain': TrafficStream_Model, 'Pretrain': TrafficStream_Model,
          'Online-AN': TrafficStream_Model, 'Online-NN': TrafficStream_Model,
          'DCRNN': DCRNN_Model, 'ASTGCN': ASTGNN_Model, 'TGCN': TGCN_Model,
          'PECPM': PECPM_Model, 'STRAP': RAP_Model, 'ST-TTC': STTTC_Model,
          'EAC': EAC_Model, 'TrafficStream': TrafficStream_Model, 'STKEC': STKEC_Model}
CONTINUAL = {'Pretrain', 'Online-AN', 'Online-NN', 'PECPM', 'STRAP', 'EAC', 'TrafficStream', 'STKEC'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(model):
    """Include algorithm state not registered as a parameter/buffer."""
    state = {'model_state_dict': copy.deepcopy(model.state_dict()), 'extra': {}}
    if isinstance(model, PECPM_Model):
        state['extra']['historical_patterns'] = copy.deepcopy(model.historical_patterns)
    if isinstance(model, RAP_Model):
        state['extra'].update(current_year=model.current_year,
                              pattern_initialized=model.pattern_initialized,
                              cache=copy.deepcopy(model.strap.pattern_manager._cache))
    if isinstance(model,EAC_Model):
        bank=copy.deepcopy(getattr(model,'sensor_bank',{}))
        for i,sid in enumerate(getattr(model,'sensor_ids',[])):bank[int(sid)]=model.U[i].detach().clone()
        state['extra'].update(sensor_ids=list(getattr(model,'sensor_ids',[])),sensor_bank=bank)
    return state


def restore(model, state):
    # RAP lazily creates a projector; instantiate it before strict restoration.
    if isinstance(model, RAP_Model) and 'strap.projector.weight' in state['model_state_dict']:
        w = state['model_state_dict']['strap.projector.weight']
        model.strap._ensure_projector(w.shape[0], next(model.parameters()).device)
    if isinstance(model,EAC_Model):
        source=state['model_state_dict']['U']
        if model.U.shape!=source.shape:
            model.U=torch.nn.Parameter(torch.empty_like(source,device=next(model.parameters()).device))
        model.num_nodes=source.shape[0]
        model.sensor_ids=list(state['extra']['sensor_ids'])
        model.sensor_bank=copy.deepcopy(state['extra']['sensor_bank'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    if isinstance(model, PECPM_Model):
        # torch.load(map_location=device) also relocates auxiliary history;
        # PECPM appends CPU rows and keeps its bounded archive on the host.
        history = state['extra']['historical_patterns']
        model.historical_patterns = None if history is None else history.detach().cpu().clone()
    if isinstance(model, RAP_Model):
        e = state['extra']
        model.strap.pattern_manager._cache = copy.deepcopy(e['cache'])
        model.current_year = e['current_year']
        model.pattern_initialized = e['pattern_initialized']
        if model.pattern_initialized and not model.strap.switch_to_year(model.current_year):
            raise ValueError('Missing STRAP pattern library in checkpoint')


def normalized_adj(raw, device):
    return torch.as_tensor(raw / (raw.sum(1, keepdims=True) + 1e-6),
                           dtype=torch.float32, device=device)


def node_context(raw_adj, centers, hops):
    chosen = set(map(int, centers))
    connected = (raw_adj != 0) | (raw_adj.T != 0)
    for _ in range(hops):
        if chosen:
            chosen |= set(np.flatnonzero(connected[list(chosen)].any(0)).tolist())
    nodes = np.array(sorted(chosen), dtype=np.int64)
    return nodes, np.isin(nodes, centers)


def tensor_batch(data, origins, device, nodes=None):
    b = window_batch(data, origins)
    if nodes is None: nodes = np.arange(len(data['sensor_ids']))
    def convert(name):
        a = b[name][:, :, nodes].transpose(0, 2, 1).copy()
        return torch.from_numpy(a).to(device)
    x, y, m = convert('x'), convert('y'), convert('y_mask')
    a = convert('x_available')
    return SimpleNamespace(x=x.reshape(-1, x.shape[-1]),
                           x_available=a.reshape(-1, a.shape[-1])), y, m, b


def masked_loss(pred, target, mask, kind='mse'):
    if not mask.any(): return None
    if not torch.isfinite(pred[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite prediction/target at an observed label')
    if kind == 'mse': return F.mse_loss(pred[mask], target[mask])
    if kind == 'huber': return F.smooth_l1_loss(pred[mask], target[mask])
    raise ValueError('Unsupported training loss: ' + kind)


def evaluate(model, data, adj, split, cohorts, *, cap=None, ttc=None, loss_nodes=None):
    """One origin at a time: no cross-origin attention/pattern lookahead.

    All cohorts share the same prediction, including TTC corrections. Validation
    and test leave the pre-evaluation model/pattern state unchanged.
    """
    saved = snapshot(model)
    was_training = model.training
    model.eval()
    H = data['metadata']['output_length']
    accum = {name: {str(h): MetricAccumulator() for h in range(1, H+1)} for name in cohorts}
    origins = split_origins(data, split)
    if cap is not None: origins = origins[:cap]
    calibration = optimizer = None
    queue = LabelDelayQueue()
    updates = []
    prediction_hash = hashlib.sha256()
    prediction_trace = []
    if ttc is not None:
        calibration = FRPlusModule(len(data['sensor_ids']), H//2+1, ttc['groups']).to(adj.device)
        optimizer = torch.optim.Adam(calibration.parameters(), lr=ttc['lr'])
    try:
        for origin in origins:
            x, y, mask, b = tensor_batch(data, [origin], adj.device)
            now = int(b['origin_time'][0])
            if calibration is not None:
                for base_old, target_old, mask_old, target_end in queue.advance(now):
                    calibration.train()
                    corr = calibration(base_old.unsqueeze(1)).squeeze(1)
                    loss = masked_loss(corr, target_old, mask_old)
                    if loss is not None:
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        updates.append({'origin': now, 'label_end': target_end})
            with torch.no_grad():
                output = model(x, adj)
                pred = (output[0] if isinstance(output,tuple) else output).reshape(y.shape)
                base = pred.detach()
                if calibration is not None:
                    calibration.eval()
                    pred = calibration(base.unsqueeze(1)).squeeze(1)
                p, target, observed = pred.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy()
            prediction_hash.update(p.tobytes())
            prediction_trace.append(hashlib.sha256(p.tobytes()).hexdigest())
            if loss_nodes is not None: observed &= loss_nodes[None, :, None]
            for name, indices in cohorts.items():
                for h in range(1, H+1):
                    accum[name][str(h)].update(p[:, indices, :h], target[:, indices, :h], observed[:, indices, :h])
            if calibration is not None:
                end = int(b['target_times'][0, -1])
                queue.add_after_prediction(end, (base, y.detach(), mask.detach(), end))
    finally:
        restore(model, saved)
        model.train(was_training)
    result = {}
    for name, metrics in accum.items():
        prefixes = {h: a.result() for h, a in metrics.items()}
        # Preserve and label historical aggregation; expose pooled score separately.
        legacy = {k: float(np.mean([r[k] for r in prefixes.values()]))
                  if all(r[k] is not None for r in prefixes.values()) else None
                  for k in ('MAE', 'RMSE', 'MAPE')}
        result[name] = {'prefixes': {h: prefixes[h] for h in ('3', '6', '12') if h in prefixes},
                        'overall': prefixes[str(H)], 'legacy_prefix_mean': legacy,
                        'sensor_count': len(cohorts[name])}
    return {'cohorts': result, 'origins': len(origins), 'prediction_sha256': prediction_hash.hexdigest(),
            'prediction_trace': prediction_trace, 'calibration_updates': len(updates), 'calibration_timing': updates,
            'timing_valid': all(r['label_end'] <= r['origin'] for r in updates)}


def validate_manifest(m):
    if m['method'] not in MODELS: raise ValueError('Method not yet integrated: '+m['method'])
    if m['purpose'] not in ('smoke', 'benchmark'): raise ValueError('Explicit run purpose required')
    years = [a['year'] for a in m['years']]
    if not years or years != list(range(years[0], years[-1]+1)):
        raise ValueError('Years must be chronological and contiguous')
    if m['district'] == 'PEMS04' and years[0] < 2002: raise ValueError('PEMS04 begins in 2002')
    if m['purpose'] == 'benchmark':
        if any(m.get(k) is not None for k in ('max_batches', 'max_eval_origins')):
            raise ValueError('Benchmark must not use smoke caps')
        if not m.get('protocol_frozen'): raise ValueError('Benchmark protocol has not been frozen')
    if m['epochs'] < 1 or m['batch_size'] < 1: raise ValueError('Invalid training budget')
    patience = m.get('early_stopping_patience')
    if patience is not None and (not isinstance(patience, int) or patience < 1):
        raise ValueError('Early stopping patience must be a positive integer')


def run(m):
    validate_manifest(m)
    out = Path(m['output']); out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(m, indent=2))
    torch.set_num_threads(int(m.get('cpu_threads', 1)))
    device = torch.device(m.get('device', 'cpu'))
    seed = int(m['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    protocol = {'schema': 'tsas-runner-v2', 'dataset_schema': 'tsas-observed-v2',
                'evaluation': 'strictly chronological single origin; explicit target mask',
                'metrics': 'cumulative 1..h; legacy_prefix_mean and overall separately named',
                'validation_selection': 'same full-graph observed MAE over all leads for every method; Online-NN training remains new-node-only',
                'model_state': 'weights plus pattern libraries; evaluation state isolated',
                'config': m['model_config'], 'method': m['method'], 'seed': seed,
                'epochs': m['epochs'], 'batch_size': m['batch_size'],
                'max_batches': m.get('max_batches'), 'max_eval_origins': m.get('max_eval_origins'),
                'early_stopping_patience': m.get('early_stopping_patience'),
                'frozen_protocol_sha256': m.get('frozen_protocol_sha256'),
                'purpose': m['purpose'], 'specialist_protocol': 'train-only stable-ID drift; legacy current-train masked Fisher; STKEC first-year differentiable cluster NLL; EAC stable-ID bank and frozen backbone after first year'}
    protocol_hash = digest(protocol)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    previous_path = None; previous_ids = []; historical_ids = set(); results = []
    previous_data = previous_adj = None
    cluster_count = None
    rng = np.random.default_rng(seed)
    for entry in m['years']:
        print(json.dumps({'event':'year_start','method':m['method'],'year':entry['year']}),flush=True)
        tic = time.time(); year = entry['year']; data = load_year(entry['data'])
        provenance = data['metadata']['provenance']
        if provenance['district'] != m['district'] or int(provenance['window_start'][:4]) != year:
            raise ValueError('Dataset district/year mismatch')
        with np.load(entry['graph'], allow_pickle=False) as z:
            raw_adj, graph_ids = z['x'], z['sensor_ids']
            graph_provenance = json.loads(str(z['metadata']))
        if not np.array_equal(graph_ids, data['sensor_ids']): raise ValueError('Graph/data ID order mismatch')
        if raw_adj.shape != (len(graph_ids), len(graph_ids)) or not np.isfinite(raw_adj).all():
            raise ValueError('Invalid adjacency')
        if m['purpose'] == 'benchmark' and graph_provenance['order_status'] != 'verified_ids':
            raise ValueError('Unverified graph node order is smoke-only')
        ids = data['sensor_ids']; changes = transition(previous_ids, ids, historical_ids)
        cohorts = {'all': np.arange(len(ids)), 'new_to_previous': changes['entered_current'] if previous_path else np.array([],dtype=int),
                   'first_ever': changes['first_seen_current'] if previous_path else np.array([],dtype=int),
                   'returning': changes['returning_current'], 'retained': changes['retained_current']}
        cfg = dict(m['model_config'], year=year, begin_year=m['years'][0]['year'], path=None, benchmark_strict=True, base_node_size=len(ids))
        cluster_labels=None
        if m['method']=='STKEC':
            if cluster_count is None:cluster_labels,cluster_count=daily_clusters(data,cfg['cluster'])
            cfg['cluster']=cluster_count
        args = SimpleNamespace(**cfg)
        model = MODELS[m['method']](args).to(device)
        parent = None
        if previous_path is not None and m['method'] in CONTINUAL:
            checkpoint = torch.load(previous_path, map_location=device, weights_only=False)
            expected = dict(district=m['district'], year=year-1, method=m['method'], seed=seed,
                            protocol_hash=protocol_hash, data_hash=sha(m['years'][len(results)-1]['data']),
                            graph_hash=sha(m['years'][len(results)-1]['graph']))
            if any(checkpoint['metadata'].get(k) != v for k,v in expected.items()):
                raise ValueError('Previous-year checkpoint provenance mismatch')
            restore(model, checkpoint)
            parent = {'path': str(previous_path), 'sha256': sha(previous_path), 'year': year-1}
            if isinstance(model, RAP_Model): model.set_year(year)
        if isinstance(model,EAC_Model):
            bind_eac_ids(model,ids)
            if previous_path is not None:
                for name,param in model.named_parameters():
                    if name not in ('U','V'):param.requires_grad=False
        adj = normalized_adj(raw_adj, device)
        skip = previous_path is not None and m['method'] == 'Pretrain'
        skip_reason = 'frozen_pretrain' if skip else None
        centers = np.arange(len(ids)); nodes = centers; center_mask = np.ones(len(ids),dtype=bool)
        if previous_path is not None and m['method'] == 'Online-NN':
            centers = changes['entered_current']
            skip = not len(centers)
            skip_reason = 'no_new_nodes' if skip else None
            if not skip:
                a,b = data['bounds'][:2]
                if not data['observed'][a+data['metadata']['input_length']:b, centers].any():
                    skip = True
                    skip_reason = 'no_observed_new_node_training_targets'
                else:
                    nodes, center_mask = node_context(raw_adj, centers, int(cfg.get('num_hops',2)))
        selection=None
        if previous_path is not None and m['method'] in ('TrafficStream','STKEC'):
            centers,selection=select_nodes(model,previous_data,data,previous_adj,adj,changes,m['method'],cfg,rng)
            skip=not len(centers);skip_reason='no_selected_nodes' if skip else None
            if not skip:
                if not data['observed'][data['metadata']['input_length']:data['bounds'][1],centers].any():
                    skip=True;skip_reason='no_observed_selected_training_targets'
                else:nodes,center_mask=node_context(raw_adj,centers,int(cfg.get('num_hops',2)))
        train_adj = normalized_adj(raw_adj[np.ix_(nodes,nodes)], device)
        train_data = dict(data)
        for key in ('x','y','observed','available'): train_data[key] = data[key][:,nodes]
        train_data['sensor_ids'] = ids[nodes]
        steps = 0; history = []; best = float('inf'); best_state = None
        fisher_batches=0;consolidation=None;auxiliary_steps=0
        if not skip and previous_path is not None and cfg.get('ewc',False):
            means,fisher,fisher_batches=fisher_state(model,data,adj,m['batch_size'],cfg.get('loss','mse'),m.get('max_batches'))
            consolidation=(means,fisher)
        if not skip:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
            stale_epochs = 0
            for epoch in range(m['epochs']):
                model.train(); origins = rng.permutation(split_origins(data,'train'))
                if m.get('max_batches') is not None: origins=origins[:m['max_batches']*m['batch_size']]
                count = 0; loss_sum = 0.
                for j in range(0,len(origins),m['batch_size']):
                    x,y,mask,_ = tensor_batch(train_data,origins[j:j+m['batch_size']],device)
                    mask &= torch.as_tensor(center_mask, device=device)[None,:,None]
                    if not mask.any(): continue
                    output = model(x,train_adj)
                    pred = (output[0] if isinstance(output,tuple) else output).reshape(y.shape)
                    loss = masked_loss(pred,y,mask,cfg.get('loss','mse'))
                    if m['method']=='STKEC' and cluster_labels is not None:
                        loss=loss+.1*cluster_loss(output[1],cluster_labels[nodes],mask)
                        auxiliary_steps+=1
                    if consolidation is not None:
                        loss=loss+ewc_penalty(model,*consolidation,cfg['ewc_lambda'],cfg.get('ewc_strategy','ewc'))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite total optimization loss')
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    steps += 1; n=int(mask.sum()); count+=n; loss_sum+=float(loss.detach())*n
                val = evaluate(model,data,adj,'val',{'all':np.arange(len(ids))},
                               cap=m.get('max_eval_origins'))
                score = val['cohorts']['all']['overall']['MAE']
                history.append({'epoch':epoch,'train_loss':loss_sum/count if count else None,'validation_MAE':score})
                if score is not None and score < best:
                    best=score; best_state=snapshot(model)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                print(json.dumps({'event':'epoch_end','method':m['method'],'year':year,
                                  'epoch':epoch+1,'validation_MAE':score,'optimizer_steps':steps}),flush=True)
                if m.get('early_stopping_patience') is not None and stale_epochs >= m['early_stopping_patience']:
                    break
            if best_state is None or not steps: raise ValueError('No trainable/valid observed labels in yearly task')
            restore(model,best_state)
        state=snapshot(model)
        state['metadata']={'district':m['district'],'year':year,'method':m['method'],'seed':seed,
                           'protocol_hash':protocol_hash,'data_hash':sha(entry['data']),'graph_hash':sha(entry['graph']),
                           'parent':parent,'training_skipped':skip,'skip_reason':skip_reason,'purpose':m['purpose']}
        checkpoint_path=out/f'{year}.pkl'
        with checkpoint_path.open('xb') as f: torch.save(state,f)
        # Reinstantiate from the saved artifact, including non-parameter state.
        reloaded=MODELS[m['method']](SimpleNamespace(**cfg)).to(device)
        restore(reloaded,torch.load(checkpoint_path,map_location=device,weights_only=False))
        ttc={'groups':cfg.get('ttc_groups',4),'lr':cfg.get('ttc_lr',1e-4)} if m['method']=='ST-TTC' else None
        metrics=evaluate(reloaded,data,adj,'test',cohorts,cap=m.get('max_eval_origins'),ttc=ttc)
        row={'year':year,'sensors':len(ids),'changes':{k:v.tolist() for k,v in changes.items()},
             'optimizer_steps':steps,'training_skipped':skip,'skip_reason':skip_reason,'validation_scope':'all_observed_nodes','history':history,'metrics':metrics,
             'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),'parent':parent,
             'seconds':time.time()-tic,'graph_provenance':graph_provenance,'selection':selection,'selected_sensor_ids':ids[centers].tolist(),'fisher_batches':fisher_batches,'auxiliary_steps':auxiliary_steps,'cluster_count':cluster_count}
        row['epochs_completed'] = len(history)
        row['stopped_early'] = bool(not skip and len(history) < m['epochs'])
        results.append(row)
        (out/'results.json').write_text(json.dumps({'status':'running','method':m['method'],'purpose':m['purpose'],'years':results},indent=2))
        previous_path=checkpoint_path;previous_ids=ids.tolist();historical_ids.update(previous_ids)
        previous_data=data;previous_adj=adj
        print(json.dumps({'event':'year_complete','method':m['method'],'year':year,
                          'epochs_completed':len(history),'seconds':row['seconds']}),flush=True)
    final={'status':'passed','method':m['method'],'purpose':m['purpose'],'protocol_hash':protocol_hash,'years':results}
    (out/'results.json').write_text(json.dumps(final,indent=2))
    return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark_manifest',required=True)
    args=parser.parse_args()
    result=run(json.loads(Path(args.benchmark_manifest).read_text()))
    print(json.dumps({'status':result['status'],'method':result['method'],'years':len(result['years'])}),flush=True)

if __name__=='__main__': main()
