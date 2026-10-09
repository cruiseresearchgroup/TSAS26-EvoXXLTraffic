"""Bounded real-data CPU smoke using the repository's existing STGNN baseline.

Two epochs and 16 batches/epoch are a wiring check, not publishable accuracy.
Uses v2 masked labels and train-only normalization; never reads old FastData.
"""
import json,sys,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .pems import read_window
from .data import prepare_year,save_year,split_origins,window_batch
from .metrics import MetricAccumulator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eac/src'))
from model.model import STGNN_Model


def main():
    torch.set_num_threads(1);torch.manual_seed(42);rng=np.random.default_rng(42)
    start=time.time()
    raw=read_window('/mnt/scratch/CRUISE/Du/data/pems/raw','05','2005-03-24',label_policy='positive_percent_observed')
    data=prepare_year(raw['values'],raw['observed'],raw['sensor_ids'],raw['timestamps'],provenance=raw['provenance'])
    data_hash=save_year('pems05_2005_smoke_v2.npz',data)
    gp=Path('/mnt/scratch/CRUISE/Du/code/evo-xxltraffic/xxltrafficdata/preprocessed/pems05/graph/2005_adj.npz')
    with np.load(gp,allow_pickle=False) as z:adj=z['x']
    N=len(raw['sensor_ids']);assert adj.shape==(N,N)
    adj=torch.tensor(adj/(adj.sum(1,keepdims=True)+1e-6),dtype=torch.float32)
    args=SimpleNamespace(gcn={'in_channel':12,'hidden_channel':64,'out_channel':12},
                         tcn={'in_channel':1,'out_channel':1,'kernel_size':3,'dilation':1},y_len=12,dropout=0.)
    model=STGNN_Model(args);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def batch(origins):
        b=window_batch(data,origins)
        x=torch.from_numpy(b['x'].transpose(0,2,1).copy()).reshape(-1,12)
        y=torch.from_numpy(b['y'].transpose(0,2,1).copy()).reshape(-1,12)
        mask=torch.from_numpy(b['y_mask'].transpose(0,2,1).copy()).reshape(-1,12)
        return x,y,mask
    def evaluate(split):
        model.eval();acc=MetricAccumulator();origins=split_origins(data,split)
        with torch.no_grad():
            for j in range(0,len(origins),64):
                x,y,m=batch(origins[j:j+64]);pred=model(SimpleNamespace(x=x),adj)
                acc.update(pred.numpy(),y.numpy(),m.numpy())
        return acc.result()
    history=[];best=float('inf');steps=0
    for epoch in range(2):
        model.train();origins=rng.permutation(split_origins(data,'train'));losses=[]
        for j in range(0,min(len(origins),16*32),32):
            x,y,m=batch(origins[j:j+32])
            if not m.any():continue
            pred=model(SimpleNamespace(x=x),adj)
            if not torch.isfinite(pred[m]).all():raise ValueError('Nonfinite prediction')
            loss=((pred[m]-y[m])**2).mean();optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append(float(loss.detach()));steps+=1
        val=evaluate('val');history.append({'epoch':epoch,'train_masked_MSE':float(np.mean(losses)),'validation':val})
        if val['MAE'] is not None and val['MAE']<best:
            best=val['MAE'];torch.save({'model_state_dict':model.state_dict(),'district':'PEMS05','year':2005,'data_hash':data_hash,'purpose':'smoke_only'},'smoke_best.pkl')
    checkpoint=torch.load('smoke_best.pkl',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict']);test=evaluate('test')
    assert steps>0 and test['observed_count']>0 and test['MAE'] is not None
    result={'status':'passed','purpose':'real-data wiring smoke; not benchmark results','model':'existing STGNN_Model','district':'PEMS05','year':2005,'sensor_ids':raw['sensor_ids'].tolist(),'seed':42,'epochs':2,'max_batches_per_epoch':16,'optimizer_steps':steps,'data_sha256':data_hash,'graph_sha256':hashlib.sha256(gp.read_bytes()).hexdigest(),'mask_policy':raw['provenance']['label_policy'],'raw_present_count':int(raw['record_present'].sum()),'observed_count':int(raw['observed'].sum()),'observed_zero_count':int((raw['observed'] & (raw['values']==0)).sum()),'history':history,'test':test,'seconds':time.time()-start,'limitations':['Only PEMS05 one year; graph ID order inherited from sorted anchor notebook, not a full graph provenance repair','No continual-method integration or TTC calibration tested','Prototype pooled-horizon metric differs from historical prefix-averaged Avg; protocol must be disclosed']}
    Path('smoke_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
