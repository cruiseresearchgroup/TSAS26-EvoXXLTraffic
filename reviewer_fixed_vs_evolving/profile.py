"""Largest requested task: one TRAIN epoch, full VALIDATION; never TEST."""
import argparse
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark_v2 import runner
from .common import sha, save
from .run_chain import manifest


def profile(root, population, label):
    root=Path(root).resolve()
    m=manifest(root,'PEMS06','Retrain',population,47)
    torch.set_num_threads(2);torch.manual_seed(47);np.random.seed(47)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert torch.__version__.split('+')[0]=='2.2.1'
    entry=m['years'][-1];data=runner.load_year(entry['data'])
    with np.load(entry['graph'],allow_pickle=False) as z:raw=z['x']
    adj=runner.normalized_adj(raw,'cuda:0')
    cfg=dict(m['model_config'],year=2025,begin_year=2018,path=None,benchmark_strict=True,base_node_size=len(data['sensor_ids']))
    model=runner.MODELS['Retrain'](SimpleNamespace(**cfg)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.03)
    assert optimizer.defaults['weight_decay']==.01
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    model.train();origins=np.random.default_rng(47).permutation(runner.split_origins(data,'train'));steps=0
    for j in range(0,len(origins),128):
        x,y,mask,_=runner.tensor_batch(data,origins[j:j+128],adj.device)
        pred=model(x,adj).reshape(y.shape);loss=runner.masked_loss(pred,y,mask)
        assert loss is not None and torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
    torch.cuda.synchronize();train_seconds=time.monotonic()-start
    result=runner.evaluate(model,data,adj,'val',{'all':np.arange(len(data['sensor_ids']))})
    assert result['origins']==1763
    assert result['cohorts']['all']['overall']['MAE'] is not None
    torch.cuda.synchronize()
    out=dict(status='PASS',label=label,population=population,scope='TRAIN/VALIDATION only',
        nodes=len(data['sensor_ids']),optimizer_steps=steps,train_seconds=train_seconds,
        seconds=time.monotonic()-start,validation_origins=result['origins'],
        validation_MAE=result['cohorts']['all']['overall']['MAE'],
        gpu=torch.cuda.get_device_name(0),gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        torch=torch.__version__,cuda=torch.version.cuda,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(root/'profile'/f'{label}.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--population',choices=['E','F'],required=True);p.add_argument('--label',required=True)
    a=p.parse_args();profile(a.root,a.population,a.label)
