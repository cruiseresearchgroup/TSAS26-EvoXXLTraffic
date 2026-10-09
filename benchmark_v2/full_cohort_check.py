"""Read-only follow-up: full 2006 test split for saved smoke checkpoints.

The first 64 test origins have no observed new-node labels. This check therefore
covers the complete test split, without retraining or replacing any artifact.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .runner import MODELS,restore,load_year,normalized_adj,evaluate,sha
from .pems import read_window

SOURCE=Path('/import/kamen/9/z5440262/iona-runs/tsas-integration-20261003-v3/integration_output')


def main():
    tic=time.time();torch.set_num_threads(1)
    original=json.loads((SOURCE/'summary.json').read_text())
    assert original['status']=='passed'
    data=load_year(SOURCE/'2006.npz');previous=load_year(SOURCE/'2005.npz')
    with np.load(SOURCE/'2006_graph.npz',allow_pickle=False) as z:raw_adj=z['x']
    adj=normalized_adj(raw_adj,'cpu')
    new=np.flatnonzero(~np.isin(data['sensor_ids'],previous['sensor_ids']))
    result={'purpose':'full-test coverage of existing smoke checkpoints, not trained benchmark accuracy','methods':[],'raw_new_node_checks':[]}
    for entry in original['methods']:
        method=entry['method'];manifest=json.loads((SOURCE/f'{method}_manifest.json').read_text())
        cfg=dict(manifest['model_config'],year=2006,begin_year=2005,path=None,benchmark_strict=True)
        model=MODELS[method](SimpleNamespace(**cfg))
        cp=SOURCE/method/'2006.pkl'
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert state['metadata']['year']==2006 and state['metadata']['method']==method
        assert state['metadata']['data_hash']==sha(SOURCE/'2006.npz')
        restore(model,state)
        ttc={'groups':cfg.get('ttc_groups',4),'lr':cfg.get('ttc_lr',1e-4)} if method=='ST-TTC' else None
        r=evaluate(model,data,adj,'test',{'all':np.arange(len(data['sensor_ids'])),'new_to_previous':new},ttc=ttc)
        assert r['cohorts']['new_to_previous']['overall']['observed_count']>0
        assert r['cohorts']['new_to_previous']['overall']['MAE'] is not None
        assert r['timing_valid']
        if method=='ST-TTC':assert r['calibration_updates']>0
        # Keep timing extrema/check plus hashes; complete traces remain unnecessary here.
        if r['calibration_timing']:
            r['minimum_label_delay_seconds']=min(x['origin']-x['label_end'] for x in r['calibration_timing'])
        del r['prediction_trace'];del r['calibration_timing']
        result['methods'].append({'method':method,'checkpoint_sha256':sha(cp),'evaluation':r})
        print('FULL TEST PASSED',method,flush=True)
    prev=set(previous['sensor_ids'].tolist())
    for year in (2006,2007):
        raw=read_window('/mnt/scratch/CRUISE/Du/data/pems/raw','05',f'{year}-01-01',label_policy='positive_percent_observed')
        new_idx=[i for i,s in enumerate(raw['sensor_ids']) if int(s) not in prev]
        T=len(raw['values']);bounds=[0,int(T*.6),int(T*.8),T]
        checks=[]
        for i in new_idx:
            item={'sensor_id':int(raw['sensor_ids'][i]),'splits':{}}
            for name,a,b in zip(('train','val','test'),bounds[:-1],bounds[1:]):
                item['splits'][name]={'present':int(raw['record_present'][a:b,i].sum()),
                                      'finite_reported':int(raw['flow_reported'][a:b,i].sum()),
                                      'positive_percent_observed':int(raw['observed'][a:b,i].sum())}
            checks.append(item)
        result['raw_new_node_checks'].append({'year':year,'nodes':checks})
        prev=set(raw['sensor_ids'].tolist())
    result.update(status='passed',seconds=time.time()-tic)
    Path('full_cohort_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'passed','methods':len(result['methods']),'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
