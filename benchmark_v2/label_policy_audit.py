"""Paired coverage and persistence diagnostics; bounded finite-flow integration.

This is a policy sensitivity diagnostic, not a trained model ranking. The same
finite-input persistence predictions are scored under both target masks. A second
diagnostic varies the input policy while scoring their common strict targets.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .pems import read_window
from .data import prepare_year,save_year,split_origins,window_batch
from .metrics import score
from .runner import ROOT,CONTINUAL,sha
from .integration_smoke import CONFIGS

GRAPH_SOURCE=Path('/import/kamen/9/z5440262/iona-runs/tsas-integration-20261003-v3/integration_output')


def main():
    tic=time.time();out=Path('label_policy_output');out.mkdir(exist_ok=False)
    years=[];rows=[];previous=set()
    for year,start in [(2005,'2005-03-24'),(2006,'2006-01-01'),(2007,'2007-01-01')]:
        raw=read_window('/mnt/scratch/CRUISE/Du/data/pems/raw','05',start,label_policy='finite_reported')
        finite=raw['flow_reported'];q=raw['percent_observed'];strict=finite & np.isfinite(q) & (q>0)
        masks={'finite_reported':finite,'positive_percent_observed':strict}
        data={name:prepare_year(raw['values'],mask,raw['sensor_ids'],raw['timestamps'],provenance=dict(raw['provenance'],label_policy=name)) for name,mask in masks.items()}
        cohorts={'all':np.arange(len(raw['sensor_ids']))}
        if previous:cohorts['new_to_previous']=np.array([i for i,s in enumerate(raw['sensor_ids']) if int(s) not in previous],dtype=int)
        row={'year':year,'sensor_ids':raw['sensor_ids'].tolist(),'coverage':{},'persistence':{},'source_files':raw['provenance']['sources']}
        for cohort,idx in cohorts.items():
            row['coverage'][cohort]={}
            for split,a,b in zip(['train','val','test'],data['finite_reported']['bounds'][:-1],data['finite_reported']['bounds'][1:]):
                present=raw['record_present'][a:b,idx];f=finite[a:b,idx];s=strict[a:b,idx];qual=q[a:b,idx];val=raw['values'][a:b,idx]
                row['coverage'][cohort][split]={'sensor_time_slots':int(present.size),'missing_record':int((~present).sum()),
                    'present_nonfinite_flow':int((present & ~f).sum()),'finite_reported':int(f.sum()),
                    'positive_percent_observed':int(s.sum()),'finite_quality_zero':int((f & (qual==0)).sum()),
                    'finite_quality_unknown':int((f & ~np.isfinite(qual)).sum()),
                    'finite_quality_negative':int((f & (qual<0)).sum()),
                    'finite_zero_flow':int((f & (val==0)).sum()),'strict_zero_flow':int((s & (val==0)).sum())}
        origin=split_origins(data['finite_reported'],'test')
        fb=window_batch(data['finite_reported'],origin);sb=window_batch(data['positive_percent_observed'],origin)
        preds={}
        for name,d in data.items():
            last=d['x'][origin]*d['metadata']['scaler_std']+d['metadata']['scaler_mean']
            preds[name]=np.repeat(last[:,None,:],12,axis=1)
        for cohort,idx in cohorts.items():
            common={name:score(p[:,:,idx],fb['y'][:,:,idx],sb['y_mask'][:,:,idx]) for name,p in preds.items()}
            filtering={name:score(preds['finite_reported'][:,:,idx],fb['y'][:,:,idx],batch['y_mask'][:,:,idx]) for name,batch in [('finite_reported',fb),('positive_percent_observed',sb)]}
            row['persistence'][cohort]={'same_predictions_different_target_masks':filtering,'same_strict_targets_different_input_policies':common}
        dp=out/f'{year}.npz';save_year(dp,data['finite_reported'])
        original=GRAPH_SOURCE/f'{year}_graph.npz'
        with np.load(original,allow_pickle=False) as z:assert np.array_equal(z['sensor_ids'],raw['sensor_ids'])
        years.append({'year':year,'data':str(dp.resolve()),'graph':str(original)})
        previous=set(raw['sensor_ids'].tolist());rows.append(row)
    (out/'coverage.json').write_text(json.dumps({'district':'PEMS05','rows':rows,'scope':'three full 31-day windows; chronological 60/20/20; persistence is descriptive, not a learned benchmark'},indent=2))
    methods=[]
    configs=dict(CONFIGS,EAC='eac.json',TrafficStream='trafficstream.json',STKEC='stkec.json')
    for method,filename in configs.items():
        cfg=json.loads((ROOT/'eac/conf/PEMS05'/filename).read_text())
        manifest={'method':method,'district':'PEMS05','purpose':'smoke','seed':42,'years':years,
                  'model_config':cfg,'epochs':2,'batch_size':16,'max_batches':4,'max_eval_origins':64,
                  'device':'cpu','output':str((out/method).resolve())}
        mp=out/f'{method}_manifest.json';mp.write_text(json.dumps(manifest,indent=2))
        subprocess.run([sys.executable,str(ROOT/'eac/main.py'),'--benchmark_manifest',str(mp)],check=True)
        result=json.loads((out/method/'results.json').read_text())
        assert result['status']=='passed' and len(result['years'])==3
        for y in result['years']:
            assert y['metrics']['timing_valid']
            assert y['metrics']['cohorts']['all']['overall']['observed_count']>0
            if y['year']>2005:
                assert y['metrics']['cohorts']['new_to_previous']['overall']['observed_count']>0
                if method in CONTINUAL:assert y['parent']['year']==y['year']-1
        methods.append(result)
        print('FINITE POLICY PASSED',method,flush=True)
    (out/'summary.json').write_text(json.dumps({'status':'passed','methods':methods,'seconds':time.time()-tic,
        'scope':'13 methods x3 PEMS05 years, finite reported labels; max8 steps/year, first64 evaluation origins; not publishable model accuracy',
        'coverage_sha256':sha(out/'coverage.json')},indent=2))


if __name__=='__main__':main()
