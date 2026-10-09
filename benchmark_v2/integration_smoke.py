"""Three real PEMS05 years, ten existing baselines; bounded CPU validation."""
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
from .pems import read_window
from .data import prepare_year,save_year
from .runner import ROOT,sha

CONFIGS={'Retrain':'retrain_st_pems05.json','Pretrain':'pretrain_st_pems05.json',
         'Online-AN':'oneline_st_an_pems05.json','Online-NN':'oneline_st_nn_pems05.json',
         'DCRNN':'retrain_dcrnn_pems05.json','ASTGCN':'retrain_astgnn_pems05.json',
         'TGCN':'retrain_tgcn_pems05.json','PECPM':'pecpm_pems05.json',
         'STRAP':'strap_pems05.json','ST-TTC':'sttc_pems05.json'}


def main():
    tic=time.time();out=Path('integration_output');out.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'benchmark_v2/tests'))
    with (out/'tests.txt').open('w') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    stats={'tests':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'passed':tests.wasSuccessful()}
    (out/'tests.json').write_text(json.dumps(stats,indent=2))
    if not tests.wasSuccessful():raise RuntimeError('Integration regression tests failed')
    years=[];coverage=[]
    for year,start in [(2005,'2005-03-24'),(2006,'2006-01-01'),(2007,'2007-01-01')]:
        raw=read_window('/mnt/scratch/CRUISE/Du/data/pems/raw','05',start,label_policy='positive_percent_observed')
        data=prepare_year(raw['values'],raw['observed'],raw['sensor_ids'],raw['timestamps'],provenance=raw['provenance'])
        dp=out/f'{year}.npz';save_year(dp,data)
        original=Path(f'/mnt/scratch/CRUISE/Du/code/evo-xxltraffic/xxltrafficdata/preprocessed/pems05/graph/{year}_adj.npz')
        with np.load(original,allow_pickle=False) as z:adj=z['x']
        assert adj.shape==(len(raw['sensor_ids']),len(raw['sensor_ids']))
        gp=out/f'{year}_graph.npz'
        with gp.open('xb') as f:np.savez(f,x=adj,sensor_ids=raw['sensor_ids'],metadata=np.array(json.dumps({'order_status':'inherited_sorted_anchor_smoke_only','original_path':str(original),'original_sha256':sha(original)})))
        years.append({'year':year,'data':str(dp.resolve()),'graph':str(gp.resolve())})
        coverage.append({'year':year,'sensors':len(raw['sensor_ids']),'present':int(raw['record_present'].sum()),'observed':int(raw['observed'].sum()),'observed_zero':int((raw['observed']&(raw['values']==0)).sum()),'splits':{s:int(data['observed'][a:b].sum()) for s,a,b in zip(['train','val','test'],data['bounds'][:-1],data['bounds'][1:])}})
    summaries=[]
    for method,filename in CONFIGS.items():
        conf=json.loads((ROOT/'eac/conf/PEMS05'/filename).read_text())
        # Actual architecture/hyperparameters from existing configs; only smoke budget is capped.
        manifest={'method':method,'district':'PEMS05','purpose':'smoke','seed':42,'years':years,
                  'model_config':conf,'epochs':2,'batch_size':16,'max_batches':4,'max_eval_origins':64,
                  'device':'cpu','output':str((out/method).resolve())}
        mp=out/f'{method}_manifest.json';mp.write_text(json.dumps(manifest,indent=2))
        print('RUN',method,flush=True)
        subprocess.run([sys.executable,str(ROOT/'eac/main.py'),'--benchmark_manifest',str(mp)],check=True)
        r=json.loads((out/method/'results.json').read_text())
        assert len(r['years'])==3 and r['status']=='passed'
        for row in r['years']:
            assert row['metrics']['cohorts']['all']['overall']['observed_count']>0
            assert row['metrics']['timing_valid']
            if row['year']>2005 and method in ('Pretrain','Online-AN','Online-NN','PECPM','STRAP'):
                assert row['parent']['year']==row['year']-1
            if method=='Pretrain' and row['year']>2005:assert row['optimizer_steps']==0
            if method=='ST-TTC':assert row['metrics']['calibration_updates']>0
            if row['year']==2007:
                assert row['metrics']['cohorts']['new_to_previous']['overall']['observed_count']==0
                assert row['metrics']['cohorts']['new_to_previous']['overall']['MAE'] is None
        summaries.append(r)
    result={'status':'passed','purpose':'bounded integration smoke, not publishable accuracy',
            'tests':stats,'coverage':coverage,'methods':summaries,'seconds':time.time()-tic,
            'limitations':['Graph order inherited from sorted anchor policy; full provenance verification pending',
                           'Only first 64 validation/test origins and up to 8 optimizer steps per trained year',
                           'Existing repository backbone variants; not a new audit of equivalence to upstream methods',
                           'EAC, TrafficStream selection/replay and STKEC not yet integrated',
                           'No full benchmark run or full annual raw rebuild']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'passed','methods':len(summaries),'method_years':sum(len(x['years']) for x in summaries),'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
