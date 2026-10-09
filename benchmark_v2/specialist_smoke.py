"""Run all 13 integrated strategies against the existing three-year smoke caches."""
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from .runner import ROOT,CONTINUAL
from .integration_smoke import CONFIGS

SOURCE=Path('/import/kamen/9/z5440262/iona-runs/tsas-integration-20261003-v3/integration_output')


def main():
    tic=time.time();out=Path('specialist_output');out.mkdir(exist_ok=False)
    tests=unittest.defaultTestLoader.discover(str(ROOT/'benchmark_v2/tests'))
    with (out/'tests.txt').open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(tests)
    stats={'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful()}
    (out/'tests.json').write_text(json.dumps(stats,indent=2))
    if not r.wasSuccessful():raise RuntimeError('Specialist regression failed')
    configs=dict(CONFIGS,EAC='eac.json',TrafficStream='trafficstream.json',STKEC='stkec.json')
    rows=[]
    base=json.loads((SOURCE/'Retrain_manifest.json').read_text())
    for method,filename in configs.items():
        m=dict(base,method=method,model_config=json.loads((ROOT/'eac/conf/PEMS05'/filename).read_text()),output=str((out/method).resolve()))
        mp=out/f'{method}_manifest.json';mp.write_text(json.dumps(m,indent=2))
        print('RUN',method,flush=True)
        subprocess.run([sys.executable,str(ROOT/'eac/main.py'),'--benchmark_manifest',str(mp)],check=True)
        result=json.loads((out/method/'results.json').read_text())
        assert result['status']=='passed' and len(result['years'])==3
        for y in result['years']:
            assert y['metrics']['timing_valid']
            assert y['metrics']['cohorts']['all']['overall']['observed_count']>0
            if y['year']>2005 and method in CONTINUAL:assert y['parent']['year']==y['year']-1
            if method in ('TrafficStream','STKEC') and y['year']>2005:
                assert y['fisher_batches']>0 and y['selection'] is not None
                assert y['selection']['current_max_timestamp'] < 1167609600 if y['year']==2006 else True
            if method=='STKEC' and y['year']==2005:assert y['auxiliary_steps']>0
            if method=='Pretrain' and y['year']>2005:assert y['optimizer_steps']==0
        rows.append(result)
    result={'status':'passed','purpose':'13 existing baseline strategies; three-year smoke, not trained accuracy','tests':stats,'methods':rows,'seconds':time.time()-tic,
            'limitations':['PEMS05 only, first64 evaluation origins and max8 optimization steps per trained year','Explicit positive-percent-observed label policy, not yet frozen for publication','Graph IDs inherited from sorted anchor policy, full verification pending','Specialist corrections change historical protocol: train-only detection, differentiable cluster loss, stable-ID EAC bank and complete state checkpoints']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'passed','methods':len(rows),'method_years':sum(len(m['years']) for m in rows),'tests':stats,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
