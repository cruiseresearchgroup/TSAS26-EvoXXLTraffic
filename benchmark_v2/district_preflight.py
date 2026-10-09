"""Allocated GPU check across all years, early transitions and late graph size."""
import argparse,gc,hashlib,json,os,socket,time,unittest
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);args=parser.parse_args()
    batch=json.loads(Path(args.batch).read_text());out=Path(batch['receipt']);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',district=batch['district'],scheduler_job_id=os.environ.get('PBS_JOBID'),host=socket.gethostname(),cases=[],data_checks=[]);tic=time.monotonic()
    def save():out.write_text(json.dumps(result,indent=2))
    try:
        import numpy as np
        import torch
        from .runner import ROOT,sha,run
        from .data import load_year,split_origins
        from .formal_contract import digest,validate_frozen_manifest
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        result.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
        spec=json.loads(Path(batch['protocol']).read_text());assert digest(spec)==batch['protocol_sha256']
        result['protocol_sha256']=digest(spec)
        for rel,h in spec['source_sha256'].items():
            if sha(ROOT/rel)!=h:raise ValueError('Frozen source changed: '+rel)
        suite=unittest.defaultTestLoader.discover(str(ROOT/'benchmark_v2/tests'))
        with out.with_suffix('.tests.txt').open('w') as f:test=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        result['tests']=dict(count=test.testsRun,passed=test.wasSuccessful())
        if not test.wasSuccessful():raise RuntimeError('Regression suite failed')
        for e in spec['data']['entries']:
            for k in ['data','graph']:
                if sha(e[k])!=e[k+'_sha256']:raise ValueError('Input checksum mismatch')
            data=load_year(e['data']);assert len(data['sensor_ids'])==e['sensors'];counts={}
            for i,split in enumerate(['train','val','test']):
                a,b=data['bounds'][i:i+2];n=int(data['observed'][a+12:b].sum());assert n>0,(e['year'],split)
                counts[split]=dict(accepted_target_slots=n,origins=len(split_origins(data,split)))
            assert counts['test']['origins']==1763
            result['data_checks'].append(dict(year=e['year'],sensors=e['sensors'],counts=counts))
            del data;gc.collect();save()
        for case in batch['cases']:
            m=json.loads(Path(case['manifest']).read_text());validate_frozen_manifest(m,spec)
            report=run(m);assert report['status']=='passed' and len(report['years'])==len(m['years'])
            for row in report['years']:
                assert row['graph_provenance']['order_status']=='verified_ids'
                assert row['metrics']['origins']==(m.get('max_eval_origins') or 1763)
                assert row['metrics']['timing_valid']
                assert row['metrics']['cohorts']['all']['overall']['observed_count']>0
                assert row['metrics']['cohorts']['all']['overall']['MAE'] is not None
            result['cases'].append(dict(name=case['name'],method=m['method'],status='passed',result=report));save()
            gc.collect();torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as e:
        import traceback
        result.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result['seconds']=time.monotonic()-tic;save();print(json.dumps({k:v for k,v in result.items() if k not in ['cases','data_checks','traceback']}),flush=True)
if __name__=='__main__':main()
