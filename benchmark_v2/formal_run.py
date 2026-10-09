"""Execute one frozen district/method/seed experiment and preserve its receipt."""
import argparse
import json
import os
import socket
import time
import traceback
import unittest
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args();m=json.loads(args.manifest.read_text())
    output=Path(m['output']);output.parent.mkdir(parents=True,exist_ok=True)
    receipt=output.parent/(output.name+'_receipt.json');tic=time.monotonic()
    result={'status':'running','method':m['method'],'district':m['district'],'seed':m['seed'],
            'purpose':m['purpose'],'host':socket.gethostname(),'scheduler_job_id':os.environ.get('PBS_JOBID')}
    torch=None
    try:
        import torch
        from .runner import ROOT,run,sha,digest
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU fallback')
        result.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
        spec=json.loads(Path(m['frozen_protocol_path']).read_text())
        if digest(spec)!=m['frozen_protocol_sha256']:raise ValueError('Frozen protocol hash mismatch')
        from .formal_contract import validate_frozen_manifest
        validate_frozen_manifest(m,spec)
        if m['purpose']=='benchmark':
            budget=spec['methods'][m['method']]
            assert m['protocol_frozen'] and m['seed'] in spec['seeds']
            assert [r['year'] for r in m['years']]==spec['data']['years']
            assert m['epochs']==100 and m['early_stopping_patience']==5
            assert m.get('max_batches') is None and m.get('max_eval_origins') is None
            assert m['batch_size']==budget['batch_size'] and m['model_config']['lr']==budget['lr']
            assert digest(m['model_config'])==budget['effective_config_sha256']
        for relative,checksum in spec['source_sha256'].items():
            if sha(ROOT/relative)!=checksum:raise ValueError('Source changed: '+relative)
        for entry in m['years']:
            for key in ['data','graph']:
                if sha(entry[key])!=entry[key+'_sha256']:raise ValueError('Cache checksum mismatch: '+entry[key])
        suite=unittest.defaultTestLoader.discover(str(ROOT/'benchmark_v2/tests'))
        with (output.parent/(output.name+'_tests.txt')).open('w') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        result['tests']={'count':tests.testsRun,'passed':tests.wasSuccessful()}
        if not tests.wasSuccessful():raise RuntimeError('Regression suite failed')
        result['preflight_seconds']=time.monotonic()-tic
        receipt.write_text(json.dumps(result,indent=2))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        report=run(m);torch.cuda.synchronize()
        assert report['status']=='passed' and len(report['years'])==len(m['years'])
        result.update(status='passed',model_seconds=time.monotonic()-start,result=report,
                      frozen_protocol_sha256=m['frozen_protocol_sha256'])
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        partial=output/'results.json'
        if partial.exists():result['partial_result']=json.loads(partial.read_text())
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['seconds']=time.monotonic()-tic
        receipt.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ['result','partial_result','traceback']}),flush=True)


if __name__=='__main__':main()
