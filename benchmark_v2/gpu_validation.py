"""Bounded full-node GPU validation; its scores are not benchmark results."""
import argparse
import json
import os
import socket
import time
import traceback
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = Path(manifest['output'])
    receipt = output.parent / (output.name + '_validation.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'purpose': 'full-node bounded GPU validation; not publishable accuracy',
              'method': manifest['method'], 'host': socket.gethostname(),
              'scheduler_job_id': os.environ.get('PBS_JOBID'), 'manifest': str(args.manifest)}
    tic = time.monotonic()
    torch = None
    try:
        import torch
        from .runner import ROOT, CONTINUAL, run, sha
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated GPU is not accessible; CPU fallback forbidden')
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        assert manifest['purpose'] == 'smoke' and manifest['device'] == 'cuda:0'
        for entry in manifest['years']:
            for key in ['data', 'graph']:
                if sha(entry[key]) != entry[key + '_sha256']:
                    raise ValueError('Transferred cache checksum mismatch: ' + entry[key])
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'benchmark_v2/tests'))
        with (output.parent / (output.name + '_tests.txt')).open('w') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result['tests'] = {'count': tests.testsRun, 'passed': tests.wasSuccessful()}
        if not tests.wasSuccessful():
            raise RuntimeError('Regression suite failed on the target environment')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        report = run(manifest)
        torch.cuda.synchronize()
        result['model_seconds'] = time.monotonic() - started
        assert report['status'] == 'passed'
        for i, row in enumerate(report['years']):
            assert row['sensors'] == manifest['years'][i]['sensors']
            assert row['metrics']['timing_valid']
            assert row['metrics']['cohorts']['all']['overall']['observed_count'] > 0
            if i and manifest['method'] in CONTINUAL:
                assert row['parent']['year'] == manifest['years'][i-1]['year']
            if manifest['method'] == 'Pretrain' and i:
                assert row['optimizer_steps'] == 0
            if manifest['method'] == 'ST-TTC':
                assert row['metrics']['calibration_updates'] > 0
        result.update(status='passed', years=[{'year': r['year'], 'sensors': r['sensors'],
                      'seconds': r['seconds'], 'optimizer_steps': r['optimizer_steps']}
                      for r in report['years']], results=str(output / 'results.json'))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['seconds'] = time.monotonic() - tic
        receipt.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
