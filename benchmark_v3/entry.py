"""Allocation-side validation entry; immutable source/data receipts are mandatory."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest
import torch
from .manifests import make, source_record, source_digest
from .runner import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(Path(a.spec).read_text())
    record = source_record(root)
    if source_digest(record) != spec['source_sha256']:
        raise ValueError('Source fingerprint differs from the submitted experiment')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'started': time.time(), 'spec': spec,
               'scheduler_job': os.environ.get('PBS_JOBID') or os.environ.get('JOB_ID'),
               'host': os.uname().nodename, 'source_sha256': source_digest(record),
               'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; never fall back silently')
        receipt['gpu'] = torch.cuda.get_device_name()
        torch.set_num_threads(2)
        suites = unittest.defaultTestLoader.discover(str(root/'benchmark_v3/tests'), top_level_dir=str(root))
        tested = unittest.TextTestRunner(verbosity=1).run(suites)
        receipt['tests'] = {'run': tested.testsRun, 'passed': tested.wasSuccessful(), 'skipped': len(tested.skipped)}
        if not tested.wasSuccessful() or tested.skipped:
            raise RuntimeError('Mechanism validation failed or was skipped')
        if spec['stage'] != 'pilot':
            raise ValueError('Formal entry remains closed until all pilot and feasibility receipts pass')
        manifest = make(root, a.data, spec['method'], spec['seed'], out/'run')
        result = run(manifest)
        if result['status'] != 'passed' or len(result['years']) != 2:
            raise ValueError('Incomplete two-year pilot')
        second = result['years'][1]
        if spec['method'] in ('PECPM', 'STRAP', 'STKEC') and not second['parent']:
            raise ValueError('Second year did not inherit its verified checkpoint')
        if spec['method'] == 'STRAP' and not any(2005 in years for years in second['pattern_origin_years'].values()):
            raise ValueError('STRAP failed to retrieve history from the preceding year')
        if spec['method'] == 'STKEC' and not second['selection']['policy'].endswith('inverse_influence'):
            raise ValueError('STKEC did not exercise influence selection')
        receipt['status'] = 'passed'
    except BaseException as e:
        receipt.update(status='failed', error=repr(e))
        raise
    finally:
        receipt['seconds'] = time.time()-receipt['started']
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
