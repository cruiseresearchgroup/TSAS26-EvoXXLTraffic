"""Execute one prepared run of the TSAS Virga batch 20261110 and preserve its receipt.

This is benchmark_v2/formal_run.py of tag tsas26-r1.2 with one check adapted. The frozen
protocol pins every yearly input by its Katana path and SHA-256, and formal_run compares the
manifest's year entries with that catalog including the path strings, so it cannot run
anywhere else. Here each local input must have exactly the catalog SHA-256 and every other
field must equal the catalog entry; validate_frozen_manifest() then runs unchanged on the
catalog paths, and the unmodified benchmark_v2.runner.run() reads the local copies.
All other checks (protocol digest, budget, frozen sources, regression tests, GPU) and the
receipt format are kept. No scientific code is changed, and nothing here sets PBS_JOBID.

On an allocated GPU, with the prepared run directory made by prepare_batch.py:
    python formal_run_relocated.py --manifest <run dir>/manifest.json
Without a GPU, to run only the input and contract checks (no receipt, no training):
    python formal_run_relocated.py --manifest <run dir>/manifest.json --check-only
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
import unittest
from pathlib import Path

LAUNCHER = 'formal_run_relocated.py (TSAS Virga batch 20261110)'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_contract(run_dir):
    """Load the frozen benchmark_v2/formal_contract.py by file (it needs only hashlib and json)."""
    path = run_dir / 'benchmark_v2' / 'formal_contract.py'
    spec = importlib.util.spec_from_file_location('tsas_frozen_formal_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(m, run_dir):
    """All formal_run.py checks except CUDA and tests; returns the local-to-catalog input map."""
    spec = json.loads(Path(m['frozen_protocol_path']).read_text())
    for relative, checksum in spec['source_sha256'].items():
        if sha(run_dir / relative) != checksum:
            raise ValueError('Source changed: ' + relative)
    contract_module = load_contract(run_dir)
    digest, validate_frozen_manifest = contract_module.digest, contract_module.validate_frozen_manifest
    if digest(spec) != m['frozen_protocol_sha256']:
        raise ValueError('Frozen protocol hash mismatch')
    catalog = {r['year']: r for r in spec['data']['entries']}
    contract = copy.deepcopy(m)
    input_map = []
    for local, frozen in zip(m['years'], contract['years']):
        ref = catalog.get(local['year'])
        if ref is None:
            raise ValueError('Year missing from frozen catalog: %s' % local['year'])
        for key in ('data', 'graph'):
            if local[key + '_sha256'] != ref[key + '_sha256'] or sha(local[key]) != ref[key + '_sha256']:
                raise ValueError('Relocated input differs from frozen catalog: ' + local[key])
            frozen[key] = ref[key]
            input_map.append({'year': local['year'], 'kind': key, 'local': local[key],
                              'catalog': ref[key], 'sha256': ref[key + '_sha256']})
    # Unchanged contract: every field other than the physical location must equal the catalog.
    validate_frozen_manifest(contract, spec)
    if m['purpose'] == 'benchmark':
        budget = spec['methods'][m['method']]
        assert m['protocol_frozen'] and m['seed'] in spec['seeds']
        assert [r['year'] for r in m['years']] == spec['data']['years']
        assert m['epochs'] == 100 and m['early_stopping_patience'] == 5
        assert m.get('max_batches') is None and m.get('max_eval_origins') is None
        assert m['batch_size'] == budget['batch_size'] and m['model_config']['lr'] == budget['lr']
        assert digest(m['model_config']) == budget['effective_config_sha256']
    return input_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    run_dir = manifest_path.parent
    sys.path.insert(0, str(run_dir))
    m = json.loads(manifest_path.read_text())
    if args.check_only:
        input_map = check(m, run_dir)
        print(json.dumps({'check_only': 'PASS', 'label': Path(m['output']).name,
                          'inputs_checked': len(input_map)}), flush=True)
        return
    output = Path(m['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.parent / (output.name + '_receipt.json')
    tic = time.monotonic()
    result = {'status': 'running', 'method': m['method'], 'district': m['district'], 'seed': m['seed'],
              'purpose': m['purpose'], 'host': socket.gethostname(),
              'scheduler_job_id': os.environ.get('PBS_JOBID'),
              'scheduler': 'slurm', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'launcher': LAUNCHER, 'inputs_relocated': True}
    torch = None
    try:
        import torch
        from benchmark_v2.runner import ROOT, run
        if Path(ROOT).resolve() != run_dir:
            raise RuntimeError('Runner is not the prepared copy: %s' % ROOT)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; no CPU fallback')
        result.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
        result['input_map'] = check(m, run_dir)
        suite = unittest.defaultTestLoader.discover(str(run_dir / 'benchmark_v2/tests'))
        with (output.parent / (output.name + '_tests.txt')).open('w') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result['tests'] = {'count': tests.testsRun, 'passed': tests.wasSuccessful()}
        if not tests.wasSuccessful():
            raise RuntimeError('Regression suite failed')
        result['preflight_seconds'] = time.monotonic() - tic
        receipt.write_text(json.dumps(result, indent=2))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        report = run(m)
        torch.cuda.synchronize()
        assert report['status'] == 'passed' and len(report['years']) == len(m['years'])
        result.update(status='passed', model_seconds=time.monotonic() - start, result=report,
                      frozen_protocol_sha256=m['frozen_protocol_sha256'])
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        partial = output / 'results.json'
        if partial.exists():
            result['partial_result'] = json.loads(partial.read_text())
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['seconds'] = time.monotonic() - tic
        receipt.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ['result', 'partial_result', 'traceback', 'input_map']}), flush=True)


if __name__ == '__main__':
    main()
