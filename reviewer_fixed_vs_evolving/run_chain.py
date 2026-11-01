"""Run one fresh 2018-2025 chain through the unmodified frozen training loop."""
import argparse
import fcntl
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from benchmark_v2 import runner
from .common import PLAN, SOURCE, YEARS, METHODS, SEEDS, sha, save


def test_cohorts(data, district, population, original):
    ids = data['sensor_ids']
    fixed = np.asarray(PLAN['districts'][district]['sensor_ids'])
    cohorts = {'S': np.flatnonzero(np.isin(ids, fixed))}
    assert len(cohorts['S']) == len(fixed)
    if population == 'E':
        year = int(data['metadata']['provenance']['window_start'][:4])
        new = original['new_to_previous']
        if year == 2018:
            # User-approved catalog-only 2017 reference; no 2017 training/checkpoint.
            previous = PLAN['districts'][district]['previous_2017_sensor_ids']
            new = np.flatnonzero(~np.isin(ids, previous))
        cohorts.update(all=np.arange(len(ids)), new=new)
    return cohorts


def manifest(root, district, method, population, seed):
    m = json.loads((SOURCE / f'benchmark_release/manifests/{district.lower()}-s{seed}-{method}.json').read_text())
    assert method in METHODS and seed in SEEDS and population in ('E', 'F')
    audit = json.loads((root / 'INPUT_AUDIT.json').read_text())
    assert audit['status'] == 'PASS'
    frozen = [r for r in m['years'] if r['year'] in YEARS]
    assert [r['year'] for r in frozen] == YEARS
    for entry in frozen:
        for kind in ('data', 'graph'):
            p = root / 'inputs' / entry[kind]
            assert sha(p) == entry[kind + '_sha256'], 'Frozen input SHA mismatch'
    if population == 'F':
        m['years'] = [dict(district=district, sensors=PLAN['districts'][district]['S_size'],
                          **{k:r[k] for k in ('year','data','graph','data_sha256','graph_sha256')})
                      for r in audit['derived_inputs'] if r['district'] == district]
    else:
        m['years'] = [dict(r, data=str(root / 'inputs' / r['data']),
                          graph=str(root / 'inputs' / r['graph'])) for r in frozen]
    for entry in m['years']:
        for kind in ('data', 'graph'):
            assert sha(entry[kind]) == entry[kind + '_sha256']
    m['model_config'].update(begin_year=2018, end_year=2025)
    m['output'] = str(root / 'results' / f'{district}-{method}-{population}-s{seed}')
    m.update(train_population=population, cpu_threads=2,
             revision_protocol='reviewer-fixed-vs-evolving-v1',
             revision_protocol_sha256=sha(SOURCE / 'reviewer_fixed_vs_evolving/PROTOCOL.json'),
             frozen_protocol_path=str(SOURCE / m['frozen_protocol_path']))
    return m


def run(root, district, method, population, seed):
    root = Path(root).resolve()
    label = f'{district}-{method}-{population}-s{seed}'
    (root / 'receipts').mkdir(exist_ok=True)
    (root / 'results').mkdir(exist_ok=True)
    lock = (root / 'receipts' / (label + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt_path = root / 'receipts' / (label + '.json')
    if receipt_path.exists() or (root / 'results' / label).exists():
        raise RuntimeError('Refusing duplicate or implicit retry: ' + label)
    receipt = dict(label=label, district=district, method=method, train_population=population,
                   seed=seed, status='RUNNING', host=socket.gethostname(),
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'), started=time.time())
    save(receipt_path, receipt)
    try:
        assert os.environ.get('SLURM_JOB_ID'), 'GPU allocation required'
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert torch.__version__.split('+')[0] == '2.2.1'
        assert np.__version__ == '1.26.4'
        code = json.loads((root / 'CODE_RECEIPT.json').read_text())
        for relative, expected in code['files'].items():
            assert sha(SOURCE / relative) == expected, 'Sealed source changed: ' + relative
        receipt.update(code_commit=code['code_commit'], gpu=torch.cuda.get_device_name(0),
                       torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                       gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
                       tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                       tf32_cudnn=torch.backends.cudnn.allow_tf32)
        m = manifest(root, district, method, population, seed)
        receipt['manifest'] = m
        save(receipt_path, receipt)
        original_evaluate = runner.evaluate

        def evaluate(model, data, adj, split, cohorts, **kwargs):
            if split == 'test':
                cohorts = test_cohorts(data, district, population, cohorts)
            return original_evaluate(model, data, adj, split, cohorts, **kwargs)

        runner.evaluate = evaluate
        torch.cuda.reset_peak_memory_stats()
        result = runner.run(m)
        assert result['status'] == 'passed' and [r['year'] for r in result['years']] == YEARS
        receipt.update(status='PASS', result_path=str(Path(m['output']) / 'results.json'),
                       result_sha256=sha(Path(m['output']) / 'results.json'),
                       years=[dict(year=r['year'], epochs=r['epochs_completed'],
                                   optimizer_steps=r['optimizer_steps'],
                                   checkpoint_sha256=r['checkpoint_sha256'], parent=r['parent'])
                              for r in result['years']])
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished=time.time(), seconds=time.time() - receipt['started'])
        if torch.cuda.is_available():
            receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(receipt_path, receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('manifest','traceback','years')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--district', choices=['PEMS05','PEMS06'], required=True)
    p.add_argument('--method', choices=METHODS, required=True); p.add_argument('--population', choices=['E','F'], required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = p.parse_args(); run(a.root, a.district, a.method, a.population, a.seed)
