"""Explicit scopes; no automatic district/method expansion."""
import copy
import hashlib
import json
from pathlib import Path

METHODS = ('DCRNN', 'ASTGCN', 'TGCN', 'PECPM', 'STRAP', 'STKEC')


def source_record(root):
    paths = sorted(p for base in ('benchmark_v3', 'benchmark_v2', 'eac/src/model')
                   for p in (Path(root)/base).rglob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def source_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def make(root, data_root, method, seed, output, *, district='pems05', purpose='smoke', tiny=False):
    root = Path(root)
    m = json.loads((root/f'benchmark_release/manifests/{district}-s{seed}-{method}.json').read_text())
    cfg = m['model_config']
    cfg.update(microbatch=1 if method == 'ASTGCN' else 4 if method == 'DCRNN' else 16)
    if method == 'DCRNN':
        cfg.update(gradient_clip=5., lr=.01)
    if method == 'PECPM':
        cfg.update(cluster=64, pattern_weight=5., ewc=True, ewc_lambda=.001,
                   strategy='incremental', num_hops=2, lr=.01)
    if method == 'STRAP':
        cfg.update(pattern_build_samples=128, history_ratio=.3, history_limit=10000,
                   use_spatial_lib=True, use_temporal_lib=True, use_spatiotemporal_lib=True)
    if method == 'STKEC':
        cfg.update(influence_damping=.1, influence_cg_steps=80)
    record = source_record(root)
    m.update(purpose=purpose, source_sha256=source_digest(record), device='cuda', output=str(output),
             protocol_frozen=False, cpu_threads=2)
    m.pop('frozen_protocol_path', None)
    m.pop('frozen_protocol_sha256', None)
    if purpose == 'smoke':
        m['years'] = m['years'][:2]
        m.update(epochs=1 if tiny else 2, max_batches=1 if tiny else 4,
                 max_eval_origins=2 if tiny else 32, early_stopping_patience=5)
    for entry in m['years']:
        for key in ('data', 'graph'):
            entry[key] = str(Path(data_root)/entry[key])
    return m
