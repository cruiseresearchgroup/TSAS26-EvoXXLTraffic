import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_v2.data import prepare_year

SOURCE = Path(__file__).resolve().parents[1]
PLAN = json.loads((Path(__file__).parent / 'plan.json').read_text())
YEARS = list(range(2018, 2026))
SEEDS = [47, 48, 49]
METHODS = ['Retrain', 'Online-AN']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def subset(data, raw_adj, fixed_ids):
    """Rebuild only F preprocessing from unfilled accepted labels, never E's scaler."""
    ids = data['sensor_ids']
    indices = np.array([np.flatnonzero(ids == sid).item() for sid in fixed_ids])
    out = prepare_year(data['y'][:, indices], data['observed'][:, indices],
                       ids[indices], data['timestamps'],
                       provenance=dict(data['metadata']['provenance'],
                                       population='fixed 2018-2025 intersection'))
    for key in ('record_present', 'flow_reported', 'percent_observed'):
        if key in data:
            out[key] = data[key][:, indices].copy()
    out['metadata']['quality_fields'] = data['metadata'].get('quality_fields', {})
    return out, raw_adj[np.ix_(indices, indices)].copy(), indices


def seed_summary(year_values):
    """Equal-year means within each fixed seed, then sample SD across seeds."""
    means = [float(np.mean(year_values[seed])) for seed in SEEDS]
    return dict(mean=float(np.mean(means)), sd=float(np.std(means, ddof=1)),
                seed_values=means)


def paired_summary(evolving, fixed):
    differences = {seed: np.asarray(evolving[seed]) - np.asarray(fixed[seed])
                   for seed in SEEDS}
    return seed_summary(differences)
