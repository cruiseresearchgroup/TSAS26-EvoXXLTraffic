"""Causal preprocessing with explicit observation provenance and compact caches.

Inputs are raw, unfilled flow and an externally sourced boolean observation mask.
A finite number, particularly zero, does not establish observation provenance.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'tsas-observed-v2'

def prepare_year(values, observed, sensor_ids, timestamps, *, provenance,
                 slots=31*288, x_len=12, y_len=12, train_rate=.6, val_rate=.2):
    values = np.asarray(values)
    observed = np.asarray(observed)
    ids = np.asarray(sensor_ids)
    timestamps = np.asarray(timestamps)
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise ValueError('values(T,N) requires an explicit boolean observed(T,N) mask')
    if not provenance or not provenance.get('observation_mask_source'):
        raise ValueError('Original observation-mask provenance is required; legacy x-only NPZ is insufficient')
    if ids.ndim != 1 or len(ids) != values.shape[1] or len(set(ids.tolist())) != len(ids):
        raise ValueError('sensor_ids must be unique and aligned with columns')
    if ids.dtype.kind not in 'iuU':
        raise ValueError('sensor IDs must be integers or non-object strings')
    if timestamps.ndim != 1 or len(timestamps) != len(values) or timestamps.dtype.kind not in 'iu':
        raise ValueError('timestamps must be aligned integer epoch seconds')
    if np.any(np.diff(timestamps) != 300):
        raise ValueError('Expected a regular increasing 5-minute grid; resolve duplicate/DST slots explicitly')
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError('Observed values must be finite')
    if not 0 < train_rate < train_rate + val_rate < 1:
        raise ValueError('Invalid chronological split rates')
    if slots <= 0 or x_len <= 0 or y_len <= 0 or len(values) < slots:
        raise ValueError('Insufficient time slots or invalid window lengths')
    values = np.asarray(values[:slots], dtype=np.float64)
    mask = observed[:slots].copy()
    bounds = np.array([0, int(slots*train_rate), int(slots*(train_rate+val_rate)), slots], dtype=np.int64)
    if np.any(np.diff(bounds) < x_len+y_len):
        raise ValueError('Every split must contain at least one full input/target window')
    train_values = values[:bounds[1]][mask[:bounds[1]]]
    if not train_values.size:
        raise ValueError('No observed training values; do not score an empty task as zero error')
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std < 1e-8:
        std = 1.0
    # Forward-only fill: unseen sensors get a train-mean placeholder and available=False.
    x = np.empty(values.shape, dtype=np.float32)
    available = np.empty(mask.shape, dtype=bool)
    last = np.full(values.shape[1], mean, dtype=np.float64)
    seen = np.zeros(values.shape[1], dtype=bool)
    for t in range(slots):
        last[mask[t]] = values[t, mask[t]]
        seen |= mask[t]
        x[t] = (last-mean)/std
        available[t] = seen
    targets = np.where(mask, values, 0).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite values after preprocessing')
    metadata = dict(schema=SCHEMA, scaler_fit='observed unique train slots only',
                    scaler_mean=mean, scaler_std=std, train_observed_count=int(train_values.size),
                    input_fill='forward only; unseen=train mean with available=False',
                    target_fill='zero placeholder; observed mask is authoritative',
                    input_length=x_len, output_length=y_len,
                    split_policy='chronological, complete windows contained within each split',
                    node_policy='caller supplies ordered anchor-time catalog; activation tracked by observations',
                    provenance=provenance)
    return dict(x=x, y=targets, observed=mask, available=available, sensor_ids=ids.copy(),
                timestamps=timestamps[:slots].copy(), bounds=bounds, metadata=metadata)

def split_origins(data, split):
    i = ['train', 'val', 'test'].index(split)
    a,b = data['bounds'][i:i+2]
    L,H = data['metadata']['input_length'],data['metadata']['output_length']
    # Origins identify the last observed input time; all H target slots lie in the split.
    return np.arange(a+L-1, b-H, dtype=np.int64)

def window_batch(data, origins):
    origins = np.asarray(origins, dtype=np.int64)
    L,H = data['metadata']['input_length'],data['metadata']['output_length']
    if origins.ndim != 1 or np.any(origins < L-1) or np.any(origins+H >= len(data['x'])):
        raise ValueError('Invalid forecast origins')
    xi = origins[:,None] + np.arange(-L+1,1)
    yi = origins[:,None] + np.arange(1,H+1)
    return {'x':data['x'][xi], 'y':data['y'][yi], 'y_mask':data['observed'][yi],
            'x_observed':data['observed'][xi], 'x_available':data['available'][xi],
            'origin_time':data['timestamps'][origins], 'target_times':data['timestamps'][yi]}

def save_year(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents replacing legacy data or a previous revision cache.
    with path.open('xb') as f:
        np.savez(f, **{k:v for k,v in data.items() if k!='metadata'},
                 metadata=np.array(json.dumps(data['metadata'],sort_keys=True)))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_year(path):
    with np.load(path,allow_pickle=False) as z:
        required={'x','y','observed','available','sensor_ids','timestamps','bounds','metadata'}
        if not required.issubset(z.files):
            raise ValueError('Not an observation-aware v2 cache; legacy FastData must be rebuilt')
        d={k:z[k] for k in required if k!='metadata'}
        for key in ('record_present','flow_reported','percent_observed'):
            if key in z.files:d[key]=z[key]
        d['metadata']=json.loads(str(z['metadata']))
    if d['metadata'].get('schema')!=SCHEMA:
        raise ValueError('Unsupported benchmark cache schema')
    if d['observed'].dtype!=np.bool_ or d['available'].dtype!=np.bool_:
        raise ValueError('Invalid mask dtype')
    for key in ('record_present','flow_reported','percent_observed'):
        if key in d and d[key].shape!=d['observed'].shape:
            raise ValueError('Unaligned original quality field: '+key)
    return d
