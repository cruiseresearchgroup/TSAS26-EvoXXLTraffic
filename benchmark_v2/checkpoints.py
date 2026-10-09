"""Numeric selection of legacy checkpoints; provenance remains a separate requirement."""
from pathlib import Path
import math

def select_numeric_checkpoint(directory):
    candidates=[]
    for p in Path(directory).glob('*.pkl'):
        try:loss=float(p.stem)
        except ValueError:continue
        if math.isfinite(loss) and p.is_file():candidates.append((loss,p.name,p))
    if not candidates:raise ValueError('No finite numeric-loss checkpoint in '+str(directory))
    return min(candidates)[2]

def validate_checkpoint_metadata(meta,*,district,year,protocol_hash,data_hash):
    expected=dict(district=district,year=year,protocol_hash=protocol_hash,data_hash=data_hash)
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError('Checkpoint provenance does not match revised data/protocol')
    if district=='PEMS04' and year<2002:raise ValueError('PEMS04 benchmark excludes 2001 checkpoints')
