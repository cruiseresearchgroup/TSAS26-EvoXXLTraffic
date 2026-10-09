"""A district-specific frozen input/config contract, independent of CUDA."""
import hashlib,json

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def validate_frozen_manifest(m,spec):
    if not spec.get('frozen') or m.get('frozen_protocol_sha256')!=digest(spec):
        raise ValueError('Unfrozen or mismatched protocol')
    if m['district']!=spec['data']['district']:
        raise ValueError('District differs from frozen protocol')
    catalog={r['year']:r for r in spec['data']['entries']}
    for r in m['years']:
        if r!=catalog.get(r['year']):raise ValueError('Input differs from frozen catalog')
    if m['purpose']=='benchmark':
        b=spec['methods'][m['method']]
        if ([r['year'] for r in m['years']]!=spec['data']['years'] or m['seed'] not in spec['seeds']
            or m['epochs']!=100 or m['early_stopping_patience']!=5
            or m.get('max_batches') is not None or m.get('max_eval_origins') is not None
            or m['batch_size']!=b['batch_size'] or digest(m['model_config'])!=b['effective_config_sha256']):
            raise ValueError('Formal budget/configuration mismatch')
