"""Fetch only 16 requested frozen yearly pairs; stop on any checksum mismatch."""
import hashlib
import json
import pathlib
import urllib.request

root = pathlib.Path(__file__).resolve().parent
plan = json.loads((root / 'INPUT_PLAN.json').read_text())
receipt = {'status': 'fetching', 'hf_revision': plan['hf_revision'], 'files': []}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

try:
    for row in plan['inputs']:
        for kind in ('data', 'graph'):
            rel = row[kind]
            target = root / 'inputs' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                url = ('https://huggingface.co/datasets/yind7/EvoXXLTraffic/resolve/'
                       + plan['hf_revision'] + '/caches/' + rel)
                with urllib.request.urlopen(url, timeout=120) as response, target.open('xb') as f:
                    while block := response.read(8 * 1024**2):
                        f.write(block)
            actual = sha(target)
            item = dict(path=rel, expected_sha256=row[kind + '_sha256'],
                        actual_sha256=actual, bytes=target.stat().st_size)
            receipt['files'].append(item)
            if actual != row[kind + '_sha256']:
                raise ValueError('Frozen input SHA256 mismatch: ' + rel)
            print(json.dumps(dict(event='verified', **item)), flush=True)
    receipt['status'] = 'PASS'
except Exception as error:
    receipt.update(status='STOPPED', error=repr(error))
    raise
finally:
    (root / 'INPUT_SHA_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
