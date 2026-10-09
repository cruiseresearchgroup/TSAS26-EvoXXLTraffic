"""Materialize one frozen experiment locally; never launch training or a scheduler."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'benchmark_release'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def prepare(label, data_root, destination):
    entries = json.loads((BUNDLE / 'index.json').read_text())
    entry = next((x for x in entries if x['label'] == label), None)
    if entry is None:
        raise ValueError('Unknown label; see benchmark_release/index.json')
    manifest = json.loads((BUNDLE / entry['manifest']).read_text())
    protocol_rel = Path(manifest['frozen_protocol_path'])
    protocol_path = ROOT / protocol_rel
    protocol = json.loads(protocol_path.read_text())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if digest != manifest['frozen_protocol_sha256']:
        raise ValueError('Frozen protocol mismatch')
    sources = []
    for relative, checksum in protocol['source_sha256'].items():
        source = ROOT / relative
        overlay = BUNDLE / 'pems04_overlay' / relative
        if entry['district'] == 'PEMS04' and overlay.exists():
            source = overlay
        if sha(source) != checksum:
            raise ValueError('Frozen source changed: ' + relative)
        sources.append((source, relative))
    for year in manifest['years']:
        for key in ('data', 'graph'):
            data = (data_root / year[key]).resolve()
            if not data.is_relative_to(data_root.resolve()):
                raise ValueError('Cache path escapes data root')
            if not data.is_file() or sha(data) != year[key + '_sha256']:
                raise ValueError('Missing or mismatched corrected artifact: ' + str(data))
            year[key] = str(data)
    # Refuse an existing output and validate all inputs before copying anything.
    destination.mkdir(parents=True, exist_ok=False)
    for source, relative in sources:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / protocol_rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_path, destination / protocol_rel)
    manifest['frozen_protocol_path'] = str(destination / protocol_rel)
    manifest['output'] = str(destination / 'results' / label)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'release_provenance.json').write_text(json.dumps({
        'label': label, 'original_manifest_sha256': entry['original_manifest_sha256'],
        'portable_manifest_sha256': entry['portable_manifest_sha256'],
        'changes': 'Only filesystem paths/output location; scientific parameters and input SHA-256 retained',
        'source_files_verified': len(sources), 'cache_files_verified': 2 * len(manifest['years'])
    }, indent=2) + '\n')
    return destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = prepare(args.label, args.data_root.resolve(), args.output.resolve())
    print('Prepared verified workspace:', out)
    print('On an allocated GPU, from that directory: python -m benchmark_v2.formal_run --manifest manifest.json')
