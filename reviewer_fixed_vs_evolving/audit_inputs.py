"""Verify all frozen bytes and actual sensor catalogs before creating F derivatives."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_v2.data import load_year, prepare_year, save_year, split_origins
from .common import PLAN, SOURCE, YEARS, sha, save, subset


def audit(root):
    root = Path(root).resolve()
    report = dict(status='RUNNING', source_commit=PLAN['release_commit'],
                  frozen_inputs=[], districts={}, derived_inputs=[])
    try:
        # Validate every source and input first. Any mismatch stops the whole batch.
        for district in ('PEMS05', 'PEMS06'):
            spec = json.loads((SOURCE / f'benchmark_release/protocols/{district.lower()}.json').read_text())
            for relative, expected in spec['source_sha256'].items():
                if sha(SOURCE / relative) != expected:
                    raise ValueError('Frozen source SHA mismatch: ' + relative)
        for entry in PLAN['inputs']:
            for kind in ('data', 'graph'):
                path = root / 'inputs' / entry[kind]
                actual = sha(path)
                report['frozen_inputs'].append(dict(path=entry[kind], sha256=actual, bytes=path.stat().st_size))
                if actual != entry[kind + '_sha256']:
                    raise ValueError('Frozen input SHA mismatch: ' + str(path))
        catalogs = json.loads((SOURCE / 'benchmark_release/ordered_catalogs.json').read_text())['catalogs']
        for district, expected in [('PEMS05', 240), ('PEMS06', 476)]:
            ids_by_year = {}
            for entry in [r for r in PLAN['inputs'] if r['district'] == district]:
                data = load_year(root / 'inputs' / entry['data'])
                with np.load(root / 'inputs' / entry['graph'], allow_pickle=False) as z:
                    graph_ids = z['sensor_ids']
                assert np.array_equal(data['sensor_ids'], graph_ids), 'Graph/data ID mismatch'
                published = next(r['sensor_ids'] for r in catalogs if r['district'] == district[-2:] and r['year'] == entry['year'])
                assert data['sensor_ids'].tolist() == published, 'Frozen catalog/data ID mismatch'
                assert data['metadata']['provenance']['label_policy'] == 'finite_reported'
                assert np.array_equal(data['observed'], data['flow_reported'])
                assert data['x'].shape[0] == 31 * 288
                assert data['bounds'].tolist() == [0, 5356, 7142, 8928]
                assert data['metadata']['input_length'] == data['metadata']['output_length'] == 12
                # Check that stored unfilled labels reproduce the original E preprocessing exactly.
                reconstructed = prepare_year(data['y'], data['observed'], data['sensor_ids'],
                                             data['timestamps'], provenance=data['metadata']['provenance'])
                for key in ('x', 'y', 'observed', 'available', 'bounds'):
                    assert np.array_equal(reconstructed[key], data[key]), ('Raw label reconstruction mismatch', key)
                ids_by_year[entry['year']] = set(data['sensor_ids'].tolist())
            fixed_ids = sorted(set.intersection(*ids_by_year.values()))
            assert len(fixed_ids) == expected, 'Fixed intersection size mismatch'
            assert fixed_ids == PLAN['districts'][district]['sensor_ids']
            ids_bytes = ''.join(str(sid) + '\n' for sid in fixed_ids).encode()
            import hashlib
            assert hashlib.sha256(ids_bytes).hexdigest() == PLAN['districts'][district]['sensor_ids_sha256']
            report['districts'][district] = dict(S_size=len(fixed_ids),
                sensor_ids_sha256=hashlib.sha256(ids_bytes).hexdigest(),
                sensor_ids_serialization=PLAN['districts'][district]['sensor_ids_serialization'],
                year_counts={str(y):len(ids_by_year[y]) for y in YEARS})
        # All required stop gates passed; derived inputs are written only in this new run.
        for entry in PLAN['inputs']:
            district = entry['district']
            data = load_year(root / 'inputs' / entry['data'])
            with np.load(root / 'inputs' / entry['graph'], allow_pickle=False) as z:
                raw, graph_meta = z['x'], json.loads(str(z['metadata']))
            fixed, graph, indices = subset(data, raw, PLAN['districts'][district]['sensor_ids'])
            assert np.array_equal(graph, raw[np.ix_(indices, indices)])
            assert np.array_equal(fixed['y'], data['y'][:, indices])
            assert np.array_equal(fixed['observed'], data['observed'][:, indices])
            target = root / 'derived_F' / district.lower() / str(entry['year'])
            target.mkdir(parents=True, exist_ok=False)
            dh = save_year(target / 'data.npz', fixed)
            graph_meta.update(induced_subgraph=True, original_graph_sha256=entry['graph_sha256'],
                              edge_weights='original induced entries; sigma not recomputed')
            with (target / 'graph.npz').open('xb') as f:
                np.savez(f, x=graph, sensor_ids=fixed['sensor_ids'], metadata=np.array(json.dumps(graph_meta, sort_keys=True)))
            report['derived_inputs'].append(dict(district=district, year=entry['year'],
                data=str(target / 'data.npz'), data_sha256=dh,
                graph=str(target / 'graph.npz'), graph_sha256=sha(target / 'graph.npz'),
                scaler_mean=fixed['metadata']['scaler_mean'], scaler_std=fixed['metadata']['scaler_std'],
                train_accepted_count=fixed['metadata']['train_observed_count'],
                origins={s:len(split_origins(fixed, s)) for s in ('train', 'val', 'test')}))
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='STOPPED', error=repr(error))
        raise
    finally:
        save(root / 'INPUT_AUDIT.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('frozen_inputs','derived_inputs')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    audit(p.parse_args().root)
