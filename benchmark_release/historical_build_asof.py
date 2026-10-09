"""Frozen D04 coordinate-time intervention; run on allocated CPU resources."""
import hashlib
import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from graph_source import (build_station_catalog, catalog_coordinates,
                          haversine_cross, snapshot_date)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asof_coordinates(ids, files, anchor):
    eligible = [p for p in files if snapshot_date(p) <= date.fromisoformat(anchor)]
    catalog = build_station_catalog(eligible)
    coords, missing = catalog_coordinates(ids, catalog, fill_missing=False)
    sources = [catalog[int(s)].coordinate_source if not missing[i] else None
               for i, s in enumerate(ids)]
    assert all(s is None or snapshot_date(Path(s)) <= date.fromisoformat(anchor)
               for s in sources)
    return coords, missing, sources, eligible


def distance_sigma(coords, block=128):
    total = 0.; count = 0
    for i in range(0, len(coords), block):
        d = haversine_cross(coords[i:i+block], coords); positive = d[d > 0]
        total += float(positive.sum(dtype=np.float64)); count += positive.size
    if not count:
        raise ValueError('No positive distances; no arbitrary bandwidth fallback')
    mean = total/count; ss = 0.
    for i in range(0, len(coords), block):
        d = haversine_cross(coords[i:i+block], coords); positive = d[d > 0]
        ss += float(np.square(positive-mean).sum(dtype=np.float64))
    sigma = float(np.sqrt(ss/count))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Undefined bandwidth; no arbitrary fallback')
    return sigma


def adjacency(coords, missing):
    known = np.flatnonzero(~missing); valid = coords[known]
    sigma = distance_sigma(valid)
    adj = np.zeros((len(coords), len(coords)), dtype=np.float32)
    for i in range(0, len(known), 128):
        d = haversine_cross(valid[i:i+128], valid)
        w = np.exp(-(d*d)/(sigma*sigma)); w[w < .1] = 0.
        adj[np.ix_(known[i:i+128], known)] = w.astype(np.float32)
    np.fill_diagonal(adj, 0.)
    assert np.isfinite(adj).all() and np.array_equal(adj, adj.T)
    assert not np.any(adj[missing]) and not np.any(adj[:, missing])
    return adj, sigma


class InterventionTests(unittest.TestCase):
    def test_date_boundary_and_future_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            early = d/'d04_text_meta_2001_08_01.txt'
            late = d/'d04_text_meta_2004_03_25.txt'
            header = 'ID\tLatitude\tLongitude\n'
            early.write_text(header+'1\t37\t-122\n2\t0\t0\n')
            late.write_text(header+'1\t38\t-123\n2\t37.2\t-122.2\n')
            ids = [2, 1, 3]
            c, m, sources, _ = asof_coordinates(ids, [late, early], '2004-03-24')
            self.assertEqual(m.tolist(), [True, False, True])
            np.testing.assert_equal(c[1], [37, -122])
            self.assertIsNone(sources[0])
            c2, m2, _, _ = asof_coordinates(ids, [early], '2004-03-24')
            np.testing.assert_equal(c, c2); np.testing.assert_equal(m, m2)
            c3, m3, _, _ = asof_coordinates(ids, [late, early], '2004-03-25')
            self.assertFalse(m3[0]); np.testing.assert_equal(c3[1], [37, -122])

    def test_isolation_and_independent_kernel(self):
        # Same-longitude distances have an independent closed-form reference.
        coords = np.array([[30., -120.], [31., -120.], [34., -120.], [np.nan, np.nan]])
        mask = np.array([False, False, False, True])
        adj, sigma = adjacency(coords, mask)
        d = np.abs(coords[:3, 0, None]-coords[None, :3, 0])*math.pi/180*6371
        expected_sigma = np.std(d[d > 0]); expected = np.exp(-d*d/expected_sigma**2)
        expected[expected < .1] = 0; np.fill_diagonal(expected, 0)
        self.assertAlmostEqual(sigma, expected_sigma, places=8)
        np.testing.assert_allclose(adj[:3, :3], expected, atol=1e-7)
        self.assertTrue(np.all(adj[3] == 0) and np.all(adj[:, 3] == 0))

    def test_undefined_bandwidth_fails(self):
        with self.assertRaises(ValueError):
            adjacency(np.array([[30., -120.], [30., -120.]]), np.array([False, False]))


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InterventionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Intervention tests failed')
    spec = json.loads(Path('inputs.json').read_text())
    files = []
    for entry in spec['metadata_sources']:
        p = Path(entry['path'])
        assert sha(p) == entry['sha256'], f'Metadata changed: {p}'
        files.append(p)
    out = Path('graphs'); out.mkdir(exist_ok=False)
    rows = []
    for r in spec['rows']:
        ids = np.asarray(r['sensor_ids'], dtype=np.int64)
        assert len(ids) == r['expected_nodes'] and len(set(ids)) == len(ids)
        coords, missing, sources, eligible = asof_coordinates(ids, files, r['start'])
        adj, sigma = adjacency(coords, missing)
        original = Path(r['graph'])
        assert sha(original) == r['graph_sha256'], f'Baseline graph changed: {original}'
        with np.load(original, allow_pickle=False) as z:
            baseline = z['x']
        assert baseline.shape == adj.shape
        base_edges = baseline != 0; new_edges = adj != 0
        row = dict(year=r['year'], anchor=r['start'], nodes=len(ids),
                   known_coordinates=int((~missing).sum()), isolated_missing=int(missing.sum()),
                   missing_coordinate_ids=ids[missing].tolist(), coordinate_sources=sources,
                   eligible_snapshots=[str(p) for p in eligible], sigma_km=sigma,
                   baseline_sigma_km=r['sigma_km'], baseline_graph_sha256=r['graph_sha256'],
                   baseline_directed_edges=int(base_edges.sum()), directed_edges=int(new_edges.sum()),
                   removed_directed_edges=int((base_edges & ~new_edges).sum()),
                   added_directed_edges=int((~base_edges & new_edges).sum()),
                   changed_weights_gt_1e_6=int((np.abs(baseline-adj) > 1e-6).sum()))
        meta = dict(order_status='verified_ids', policy='asof_coordinate_isolation_v1',
                    anchor=r['start'], sigma_km=sigma, threshold=.1,
                    coordinate_timing='earliest valid coordinate among frozen snapshots dated <= anchor',
                    missing_policy='zero spatial row and column before unchanged model self-loops',
                    bandwidth='population std of positive directed pairwise distances among known coordinates',
                    catalog='unchanged retrospective anchor-day catalog',
                    joint_intervention=True, source_files=spec['metadata_sources'])
        p = out/f"{r['year']}.npz"
        np.savez_compressed(p, x=adj, sensor_ids=ids, metadata=json.dumps(meta, sort_keys=True),
                            coordinates=coords, coordinate_missing=missing)
        row.update(file=str(p), sha256=sha(p)); rows.append(row)
        print(json.dumps({k:row[k] for k in ['year','nodes','known_coordinates','isolated_missing','directed_edges']}), flush=True)
    assert [r['year'] for r in rows] == list(range(2002, 2026))
    Path('graph_audit.json').write_text(json.dumps(dict(status='passed', tests=result.testsRun,
        source_sha256={p.name:sha(p) for p in [Path('build_asof.py'), Path('graph_source.py'), Path('inputs.json')]},
        metadata_sources=spec['metadata_sources'], rows=rows), indent=2))


if __name__ == '__main__':
    main()
