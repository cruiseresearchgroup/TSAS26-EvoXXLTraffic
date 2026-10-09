"""Reproduce equal-year, then across-seed statistics from the published metric export."""
import csv
import gzip
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def summarize():
    per_seed = defaultdict(list)
    with gzip.open(ROOT / 'results/yearly_metrics.csv.gz', 'rt') as f:
        for r in csv.DictReader(f):
            if int(r['denominator']) == 0:
                assert r['value'] == ''
                continue
            key = tuple(r[k] for k in ('district','method','variant','cohort','horizon','metric'))
            for period in ('full', '2018-2025' if int(r['year']) >= 2018 else 'before-2018'):
                per_seed[key + (period, int(r['seed']))].append(float(r['value']))
    groups = defaultdict(list)
    for key, values in per_seed.items():
        groups[key[:-1]].append((key[-1], statistics.mean(values), len(values)))
    rows = []
    for key, values in sorted(groups.items()):
        assert len({v[2] for v in values}) == 1
        row = dict(zip(('district','method','variant','cohort','horizon','metric','period'), key))
        row.update(seeds=len(values), years=values[0][2], mean=statistics.mean(v[1] for v in values), std=statistics.stdev(v[1] for v in values))
        rows.append(row)
    return rows

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = summarize()
    with args.output.open('x') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Wrote', len(rows), 'aggregate rows to', args.output)
