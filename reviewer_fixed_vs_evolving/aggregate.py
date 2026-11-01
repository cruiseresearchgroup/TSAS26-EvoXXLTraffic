"""Rebuild all deliverables only from saved annual TEST metrics and receipts."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .common import PLAN, SOURCE, YEARS, SEEDS, METHODS, sha, save, seed_summary, paired_summary


def aggregate(root):
    root = Path(root).resolve()
    annual = []
    receipts = []
    series = {}
    artifacts = []
    for district, method, pop, seed in itertools.product(['PEMS05','PEMS06'],METHODS,['E','F'],SEEDS):
        label = f'{district}-{method}-{pop}-s{seed}'
        receipt_path = root / 'receipts' / (label + '.json')
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'PASS', ('Incomplete/failed chain', label)
        result_path = root / 'results' / label / 'results.json'
        assert sha(result_path) == receipt['result_sha256']
        result = json.loads(result_path.read_text())
        assert result['status'] == 'passed' and [r['year'] for r in result['years']] == YEARS
        assert receipt['manifest']['max_batches'] is None and receipt['manifest']['max_eval_origins'] is None
        assert len(receipt['years']) == 8
        artifacts.append(dict(path=str(result_path), sha256=sha(result_path)))
        for row in result['years']:
            assert row['epochs_completed'] == len(row['history']) and 1 <= row['epochs_completed'] <= 100
            assert row['optimizer_steps'] > 0 and not row['training_skipped']
            assert sha(root / 'results' / label / f"{row['year']}.pkl") == row['checkpoint_sha256']
            if row['year'] == 2018 or method == 'Retrain':
                assert row['parent'] is None
            else:
                assert row['parent']['year'] == row['year'] - 1
                assert Path(row['parent']['path']).parent == result_path.parent
            expected = ['S','all','new'] if pop == 'E' else ['S']
            assert set(row['metrics']['cohorts']) == set(expected)
            assert row['metrics']['origins'] == 1763
            for cohort in expected:
                scores = row['metrics']['cohorts'][cohort]['overall']
                for metric in ['MAE','RMSE','MAPE']:
                    value = scores[metric]
                    n = scores['mape_count' if metric == 'MAPE' else 'observed_count']
                    assert value is not None and np.isfinite(value) and n > 0, ('Empty/nonfinite metric',label,row['year'],cohort,metric)
                    annual.append(dict(district=district, method=method, train_population=pop,
                        eval_population=cohort, seed=seed, year=row['year'], metric=metric,
                        value=value, n_targets=n))
                    key=(district,method,pop,cohort,metric)
                    series.setdefault(key,{}).setdefault(seed,[]).append(value)
        receipts.append({k:v for k,v in receipt.items() if k != 'manifest'})
    assert len(annual) == 1152
    summary=[]
    for key, by_seed in sorted(series.items()):
        assert set(by_seed)==set(SEEDS) and all(len(v)==8 for v in by_seed.values())
        result=seed_summary(by_seed)
        summary.append(dict(zip(['district','method','train_population','eval_population','metric'],key),
                            statistic='population',mean=result['mean'],sd=result['sd'],n_seeds=3,n_years=8))
    for district,method,metric in itertools.product(['PEMS05','PEMS06'],METHODS,['MAE','RMSE','MAPE']):
        result=paired_summary(series[(district,method,'E','S',metric)],series[(district,method,'F','S',metric)])
        summary.append(dict(district=district,method=method,train_population='E-F',eval_population='S',
                            metric=metric,statistic='paired_difference',mean=result['mean'],sd=result['sd'],n_seeds=3,n_years=8))
    assert len(summary)==60
    output=root/'deliverables';output.mkdir(exist_ok=True)
    for name, rows in [('results_fixed_vs_evolving.csv',annual),('summary_fixed_vs_evolving.csv',summary)]:
        target=output/name
        with target.with_suffix('.tmp').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        target.with_suffix('.tmp').replace(target)
    lookup={(r['district'],r['method'],r['train_population'],r['metric']):r
            for r in summary if r['eval_population']=='S'}
    sentences=[]
    for district,method in itertools.product(['PEMS05','PEMS06'],METHODS):
        pieces=[]
        for metric in ['MAE','RMSE','MAPE']:
            e,f,diff=[lookup[(district,method,p,metric)] for p in ['E','F','E-F']]
            unit='%' if metric=='MAPE' else ''
            delta_unit=' percentage points' if metric=='MAPE' else ''
            pieces.append(f"{metric} of {e['mean']:.3f} ± {e['sd']:.3f}{unit} versus {f['mean']:.3f} ± {f['sd']:.3f}{unit} (paired E − F: {diff['mean']:.3f} ± {diff['sd']:.3f}{delta_unit})")
        sentences.append(f"For {district} with {method}, E versus F on the fixed sensor intersection yielded TEST " + ', '.join(pieces) + ', with eight years averaged equally within each of three seeds (mean ± sample SD).')
    (output/'RESULTS_EN.txt').write_text('\n'.join(sentences)+'\n')
    audit=json.loads((root/'INPUT_AUDIT.json').read_text())
    code=json.loads((root/'CODE_RECEIPT.json').read_text())
    receipt=dict(status='COMPLETE',run_id=root.name,code_commit=code['code_commit'],
                 base_release_commit=PLAN['release_commit'],release_tag=PLAN['release_tag'],
                 hf_revision=PLAN['hf_revision'],evaluation_split='TEST',
                 protocol=json.loads((SOURCE/'reviewer_fixed_vs_evolving/PROTOCOL.json').read_text()),
                 protocol_sha256=sha(SOURCE/'reviewer_fixed_vs_evolving/PROTOCOL.json'),
                 input_data_graph_sha256=audit['frozen_inputs'],fixed_sensor_sets=audit['districts'],
                 derived_inputs=audit['derived_inputs'],chains=receipts,annual_result_artifacts=artifacts,
                 gpu_models=sorted(set(r['gpu'] for r in receipts)),
                 row_counts={'annual':len(annual),'summary':len(summary)},
                 deliverable_sha256={p.name:sha(p) for p in output.iterdir() if p.name in ['results_fixed_vs_evolving.csv','summary_fixed_vs_evolving.csv','RESULTS_EN.txt']})
    save(output/'RECEIPT.json',receipt)
    print(json.dumps({'status':'COMPLETE','directory':str(output),'chains':len(receipts),'annual_rows':len(annual),'summary_rows':len(summary)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    aggregate(p.parse_args().root)
