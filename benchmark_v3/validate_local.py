"""Run only inside a bounded IONA GPU allocation."""
import argparse
import json
import pathlib
import unittest
import torch
from .manifests import METHODS, make, source_record, source_digest
from .runner import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out = pathlib.Path(args.output)
    out.mkdir(exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    report = {'status': 'running', 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'source_sha256': source_digest(source_record(root)),
              'methods': {}}
    for folder in ('benchmark_v3/tests', 'benchmark_v2/tests'):
        suite = unittest.TestLoader().discover(str(root/folder))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        report[folder] = {'ran': result.testsRun, 'passed': result.wasSuccessful(), 'skipped': len(result.skipped)}
        if not result.wasSuccessful():
            raise RuntimeError('Mechanism/regression tests failed')
    for method in METHODS:
        manifest = make(root, args.data, method, 47, out/method, tiny=True)
        result = run(manifest)
        report['methods'][method] = {'status': result['status'], 'years': [r['year'] for r in result['years']]}
        (out/'receipt.json').write_text(json.dumps(report, indent=2))
    report['status'] = 'passed'
    (out/'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
