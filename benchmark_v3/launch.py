"""Standard-library launcher, entered only after the IONA single-winner gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def extract(path, expected, target):
    if sha(path) != expected:
        raise ValueError('Archive SHA mismatch: '+str(path))
    target.mkdir(exist_ok=False)
    # Every archive is an explicitly SHA-bound locally prepared input, never a
    # file downloaded or chosen by the allocated process.
    subprocess.run(['tar', '-xf', str(path), '-C', str(target)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--site', choices=['katana', 'glab', 'nci'], required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--environment', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not (os.environ.get('PBS_JOBID') or os.environ.get('JOB_ID')):
        raise RuntimeError('Scheduler allocation required')
    if a.site == 'nci' and not os.environ.get('PBS_JOBFS'):
        raise RuntimeError('NCI environment and data must be unpacked in allocated jobfs')
    spec_path = Path(a.spec).resolve()
    if sha(spec_path) != a.spec_sha256:
        raise ValueError('Submitted specification changed')
    spec = json.loads(spec_path.read_text())
    here = Path(__file__).resolve().parent
    assets = Path(a.assets).resolve()
    manifest = json.loads((here/'artifacts.json').read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = Path(os.environ.get('PBS_JOBFS') or os.environ.get('TMPDIR') or '/tmp')
    if shutil.disk_usage(scratch).free < 24*1024**3:
        raise RuntimeError('Insufficient allocated temporary space')
    receipt = {'status': 'preparing', 'site': a.site, 'spec_sha256': a.spec_sha256,
               'started': time.time(), 'scheduler_job': os.environ.get('PBS_JOBID') or os.environ.get('JOB_ID')}
    try:
        with tempfile.TemporaryDirectory(prefix='tsas-v3-', dir=scratch) as folder:
            temp = Path(folder)
            extract(a.environment, manifest['environment_sha256'], temp/'env')
            python = temp/'env/bin/python'
            subprocess.run([str(python), str(temp/'env/bin/conda-unpack')], check=True)
            extract(assets/manifest['deps_archive'], manifest['deps_sha256'], temp/'overlay')
            extract(here/'source.tar.gz', manifest['source_archive_sha256'], temp/'source')
            district = spec.get('district', 'pems05')
            extract(assets/f'{district}-frozen-inputs.tar.gz', manifest['input_archives'][district], temp/'data')
            env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONPATH=str(temp/'overlay/deps')+':'+str(temp/'source'),
                       TMPDIR=str(temp), XDG_CACHE_HOME=str(temp/'cache'))
            command = [str(python), '-B', '-u', '-m', 'benchmark_v3.entry', '--spec', str(spec_path),
                       '--data', str(temp/'data'), '--output', str(out/'validation')]
            with (out/'validation.log').open('w') as log:
                result = subprocess.run(command, cwd=temp/'source', env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((out/'validation.log').read_text()[-6000:], flush=True)
                raise RuntimeError(f'Validation process exited {result.returncode}')
            receipt['status'] = 'passed'
    except BaseException as e:
        receipt.update(status='failed', error=repr(e))
        raise
    finally:
        receipt['seconds'] = time.time()-receipt['started']
        (out/'launch-receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
