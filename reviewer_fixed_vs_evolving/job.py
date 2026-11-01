"""A single allocation, at most two separate experiment processes; record both exits."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import SOURCE, save


def execute(root, mode, district=None, method=None, seed=None):
    root=Path(root).resolve()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    base=Path(os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp')
    with tempfile.TemporaryDirectory(prefix='tsas-r2-'+os.environ['SLURM_JOB_ID']+'-',dir=base) as tmp:
        children=[];streams=[]
        def launch(module,pop,label,extra):
            own=Path(tmp)/label;own.mkdir()
            child_env=dict(env,TMPDIR=str(own),XDG_CACHE_HOME=str(own/'cache'),TORCH_HOME=str(own/'torch'))
            folder=root/('profile' if mode=='profile' else 'logs');folder.mkdir(exist_ok=True)
            stream=(folder/(label+'.log')).open('x');streams.append(stream)
            command=[sys.executable,'-B','-u','-m',module,'--root',str(root),'--population',pop]+extra
            proc=subprocess.Popen(command,cwd=SOURCE,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            return dict(label=label,process=proc,command=command)
        if mode=='profile':
            (root/'profile').mkdir(exist_ok=False)
            subprocess.run([sys.executable,'-B','-m','unittest','reviewer_fixed_vs_evolving.test_contracts','-v'],cwd=SOURCE,env=env,check=True)
            for pop in ['E','F']:
                label='single-'+pop;c=launch('reviewer_fixed_vs_evolving.profile',pop,label,['--label',label])
                assert c['process'].wait()==0,'Single resource profile failed'
            single=[json.loads((root/'profile'/('single-'+p+'.json')).read_text()) for p in ['E','F']]
            capacity=min(r['gpu_capacity_bytes'] for r in single)
            assert sum(r['peak_reserved_bytes'] for r in single)+.2*capacity<=capacity
            assert sum(r['max_rss_bytes'] for r in single)<16*1024**3
            for pop in ['E','F']:
                label='paired-'+pop
                children.append(launch('reviewer_fixed_vs_evolving.profile',pop,label,['--label',label]))
        else:
            gate=json.loads((root/'profile/PASS.json').read_text());assert gate['status']=='PASS'
            # Recheck measured demand against this allocation's actual device.
            import torch
            assert torch.cuda.is_available() and torch.cuda.device_count()==1
            capacity=torch.cuda.get_device_properties(0).total_memory
            demand=sum(r['peak_reserved_bytes'] for r in gate['paired'])
            assert demand+.2*capacity<=capacity, 'Allocated GPU fails measured packing margin'
            save(root/(f"resource_gate_{os.environ['SLURM_JOB_ID']}.json"),dict(
                status='PASS',gpu=torch.cuda.get_device_name(0),capacity_bytes=capacity,
                measured_pair_reserved_bytes=demand,safety_margin=.2,
                profile_job_id=gate['slurm_job_id']))
            for pop in ['E','F']:
                label=f'{district}-{method}-{pop}-s{seed}'
                children.append(launch('reviewer_fixed_vs_evolving.run_chain',pop,label,
                    ['--district',district,'--method',method,'--seed',str(seed)]))
        exits=[dict(label=c['label'],exit_code=c['process'].wait()) for c in children]
        for stream in streams:stream.close()
        save(root/(f"{mode}_{os.environ['SLURM_JOB_ID']}_exits.json"),exits)
        assert all(r['exit_code']==0 for r in exits),exits
        if mode=='profile':
            paired=[json.loads((root/'profile'/('paired-'+p+'.json')).read_text()) for p in ['E','F']]
            assert sum(r['peak_reserved_bytes'] for r in paired)+.2*capacity<=capacity
            assert sum(r['max_rss_bytes'] for r in paired)<16*1024**3
            # Paired wall time must beat serial wall time with <=10% tolerance.
            assert max(r['seconds'] for r in paired)<=1.1*sum(r['seconds'] for r in single), 'Pairing loses throughput'
            save(root/'profile/PASS.json',dict(status='PASS',single=single,paired=paired,
                slurm_job_id=os.environ['SLURM_JOB_ID'],gpu_safety_margin=.2,
                cpu_threads_per_process=2,processes_per_gpu=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=['profile','formal'],required=True)
    p.add_argument('--district');p.add_argument('--method');p.add_argument('--seed',type=int)
    a=p.parse_args();execute(a.root,a.mode,a.district,a.method,a.seed)
