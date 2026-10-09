"""Build revision caches in new directories; never alter legacy NPZ or raw files."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from .pems_fast import read_window,sha
from .data import prepare_year,save_year,load_year,split_origins


def build(spec,out):
    tic=time.time();ds,year=spec['district'],spec['year']
    if ds=='04' and year<2002:raise ValueError('PEMS04 experiments start in 2002')
    raw=read_window('/mnt/scratch/CRUISE/Du/data/pems/raw',ds,spec['start'],label_policy='finite_reported',file_overrides=spec.get('raw_file_overrides'))
    if raw['sensor_ids'].tolist()!=spec['sensor_ids']:raise ValueError('Original anchor ID membership changed')
    if raw['provenance']['sources'][0]['sha256']!=spec['anchor_sha256']:raise ValueError('Original anchor file changed since audit')
    data=prepare_year(raw['values'],raw['observed'],raw['sensor_ids'],raw['timestamps'],provenance=raw['provenance'])
    # Keep original quality separate from the chosen main target eligibility mask.
    data.update(record_present=raw['record_present'],flow_reported=raw['flow_reported'],percent_observed=raw['percent_observed'])
    data['metadata']['quality_fields']={'record_present':'Original station/time record exists',
        'flow_reported':'Present record with finite published Total Flow; includes valid zero and upstream imputation',
        'percent_observed':'Original PeMS field; may be NaN; positive is not equivalent to 100% directly observed',
        'observed':'Internal loss-mask name: eligible published targets under finite_reported policy, not a direct-observation claim'}
    coverage={}
    for split,a,b in zip(['train','val','test'],data['bounds'][:-1],data['bounds'][1:]):
        mask=raw['observed'][a:b];quality=raw['percent_observed'][a:b]
        coverage[split]={'slots':int(mask.size),'finite_reported':int(mask.sum()),
            'positive_percent_observed':int((mask & np.isfinite(quality)&(quality>0)).sum()),
            'missing_record':int((~raw['record_present'][a:b]).sum()),
            'present_nonfinite_flow':int((raw['record_present'][a:b]&~mask).sum()),
            'zero_flow':int((mask & (raw['values'][a:b]==0)).sum()),
            'origins':len(split_origins(data,split))}
        if not mask.any():raise ValueError(f'No finite published labels in {split}; cannot declare this task ready')
    dest=out/f'pems{ds}'/str(year);dest.mkdir(parents=True,exist_ok=False)
    path=dest/'data.npz';data_hash=save_year(path,data)
    n=len(raw['sensor_ids']);ids=raw['sensor_ids'];graph=Path(spec['graph'])
    if sha(graph)!=spec['graph_sha256']:raise ValueError('Graph changed since ID verification')
    with np.load(graph,allow_pickle=False) as z:adj=z['x']
    assert adj.shape==(n,n)
    gm={'order_status':'verified_ids','verification':'Original anchor IDs and full notebook-kernel reproduction',
        'original_path':str(graph),'original_sha256':spec['graph_sha256'],
        'coordinate_source':spec['coordinate_source'],'missing_coordinate_ids':spec['missing_coordinate_ids'],
        'future_coordinate_ids':spec['future_coordinate_ids'],'temporal_metadata_policy_frozen':False,
        'semantics':'Retrospective geographic proximity; not physical road connectivity or proof of contemporaneous metadata availability'}
    gp=dest/'graph.npz'
    with gp.open('xb') as f:np.savez(f,x=adj,sensor_ids=ids,metadata=np.array(json.dumps(gm)))
    del raw,data,adj;gc.collect()
    check=load_year(path)
    assert np.array_equal(check['sensor_ids'],ids)
    assert np.isfinite(check['x']).all() and np.isfinite(check['y']).all()
    with np.load(path,allow_pickle=False) as z:
        assert np.array_equal(z['flow_reported'],check['observed'])
        assert z['percent_observed'].shape==check['observed'].shape
    row={'district':'PEMS'+ds,'year':year,'data':str(path.resolve()),'data_sha256':data_hash,
        'graph':str(gp.resolve()),'graph_sha256':sha(gp),'sensors':n,'bytes':path.stat().st_size+gp.stat().st_size,
        'coverage':coverage,'seconds':time.time()-tic,'status':'passed',
        'label_policy':'finite_reported','full_experiment_ready':False}
    (dest/'summary.json').write_text(json.dumps(row,indent=2));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();specs=json.loads(args.manifest.read_text());args.output.mkdir(parents=True,exist_ok=False)
    rows=[];tic=time.time()
    for spec in specs:
        try:row=build(spec,args.output)
        except Exception as e:row={'district':'PEMS'+spec['district'],'year':spec['year'],'status':'failed','error':f'{type(e).__name__}: {e}'}
        rows.append(row);gc.collect()
        report={'rows':rows,'expected_years':len(specs),'complete':len(rows)==len(specs),'seconds':time.time()-tic,
            'status':'passed' if len(rows)==len(specs) and all(r['status']=='passed' for r in rows) else 'incomplete_or_failed',
            'scope':'Full first31-day caches; original quality preserved; no model training; graph temporal policy pending'}
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ['coverage']}),flush=True)
    if any(r['status']!='passed' for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
