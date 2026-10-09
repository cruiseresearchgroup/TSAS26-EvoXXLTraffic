"""Read-only verification of saved graphs using IDs from original anchor files.

Graph reproduction uses notebook coordinates (including its centroid fallback)
and independently computes sigma, rather than fitting sigma to saved weights.
Historical metadata may postdate the anchor: report this, do not hide it.
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .graph_source import pems_meta_sources, build_station_catalog, catalog_coordinates, haversine_cross, validate_full_graph
from .runner import sha

REPO=Path('/mnt/scratch/CRUISE/Du/code/evo-xxltraffic')
RAW=Path('/mnt/scratch/CRUISE/Du/data/pems/raw')


def distance_sigma(coords, block_size=128):
    """Two-pass population std of positive distances, without N-by-N temporaries."""
    total=0.;count=0
    for i in range(0,len(coords),block_size):
        d=haversine_cross(coords[i:i+block_size],coords);positive=d[d>0]
        total+=float(positive.sum(dtype=np.float64));count+=positive.size
    if not count:raise ValueError('No positive inter-station distances')
    mean=total/count;ss=0.
    for i in range(0,len(coords),block_size):
        d=haversine_cross(coords[i:i+block_size],coords);positive=d[d>0]
        ss+=float(np.square(positive-mean).sum(dtype=np.float64))
    sigma=float(np.sqrt(ss/count))
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Undefined distance kernel scale')
    return sigma


def main():
    tic=time.time();rows=[];catalogs={};sources={}
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,default=Path(__file__).with_name('graph_inputs.json'))
    inputs=json.loads(parser.parse_args().inputs.read_text())
    out=Path('graph_identity_output');out.mkdir(exist_ok=False)
    for spec in inputs:
        ds,year,start=spec['district'],spec['year'],spec['start']
        row=dict(spec)
        try:
            anchor=RAW/f'pems{ds}'/f'd{ds}_text_station_5min_{start.replace("-","_")}.txt.gz'
            ids=np.sort(pd.read_csv(anchor,header=None,usecols=[1])[1].unique().astype(np.int64))
            row.update(anchor=str(anchor),anchor_sha256=sha(anchor),sensor_ids=ids.tolist(),nodes=len(ids))
            assert len(ids)==spec['expected_nodes'], 'Anchor ID count differs from original notebook'
            if ds not in catalogs:
                _,files,label=pems_meta_sources(REPO,ds)
                catalogs[ds]=build_station_catalog(files)
                sources[ds]={'label':label,'files':[{'path':str(p),'sha256':sha(p)} for p in files]}
            cat=catalogs[ds];coords,missing=catalog_coordinates(ids,cat,fill_missing=True)
            sigma=distance_sigma(coords)
            graph=REPO/'xxltrafficdata/preprocessed'/f'pems{ds}'/'graph'/f'{year}_adj.npz'
            with np.load(graph,allow_pickle=False) as z:adj=z['x']
            assert adj.shape==(len(ids),len(ids)), 'Graph shape differs from anchor catalog'
            v=validate_full_graph(adj,coords,sigma,128)
            future=[int(s) for s in ids if s in cat and cat[s].coordinate_source and Path(cat[s].coordinate_source).stem[-10:].replace('_','-')>start]
            row.update(graph=str(graph),graph_sha256=sha(graph),sigma_km=sigma,
                       missing_coordinate_ids=ids[missing].tolist(),future_coordinate_ids=future,
                       coordinate_source=sources[ds]['label'],compared_entries=v.compared_entries,
                       max_abs_error=v.max_abs_error,topology_mismatches=v.topology_mismatches,
                       status='verified_anchor_ids' if v.passed else 'graph_mismatch')
            if ds=='07' and year==2017:
                row['previous_ambiguous_column_4804_id']=int(ids[4804])
                row['candidate_775726_present']=bool(775726 in ids)
                row['candidate_775729_present']=bool(775729 in ids)
        except Exception as exc:
            row.update(status='unresolved',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        result={'scope':f'{len(inputs)} anchor files and full graph matrices; no training or raw-data changes',
                'rows':rows,'metadata_sources':sources,'seconds':time.time()-tic,
                'complete':len(rows)==len(inputs),'verified':sum(r['status']=='verified_anchor_ids' for r in rows),
                'caveat':'Proves compatibility with the declared anchor-ID ordering. Co-located IDs cannot be distinguished by adjacency alone; original anchor records resolve catalog membership. Does not independently prove every legacy RawData column.'}
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(ds,year,row['status'],flush=True)


if __name__=='__main__':main()
