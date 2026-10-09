"""Release-candidate utility: unthresholded geographic distances, never road links.

Input CSV must contain sensor_index, station_id, latitude, longitude. IDs must
match a JSON list exactly. Missing coordinates remain NaN; no centroid is added.
Full-dataset execution belongs on allocated compute resources. The built-in
self-test uses three synthetic sensors only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_coordinates(path, expected_ids):
    rows = list(csv.DictReader(path.open()))
    if [int(r['sensor_index']) for r in rows] != list(range(len(rows))):
        raise ValueError('sensor_index must follow the saved array order')
    ids = [int(r['station_id']) for r in rows]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError('Station IDs do not match the declared ordered catalog')
    coords = []
    for r in rows:
        pair = tuple(float(r[k]) if r[k].strip() else math.nan for k in ('latitude', 'longitude'))
        if all(math.isfinite(v) for v in pair):
            if not (-90 <= pair[0] <= 90 and -180 <= pair[1] <= 180):
                raise ValueError('Coordinate outside valid geographic bounds')
        coords.append(pair)
    return coords


def distance(a, b):
    if not all(math.isfinite(x) for x in (*a, *b)):
        return math.nan
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def self_test():
    a, b, missing = (0., 0.), (0., 1.), (math.nan, 1.)
    assert distance(a,a) == 0
    assert abs(distance(a,b)-111.19492664455873) < 1e-9
    assert distance(a,b) == distance(b,a)
    assert math.isnan(distance(a,missing))
    # Identity ordering must fail closed, even when both coordinates coincide.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'coords.csv'
        p.write_text('sensor_index,station_id,latitude,longitude\n0,12,0,0\n1,11,0,0\n')
        try: read_coordinates(p,[11,12])
        except ValueError: pass
        else: raise AssertionError('accepted reordered IDs')
    return {'passed': True, 'checks': 5, 'scope': 'synthetic geometry, missingness, and identity only'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates',type=Path)
    parser.add_argument('--ordered-ids',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return
    if not all((args.coordinates,args.ordered_ids,args.output)) or args.block_size <= 0:
        parser.error('coordinates, ordered-ids, output and positive block-size are required')
    ids=json.loads(args.ordered_ids.read_text())
    coords=read_coordinates(args.coordinates,ids)
    if not coords: raise ValueError('Empty catalog')
    args.output.mkdir(parents=True,exist_ok=False)
    # NumPy is required only for real export, not synthetic self-tests.
    import numpy as np
    files=[]
    for start in range(0,len(ids),args.block_size):
        end=min(start+args.block_size,len(ids))
        values=np.array([[distance(a,b) for b in coords] for a in coords[start:end]],dtype=np.float64)
        path=args.output/f'distance_{start:06d}_{end:06d}.npy'
        np.save(path,values,allow_pickle=False)
        files.append({'path':path.name,'row_start':start,'row_end':end,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={'units':'km','earth_radius_km':6371.0,'station_ids':ids,'shape':[len(ids),len(ids)],
              'threshold':None,'coordinate_fallback':None,'missing_distance':'NaN',
              'coordinate_sha256':hashlib.sha256(args.coordinates.read_bytes()).hexdigest(),
              'ordered_ids_sha256':hashlib.sha256(args.ordered_ids.read_bytes()).hexdigest(),
              'metadata_time_status':'must be established from the coordinate source; no as-of claim',
              'road_connectivity':False,'blocks':files}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
