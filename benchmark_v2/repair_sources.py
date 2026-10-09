"""Verify failed-year daily gzip files, fetch separate official copies and rebuild.

Original files and completed caches remain untouched. Credentials stay inside the
existing remote downloader source; they are neither staged nor logged.
"""
import argparse
import ast
import gzip
import json
import time
from datetime import datetime,timedelta
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import requests
from .pems_fast import sha
from .build_caches import build

RAW=Path('/mnt/scratch/CRUISE/Du/data/pems/raw')


def integrity(path):
    try:
        with gzip.open(path,'rb') as f:
            for _ in iter(lambda:f.read(1048576),b''):pass
        return None
    except Exception as e:return f'{type(e).__name__}: {e}'


def session():
    tree=ast.parse(Path('/mnt/scratch/CRUISE/Du/code/evo-xxltraffic/claude/download_pems.py').read_text())
    credentials={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['BASE_URL','USERNAME','PASSWORD']:
                    credentials[target.id]=ast.literal_eval(node.value)
    assert urlsplit(credentials['BASE_URL']).hostname=='pems.dot.ca.gov'
    s=requests.Session();s.headers['User-Agent']='Mozilla/5.0'
    s.post(credentials['BASE_URL'],data={'redirect':'','username':credentials['USERNAME'],'password':credentials['PASSWORD'],'login':'Login'},timeout=30).raise_for_status()
    if 'Logout' not in s.get(credentials['BASE_URL'],timeout=30).text:
        raise RuntimeError('Existing PeMS login did not authenticate; no credentials changed')
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reuse-downloads',type=Path);args=p.parse_args()
    specs=json.loads(args.manifest.read_text());out=Path('repaired_output');out.mkdir(exist_ok=False)
    rows=[];audits=[];client=None;tic=time.time()
    for spec in specs:
        try:
            ds=spec['district'];year=spec['year'];day0=datetime.strptime(spec['start'],'%Y-%m-%d')
            catalog=json.loads((RAW/'.a2tta-backfill-20260925/catalog'/f'd{ds}-{year}.json').read_text())
            catalog={r['name']:r for r in catalog};overrides={}
            for k in range(31):
                day=day0+timedelta(days=k);name=f'd{ds}_text_station_5min_{day:%Y_%m_%d}.txt.gz';original=RAW/f'pems{ds}'/name
                if not original.exists():continue # Existing absent-day policy retains an explicit false mask.
                error=integrity(original)
                if error is None:continue
                record={'district':ds,'year':year,'name':name,'original_sha256':sha(original),'original_bytes':original.stat().st_size,'integrity_error':error,'status':'unresolved'}
                audits.append(record)
                entry=catalog[name];url=entry['url']
                if urlsplit(url).hostname!='pems.dot.ca.gov':raise ValueError('Unexpected official source host')
                dest=args.reuse_downloads/f'pems{ds}'/name if args.reuse_downloads else None
                reused=bool(dest and dest.is_file())
                if not reused:
                    if client is None:client=session()
                    dest=Path('repairs')/f'pems{ds}'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    with client.get(url,stream=True,timeout=45) as response:
                        response.raise_for_status()
                        if urlsplit(response.url).hostname!='pems.dot.ca.gov':raise ValueError('Unexpected download redirect')
                        with dest.open('xb') as f:
                            for block in response.iter_content(1048576):f.write(block)
                file_id=entry.get('file_id',parse_qs(urlsplit(url).query).get('download',[None])[0])
                record.update(candidate_sha256=sha(dest),candidate_bytes=dest.stat().st_size,official_file_id=file_id,official_catalog_bytes=entry['bytes'],reused_download=reused)
                if dest.stat().st_size!=int(entry['bytes']):raise ValueError('Official download size disagrees with saved catalog')
                check=integrity(dest)
                if check:raise ValueError('Official candidate also fails gzip integrity: '+check)
                record.update(status='verified_replacement',replacement=str(dest.resolve()))
                overrides[name]=str(dest.resolve());print('VERIFIED REPLACEMENT',name,flush=True)
                (out/'repairs.json').write_text(json.dumps(audits,indent=2));time.sleep(1)
            updated=dict(spec,raw_file_overrides=overrides)
            row=build(updated,out)
            row['replaced_source_files']=len(overrides)
        except Exception as e:
            row={'district':'PEMS'+spec['district'],'year':spec['year'],'status':'failed','error':f'{type(e).__name__}: {e}'}
        rows.append(row)
        (out/'repairs.json').write_text(json.dumps(audits,indent=2))
        (out/'summary.json').write_text(json.dumps({'rows':rows,'complete':len(rows)==len(specs),'status':'passed' if len(rows)==len(specs) and all(r['status']=='passed' for r in rows) else 'incomplete_or_failed','seconds':time.time()-tic},indent=2))
        print('REBUILD',spec['district'],spec['year'],row['status'],flush=True)
    if any(r['status']!='passed' for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
