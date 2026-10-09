"""Read existing original PeMS daily files without changing or downloading them.

The caller explicitly chooses a label policy. Both finite reported flows and the
raw % Observed field are preserved. Local wall-clock slots are encoded on a naive
calendar grid; they are not asserted to be UTC. Duplicate slots fail explicitly.
"""
import csv,gzip,hashlib,math
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np


def _float(s):
    try:return float(s)
    except (TypeError,ValueError):return float('nan')

def read_window(root,district,start,days=31,*,label_policy):
    if label_policy not in ['finite_reported','positive_percent_observed']:
        raise ValueError('Select and report an explicit PeMS label policy')
    day0=datetime.strptime(start,'%Y-%m-%d')
    ds=str(district).removeprefix('PEMS').removeprefix('pems').zfill(2)
    root=Path(root)/('pems'+ds)
    def path(day):return root/('d'+ds+'_text_station_5min_'+day.strftime('%Y_%m_%d')+'.txt.gz')
    first=path(day0)
    if not first.is_file():raise ValueError('Missing anchor-day file: '+str(first))
    with gzip.open(first,'rt',newline='') as f:
        ids=np.array(sorted({int(r[1]) for r in csv.reader(f)}),dtype=np.int64)
    if not len(ids):raise ValueError('Empty anchor catalog')
    col={int(s):i for i,s in enumerate(ids)};shape=(days*288,len(ids))
    values=np.full(shape,np.nan,dtype=np.float32);quality=np.full(shape,np.nan,dtype=np.float32)
    present=np.zeros(shape,dtype=bool);sources=[];off_anchor=0
    for dayidx in range(days):
        day=day0+timedelta(days=dayidx);p=path(day)
        rec={'path':str(p),'exists':p.is_file()};sources.append(rec)
        if not p.is_file():continue  # Mask remains false; never manufacture a valid zero.
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):digest.update(block)
        rec['sha256']=digest.hexdigest()
        with gzip.open(p,'rt',newline='') as f:
            for r in csv.reader(f):
                if len(r)<12:raise ValueError('Short record in '+str(p))
                sid=int(r[1])
                if sid not in col:off_anchor+=1;continue
                dt=datetime.strptime(r[0],'%m/%d/%Y %H:%M:%S')
                offset=(dt-day).total_seconds()
                if offset<0 or offset>=86400 or offset%300:raise ValueError('Timestamp outside day/grid')
                i=dayidx*288+int(offset/300);j=col[sid]
                if present[i,j]:raise ValueError('Duplicate timestamp/station; define DST or duplicate policy explicitly')
                present[i,j]=True;values[i,j]=_float(r[9]);quality[i,j]=_float(r[8])
    reported=present & np.isfinite(values)
    observed=reported.copy()
    if label_policy=='positive_percent_observed':observed &= np.isfinite(quality) & (quality>0)
    origin=int((day0-datetime(1970,1,1)).total_seconds())
    return dict(values=values,observed=observed,sensor_ids=ids,
                timestamps=origin+np.arange(days*288,dtype=np.int64)*300,
                record_present=present,flow_reported=reported,percent_observed=quality,
                provenance={'observation_mask_source':'original PeMS daily station records',
                            'label_policy':label_policy,'sources':sources,'district':'PEMS'+ds,
                            'window_start':start,'days':days,'off_anchor_records_excluded':off_anchor,
                            'sensor_order':'sorted anchor-day station IDs, matching yearly_nodes notebook policy',
                            'time_basis':'naive PeMS wall-clock calendar seconds; not UTC; duplicate slots rejected'})
