"""Vectorized daily-file reader for the full cache rebuild.

Same selected fields, anchor rule, duplicate rejection and masks as pems.py.
Each day is processed separately to bound memory; no model imports are needed.
"""
import hashlib
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read_window(root,district,start,days=31,*,label_policy,file_overrides=None):
    if label_policy not in ('finite_reported','positive_percent_observed'):
        raise ValueError('Select and report an explicit PeMS label policy')
    day0=datetime.strptime(start,'%Y-%m-%d')
    ds=str(district).removeprefix('PEMS').removeprefix('pems').zfill(2)
    base=Path(root)/('pems'+ds)
    file_overrides=file_overrides or {}
    def original_path(day):return base/f'd{ds}_text_station_5min_{day:%Y_%m_%d}.txt.gz'
    def path(day):
        original=original_path(day)
        replacement=Path(file_overrides.get(original.name,original))
        if replacement.name!=original.name:raise ValueError('Replacement must retain original daily filename')
        return replacement
    first=path(day0)
    if not first.is_file():raise ValueError('Missing anchor-day file: '+str(first))
    ids=np.sort(pd.read_csv(first,header=None,usecols=[1],dtype={1:np.int64})[1].unique())
    if not len(ids):raise ValueError('Empty anchor catalog')
    shape=(days*288,len(ids));values=np.full(shape,np.nan,dtype=np.float32)
    quality=np.full(shape,np.nan,dtype=np.float32);present=np.zeros(shape,dtype=bool)
    sources=[];off_anchor=0
    for k in range(days):
        day=day0+timedelta(days=k);p=path(day);rec={'path':str(p),'exists':p.is_file()};sources.append(rec)
        if p!=original_path(day):
            rec['original_path']=str(original_path(day))
            rec['original_sha256']=sha(original_path(day))
            rec['replacement_reason']='Verified official re-download after original gzip integrity failure'
        if not p.is_file():continue
        rec['sha256']=sha(p)
        df=pd.read_csv(p,header=None,usecols=[0,1,8,9],dtype={0:str,1:np.int64,8:str,9:str},keep_default_na=False)
        selected=df[1].isin(ids);off_anchor+=int((~selected).sum());df=df.loc[selected]
        dt=pd.to_datetime(df[0],format='%m/%d/%Y %H:%M:%S',errors='raise',exact=True)
        offset=(dt-pd.Timestamp(day)).dt.total_seconds().to_numpy()
        if np.any((offset<0)|(offset>=86400)|(offset%300!=0)):
            raise ValueError('Timestamp outside day/grid')
        j=np.searchsorted(ids,df[1].to_numpy());i=offset.astype(np.int64)//300
        flat=i*len(ids)+j
        if len(np.unique(flat))!=len(flat):
            raise ValueError('Duplicate timestamp/station; define DST or duplicate policy explicitly')
        i+=k*288
        present[i,j]=True
        values[i,j]=pd.to_numeric(df[9],errors='coerce').to_numpy(dtype=np.float32)
        quality[i,j]=pd.to_numeric(df[8],errors='coerce').to_numpy(dtype=np.float32)
    reported=present & np.isfinite(values);observed=reported.copy()
    if label_policy=='positive_percent_observed':observed &= np.isfinite(quality)&(quality>0)
    origin=int((day0-datetime(1970,1,1)).total_seconds())
    return dict(values=values,observed=observed,sensor_ids=ids,
        timestamps=origin+np.arange(days*288,dtype=np.int64)*300,
        record_present=present,flow_reported=reported,percent_observed=quality,
        provenance={'observation_mask_source':'original PeMS daily station records','label_policy':label_policy,
            'sources':sources,'district':'PEMS'+ds,'window_start':start,'days':days,
            'off_anchor_records_excluded':off_anchor,'reader':'vectorized daily pandas; fields0,1,8,9',
            'sensor_order':'sorted anchor-day station IDs, matching yearly_nodes notebook policy',
            'time_basis':'naive PeMS wall-clock calendar seconds; not UTC; duplicate slots rejected'})
