"""Bounded read-only checks of original PeMS records on allocated CPU resources."""
import csv,gzip,json,pathlib,datetime,math,hashlib,unittest,io
ROOT=pathlib.Path('/mnt/scratch/CRUISE/Du/data/pems/raw')
CASES=[('04','2001_09_01'),('04','2001_12_01'),('04','2002_01_01'),('04','2002_01_15'),('04','2002_01_31'),('05','2005_03_24'),('06','2005_09_01'),('11','1999_07_01')]

def number(s):
    try:return float(s)
    except (ValueError,TypeError):return float('nan')

def main():
    rows=[]
    for ds,day in CASES:
        p=ROOT/('pems'+ds)/('d'+ds+'_text_station_5min_'+day+'.txt.gz')
        r={'district':'PEMS'+ds,'day':day,'path':str(p),'exists':p.is_file()}
        if not p.is_file():rows.append(r);continue
        r.update(records=0,blank_flow=0,finite_zero_flow=0,finite_nonzero_flow=0,
                 finite_flow_with_positive_percent_observed=0,finite_flow_with_zero_percent_observed=0,
                 finite_flow_with_missing_percent_observed=0,negative_flow=0)
        stations=set();example=[]
        with gzip.open(p,'rt',newline='') as f:
            for rec in csv.reader(f):
                if len(rec)<12:raise ValueError('Short record in '+str(p))
                r['records']+=1;stations.add(rec[1]);v=number(rec[9]);obs=number(rec[8])
                if not math.isfinite(v):
                    r['blank_flow']+=1
                    if len(example)<2:example.append({'time':rec[0],'station':rec[1],'samples':rec[7],'percent_observed':rec[8],'total_flow':rec[9]})
                else:
                    r['finite_zero_flow' if v==0 else 'finite_nonzero_flow']+=1
                    if v<0:r['negative_flow']+=1
                    key='finite_flow_with_missing_percent_observed' if not math.isfinite(obs) else 'finite_flow_with_positive_percent_observed' if obs>0 else 'finite_flow_with_zero_percent_observed'
                    r[key]+=1
        r['stations']=len(stations);r['blank_examples']=example
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        r['sha256']=h.hexdigest();rows.append(r)
        print(json.dumps(r),flush=True)
    import pandas as pd
    probe=pd.DataFrame({'Timestamp':['2002-01-01']*2,'Station':[1,2],'Total Flow':[float('nan'),0.]})
    piv=probe.pivot_table(index='Timestamp',columns='Station',values='Total Flow',aggfunc='sum')
    summary={'utc':datetime.datetime.utcnow().isoformat(),'scope':'8 specified raw daily files; no source modification','files':rows,
             'pandas_version':pd.__version__,'notebook_sum_probe':{'blank_input_output':float(piv[1].iloc[0]),'true_zero_input_output':float(piv[2].iloc[0])}}
    pathlib.Path('raw_record_audit.json').write_text(json.dumps(summary,indent=2))
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover('benchmark_v2/tests')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pathlib.Path('protocol_tests.txt').write_text(stream.getvalue());print(stream.getvalue(),flush=True)
    pathlib.Path('protocol_tests.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()},indent=2))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
