"""Compare vectorized reader to reference on real days, then build two full years."""
import json
from pathlib import Path
import numpy as np
from .pems import read_window as reference
from .pems_fast import read_window as fast


def main():
    result=[]
    for ds,start in [('04','2002-01-01'),('05','2005-03-24')]:
        for policy in ('finite_reported','positive_percent_observed'):
            kwargs=dict(root='/mnt/scratch/CRUISE/Du/data/pems/raw',district=ds,start=start,days=2,label_policy=policy)
            a=reference(**kwargs);b=fast(**kwargs)
            for key in ['values','observed','sensor_ids','timestamps','record_present','flow_reported','percent_observed']:
                assert np.array_equal(a[key],b[key],equal_nan=True),(ds,start,policy,key)
            assert a['provenance']['sources']==b['provenance']['sources']
            assert a['provenance']['off_anchor_records_excluded']==b['provenance']['off_anchor_records_excluded']
            result.append({'district':ds,'start':start,'days':2,'policy':policy,'status':'passed'})
            print('READER EQUIVALENCE PASSED',ds,start,policy,flush=True)
    Path('reader_equivalence.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
