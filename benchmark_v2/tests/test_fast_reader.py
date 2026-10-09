import gzip
import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark_v2.pems import read_window as reference
from benchmark_v2.pems_fast import read_window as fast


class FastReaderTests(unittest.TestCase):
    def test_masks_zero_quality_and_missing_day_match_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pems05';p.mkdir()
            with gzip.open(p/'d05_text_station_5min_2006_01_01.txt.gz','wt') as f:
                f.write('01/01/2006 00:00:00,500001,5,1,N,ML,1,10,100,0,0,0\n')
                f.write('01/01/2006 00:00:00,500002,5,1,N,ML,1,0,0,17,0,0\n')
                f.write('01/01/2006 00:05:00,500001,5,1,N,ML,1,0,0,,0,0\n')
            for policy in ['finite_reported','positive_percent_observed']:
                a=reference(tmp,'05','2006-01-01',days=2,label_policy=policy)
                b=fast(tmp,'05','2006-01-01',days=2,label_policy=policy)
                for k in ['values','observed','sensor_ids','timestamps','record_present','flow_reported','percent_observed']:
                    self.assertTrue(np.array_equal(a[k],b[k],equal_nan=True),k)
                self.assertEqual(a['provenance']['sources'],b['provenance']['sources'])
                self.assertFalse(b['record_present'][288:].any())

    def test_duplicate_and_off_grid_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pems05';p.mkdir();file=p/'d05_text_station_5min_2006_01_01.txt.gz'
            for text in ['01/01/2006 00:00:00,500001,5,1,N,ML,1,10,100,0,0,0\n'*2,
                         '01/01/2006 00:01:00,500001,5,1,N,ML,1,10,100,0,0,0\n']:
                with gzip.open(file,'wt') as f:f.write(text)
                with self.assertRaises(ValueError):fast(tmp,'05','2006-01-01',days=1,label_policy='finite_reported')

    def test_explicit_repair_keeps_original_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pems05';p.mkdir();q=Path(tmp)/'verified';q.mkdir()
            name='d05_text_station_5min_2006_01_02.txt.gz'
            with gzip.open(p/'d05_text_station_5min_2006_01_01.txt.gz','wt') as f:f.write('01/01/2006 00:00:00,500001,5,1,N,ML,1,10,100,0,0,0\n')
            (p/name).write_bytes(b'bad-original')
            with gzip.open(q/name,'wt') as f:f.write('01/02/2006 00:00:00,500001,5,1,N,ML,1,10,100,19,0,0\n')
            r=fast(tmp,'05','2006-01-01',days=2,label_policy='finite_reported',file_overrides={name:str(q/name)})
            self.assertEqual(r['values'][288,0],19)
            self.assertEqual((p/name).read_bytes(),b'bad-original')
            self.assertEqual(r['provenance']['sources'][1]['original_path'],str(p/name))
            self.assertNotEqual(r['provenance']['sources'][1]['original_sha256'],r['provenance']['sources'][1]['sha256'])
