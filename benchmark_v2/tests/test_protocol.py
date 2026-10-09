import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark_v2.data import prepare_year, split_origins, window_batch, save_year, load_year
from benchmark_v2.identity import transition
from benchmark_v2.metrics import score, MetricAccumulator, horizons
from benchmark_v2.checkpoints import select_numeric_checkpoint, validate_checkpoint_metadata
from benchmark_v2.online import LabelDelayQueue

class ProtocolTests(unittest.TestCase):
    def setup_data(self):
        v=np.arange(600,dtype=float).reshape(300,2);m=np.ones_like(v,dtype=bool)
        return v,m
    def prep(self,v,m):
        return prepare_year(v,m,[10,20],np.arange(len(v))*300,slots=300,
                            provenance={'observation_mask_source':'synthetic test truth'})
    def test_test_values_cannot_change_training_or_scaler(self):
        v,m=self.setup_data();a=self.prep(v,m);v[240:]*=100000;b=self.prep(v,m)
        np.testing.assert_array_equal(a['x'][:240],b['x'][:240])
        self.assertEqual(a['metadata']['scaler_mean'],b['metadata']['scaler_mean'])
        self.assertEqual(a['metadata']['scaler_std'],b['metadata']['scaler_std'])
    def test_no_backward_fill_and_zero_observation_is_retained(self):
        v,m=self.setup_data();m[:250,1]=False;v[250,1]=1000
        d=self.prep(v,m)
        self.assertFalse(d['available'][:250,1].any());self.assertTrue(d['available'][250,1])
        self.assertTrue((d['x'][:250,1]==0).all());self.assertTrue(d['observed'][0,0])
        m[50:60,0]=False;d=self.prep(v,m)
        np.testing.assert_array_equal(d['x'][50:60,0],np.repeat(d['x'][49,0],10))
        self.assertFalse(d['observed'][50:60,0].any())
    def test_time_order_boundaries_and_last_window(self):
        v,m=self.setup_data();d=self.prep(v,m)
        for i,s in enumerate(['train','val','test']):
            o=split_origins(d,s);self.assertTrue((np.diff(o)==1).all())
            b=window_batch(d,o);self.assertEqual(int(o[0]-11),d['bounds'][i])
            self.assertEqual(int(o[-1]+12),d['bounds'][i+1]-1)
            self.assertTrue((b['target_times'][:,0]>b['origin_time']).all())
    def test_empty_training_and_legacy_cache_rejected(self):
        v,m=self.setup_data();m[:180]=False
        with self.assertRaises(ValueError):self.prep(v,m)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'old.npz';np.savez(p,x=v)
            with self.assertRaises(ValueError):load_year(p)
    def test_roundtrip_and_no_overwrite(self):
        v,m=self.setup_data();d=self.prep(v,m)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'new.npz';save_year(p,d);a=load_year(p)
            np.testing.assert_array_equal(a['observed'],m)
            self.assertEqual(a['metadata'],d['metadata'])
            with self.assertRaises(FileExistsError):save_year(p,d)
    def test_id_reordering_contraction_return(self):
        r=transition([10,20,30,40],[30,5,10],[5,10,20,30,40])
        self.assertEqual(r['entered_current'].tolist(),[1]);self.assertEqual(r['returning_current'].tolist(),[1])
        self.assertEqual(r['first_seen_current'].tolist(),[])
        self.assertEqual(r['retained_previous'].tolist(),[2,0])
        self.assertEqual(r['departed_previous'].tolist(),[1,3])
    def test_masks_zero_labels_empty_and_nonfinite(self):
        y=np.array([0.,2.,999.]);p=np.array([2.,4.,np.nan]);m=np.array([True,True,False])
        r=score(p,y,m);self.assertEqual(r['MAE'],2);self.assertEqual(r['RMSE'],2)
        self.assertEqual(r['MAPE'],100);self.assertEqual(r['observed_count'],2);self.assertEqual(r['mape_count'],1)
        self.assertIsNone(score(p,y,np.zeros(3,dtype=bool))['MAE'])
        p[0]=np.nan
        with self.assertRaises(ValueError):score(p,y,m)
    def test_pooled_metrics_are_batch_invariant(self):
        y=np.arange(20,dtype=float);p=y+np.arange(20)/10;m=np.arange(20)%3!=0
        a=MetricAccumulator();a.update(p[:3],y[:3],m[:3]);a.update(p[3:],y[3:],m[3:])
        for k in ['MAE','RMSE','MAPE']:self.assertAlmostEqual(a.result()[k],score(p,y,m)[k])
    def test_avg_does_not_repeat_short_leads(self):
        y=np.ones((1,12,1));p=y+np.arange(1,13)[None,:,None];m=np.ones_like(y,dtype=bool)
        r=horizons(p,y,m);self.assertEqual(r['Avg']['MAE'],6.5);self.assertEqual(r['3']['MAE'],2)
    def test_numeric_checkpoint_and_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            for n in ['10.0.pkl','9.0.pkl','nan.pkl','README.txt']:(Path(t)/n).touch()
            self.assertEqual(select_numeric_checkpoint(t).name,'9.0.pkl')
        with self.assertRaises(ValueError):validate_checkpoint_metadata({},district='PEMS04',year=2002,protocol_hash='a',data_hash='b')
    def test_online_labels_available_only_after_final_target(self):
        q=LabelDelayQueue();self.assertEqual(q.advance(100),[]);q.add_after_prediction(3700,'forecast')
        self.assertEqual(q.advance(3400),[]);self.assertEqual(q.advance(3700),['forecast'])
        with self.assertRaises(ValueError):q.advance(3600)

if __name__=='__main__':unittest.main()
