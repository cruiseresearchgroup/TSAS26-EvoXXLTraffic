import unittest
import numpy as np
from benchmark_v2.data import prepare_year, split_origins
from benchmark_v2.metrics import MetricAccumulator
from .common import subset, seed_summary, paired_summary
from .run_chain import test_cohorts


class Contracts(unittest.TestCase):
    def test_fixed_scaler_and_induced_graph(self):
        values = np.arange(90 * 3, dtype=float).reshape(90,3)
        values[:,2] += 10000
        observed = np.ones_like(values, dtype=bool)
        observed[:5,0] = False
        data = prepare_year(values, observed, [3,8,9], np.arange(90)*300,
                            slots=90, x_len=2, y_len=2,
                            provenance={'observation_mask_source':'synthetic'})
        # subset uses the real 31-day specification; build a full synthetic task.
        v = np.tile(values, (100,1))[:8928]
        m = np.tile(observed, (100,1))[:8928]
        data = prepare_year(v,m,[3,8,9],np.arange(8928)*300,
                            provenance={'observation_mask_source':'synthetic'})
        raw = np.array([[1.,.2,.7],[.2,1.,.9],[.7,.9,1.]])
        fixed, graph, indices = subset(data,raw,[3,8])
        np.testing.assert_array_equal(graph, [[1.,.2],[.2,1.]])
        self.assertAlmostEqual(fixed['metadata']['scaler_mean'],v[:5356,:2][m[:5356,:2]].mean())
        self.assertNotEqual(fixed['metadata']['scaler_mean'],data['metadata']['scaler_mean'])
        self.assertTrue(np.all(fixed['x'][:5,0] == 0))
        self.assertTrue(np.all(~fixed['available'][:5,0]))
        for split, bounds in zip(['train','val','test'],[(0,5356),(5356,7142),(7142,8928)]):
            origins=split_origins(fixed,split)
            self.assertEqual(origins[0]-11,bounds[0])
            self.assertEqual(origins[-1]+12,bounds[1]-1)

    def test_metrics_pool_each_lead_once_and_zero_denominator(self):
        acc=MetricAccumulator()
        y=np.array([[[0.,2.,4.,1e-6]]]); p=np.array([[[1.,4.,2.,1e-6]]])
        acc.update(p,y,np.ones_like(y,dtype=bool));r=acc.result()
        self.assertEqual(r['observed_count'],4);self.assertEqual(r['mape_count'],2)
        self.assertAlmostEqual(r['MAE'],1.25);self.assertAlmostEqual(r['RMSE'],1.5)
        self.assertAlmostEqual(r['MAPE'],75.)

    def test_two_stage_paired_sample_sd(self):
        e={s:np.arange(8,dtype=float)+i for i,s in enumerate([47,48,49])}
        f={s:np.arange(8,dtype=float) for s in [47,48,49]}
        self.assertEqual(seed_summary(e)['mean'],4.5)
        self.assertEqual(seed_summary(e)['sd'],1.)
        self.assertEqual(paired_summary(e,f)['mean'],1.)
        self.assertEqual(paired_summary(e,f)['sd'],1.)

    def test_2018_new_uses_2017_catalog_without_checkpoint(self):
        from .common import PLAN
        d='PEMS05'; ids=np.asarray(PLAN['districts'][d]['sensor_ids'])
        data={'sensor_ids':ids,'metadata':{'provenance':{'window_start':'2018-01-01'}}}
        c=test_cohorts(data,d,'E',{'new_to_previous':np.array([],dtype=int)})
        np.testing.assert_array_equal(c['new'], np.flatnonzero(~np.isin(ids,PLAN['districts'][d]['previous_2017_sensor_ids'])))


if __name__ == '__main__': unittest.main()
