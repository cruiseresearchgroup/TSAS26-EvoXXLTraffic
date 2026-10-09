import ast
import copy
import csv
import gzip
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from benchmark_v2.data import prepare_year
from benchmark_v2.pems import read_window
from benchmark_v2.runner import (MODELS, snapshot, restore, tensor_batch, evaluate,
                                 masked_loss, node_context, validate_manifest, ROOT)


def args():
    return SimpleNamespace(gcn={'in_channel':12,'hidden_channel':64,'out_channel':12},
                           tcn={'in_channel':1,'out_channel':1,'kernel_size':3,'dilation':1},
                           y_len=12,dropout=0.,year=2005,begin_year=2005,path=None,
                           benchmark_strict=True,use_strap=True,backbone_type='stgnn')


def data():
    v=np.arange(720,dtype=float).reshape(360,2)/10
    m=np.ones_like(v,dtype=bool)
    return prepare_year(v,m,[10,20],np.arange(360)*300,slots=360,
                        provenance={'observation_mask_source':'synthetic regression truth'})


class IntegrationTests(unittest.TestCase):
    def test_masked_gradient_preserves_true_zero_and_ignores_placeholder(self):
        pred=torch.tensor([2.,1000.,4.],requires_grad=True)
        loss=masked_loss(pred,torch.tensor([0.,99999.,2.]),torch.tensor([True,False,True]))
        loss.backward()
        torch.testing.assert_close(pred.grad,torch.tensor([2.,0.,2.]))
        self.assertIsNone(masked_loss(pred,torch.zeros(3),torch.zeros(3,dtype=torch.bool)))

    def test_pecpm_validation_cannot_mutate_training_memory(self):
        torch.manual_seed(7);model=MODELS['PECPM'](args());d=data();adj=torch.eye(2)
        x,_,_,_=tensor_batch(d,[11,12],adj.device);model.train();model(x,adj)
        before=snapshot(model)
        result=evaluate(model,d,adj,'val',{'all':np.arange(2)},cap=24)
        self.assertGreater(result['cohorts']['all']['overall']['observed_count'],0)
        torch.testing.assert_close(model.historical_patterns,before['extra']['historical_patterns'])
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before['model_state_dict'][k])

    def test_pecpm_unavailable_rows_do_not_enter_memory(self):
        model=MODELS['PECPM'](args());x=SimpleNamespace(x=torch.ones(2,12),x_available=torch.tensor([[False]*12,[True]*12]))
        model(x,torch.eye(2));self.assertEqual(len(model.historical_patterns),1)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU checkpoint relocation regression')
    def test_pecpm_gpu_checkpoint_reloads_host_history(self):
        torch.manual_seed(7)
        model=MODELS['PECPM'](args()).cuda();d=data();adj=torch.eye(2,device='cuda')
        x,_,_,_=tensor_batch(d,[11,12],adj.device);model.train();model(x,adj)
        state=snapshot(model);model.eval();expected=model(x,adj).detach()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';torch.save(state,path)
            loaded=torch.load(path,map_location='cuda',weights_only=False)
        self.assertEqual(loaded['extra']['historical_patterns'].device.type,'cuda')
        other=MODELS['PECPM'](args()).cuda();restore(other,loaded);other.eval()
        self.assertEqual(other.historical_patterns.device.type,'cpu')
        torch.testing.assert_close(other(x,adj),expected)
        self.assertEqual(other.historical_patterns.device.type,'cpu')

    def test_strap_checkpoint_restores_prediction_and_library(self):
        torch.manual_seed(7);model=MODELS['STRAP'](args());d=data();adj=torch.eye(2)
        x,_,_,_=tensor_batch(d,[11,12],adj.device);model.train();model(x,adj)
        model.eval();expected=model(x,adj).detach();state=snapshot(model)
        other=MODELS['STRAP'](args());restore(other,state);other.eval()
        torch.testing.assert_close(other(x,adj),expected)
        self.assertTrue(other.pattern_initialized)
        self.assertTrue(other.strap.pattern_manager._cache)

    def test_ttc_future_labels_cannot_change_earlier_predictions(self):
        torch.manual_seed(7);model=MODELS['ST-TTC'](args());d=data();adj=torch.eye(2)
        other=copy.deepcopy(d);changed=int(d['bounds'][2])+32;other['y'][changed:]+=5000
        kw=dict(split='test',cohorts={'all':np.arange(2)},cap=48,ttc={'groups':4,'lr':.001})
        a=evaluate(model,d,adj,**kw);b=evaluate(model,other,adj,**kw)
        # First origin is test start+11, hence indices 0..20 precede changed labels.
        self.assertEqual(a['prediction_trace'][:21],b['prediction_trace'][:21])
        self.assertNotEqual(a['prediction_trace'][-1],b['prediction_trace'][-1])
        self.assertTrue(a['timing_valid']);self.assertGreater(a['calibration_updates'],0)

    def test_new_node_cohort_uses_same_ttc_predictions(self):
        model=MODELS['ST-TTC'](args());d=data()
        r=evaluate(model,d,torch.eye(2),'test',{'all':np.arange(2),'new':np.arange(2)},cap=32,ttc={'groups':4,'lr':.001})
        self.assertEqual(r['cohorts']['all'],r['cohorts']['new'])

    def test_unobserved_new_cohort_is_missing_not_zero_error(self):
        model=MODELS['Retrain'](args());d=data()
        d['observed'][d['bounds'][2]:,1]=False
        r=evaluate(model,d,torch.eye(2),'test',{'all':np.arange(2),'new':np.array([1])},cap=24)
        self.assertGreater(r['cohorts']['all']['overall']['observed_count'],0)
        self.assertEqual(r['cohorts']['new']['overall']['observed_count'],0)
        self.assertIsNone(r['cohorts']['new']['overall']['MAE'])

    def test_isolated_new_node_is_kept_in_context(self):
        nodes,mask=node_context(np.zeros((3,3)),[1],2)
        np.testing.assert_array_equal(nodes,[1]);np.testing.assert_array_equal(mask,[True])

    def test_raw_reader_distinguishes_missing_zero_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pems05';p.mkdir();f=p/'d05_text_station_5min_2005_01_01.txt.gz'
            rows=[['01/01/2005 00:00:00',1,5,1,'N','ML',1,1,100,'0','',''],
                  ['01/01/2005 00:00:00',2,5,1,'N','ML',1,0,0,'','','']]
            with gzip.open(f,'wt',newline='') as z:csv.writer(z).writerows(rows)
            d=read_window(tmp,5,'2005-01-01',days=2,label_policy='positive_percent_observed')
            self.assertTrue(d['observed'][0,0]);self.assertEqual(d['values'][0,0],0)
            self.assertFalse(d['observed'][0,1]);self.assertFalse(d['observed'][288:].any())
            with gzip.open(f,'at',newline='') as z:csv.writer(z).writerow(rows[0])
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_window(tmp,5,'2005-01-01',days=2,label_policy='finite_reported')

    def test_legacy_checkpoint_progresses_after_first_transition(self):
        # Execute the production helper functions without unrelated Bio imports.
        source=ast.parse((ROOT/'eac/utils/common_tools.py').read_text())
        wanted={'_numeric_checkpoint','load_best_model','load_test_best_model'}
        funcs=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[])
        namespace={'os':os,'osp':os.path,'np':np,'torch':torch}
        exec(compile(funcs,'common_tools.py','exec'),namespace)
        with tempfile.TemporaryDirectory() as tmp:
            external=Path(tmp)/'12.345.pkl';torch.save({'model_state_dict':{'epoch':2005}},external)
            p=Path(tmp)/'m-42'/'2006';p.mkdir(parents=True)
            for name,epoch in [('10.0.pkl',10),('9.0.pkl',9)]:torch.save({'model_state_dict':{'epoch':epoch}},p/name)
            a=SimpleNamespace(load_first_year=True,begin_year=2005,year=2006,first_year_model_path=str(external),
                              model_path=tmp,logname='m',seed=42,logger=Mock(),device='cpu',method='X')
            class Dummy:
                def __init__(self,a):pass
                def load_state_dict(self,s):self.state=s
                def to(self,d):return self
            a.methods={'X':Dummy}
            model,loss=namespace['load_best_model'](a)
            self.assertEqual(model.state['epoch'],2005);self.assertEqual(loss,'12.345')
            a.year=2007;model,loss=namespace['load_best_model'](a)
            self.assertEqual(model.state['epoch'],9);self.assertEqual(loss,'9.0')

    def test_pems04_2001_and_unintegrated_methods_rejected(self):
        m={'method':'Retrain','purpose':'smoke','district':'PEMS04','years':[{'year':2001}],'epochs':1,'batch_size':2}
        with self.assertRaisesRegex(ValueError,'2002'):validate_manifest(m)
        m['method']='UnsupportedMethod'
        with self.assertRaisesRegex(ValueError,'not yet integrated'):validate_manifest(m)

if __name__=='__main__':unittest.main()
