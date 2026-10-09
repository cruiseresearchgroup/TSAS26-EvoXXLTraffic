import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_v2.data import prepare_year, save_year
from benchmark_v2.runner import run, evaluate, validate_manifest


class TrainingBudgetTests(unittest.TestCase):
    def test_validation_plateau_stops_training_and_keeps_test_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values=np.arange(720,dtype=float).reshape(360,2)/10
            d=prepare_year(values,np.ones_like(values,dtype=bool),[10,20],np.arange(360)*300,
                           slots=360,provenance={'observation_mask_source':'synthetic truth',
                                               'district':'PEMS05','window_start':'2005-01-01'})
            save_year(root/'data.npz',d)
            np.savez(root/'graph.npz',x=np.eye(2),sensor_ids=np.array([10,20]),
                     metadata=np.array(json.dumps({'order_status':'verified_ids'})))
            config={'gcn':{'in_channel':12,'hidden_channel':8,'out_channel':12},
                    'tcn':{'in_channel':1,'out_channel':1,'kernel_size':3,'dilation':1},
                    'y_len':12,'dropout':0.,'lr':.001,'loss':'mse'}
            m={'method':'Retrain','district':'PEMS05','purpose':'smoke','seed':47,
               'years':[{'year':2005,'data':str(root/'data.npz'),'graph':str(root/'graph.npz')}],
               'model_config':config,'epochs':20,'batch_size':2,'max_batches':1,
               'max_eval_origins':4,'early_stopping_patience':2,'output':str(root/'out')}
            splits=[]
            def controlled(*args,**kwargs):
                result=evaluate(*args,**kwargs);split=args[3];splits.append(split)
                if split=='val':result['cohorts']['all']['overall']['MAE']=1.
                return result
            with patch('benchmark_v2.runner.evaluate',side_effect=controlled):r=run(m)
            row=r['years'][0]
            self.assertEqual(row['epochs_completed'],3)
            self.assertEqual(row['optimizer_steps'],3)
            self.assertTrue(row['stopped_early'])
            self.assertEqual(splits,['val','val','val','test'])
            self.assertGreater(row['metrics']['cohorts']['all']['overall']['observed_count'],0)

    def test_invalid_patience_and_capped_benchmark_rejected(self):
        m={'method':'Retrain','purpose':'smoke','district':'PEMS04','years':[{'year':2002}],
           'epochs':100,'batch_size':128,'early_stopping_patience':0}
        with self.assertRaisesRegex(ValueError,'patience'):validate_manifest(m)
        m.update(purpose='benchmark',protocol_frozen=True,early_stopping_patience=5,max_batches=8)
        with self.assertRaisesRegex(ValueError,'smoke caps'):validate_manifest(m)
