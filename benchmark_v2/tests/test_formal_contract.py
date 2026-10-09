import copy,unittest
from benchmark_v2.formal_contract import digest,validate_frozen_manifest
class FormalContractTests(unittest.TestCase):
    def setUp(self):
        entry=dict(year=2005,district='PEMS05',sensors=6,data='data',graph='graph',data_sha256='a',graph_sha256='b')
        cfg=dict(lr=.03)
        self.spec=dict(frozen=True,data=dict(district='PEMS05',years=[2005],entries=[entry]),seeds=[47],methods={'Retrain':dict(batch_size=128,effective_config_sha256=digest(cfg))})
        self.m=dict(district='PEMS05',years=[entry],seed=47,method='Retrain',purpose='benchmark',model_config=cfg,epochs=100,early_stopping_patience=5,batch_size=128,max_batches=None,max_eval_origins=None,frozen_protocol_sha256=digest(self.spec))
    def test_original_allowed(self):validate_frozen_manifest(self.m,self.spec)
    def test_wrong_district(self):
        self.m['district']='PEMS04'
        with self.assertRaises(ValueError):validate_frozen_manifest(self.m,self.spec)
    def test_tampered_source_identity(self):
        self.m['years']=copy.deepcopy(self.m['years']);self.m['years'][0]['data_sha256']='changed'
        with self.assertRaises(ValueError):validate_frozen_manifest(self.m,self.spec)
    def test_missing_year(self):
        self.m['years']=[]
        with self.assertRaises(ValueError):validate_frozen_manifest(self.m,self.spec)
    def test_smoke_cap_rejected_for_formal(self):
        self.m['max_eval_origins']=64
        with self.assertRaises(ValueError):validate_frozen_manifest(self.m,self.spec)
    def test_changed_config(self):
        self.m['model_config']={'lr':.01}
        with self.assertRaises(ValueError):validate_frozen_manifest(self.m,self.spec)
