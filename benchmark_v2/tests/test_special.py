import copy
import unittest
import numpy as np
import torch
from benchmark_v2.data import prepare_year
from benchmark_v2.identity import transition
from benchmark_v2.runner import MODELS,snapshot,restore,tensor_batch
from benchmark_v2.special import (bind_eac_ids,cluster_loss,daily_clusters,
                                  select_nodes,fisher_state,ewc_penalty)
from test_integration import args,data


class SpecialistTests(unittest.TestCase):
    def test_eac_reorder_departure_and_return_preserve_sensor_rows(self):
        a=args();a.rank=3;a.base_node_size=2
        model=MODELS['EAC'](a);bind_eac_ids(model,[10,20])
        with torch.no_grad():model.U.copy_(torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
        bind_eac_ids(model,[20,30])
        torch.testing.assert_close(model.U[0],torch.tensor([4.,5.,6.]))
        bind_eac_ids(model,[30,10,20])
        torch.testing.assert_close(model.U[1],torch.tensor([1.,2.,3.]))
        torch.testing.assert_close(model.U[2],torch.tensor([4.,5.,6.]))
        other=MODELS['EAC'](a);restore(other,snapshot(model))
        self.assertEqual(other.sensor_ids,[30,10,20]);torch.testing.assert_close(other.U,model.U)

    def test_eac_predictions_follow_id_permutation(self):
        a=args();a.rank=3;a.base_node_size=2
        model=MODELS['EAC'](a);bind_eac_ids(model,[10,20]);model.eval()
        d=data();x,_,_,_=tensor_batch(d,[11],torch.device('cpu'))
        before=model(x,torch.eye(2)).detach()
        bind_eac_ids(model,[20,10]);x.x=x.x.flip(0)
        after=model(x,torch.eye(2)).detach()
        torch.testing.assert_close(after,before.flip(0))

    def test_stkec_auxiliary_loss_has_gradient_and_correct_class_ids(self):
        logits=torch.tensor([[2.,-1.],[-1.,2.],[2.,-1.],[-1.,2.]],requires_grad=True)
        scores=logits.softmax(-1);mask=torch.ones((2,2,12),dtype=torch.bool)
        loss=cluster_loss(scores,np.array([0,1]),mask)
        self.assertLess(float(loss),.1);loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        logits2=logits.detach().clone().requires_grad_();mask[:,1]=False
        loss=cluster_loss(logits2.softmax(-1),np.array([0,-1]),mask);loss.backward()
        self.assertEqual(float(logits2.grad[[1,3]].abs().sum()),0)

    def test_stkec_clustering_ignores_future_and_fully_unobserved_node(self):
        v=np.arange(9000,dtype=float).reshape(3000,3);m=np.ones_like(v,dtype=bool);m[:,2]=False
        d=prepare_year(v,m,[1,2,3],np.arange(3000)*300,slots=3000,provenance={'observation_mask_source':'synthetic truth'})
        labels,k=daily_clusters(d,6);self.assertEqual(k,2);self.assertEqual(labels[2],-1)
        other=copy.deepcopy(d);other['y'][d['bounds'][1]:]=1e8;other['observed'][d['bounds'][1]:]=True
        l2,k2=daily_clusters(other,6);np.testing.assert_array_equal(labels,l2);self.assertEqual(k,k2)

    def test_drift_is_stable_id_aligned_and_train_only(self):
        old=data();cur=copy.deepcopy(old)
        # Deliberately reorder both the catalog and columns.
        cur['sensor_ids']=cur['sensor_ids'][::-1].copy()
        for k in ('x','y','observed','available'):cur[k]=cur[k][:,::-1].copy()
        changes=transition(old['sensor_ids'],cur['sensor_ids'],old['sensor_ids'])
        a=args();model=MODELS['TrafficStream'](a)
        cfg={'infl_topk':2,'increase':True,'detect':True,'replay':True,'replay_strategy':'inforeplay'}
        selected,r=select_nodes(model,old,cur,torch.eye(2),torch.eye(2),changes,'STKEC',cfg,np.random.default_rng(1))
        np.testing.assert_allclose(r['scores'],0,atol=1e-6)
        modified=copy.deepcopy(cur);modified['x'][cur['bounds'][1]:]=100000
        selected2,r2=select_nodes(model,old,modified,torch.eye(2),torch.eye(2),changes,'STKEC',cfg,np.random.default_rng(1))
        self.assertEqual(r,r2);np.testing.assert_array_equal(selected,selected2)

    def test_fisher_cap_counts_valid_batches_after_missing_prefix(self):
        d=data();d['observed'][:100]=False
        model=MODELS['TrafficStream'](args())
        _,f,n=fisher_state(model,d,torch.eye(2),8,'mse',2)
        self.assertEqual(n,2)
        self.assertGreater(sum(float(v.sum()) for v in f.values()),0)

    def test_fisher_and_penalty_use_observed_training_targets(self):
        d=data();d['observed'][:80,1]=False;other=copy.deepcopy(d);other['y'][:80,1]=1e8
        model=MODELS['TrafficStream'](args());adj=torch.eye(2)
        m,f,n=fisher_state(model,d,adj,8,'mse',2)
        _,g,n2=fisher_state(model,other,adj,8,'mse',2)
        self.assertEqual(n,n2)
        for k in f:torch.testing.assert_close(f[k],g[k])
        self.assertEqual(float(ewc_penalty(model,m,f,.1,'ewc')),0)
        with torch.no_grad():
            for p in model.parameters():p.add_(.001)
        self.assertGreater(float(ewc_penalty(model,m,f,.1,'ewc')),0)

if __name__=='__main__':unittest.main()
