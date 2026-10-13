import copy
import logging
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from benchmark_v2.data import prepare_year
from benchmark_v3.recurrent import DCRNN, TGCN, DiffusionConv, random_walk
from benchmark_v3.astgcn import ASTGCN
from benchmark_v3.pecpm import PECPM
from benchmark_v3.strap import STRAP
from benchmark_v3.influence import conjugate_gradient, select_influence
from benchmark_v3.runner import tensor_batch, snapshot, restore, normalized_adj, backward_logical_batch
from benchmark_v2.special import cluster_loss
from benchmark_v3.vendor.astgcn_graph import cheb_polynomial
from benchmark_v3.vendor.pecpm_reference import Basic_Model
from benchmark_v3.vendor.pattern_features import nearest_hop_nodes


def args(path='.', n=6):
    return SimpleNamespace(gcn={'in_channel': 12, 'hidden_channel': 64, 'out_channel': 12},
        tcn={'in_channel': 1, 'out_channel': 1, 'kernel_size': 3, 'dilation': 1},
        x_len=12, y_len=12, dropout=0., year=2005, begin_year=2005,
        base_node_size=n, path=path, batch_size=16, pattern_build_samples=16,
        logger=logging.getLogger('test'), cluster=64, k_neighbors=4)


def sample(b=2, n=6, device='cpu'):
    return SimpleNamespace(x=torch.randn(b*n, 12, device=device),
        x_available=torch.ones(b*n, 12, dtype=torch.bool, device=device),
        time_features=torch.rand(b*n, 24, device=device))


def graph(n=6):
    a = torch.zeros(n, n)
    for i in range(n-1):
        a[i, i+1] = a[i+1, i] = 1
    return a


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_bidirectional_walks_normalized_separately(self):
        a = torch.tensor([[0., 2., 1.], [4., 0., 0.], [1., 1., 0.]])
        m = DCRNN(args(n=3))
        m.prepare_graph(a)
        torch.testing.assert_close(m.supports[0].to_dense(), random_walk(a).T)
        torch.testing.assert_close(m.supports[1].to_dense(), random_walk(a.T).T)
        self.assertFalse(torch.equal(m.supports[1].to_dense(), random_walk(a)))

    def test_diffusion_matches_independent_numpy_polynomials(self):
        a = np.array([[0., 1., 0.], [1., 0., .5], [0., .5, 0.]], dtype='float32')
        supports = [a, a.T/2]
        x = np.arange(6, dtype='float32').reshape(1, 3, 2)
        conv = DiffusionConv(2, 2)
        terms = [x]
        for s in supports:
            terms.extend([s @ x, (2*s @ s-np.eye(3)) @ x])
        feature = np.stack(terms, -1).reshape(1, 3, -1)
        expected = feature @ conv.weight.detach().numpy()+conv.bias.detach().numpy()
        actual = conv(torch.from_numpy(x), [torch.from_numpy(s) for s in supports])
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_dcrnn_uses_every_input_step(self):
        m, a = DCRNN(args()), graph()
        m.prepare_graph(a)
        x = sample()
        x.x.requires_grad_()
        m(x, a).square().sum().backward()
        self.assertTrue((x.x.grad.abs().sum(0) > 0).all())
        self.assertGreater(float(m.encoder[0].candidate.weight.grad.abs().sum()), 0)

    def test_teacher_forcing_is_training_only_and_masked(self):
        m, a, x = DCRNN(args()), graph(), sample()
        m.prepare_graph(a)
        labels = torch.randn(2, 6, 12)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        p = m(x, a, labels=labels, label_mask=mask, sampling=[True]*12)
        q = m(x, a, labels=labels+999, label_mask=mask, sampling=[True]*12)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        m.eval()
        mask.fill_(True)
        torch.testing.assert_close(m(x, a), m(x, a, labels=labels, label_mask=mask), rtol=0, atol=0)

    def test_tgcn_reset_gate_receives_gradient_from_nonzero_history(self):
        m, a, x = TGCN(args()), graph(), sample()
        m.prepare_graph(a)
        x.x.requires_grad_()
        m(x, a).square().sum().backward()
        self.assertGreater(float(m.cell.gates.weight.grad[:64].abs().sum()), 0.)
        self.assertTrue((x.x.grad.abs().sum(0) > 0).all())

    def test_tgcn_cell_matches_equations(self):
        m, a = TGCN(args()), graph()
        m.prepare_graph(a)
        x, h = torch.randn(2, 6, 1), torch.randn(2, 6, 64)
        d = (a+torch.eye(6)).sum(1).rsqrt()
        norm = d[:, None]*(a+torch.eye(6))*d[None, :]
        r, u = torch.sigmoid((norm @ torch.cat((x, h), -1)) @ m.cell.gates.weight.T+m.cell.gates.bias).chunk(2, -1)
        c = torch.tanh((norm @ torch.cat((x, r*h), -1)) @ m.cell.candidate.weight.T+m.cell.candidate.bias)
        torch.testing.assert_close(m.cell(x, h, m.graph), u*h+(1-u)*c)

    def test_dense_and_sparse_graphs_preserve_predictions_and_gradients(self):
        for constructor in (DCRNN, TGCN):
            dense, a, x = constructor(args()), graph(), sample()
            dense.prepare_graph(a)
            sparse = copy.deepcopy(dense)
            if constructor is DCRNN:
                dense.supports = [s.to_dense() for s in dense.supports]
                sparse.supports = [s.to_sparse().coalesce() for s in dense.supports]
            else:
                dense.graph = dense.graph.to_dense()
                sparse.graph = dense.graph.to_sparse().coalesce()
            p, q = dense(x, a), sparse(x, a)
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
            p.square().mean().backward()
            q.square().mean().backward()
            for (name, left), (_, right) in zip(dense.named_parameters(), sparse.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-7, msg=name)


class AccumulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.n = 6
        self.x = sample(b=4)
        self.x.x_available[7:10, -1] = False
        self.y = torch.randn(4, 6, 12)
        self.mask = torch.ones_like(self.y, dtype=torch.bool)
        self.mask[0, :, 1:] = False
        self.mask[1, 0] = False
        self.mask[3] = False
        self.a = graph()

    def chunks(self, size):
        batches = []
        for start in range(0, 4, size):
            end = min(4, start+size)
            x = SimpleNamespace(**{name: getattr(self.x, name)[start*self.n:end*self.n]
                for name in ('x', 'x_available', 'time_features')})
            batches.append((x, self.y[start:end], self.mask[start:end]))
        return batches

    def compare_gradients(self, model, reference_loss, cfg, labels=None):
        reference = copy.deepcopy(model)
        reference_loss(reference).backward()
        for size in (1, 2, 4):
            tested = copy.deepcopy(model)
            backward_logical_batch(tested, self.chunks(size), self.a, cfg, labels)
            for (name, left), (_, right) in zip(reference.named_parameters(), tested.named_parameters()):
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-7, msg=name)

    def test_pattern_loss_matches_direct_masked_objective(self):
        model = PECPM(args())
        model.centroids.normal_()
        model.patterns_ready.fill_(True)
        model.centroid_count.fill_(64)
        def direct(m):
            pred, attention = m(self.x, self.a)
            mse = (pred.reshape(self.y.shape)[self.mask]-self.y[self.mask]).square().mean()
            return mse+5*m.pattern_loss(self.x, attention, self.mask)
        self.compare_gradients(model, direct, {'pattern_weight': 5})

    def test_cluster_loss_matches_direct_masked_objective(self):
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = torch.nn.Linear(12, 12)
                self.classifier = torch.nn.Linear(12, 3)
            def forward(self, x, a):
                return self.predictor(x.x), self.classifier(x.x).softmax(-1)
        labels = np.array([0, -1, 2, 1, 2, 0])
        def direct(m):
            pred, scores = m(self.x, self.a)
            mse = (pred.reshape(self.y.shape)[self.mask]-self.y[self.mask]).square().mean()
            return mse+.1*cluster_loss(scores, labels, self.mask)
        self.compare_gradients(Heads(), direct, {}, labels)

    def test_sampling_is_shared_across_memory_chunks(self):
        model = DCRNN(args())
        model.prepare_graph(self.a)
        model.batches_seen.fill_(6908)  # A mix of prediction and teacher feedback.
        reference = None
        for size in (1, 2, 4):
            tested = copy.deepcopy(model)
            torch.manual_seed(87)
            backward_logical_batch(tested, self.chunks(size), self.a, {'y_len': 12})
            gradients = {name: p.grad for name, p in tested.named_parameters()}
            if reference is None:
                reference = gradients
            else:
                for name, g in gradients.items():
                    torch.testing.assert_close(g, reference[name], rtol=3e-4, atol=3e-7, msg=name)


class AttentionTests(unittest.TestCase):
    def test_hop_prefix_matches_full_bfs_including_ties_loops_and_isolates(self):
        import networkx as nx
        graphs = [nx.path_graph(40), nx.complete_graph(40), nx.empty_graph(40)]
        graphs += [nx.gnp_random_graph(40, p, seed=seed) for p in (.04, .15, .7) for seed in (47, 48, 49)]
        for graph in graphs:
            graph.add_edge(0, 0)
            for hops in (1, 2, 3):
                for limit in (1, 3, 20, 50):
                    for node in graph:
                        distance = nx.single_source_shortest_path_length(graph, node, cutoff=hops)
                        expected = sorted((n for n in distance if n != node), key=lambda n: (distance[n], n))[:limit]
                        self.assertEqual(nearest_hop_nodes(graph, node, hops, limit), expected)

    def test_astgcn_has_sample_specific_temporal_attention(self):
        torch.manual_seed(47)
        m, a, x = ASTGCN(args()), graph(), sample()
        m.prepare_graph(a)
        m.eval()
        y = m(x, a)
        single = sample(b=1)
        single.x = x.x[:6]
        torch.testing.assert_close(y[:1], m(single, a), rtol=1e-5, atol=1e-5)
        y.square().sum().backward()
        self.assertGreater(float(m.net.BlockList[0].TAt.Ve.grad.abs().sum()), 0.)
        self.assertGreater(float(m.net.BlockList[0].SAt.Vs.grad.abs().sum()), 0.)

    def test_chebyshev_second_order_reaches_two_hop_node(self):
        a = graph(3).numpy()
        p = cheb_polynomial(a, 3)
        self.assertEqual(p[2][0, 2], 2.)
        np.testing.assert_array_equal(p[2], 2*a @ a-np.eye(3))

    def test_pecpm_predictor_matches_author_forward(self):
        torch.manual_seed(47)
        cfg = args()
        repaired = PECPM(cfg)
        reference_cfg = copy.deepcopy(cfg)
        reference_cfg.dropout = [0.]
        reference = Basic_Model(reference_cfg)
        reference.load_state_dict({k: v for k, v in repaired.state_dict().items() if k in reference.state_dict()})
        x = sample()
        triple = torch.cat((x.x[..., None], x.time_features.reshape(-1, 12, 2)), -1)
        a = normalized_adj(graph().numpy(), 'cpu')
        p, attention = repaired(x, a)
        q, reference_attention = reference(SimpleNamespace(x=triple.reshape(-1, 36)), a)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(attention, reference_attention, rtol=0, atol=0)

    def test_pecpm_memory_temporal_and_both_gates_train(self):
        m, x, a = PECPM(args()), sample(), graph()
        m.fit_patterns(x.x.numpy(), 47)
        p, attention = m(x, a)
        loss = p.square().mean()+m.pattern_loss(x, attention, torch.ones(2, 6, 12, dtype=torch.bool))
        loss.backward()
        for layer in (m.memory, m.w1.weight, m.w2.weight, m.w3.weight, m.tcn2.weight):
            self.assertGreater(float(layer.grad.abs().sum()), 0.)

    def test_strap_pairs_history_and_checkpoint(self):
        torch.manual_seed(47)
        with tempfile.TemporaryDirectory() as path:
            cfg, a = args(path), normalized_adj(graph().numpy(), 'cpu')
            m = STRAP(cfg)
            train = sample(b=16)
            m.initialize_patterns(train, a)
            first = copy.deepcopy(m.strap.pattern_manager.payloads[2005])
            m.set_year(2006)
            m.initialize_patterns(train, a)
            second = m.strap.pattern_manager.payloads[2006]
            for kind, lib in second['libraries'].items():
                self.assertIn(2005, lib['origin_year'].tolist())
                history = lib['origin_year'] == 2005
                for key, value in zip(lib['keys'][history], lib['values'][history]):
                    src = first['libraries'][kind]
                    paired = ((src['keys'] == key).all(-1) & (src['values'] == value).all(-1)).any()
                    self.assertTrue(paired)
            x = sample()
            m.eval()
            p = m(x, a)
            restored = STRAP(cfg)
            restore(restored, snapshot(m))
            restored.eval()
            torch.testing.assert_close(p, restored(x, a), rtol=0, atol=0)
            self.assertEqual(restored.current_year, 2006)


class InfluenceTests(unittest.TestCase):
    def test_inverse_solve_against_dense_reference(self):
        torch.manual_seed(47)
        a = torch.randn(8, 8, dtype=torch.float64)
        h = a.T @ a+torch.eye(8)*.1
        b = torch.randn(8, dtype=torch.float64)
        x, receipt = conjugate_gradient(lambda v: h @ v, b, iterations=30, tolerance=1e-8)
        torch.testing.assert_close(x, torch.linalg.solve(h, b), rtol=1e-6, atol=1e-6)
        self.assertLess(receipt['relative_residual'], 1e-6)

    def test_influence_and_calendar_ignore_future_values(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(12, 12)
            def forward(self, x, a):
                return self.fc(x.x)
        torch.manual_seed(47)
        rng = np.random.default_rng(47)
        n, slots = 3, 720
        values = rng.normal(20, 3, (slots, n))
        data = prepare_year(values, np.ones((slots, n), dtype=bool), np.arange(n),
            np.arange(slots)*300, slots=slots, provenance={'observation_mask_source': 'synthetic test'})
        altered = copy.deepcopy(data)
        for key in ('x', 'y'):
            altered[key][data['bounds'][1]:] += 9999
        model, adj = Linear(), torch.eye(n)
        changes = {'retained_current': np.arange(n), 'entered_current': np.array([], dtype=int)}
        cfg = {'influe_length': 12, 'infl_topk': 1, 'influence_cg_steps': 80}
        c, r = select_influence(model, data, adj, changes, cfg, np.random.default_rng(47), tensor_batch)
        d, s = select_influence(model, altered, adj, changes, cfg, np.random.default_rng(47), tensor_batch)
        np.testing.assert_array_equal(c, d)
        self.assertEqual(r, s)
        self.assertLess(max(r['pool_origins'])+12, min(r['query_origins'])-12)
        self.assertLess(max(r['query_origins'])+12, data['bounds'][1])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
