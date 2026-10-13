"""Time-unrolled graph recurrent predictors, with explicit graph conventions.

DCRNN: Li et al., ICLR 2018; DCGRU with dual random walks, K=2,
two recurrent layers, autoregressive decoder and inverse-sigmoid sampling.
T-GCN: Zhao et al.; self-loop normalized GCN gates at every input step.
Source commits and port differences are recorded in provenance.json.
"""
import math
import torch
from torch import nn


def graph_mm(a, x):
    """A[N,N] times x[B,N,C], allowing sparse full graphs."""
    if a.layout == torch.strided:
        return a @ x
    b, n, c = x.shape
    return torch.sparse.mm(a, x.permute(1, 0, 2).reshape(n, b*c)).reshape(n, b, c).permute(1, 0, 2)


def random_walk(a):
    degree = a.sum(1, keepdim=True)
    return torch.where(degree > 0, a / degree.clamp_min(1e-12), torch.zeros_like(a))


class DiffusionConv(nn.Module):
    def __init__(self, inputs, outputs, order=2, bias=0.):
        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.empty(inputs*(2*order+1), outputs))
        self.bias = nn.Parameter(torch.full((outputs,), float(bias)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, supports):
        terms = [x]
        for a in supports:
            # Each directional polynomial starts from the same signal.
            prev, curr = x, graph_mm(a, x)
            if self.order:
                terms.append(curr)
            for _ in range(2, self.order+1):
                prev, curr = curr, 2*graph_mm(a, curr)-prev
                terms.append(curr)
        features = torch.stack(terms, -1).flatten(-2)
        return features @ self.weight + self.bias


class DCGRUCell(nn.Module):
    def __init__(self, inputs, hidden, order=2):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(inputs+hidden, 2*hidden, order, bias=1.)
        self.candidate = DiffusionConv(inputs+hidden, hidden, order)

    def forward(self, x, h, supports):
        reset, update = self.gates(torch.cat((x, h), -1), supports).sigmoid().chunk(2, -1)
        candidate = self.candidate(torch.cat((x, reset*h), -1), supports).tanh()
        return update*h+(1-update)*candidate


class DCRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        h = args.gcn['hidden_channel']
        self.encoder = nn.ModuleList([DCGRUCell(1, h), DCGRUCell(h, h)])
        self.decoder = nn.ModuleList([DCGRUCell(1, h), DCGRUCell(h, h)])
        self.projection = nn.Linear(h, 1)
        self.register_buffer('batches_seen', torch.zeros((), dtype=torch.long))
        self.register_buffer('output_mean', torch.tensor(0.))
        self.register_buffer('output_std', torch.tensor(1.))
        self.supports = None

    def prepare_graph(self, raw):
        self.supports = [random_walk(a).T.to_sparse().coalesce() for a in (raw, raw.T)]

    def forward(self, data, adj, labels=None, label_mask=None, sampling=None):
        if self.supports is None:
            raise RuntimeError('DCRNN needs the unnormalized graph before computing both walks')
        x = data.x.reshape(-1, adj.shape[0], self.args.x_len).transpose(1, 2).unsqueeze(-1)
        states = [x.new_zeros(x.shape[0], x.shape[2], c.hidden) for c in self.encoder]
        for t in range(x.shape[1]):
            value = x[:, t]
            for i, cell in enumerate(self.encoder):
                states[i] = cell(value, states[i], self.supports)
                value = states[i]
        value = x.new_zeros(x.shape[0], x.shape[2], 1)
        predictions = []
        probability = 1000/(1000+math.exp(min(float(self.batches_seen)/1000, 50)))
        for t in range(self.args.y_len):
            for i, cell in enumerate(self.decoder):
                states[i] = cell(value, states[i], self.supports)
                value = states[i]
            value = self.projection(value)
            predictions.append(value[..., 0])
            if self.training and labels is not None:
                use_truth = bool(sampling[t]) if sampling is not None else bool(torch.rand((), device=x.device) < probability)
                if use_truth:
                    truth = ((labels[..., t]-self.output_mean)/self.output_std).unsqueeze(-1)
                    value = torch.where(label_mask[..., t, None], truth, value)
        return torch.stack(predictions, -1)*self.output_std+self.output_mean


class TGCNCell(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gates = nn.Linear(hidden+1, 2*hidden)
        self.candidate = nn.Linear(hidden+1, hidden)
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.ones_(self.gates.bias)
        nn.init.xavier_uniform_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, x, h, adj):
        reset, update = self.gates(graph_mm(adj, torch.cat((x, h), -1))).sigmoid().chunk(2, -1)
        candidate = self.candidate(graph_mm(adj, torch.cat((x, reset*h), -1))).tanh()
        return update*h+(1-update)*candidate


class TGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden = args.gcn['hidden_channel']
        self.cell = TGCNCell(self.hidden)
        self.head = nn.Linear(self.hidden, args.y_len)
        self.graph = None

    def prepare_graph(self, raw):
        a = raw+torch.eye(len(raw), device=raw.device, dtype=raw.dtype)
        d = a.sum(1).clamp_min(1e-12).rsqrt()
        self.graph = (d[:, None]*a*d[None, :]).to_sparse().coalesce()

    def forward(self, data, adj):
        if self.graph is None:
            raise RuntimeError('TGCN needs the raw graph, including self-loop normalization')
        x = data.x.reshape(-1, adj.shape[0], self.args.x_len)
        h = x.new_zeros(x.shape[0], x.shape[1], self.hidden)
        for t in range(self.args.x_len):
            h = self.cell(x[..., t, None], h, self.graph)
        return self.head(h)
