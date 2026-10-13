"""Adapter for the pinned author's ASTGCN recent-sequence model."""
import torch
from torch import nn
from .vendor.astgcn import ASTGCN_submodule
from .vendor.astgcn_graph import scaled_Laplacian, cheb_polynomial


class ASTGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        n = args.base_node_size
        # Graph tensors are supplied before the first forward, never inferred
        # from a row-normalized graph or saved across a changed node catalog.
        terms = [torch.zeros(n, n) for _ in range(3)]
        self.net = ASTGCN_submodule(torch.device('cpu'), 2, 1, 3, 64, 64, 1,
                                    terms, args.y_len, args.x_len, n)
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.ndim > 1 else nn.init.uniform_(p)
        self.graph_ready = False

    def prepare_graph(self, raw):
        terms = [torch.as_tensor(a, dtype=raw.dtype, device=raw.device)
                 for a in cheb_polynomial(scaled_Laplacian(raw.detach().cpu().numpy()), 3)]
        for block in self.net.BlockList:
            block.cheb_conv_SAt.cheb_polynomials = terms
            block.cheb_conv_SAt.DEVICE = raw.device
        self.graph_ready = True

    def forward(self, data, adj):
        if not self.graph_ready:
            raise RuntimeError('Initialize ASTGCN graph polynomials before inference')
        x = data.x.reshape(-1, adj.shape[0], 1, self.args.x_len)
        return self.net(x)
