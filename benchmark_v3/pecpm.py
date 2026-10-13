"""PECPM predictor and train-only reconstruction of its pattern supervision.

The author release depends on unpublished year-specific attention arrays. This
adapter regenerates fixed-width cosine/top-k targets from training prefixes;
it is an explicitly documented benchmark adaptation, not those original arrays.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.gcn_conv import BatchGCNConv


class PECPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gcn1 = BatchGCNConv(12, 64, bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(64, 12, bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(1, 1, 3, padding=1)
        self.tcn2 = nn.Conv1d(1, 1, 3, padding=1)
        self.w1, self.w2, self.w3 = nn.Linear(64, 64), nn.Linear(12, 12), nn.Linear(24, 64)
        self.fc = nn.Linear(12, 12)
        self.memory = nn.Parameter(torch.empty(64, 12))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        self.register_buffer('centroids', torch.zeros(64, 12))
        self.register_buffer('patterns_ready', torch.tensor(False))
        self.register_buffer('centroid_count', torch.zeros((), dtype=torch.long))

    def feature(self, data, adj):
        n = len(adj)
        x = data.x.reshape(-1, n, 12)
        times = data.time_features.reshape(-1, n, 24)
        x = F.relu(self.gcn1(x, adj)+self.w3(times))
        x = self.tcn1(x.reshape(-1, 1, 64))
        x = x*self.w1(x).sigmoid()
        x = self.gcn2(x.reshape(-1, n, 64), adj)
        x = self.tcn2(x.reshape(-1, 1, 12))
        return (x*self.w2(x).sigmoid()).reshape(-1, 12)

    def forward(self, data, adj):
        x = self.feature(data, adj)
        attention = (x @ self.memory.T).softmax(-1)
        pred = self.fc(F.gelu(x+data.x+attention @ self.memory))
        return F.dropout(pred, p=self.args.dropout, training=self.training), attention

    @torch.no_grad()
    def fit_patterns(self, vectors, seed):
        from sklearn.cluster import KMeans
        values = np.asarray(vectors, dtype=np.float32)
        if not len(values):
            raise ValueError('No available training input windows for PECPM patterns')
        # Fixed cap determined before reading outcomes; previous centroids seed
        # the next year's consolidation, preserving slot identity across years.
        if len(values) > 65536:
            values = values[np.random.default_rng(seed).choice(len(values), 65536, replace=False)]
        k = min(64, len(np.unique(values, axis=0)))
        init = self.centroids[:k].detach().cpu().numpy() if bool(self.patterns_ready) and int(self.centroid_count) == k else 'k-means++'
        fit = KMeans(n_clusters=k, init=init, n_init=1 if not isinstance(init, str) else 10,
                     max_iter=100, random_state=seed).fit(values)
        centers = fit.cluster_centers_[np.arange(64) % k]
        self.centroids.copy_(torch.as_tensor(centers, device=self.centroids.device))
        self.centroid_count.fill_(k)
        self.patterns_ready.fill_(True)

    def pattern_loss(self, data, attention, observed_targets):
        if not bool(self.patterns_ready):
            raise RuntimeError('Build PECPM pattern labels from the training split first')
        similarity = F.normalize(data.x, dim=-1) @ F.normalize(self.centroids, dim=-1).T
        idx = similarity.topk(min(5, int(self.centroid_count)), dim=-1).indices
        label = torch.zeros_like(similarity).scatter(1, idx, similarity.gather(1, idx)).detach()
        valid = data.x_available.all(-1) & observed_targets.any(-1).reshape(-1)
        return F.mse_loss(attention[valid], label[valid]) if valid.any() else attention.sum()*0


def select_pecpm(previous, current, changes, rng):
    """Author's daily Wasserstein drift/replay, restricted to observed train days."""
    from scipy.stats import wasserstein_distance
    def profiles(data):
        days = int(data['bounds'][1])//288
        val = data['y'][:days*288].reshape(days, 288, -1)
        mask = data['observed'][:days*288].reshape(days, 288, -1)
        count = mask.sum(1)
        mean = np.divide((val*mask).sum(1), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        return mean, count > 0
    a, am = profiles(previous)
    b, bm = profiles(current)
    scores = []
    for old, new in zip(changes['retained_previous'], changes['retained_current']):
        if am[:, old].any() and bm[:, new].any():
            scores.append((int(new), float(wasserstein_distance(a[am[:, old], old], b[bm[:, new], new]))))
    scores.sort(key=lambda row: (row[1], int(current['sensor_ids'][row[0]])))
    n = len(current['sensor_ids'])
    evolved, replay = int(.01*n), int(.09*n)
    selected = set(map(int, changes['entered_current']))
    if evolved:
        selected.update(i for i, _ in scores[-evolved:])
    if replay and scores:
        stable = min(len(scores), int(.1*replay))
        selected.update(i for i, _ in scores[:stable])
        selected.update(rng.choice([i for i, _ in scores], replay-stable, replace=True).tolist())
    if len(selected) > int(.2*n):
        selected = set(rng.choice(sorted(selected), max(1, int(.15*n)), replace=False).tolist())
    return np.array(sorted(selected), dtype=int), {'policy': 'observed_train_daily_wasserstein_1pct_evolved_9pct_replay',
                                                 'scores': scores, 'uses_test': False}
