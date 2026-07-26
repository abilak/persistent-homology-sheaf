"""
Same-protocol baselines for the rebuttal (Q3): GCN, GIN, and an MLP (a GNN with
NO message passing = DeepSets over nodes). All three consume the *identical*
input tensor produced by the PPGN data loader (channel 0 = normalized
adjacency D^-1/2 A D^-1/2; diagonals of channels 1.. = node-label one-hots),
use the same DataGenerator, splits, batching, optimizer and training loop as the
PPGN baseline and the topology model, so the comparison is strictly apples-to-
apples.

Selected via config.architecture.baseline_type in {"gcn","gin","mlp"}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# --- GSN structural features: per-node cycle counts (lengths 3..6), the
# canonical subgraph-counting features of Bouritsas et al. (GSN). Cached per
# graph (keyed by binary adjacency) since they are graph-only. ---
_CYCLE_CACHE = {}
_CYCLE_LENGTHS = (3, 4, 5, 6)


def structural_counts(A_bin_np):
    """Return M x len(_CYCLE_LENGTHS) per-node simple-cycle counts."""
    M = A_bin_np.shape[0]
    key = (M, A_bin_np.astype(np.uint8).tobytes())
    cached = _CYCLE_CACHE.get(key)
    if cached is not None:
        return cached
    G = nx.from_numpy_array(A_bin_np)
    counts = np.zeros((M, len(_CYCLE_LENGTHS)), dtype=np.float32)
    lidx = {L: i for i, L in enumerate(_CYCLE_LENGTHS)}
    try:
        for cyc in nx.simple_cycles(G, length_bound=max(_CYCLE_LENGTHS)):
            L = len(cyc)
            if L in lidx:
                for v in cyc:
                    counts[v, lidx[L]] += 1
    except Exception:
        pass
    if len(_CYCLE_CACHE) < 200000:
        _CYCLE_CACHE[key] = counts
    return counts


def gsn_node_features(A_bin):
    """A_bin: B x M x M tensor -> B x M x len(_CYCLE_LENGTHS) cycle counts."""
    B, M, _ = A_bin.shape
    arr = A_bin.detach().cpu().numpy()
    feats = np.stack([structural_counts(arr[b]) for b in range(B)], axis=0)
    return torch.from_numpy(feats).to(A_bin.device)


def extract_nodes(input):
    """input: B x C x M x M -> (node_features B x M x F, norm_adj B x M x M,
    binary_adj B x M x M). Node features = label one-hots (diag of channels 1:)
    plus node degree; for label-free datasets this is just the degree."""
    B, C, M, _ = input.shape
    diag = torch.diagonal(input, dim1=2, dim2=3).transpose(1, 2)  # B x M x C
    A_norm = input[:, 0]                                          # B x M x M
    A_bin = (input[:, 0].abs() > 1e-6).float()
    A_bin = A_bin * (1 - torch.eye(M, device=input.device).unsqueeze(0))
    deg = A_bin.sum(-1, keepdim=True)                            # B x M x 1
    feats = torch.cat([diag[:, :, 1:], deg], dim=-1)            # B x M x C
    return feats, A_norm, A_bin


class MLPBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin1 = nn.Linear(din, dout)
        self.lin2 = nn.Linear(dout, dout)
        self.bn = nn.BatchNorm1d(dout)

    def forward(self, h):  # h: B x M x d
        B, M, _ = h.shape
        h = F.relu(self.lin1(h))
        h = self.lin2(h)
        h = self.bn(h.reshape(B * M, -1)).reshape(B, M, -1)
        return F.relu(h)


class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kind = config.architecture.baseline_type
        in_dim = config.node_labels + 1
        if self.kind == 'gsn':
            in_dim += len(_CYCLE_LENGTHS)   # append per-node cycle counts
        hidden = config.architecture.block_features[0]
        self.n_layers = len(config.architecture.block_features)
        self.num_classes = config.num_classes

        self.layers = nn.ModuleList()
        self.eps = nn.ParameterList()
        d = in_dim
        for _ in range(self.n_layers):
            self.layers.append(MLPBlock(d, hidden))
            self.eps.append(nn.Parameter(torch.zeros(1)))
            d = hidden
        # jumping-knowledge: classify from sum-pool of every layer (+input)
        self.classifiers = nn.ModuleList()
        dims = [in_dim] + [hidden] * self.n_layers
        for dd in dims:
            self.classifiers.append(nn.Linear(dd, self.num_classes))
        self.dropout = nn.Dropout(0.5)

    def _aggregate(self, h, A_norm, A_bin, layer_i):
        if self.kind == 'mlp':
            return h                                   # no message passing
        if self.kind == 'gcn':
            M = h.shape[1]
            A_hat = A_norm + torch.eye(M, device=h.device).unsqueeze(0)
            return torch.bmm(A_hat, h)                 # spectral conv
        if self.kind in ('gin', 'gsn'):
            return (1 + self.eps[layer_i]) * h + torch.bmm(A_bin, h)  # sum agg
        raise ValueError(self.kind)

    def forward(self, input):
        h, A_norm, A_bin = extract_nodes(input)
        if self.kind == 'gsn':
            h = torch.cat([h, gsn_node_features(A_bin)], dim=-1)  # + cycle counts
        pooled = [h.sum(dim=1)]                         # input layer readout
        for i, layer in enumerate(self.layers):
            m = self._aggregate(h, A_norm, A_bin, i)
            h = layer(m)
            pooled.append(h.sum(dim=1))                 # sum readout per layer
        score = 0
        for cl, p in zip(self.classifiers, pooled):
            score = score + self.dropout(cl(p))
        return score
