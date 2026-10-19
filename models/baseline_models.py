import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

CYCLE_LENGTHS = (3, 4, 5, 6)
_cycles = {}

def cycles(A):
    key = (A.shape[0], A.astype(np.uint8).tobytes())
    if(key in _cycles):
        return _cycles[key]
    counts = np.zeros((A.shape[0], len(CYCLE_LENGTHS)), dtype=np.float32)
    col = {L: i for i, L in enumerate(CYCLE_LENGTHS)}
    # needs networkx >= 3.1
    for cyc in nx.simple_cycles(nx.from_numpy_array(A), length_bound=max(CYCLE_LENGTHS)):
        if(len(cyc) in col):
            for v in cyc:
                counts[v, col[len(cyc)]] += 1
    if(len(_cycles) < 50000):
        _cycles[key] = counts
    return counts

def gsn_feats(A_bin, n_real=None):
    B, M, _ = A_bin.shape
    arr = A_bin.detach().cpu().numpy()
    feats = np.zeros((B, M, len(CYCLE_LENGTHS)), dtype=np.float32)
    for b in range(B):
        n = M if n_real is None else int(n_real[b])
        feats[b, :n] = cycles(arr[b, :n, :n])
    return torch.from_numpy(feats).to(A_bin.device)

def node_feats(input):
    B, C, M, _ = input.shape
    diag = torch.diagonal(input, dim1=2, dim2=3).transpose(1, 2)
    A_bin = (input[:, 0].abs() > 1e-6).float()
    A_bin = A_bin * (1 - torch.eye(M, device=input.device).unsqueeze(0))
    deg = A_bin.sum(-1, keepdim=True)
    return torch.cat([diag[:, :, 1:], deg], dim=-1), A_bin

class MLPBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin1 = nn.Linear(din, dout)
        self.lin2 = nn.Linear(dout, dout)
        self.bn = nn.BatchNorm1d(dout)

    def forward(self, h, idx=None, mask=None):
        B, M, _ = h.shape
        h = self.lin2(F.relu(self.lin1(h)))
        flat = h.reshape(B * M, -1)
        if(idx is None):
            flat = self.bn(flat)
        else:
            flat = torch.zeros_like(flat).index_copy(0, idx, self.bn(flat.index_select(0, idx)))
        h = F.relu(flat.reshape(B, M, -1))
        return h if mask is None else h * mask[:, :, None]

class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.architecture.baseline_type
        in_dim = config.node_labels + 1
        if(self.kind == 'gsn'):
            in_dim += len(CYCLE_LENGTHS)
        hidden = config.architecture.block_features[0]
        self.n_layers = config.architecture.get('baseline_layers', 5)

        self.layers = nn.ModuleList()
        self.eps = nn.ParameterList()
        d = in_dim
        for _ in range(self.n_layers):
            self.layers.append(MLPBlock(d, hidden))
            self.eps.append(nn.Parameter(torch.zeros(1)))
            d = hidden
        self.classifiers = nn.ModuleList(nn.Linear(dd, config.num_classes) for dd in [in_dim] + [hidden] * self.n_layers)
        self.dropout = nn.Dropout(0.5)

    def aggregate(self, h, A_bin, i):
        if(self.kind == 'mlp'):
            return h
        if(self.kind == 'gcn'):
            A = A_bin + torch.eye(h.shape[1], device=h.device).unsqueeze(0)
            dinv = torch.rsqrt(A.sum(-1).clamp(min=1e-12))
            return torch.bmm(A * dinv.unsqueeze(-1) * dinv.unsqueeze(-2), h)
        if(self.kind in ('gin', 'gsn')):
            return (1 + self.eps[i]) * h + torch.bmm(A_bin, h)
        raise ValueError(self.kind)

    def forward(self, input):
        n_real = getattr(input, '_n_real', None)
        mask = getattr(input, '_node_mask', None)
        if(n_real is not None and mask is None):
            M = input.shape[-1]
            mask = torch.arange(M, device=input.device)[None, :] < torch.as_tensor(n_real, device=input.device)[:, None]
        idx = mask.reshape(-1).nonzero().squeeze(1) if mask is not None else None

        h, A_bin = node_feats(input)
        if(self.kind == 'gsn'):
            h = torch.cat([h, gsn_feats(A_bin, n_real)], dim=-1)
        if(mask is not None):
            h = h * mask[:, :, None]
        pooled = [h.sum(dim=1)]
        for(i, layer) in enumerate(self.layers):
            h = layer(self.aggregate(h, A_bin, i), idx, mask)
            pooled.append(h.sum(dim=1))
        score = 0
        for(cl, p) in zip(self.classifiers, pooled):
            score = score + self.dropout(cl(p))
        return score
