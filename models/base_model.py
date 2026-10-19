import os
import torch
import torch.nn as nn

import layers.modules as modules
from layers.topology import TopologyLayer, build_structs, betti_vec

def get_device():
    if(os.environ.get('DEVICE')):
        return torch.device(os.environ['DEVICE'])
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = get_device()

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        arch = config.architecture
        block_features = arch.block_features
        in_f = config.node_labels + 1

        self.use_topo = arch.get('use_topology', False)
        self.max_dim = arch.get('topo_max_simplex_dim', 2)
        self.betti_control = not self.use_topo and arch.get('betti_control', False)
        if(self.betti_control):
            in_f += self.max_dim + 1

        if(arch.get('topo_apply_layers', 'all') == 'last'):
            topo_at = {len(block_features) - 1}
        else:
            topo_at = set(range(len(block_features)))

        self.reg_blocks = nn.ModuleList()
        self.topo_layers = nn.ModuleList() if self.use_topo else None
        last = in_f
        for(i, nf) in enumerate(block_features):
            self.reg_blocks.append(modules.RegularBlock(config, last, nf))
            if(self.use_topo):
                if(i not in topo_at):
                    self.topo_layers.append(nn.Identity())
                else:
                    self.topo_layers.append(TopologyLayer(nf, hid=arch.get('topo_hidden_dim', 64), pdim=arch.get('topo_max_ph_dim', 1), nstat=arch.get('topo_num_stats', 4), norm_st=arch.get('topo_norm_stats', False), use_ess=arch.get('topo_essential', False), squash=arch.get('topo_filt_squash', False), static=arch.get('topo_static', False), ph_backend=arch.get('topo_ph_backend', 'auto')))
            last = nf

        self.topo_readout = self.use_topo and arch.get('topo_readout', False)
        self.topo_fc = nn.ModuleList()
        if(self.topo_readout):
            for tl in self.topo_layers:
                if(isinstance(tl, nn.Identity)):
                    self.topo_fc.append(nn.Identity())
                    continue
                head = nn.Linear(tl.gdim, config.num_classes)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.topo_fc.append(head)
        self.betti_fc = nn.Linear(self.max_dim + 1, config.num_classes) if self.betti_control else None

        self.fc_layers = nn.ModuleList()
        for nf in block_features:
            self.fc_layers.append(modules.FullyConnected(2 * nf, config.num_classes, act=None))

    def graph_structs(self, input):
        adj = getattr(input, '_host_adj', None)
        if(adj is None):
            adj = input[:, 0, :, :].detach().cpu().numpy()
        n_real = getattr(input, '_n_real', None)
        if(n_real is not None):
            adj = [adj[b, :n, :n] for b, n in enumerate(n_real)]
        return build_structs(adj, self.max_dim)

    def forward(self, input):
        x = input
        scores = x.new_zeros(())
        tscores = x.new_zeros(())
        structs = self.graph_structs(input) if self.use_topo else None
        nmask = getattr(input, '_node_mask', None)

        betti = None
        if(self.betti_control):
            bettis = [betti_vec(st, self.max_dim) for st in self.graph_structs(input)]
            betti = torch.log1p(torch.tensor(bettis, dtype=x.dtype, device=x.device))
            B, _, M, _ = x.shape
            eye = torch.eye(M, dtype=x.dtype, device=x.device)
            if(nmask is not None):
                eye = eye[None] * nmask[:, None, :].to(x.dtype)
            else:
                eye = eye[None].expand(B, M, M)
            x = torch.cat([x, betti[:, :, None, None] * eye[:, None]], dim=1)

        mask2d = None
        if(nmask is not None):
            mask2d = (nmask[:, :, None] & nmask[:, None, :])[:, None].to(x.dtype)
            pool = lambda t: modules.diag_offdiag_maxpool_masked(t, nmask)
        else:
            pool = modules.diag_offdiag_maxpool

        for(i, block) in enumerate(self.reg_blocks):
            x = block(x, mask2d)
            if(self.use_topo and not isinstance(self.topo_layers[i], nn.Identity)):
                x = self.topo_layers[i](x, structs)
                if(self.topo_readout):
                    tscores = self.topo_fc[i](self.topo_layers[i].last_g) + tscores
            scores = self.fc_layers[i](pool(x)) + scores

        if(betti is not None):
            scores = scores + self.betti_fc(betti)
        return scores + tscores

def build_model(config):
    if(config.architecture.get('baseline_type')):
        from models.baseline_models import BaselineModel
        return BaselineModel(config).to(DEVICE)
    return BaseModel(config).to(DEVICE)
