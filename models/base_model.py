import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules
from layers.topology import TopologyLayer, build_clique_complex


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        use_new_suffix = config.architecture.new_suffix  # True or False
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # Topology config (disabled by default for backward compat)
        self.use_topology = getattr(config.architecture, 'use_topology', False)

        # First part - sequential equivariant blocks + optional topology
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        self.topo_layers = nn.ModuleList() if self.use_topology else None
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            if self.use_topology:
                topo_hidden = getattr(config.architecture, 'topo_hidden_dim', 64)
                topo_ph_dim = getattr(config.architecture, 'topo_max_ph_dim', 1)
                topo_stats = getattr(config.architecture, 'topo_num_stats', 4)
                self.topo_layers.append(
                    TopologyLayer(eqv_features=next_layer_features,
                                  hidden_dim=topo_hidden,
                                  max_ph_dim=topo_ph_dim,
                                  num_stats=topo_stats)
                )
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2*output_features, self.config.num_classes, activation_fn=None)
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))

        # Cache for simplicial complexes (keyed by adjacency bytes)
        self._simplicial_cache = {}

    def _build_simplicial_complexes(self, input):
        """Extract adjacency from channel 0 and build clique complexes per graph, with caching."""
        adj_batch = input[:, 0, :, :].detach().cpu().numpy()
        max_dim = getattr(self.config.architecture, 'topo_max_simplex_dim', 2)
        results = []
        for adj in adj_batch:
            key = adj.tobytes()
            if key not in self._simplicial_cache:
                self._simplicial_cache[key] = build_clique_complex(adj, max_dim=max_dim)
            results.append(self._simplicial_cache[key])
        return results

    def forward(self, input, ablation_mode=None):
        x = input
        scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        # Build simplicial complexes once from input adjacency
        simplices_batch = None
        if self.use_topology:
            simplices_batch = self._build_simplicial_complexes(input)

        for i, block in enumerate(self.reg_blocks):

            # Step 1: equivariant update: X^(l+1/2) = EqvLayer(X^(l))
            x = block(x)

            # Steps 2-5: topology (filtration on X^(l+1/2) -> PH -> broadcast -> fuse)
            if self.use_topology:
                x = self.topo_layers[i](x, simplices_batch, ablation_mode=ablation_mode)

            if self.config.architecture.new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        return scores
