import torch
import torch.nn as nn
import numpy as np
from itertools import combinations, product
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def build_clique_complex(adj, max_dim=2):
    """
    Build a clique complex from an adjacency matrix.

    Args:
        adj: M x M numpy array (non-zero where edges exist)
        max_dim: maximum simplex dimension (1=edges only, 2=edges+triangles)

    Returns:
        dict: {0: [(i,), ...], 1: [(i,j), ...], 2: [(i,j,k), ...]}
    """
    M = adj.shape[0]
    simplices = {0: [(i,) for i in range(M)]}

    # 1-simplices: edges (upper triangle, non-zero entries)
    edges = []
    for i in range(M):
        for j in range(i + 1, M):
            if abs(adj[i, j]) > 1e-6:
                edges.append((i, j))
    simplices[1] = edges

    # 2-simplices: triangles (3-cliques)
    if max_dim >= 2:
        adj_set = set(edges)
        triangles = []
        for u, v in edges:
            for w in range(v + 1, M):
                if (u, w) in adj_set and (v, w) in adj_set:
                    triangles.append((u, v, w))
        simplices[2] = triangles

    return simplices


def ordered_pairs(sigma):
    """
    Canonical set of ordered pairs P(sigma) = {(i, j) : i in sigma, j in sigma}.

    Examples:
        vertex (u,):      P = {(u,u)}
        edge (u,v):       P = {(u,u),(u,v),(v,u),(v,v)}
        triangle (u,v,w): all 9 ordered pairs among {u,v,w}
    """
    verts = list(sigma)
    return list(product(verts, repeat=2))


class LearnedFiltration(nn.Module):
    """
    Learned filtration function on simplices:

        f(sigma) = rho( sum_{(i,j) in P(sigma)} X_ij )

    where rho is a shared MLP and P(sigma) is the set of all ordered
    pairs of vertices in sigma (the Cartesian product sigma x sigma).
    """
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pair_features, simplices_batch):
        """
        Args:
            pair_features: B x d x M x M
            simplices_batch: list of B dicts from build_clique_complex

        Returns:
            list of B lists of (simplex_tuple, scalar_tensor)
        """
        B, d, M, _ = pair_features.shape

        # Gather all simplex features across the batch for a single batched MLP call
        all_feats = []
        all_sigmas = []  # (batch_idx, sigma) for re-splitting
        for b in range(B):
            for dim_k in sorted(simplices_batch[b].keys()):
                for sigma in simplices_batch[b][dim_k]:
                    pairs = ordered_pairs(sigma)
                    feat = sum(pair_features[b, :, i, j] for i, j in pairs)
                    all_feats.append(feat / len(pairs))
                    all_sigmas.append((b, sigma))

        if len(all_feats) == 0:
            return [[] for _ in range(B)]

        # Single batched MLP forward pass
        all_vals = self.mlp(torch.stack(all_feats)).squeeze(-1)  # (total_simplices,)

        # Re-split into per-graph lists
        result = [[] for _ in range(B)]
        for idx, (b, sigma) in enumerate(all_sigmas):
            result[b].append((sigma, all_vals[idx]))

        return result


class DifferentiablePH(nn.Module):
    """
    Differentiable persistent homology with learned vectorization.

    Uses gudhi for the combinatorial PH computation, but maintains gradient
    flow by:
    1. Applying a differentiable make_filtration_non_decreasing to align
       the computation graph with gudhi's adjusted filtration.
    2. Tracking birth/death simplex pairs and indexing back into the
       adjusted differentiable filtration tensor for lifetime computation.
    3. Using learned attention-pooling over lifetime embeddings per homology
       dimension.
    """
    def __init__(self, max_ph_dim=1, vec_dim=16):
        super().__init__()
        self.max_ph_dim = max_ph_dim
        self.vec_dim = vec_dim
        self.out_features = (max_ph_dim + 1) * vec_dim
        self._pool = ThreadPoolExecutor(max_workers=8)

        # Per-dimension learned vectorizers
        self.embeds = nn.ModuleList()
        self.attns = nn.ModuleList()
        for _ in range(max_ph_dim + 1):
            self.embeds.append(nn.Sequential(
                nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)
            ))
            self.attns.append(nn.Sequential(
                nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)
            ))

    @staticmethod
    def _gudhi_worker(simplices_list, filt_detached):
        """Run gudhi PH on a single graph. Called in a thread pool.
        
        Args:
            simplices_list: list of simplex tuples
            filt_detached: list of float filtration values (detached)
            
        Returns:
            list of persistence pairs from gudhi
        """
        st = gudhi.SimplexTree()
        for i, sigma in enumerate(simplices_list):
            st.insert(list(sigma), filtration=filt_detached[i])
        st.make_filtration_non_decreasing()
        st.persistence()
        return st.persistence_pairs()

    @staticmethod
    def _make_non_decreasing(filt_tensor, simplices_list, simplex_to_idx):
        """
        Differentiable filtration non-decreasing correction.

        Ensures f(face) <= f(coface) by propagating max values up the
        simplex hierarchy using torch.max (which has subgradients).
        """
        adjusted = list(filt_tensor)  # list of scalar tensors

        # Group simplices by dimension
        by_dim = {}
        for idx, sigma in enumerate(simplices_list):
            by_dim.setdefault(len(sigma) - 1, []).append((idx, sigma))

        for dim in sorted(by_dim.keys()):
            if dim == 0:
                continue
            for idx, sigma in by_dim[dim]:
                face_vals = []
                for k in range(1, len(sigma)):
                    for face in combinations(sigma, k):
                        face_key = tuple(sorted(face))
                        if face_key in simplex_to_idx:
                            face_vals.append(adjusted[simplex_to_idx[face_key]])
                if face_vals:
                    max_face = torch.stack(face_vals).max()
                    adjusted[idx] = torch.max(adjusted[idx], max_face)

        return torch.stack(adjusted)

    def forward(self, filtration_batch, device, num_nodes=None):
        """
        Args:
            filtration_batch: list of B lists of (simplex, filtration_value_tensor)
            device: torch device
            num_nodes: M (padded graph size) for node-level features

        Returns:
            graph_vec: B x out_features tensor
            node_vec:  B x out_features x M tensor (None if num_nodes is None)
        """
        B = len(filtration_batch)

        if not GUDHI_AVAILABLE:
            gv = torch.zeros(B, self.out_features, device=device)
            nv = torch.zeros(B, self.out_features, num_nodes,
                             device=device) if num_nodes else None
            return gv, nv

        vectors = []
        node_vectors = []

        # --- Phase 1: prepare per-graph data (differentiable) ---
        graph_data = []
        for graph_filt in filtration_batch:
            simplex_to_idx = {}
            filt_values = []
            simplices_list = []
            for idx, (sigma, val) in enumerate(graph_filt):
                simplex_to_idx[tuple(sorted(sigma))] = idx
                filt_values.append(val)
                simplices_list.append(sigma)

            filt_tensor = torch.stack(filt_values)
            filt_adjusted = self._make_non_decreasing(
                filt_tensor, simplices_list, simplex_to_idx)
            filt_detached = filt_adjusted.detach().cpu().tolist()

            graph_data.append({
                'simplex_to_idx': simplex_to_idx,
                'simplices_list': simplices_list,
                'filt_adjusted': filt_adjusted,
                'filt_detached': filt_detached,
            })

        # --- Phase 2: parallel gudhi calls across batch ---
        futures = [
            self._pool.submit(
                self._gudhi_worker,
                gd['simplices_list'],
                gd['filt_detached'])
            for gd in graph_data
        ]
        all_pairs = [f.result() for f in futures]

        # --- Phase 3: differentiable lifetime + node attribution ---
        for b, (gd, pairs) in enumerate(zip(graph_data, all_pairs)):
            simplex_to_idx = gd['simplex_to_idx']
            simplices_list = gd['simplices_list']
            filt_adjusted = gd['filt_adjusted']
            filt_detached = gd['filt_detached']

            # Build tie-merge lookup
            val_to_nodes = defaultdict(set)
            for idx, sigma in enumerate(simplices_list):
                key = (len(sigma), round(filt_detached[idx], 6))
                for v in sigma:
                    val_to_nodes[key].add(v)

            # Differentiable lifetimes with node-involvement tracking
            dim_data = {d: [] for d in range(self.max_ph_dim + 1)}
            for birth_simplex, death_simplex in pairs:
                if len(death_simplex) == 0:
                    continue  # infinite persistence
                dim = len(birth_simplex) - 1
                if dim > self.max_ph_dim:
                    continue
                birth_key = tuple(sorted(birth_simplex))
                death_key = tuple(sorted(death_simplex))
                if birth_key in simplex_to_idx and death_key in simplex_to_idx:
                    lifetime = torch.clamp(
                        filt_adjusted[simplex_to_idx[death_key]]
                        - filt_adjusted[simplex_to_idx[birth_key]], min=0)
                    # Merge nodes from ALL simplices that share the same
                    # (dimension, filt_value) as the birth or death simplex.
                    # This makes node attribution invariant to gudhi's
                    # labeling-dependent tie-breaking.
                    bv_r = round(filt_detached[simplex_to_idx[birth_key]], 6)
                    dv_r = round(filt_detached[simplex_to_idx[death_key]], 6)
                    involved = (val_to_nodes[(len(birth_simplex), bv_r)]
                                | val_to_nodes[(len(death_simplex), dv_r)])
                    dim_data[dim].append((lifetime, involved))

            # Attention-pooling: graph-level + node-level per dimension
            graph_parts = []
            node_parts = []
            M = num_nodes
            for d in range(self.max_ph_dim + 1):
                entries = dim_data[d]
                if len(entries) == 0:
                    graph_parts.append(torch.zeros(self.vec_dim, device=device))
                    if M is not None:
                        node_parts.append(
                            torch.zeros(self.vec_dim, M, device=device))
                    continue

                lts = torch.stack(
                    [e[0] for e in entries]).unsqueeze(-1)      # (N, 1)
                embeds = self.embeds[d](lts)                    # (N, vec_dim)
                logits = self.attns[d](lts)                     # (N, 1)

                # Graph-level: attention pool over all pairs
                weights = torch.softmax(logits, dim=0)          # (N, 1)
                graph_parts.append(
                    (weights * embeds).sum(dim=0))              # (vec_dim,)

                # Node-level: masked attention pool per node
                if M is not None:
                    N = len(entries)
                    involve = torch.zeros(N, M, device=device)
                    for k, (_, ns) in enumerate(entries):
                        for n in ns:
                            if n < M:
                                involve[k, n] = 1.0
                    logits_exp = logits.expand(-1, M)           # (N, M)
                    masked = logits_exp.masked_fill(
                        involve == 0, float('-inf'))
                    node_w = torch.softmax(
                        masked, dim=0).nan_to_num(0.0)          # (N, M)
                    node_parts.append(
                        torch.mm(embeds.t(), node_w))           # (vec_dim, M)

            vectors.append(torch.cat(graph_parts))
            if M is not None:
                node_vectors.append(torch.cat(node_parts, dim=0))

        graph_out = torch.stack(vectors)
        node_out = torch.stack(node_vectors) if node_vectors else None
        return graph_out, node_out


class TopologyLayer(nn.Module):
    """
    Full topology branch for one network layer:

        1. Filtration on post-equivariant features:
           f(sigma) = rho( sum_{(i,j) in P(sigma)} X^(l+1/2)_ij )
        2. Persistent homology with differentiable vectorization:
           T_graph, {t_i} = Phi_PH(K(G), f)
        3. Broadcast topology to all pairs (graph + node-level):
           T_ij = [T_graph, t_i, t_j]
        4. Gated residual fusion:
           X^(l+1) = X^(l+1/2) + gate * psi( X^(l+1/2) || T_ij )

    Node-level features t_i are computed by attention-pooling over
    persistence pairs involving node i (parameter-shared with graph pool).
    """
    def __init__(self, eqv_features, hidden_dim=64,
                 max_ph_dim=1, num_stats=4):
        super().__init__()
        self.filtration = LearnedFiltration(eqv_features, hidden_dim)
        self.ph = DifferentiablePH(max_ph_dim, num_stats)

        topo_dim = self.ph.out_features
        self.topo_norm = nn.LayerNorm(topo_dim)
        self.node_norm = nn.LayerNorm(topo_dim)

        # Fusion input: x_eqv || T_graph || t_row || t_col
        fuse_in = eqv_features + 3 * topo_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in, eqv_features, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(eqv_features, eqv_features, kernel_size=1, bias=True),
        )
        # First layer: Xavier init for good signal propagation
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        # Last layer: small-scale init so residual starts near identity
        # but gradients flow to topology branch from step 1
        nn.init.normal_(self.fusion[2].weight, std=0.01)
        nn.init.zeros_(self.fusion[2].bias)

        # Learned gate: per-position control of topology contribution
        self.gate_conv = nn.Conv2d(fuse_in, eqv_features,
                                   kernel_size=1, bias=True)
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, x_eqv, simplices_batch):
        """
        Args:
            x_eqv:  B x d_eqv x M x M  (post-equivariant features)
            simplices_batch: list of B simplex dicts

        Returns:
            B x d_eqv x M x M
        """
        B, d, M, _ = x_eqv.shape

        # Steps 1-2: filtration on X^(l+1/2) → PH → graph + node vectors
        filt_batch = self.filtration(x_eqv, simplices_batch)
        graph_vec, node_vec = self.ph(
            filt_batch, device=x_eqv.device, num_nodes=M)

        # Normalize topology features
        graph_vec = self.topo_norm(graph_vec)                     # (B, t)
        node_vec = self.node_norm(
            node_vec.permute(0, 2, 1)).permute(0, 2, 1)           # (B, t, M)

        # Step 3: build per-pair topology features
        graph_bc = graph_vec.unsqueeze(-1).unsqueeze(-1).expand(
            B, -1, M, M)                                          # T_graph
        node_row = node_vec.unsqueeze(-1).expand(B, -1, M, M)    # t_i
        node_col = node_vec.unsqueeze(-2).expand(B, -1, M, M)    # t_j

        # Step 4: gated residual fusion
        combined = torch.cat(
            [x_eqv, graph_bc, node_row, node_col], dim=1)
        gate = torch.sigmoid(self.gate_conv(combined))
        delta = self.fusion(combined)
        out = x_eqv + gate * delta

        return out
