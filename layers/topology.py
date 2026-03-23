import torch
import torch.nn as nn
import numpy as np
from itertools import combinations, product

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
        result = []

        for b in range(B):
            graph_filt = []
            for dim_k in sorted(simplices_batch[b].keys()):
                for sigma in simplices_batch[b][dim_k]:
                    pairs = ordered_pairs(sigma)
                    feat = sum(pair_features[b, :, i, j] for i, j in pairs)
                    feat = feat / len(pairs)  # average over pairs to keep scale stable
                    val = self.mlp(feat).squeeze(-1)
                    graph_filt.append((sigma, val))

            result.append(graph_filt)

        return result


class DifferentiablePH(nn.Module):
    """
    Differentiable persistent homology vectorization.

    Uses gudhi for the combinatorial PH computation, but maintains gradient
    flow by tracking which simplices cause births/deaths in the persistence
    diagram and indexing back into the differentiable filtration tensor.

    Vectorization: per homology dimension, computes
        [count, sum_of_lifetimes, max_lifetime, mean_lifetime]
    """
    def __init__(self, max_ph_dim=1, num_stats=4):
        super().__init__()
        self.max_ph_dim = max_ph_dim
        self.num_stats = num_stats
        self.out_features = (max_ph_dim + 1) * num_stats

    def forward(self, filtration_batch, device):
        """
        Args:
            filtration_batch: list of B lists of (simplex, filtration_value_tensor)
            device: torch device

        Returns:
            B x out_features tensor (differentiable w.r.t. filtration values)
        """
        B = len(filtration_batch)

        if not GUDHI_AVAILABLE:
            return torch.zeros(B, self.out_features, device=device)

        vectors = []
        for graph_filt in filtration_batch:
            # Map each simplex to an index for differentiable lookup
            simplex_to_idx = {}
            filt_values = []
            for idx, (sigma, val) in enumerate(graph_filt):
                simplex_to_idx[tuple(sorted(sigma))] = idx
                filt_values.append(val)

            filt_tensor = torch.stack(filt_values)

            # Build gudhi simplex tree with detached values
            st = gudhi.SimplexTree()
            for sigma, val in graph_filt:
                st.insert(list(sigma), filtration=val.detach().cpu().item())
            st.make_filtration_non_decreasing()
            st.persistence()
            pairs = st.persistence_pairs()

            # Collect differentiable lifetimes per homology dimension
            dim_lifetimes = {d: [] for d in range(self.max_ph_dim + 1)}
            for birth_simplex, death_simplex in pairs:
                if len(death_simplex) == 0:
                    continue  # infinite persistence
                dim = len(birth_simplex) - 1
                if dim > self.max_ph_dim:
                    continue
                birth_key = tuple(sorted(birth_simplex))
                death_key = tuple(sorted(death_simplex))
                if birth_key in simplex_to_idx and death_key in simplex_to_idx:
                    lifetime = (filt_tensor[simplex_to_idx[death_key]]
                                - filt_tensor[simplex_to_idx[birth_key]])
                    dim_lifetimes[dim].append(torch.clamp(lifetime, min=0))

            # Vectorize: [count, sum, max, mean] per dimension
            vec_parts = []
            for d in range(self.max_ph_dim + 1):
                lts = dim_lifetimes[d]
                if len(lts) == 0:
                    vec_parts.append(torch.zeros(self.num_stats, device=device))
                else:
                    lt_stack = torch.stack(lts)
                    count = torch.tensor(float(len(lts)), device=device)
                    total = lt_stack.sum()
                    maximum = lt_stack.max()
                    mean = lt_stack.mean()
                    vec_parts.append(torch.stack([count, total, maximum, mean]))

            vectors.append(torch.cat(vec_parts))

        return torch.stack(vectors)


class TopologyLayer(nn.Module):
    """
    Full topology branch for one network layer:

        1. Filtration on pre-equivariant features:
           f(sigma) = rho( sum_{(i,j) in P(sigma)} X^(l)_ij )
        2. Persistent homology with differentiable vectorization:
           T = Phi_PH(K(G), f)
        3. Broadcast topology to all pairs:
           T_uv = T  for all (u,v)
        4. Fuse post-equivariant features with topology:
           X^(l+1) = psi( X^(l+1/2) || T )

    The filtration uses pre-equivariant features X^(l), while the
    fusion combines post-equivariant features X^(l+1/2) with topology.
    """
    def __init__(self, eqv_features, filt_features, hidden_dim=64,
                 max_ph_dim=1, num_stats=4):
        super().__init__()
        self.filtration = LearnedFiltration(filt_features, hidden_dim)
        self.ph = DifferentiablePH(max_ph_dim, num_stats)

        topo_dim = self.ph.out_features
        self.topo_norm = nn.LayerNorm(topo_dim)
        # Fusion MLP psi_ell: applied per pair via 1x1 convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(eqv_features + topo_dim, eqv_features,
                      kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(eqv_features, eqv_features,
                      kernel_size=1, bias=True),
        )
        for m in self.fusion:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_eqv, x_filt, simplices_batch):
        """
        Args:
            x_eqv:  B x d_eqv x M x M  (post-equivariant features)
            x_filt: B x d_filt x M x M  (pre-equivariant features)
            simplices_batch: list of B simplex dicts

        Returns:
            B x d_eqv x M x M
        """
        B, d, M, _ = x_eqv.shape

        # Steps 1-2: filtration on pre-equivariant features, then PH
        filt_batch = self.filtration(x_filt, simplices_batch)
        topo_vec = self.ph(filt_batch, device=x_eqv.device)  # B x t

        # Normalize PH statistics to match equivariant feature scale
        topo_vec = self.topo_norm(topo_vec)

        # Step 3: broadcast to all pairs
        topo_broadcast = topo_vec.unsqueeze(-1).unsqueeze(-1).expand(B, -1, M, M)

        # Step 4: fuse post-equivariant features with topology
        combined = torch.cat([x_eqv, topo_broadcast], dim=1)
        out = self.fusion(combined)

        return out
