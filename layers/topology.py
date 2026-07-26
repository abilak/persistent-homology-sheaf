import torch
import torch.nn as nn
import numpy as np
from itertools import combinations, product
from collections import defaultdict

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
        max_dim: maximum simplex dimension (1=edges, 2=+triangles, 3=+tetrahedra)

    Returns:
        dict: {0: [(i,), ...], 1: [(i,j), ...], 2: [(i,j,k), ...], ...}
    """
    M = adj.shape[0]
    simplices = {0: [(i,) for i in range(M)]}

    edges = []
    for i in range(M):
        for j in range(i + 1, M):
            if abs(adj[i, j]) > 1e-6:
                edges.append((i, j))
    simplices[1] = edges

    if max_dim >= 2:
        adj_set = set(edges)
        triangles = []
        for u, v in edges:
            for w in range(v + 1, M):
                if (u, w) in adj_set and (v, w) in adj_set:
                    triangles.append((u, v, w))
        simplices[2] = triangles

    # 3-simplices: tetrahedra (4-cliques) — needed to expose the 4-clique
    # difference that 3-WL cannot detect (e.g. Rook(4,4) vs Shrikhande).
    if max_dim >= 3:
        tetra = []
        for (u, v, w) in simplices[2]:
            for x in range(w + 1, M):
                if (u, x) in adj_set and (v, x) in adj_set and (w, x) in adj_set:
                    tetra.append((u, v, w, x))
        simplices[3] = tetra

    return simplices


def ordered_pairs(sigma):
    """Canonical ordered pairs P(sigma) = sigma x sigma."""
    verts = list(sigma)
    return list(product(verts, repeat=2))


# --------------------------------------------------------------------------- #
#  Per-graph structure cache.
#
#  Everything that depends ONLY on the graph (the clique complex, the
#  gather/scatter indices used by the learned filtration, and the face-index
#  tables used by the non-decreasing correction) is built once and cached,
#  keyed by the binary adjacency pattern. Across epochs the same graph reuses
#  its structure, so the per-forward path contains only tensor ops + the gudhi
#  persistence call — no Python complex rebuilds and no per-simplex host syncs.
# --------------------------------------------------------------------------- #
class GraphStruct:
    """Cached, graph-only structure for the topology branch (device-agnostic
    index tensors are materialized lazily per device)."""
    __slots__ = ("simplices_list", "simplex_to_idx", "S", "M",
                 "f_i_cpu", "f_j_cpu", "f_seg_cpu", "f_counts_cpu",
                 "face_tables_cpu", "_dev_cache")

    def __init__(self, adj, max_dim):
        M = adj.shape[0]
        self.M = M
        comp = build_clique_complex(adj, max_dim=max_dim)

        simplices_list = []
        for dim_k in sorted(comp.keys()):
            simplices_list.extend(comp[dim_k])
        self.simplices_list = simplices_list
        self.simplex_to_idx = {tuple(sorted(s)): i
                               for i, s in enumerate(simplices_list)}
        self.S = len(simplices_list)

        # gather/scatter indices for f(sigma)=rho(mean_{(i,j) in sigmaxsigma} X_ij)
        f_i, f_j, f_seg, f_counts = [], [], [], []
        for s_idx, sigma in enumerate(simplices_list):
            f_counts.append(len(sigma) * len(sigma))
            for i in sigma:
                for j in sigma:
                    f_i.append(i); f_j.append(j); f_seg.append(s_idx)
        self.f_i_cpu = torch.tensor(f_i, dtype=torch.long)
        self.f_j_cpu = torch.tensor(f_j, dtype=torch.long)
        self.f_seg_cpu = torch.tensor(f_seg, dtype=torch.long)
        self.f_counts_cpu = torch.tensor(f_counts, dtype=torch.float32)

        # face-index tables for the differentiable non-decreasing correction,
        # grouped by dimension (ascending), enumeration matching the reference.
        by_dim = defaultdict(list)
        for idx, sigma in enumerate(simplices_list):
            by_dim[len(sigma) - 1].append(idx)
        face_tables = []
        max_d = max(by_dim.keys()) if by_dim else 0
        for dim in range(1, max_d + 1):
            sim_idx_list = by_dim.get(dim, [])
            if not sim_idx_list:
                continue
            face_rows = []
            for idx in sim_idx_list:
                sigma = simplices_list[idx]
                row = []
                for k in range(1, len(sigma)):
                    for face in combinations(sigma, k):
                        row.append(self.simplex_to_idx[tuple(sorted(face))])
                face_rows.append(row)
            face_tables.append((torch.tensor(sim_idx_list, dtype=torch.long),
                                torch.tensor(face_rows, dtype=torch.long)))
        self.face_tables_cpu = face_tables
        self._dev_cache = {}

    def on(self, device):
        """Return (f_i,f_j,f_seg,f_counts,face_tables) on the given device."""
        key = str(device)
        if key not in self._dev_cache:
            ft = [(a.to(device), b.to(device)) for a, b in self.face_tables_cpu]
            self._dev_cache[key] = (
                self.f_i_cpu.to(device), self.f_j_cpu.to(device),
                self.f_seg_cpu.to(device), self.f_counts_cpu.to(device), ft)
        return self._dev_cache[key]


_STRUCT_CACHE = {}
_STRUCT_CACHE_MAX = 200000


def get_graph_struct(adj, max_dim):
    """Cache lookup keyed by binary adjacency pattern."""
    adj_bin = (np.abs(adj) > 1e-6)
    key = (adj_bin.shape[0], adj_bin.astype(np.uint8).tobytes())
    st = _STRUCT_CACHE.get(key)
    if st is None:
        st = GraphStruct(adj, max_dim)
        if len(_STRUCT_CACHE) < _STRUCT_CACHE_MAX:
            _STRUCT_CACHE[key] = st
    return st


def build_graph_structs(adj_batch, max_dim):
    return [get_graph_struct(adj, max_dim) for adj in adj_batch]


class LearnedFiltration(nn.Module):
    """
    Learned filtration f(sigma) = rho( mean_{(i,j) in sigma x sigma} X_ij ),
    rho a shared MLP. Uses cached per-graph gather/scatter indices so the
    per-forward cost is a single gather + segment-sum + batched MLP.
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

    def forward(self, pair_features, structs):
        """
        Args:
            pair_features: B x d x M x M
            structs: list of B GraphStruct
        Returns:
            list of B (struct, filtration_values_tensor[S])
        """
        B, d, M, _ = pair_features.shape
        device = pair_features.device
        out = []
        for b in range(B):
            st = structs[b]
            if st.S == 0:
                out.append((st, torch.zeros(0, device=device)))
                continue
            f_i, f_j, f_seg, f_counts, _ = st.on(device)
            pair_vals = pair_features[b][:, f_i, f_j].transpose(0, 1)  # (P, d)
            sums = torch.zeros(st.S, d, device=device, dtype=pair_vals.dtype)
            sums = sums.index_add(0, f_seg, pair_vals)
            means = sums / f_counts.unsqueeze(-1)
            vals = self.mlp(means).squeeze(-1)  # (S,)
            out.append((st, vals))
        return out


class DifferentiablePH(nn.Module):
    """
    Differentiable persistent homology with learned attention-pooling
    vectorization. Uses gudhi for the combinatorial pairing while keeping
    gradient flow through the filtration values. Consumes cached GraphStructs.
    """
    def __init__(self, max_ph_dim=1, vec_dim=16):
        super().__init__()
        self.max_ph_dim = max_ph_dim
        self.vec_dim = vec_dim
        self.out_features = (max_ph_dim + 1) * vec_dim
        self.embeds = nn.ModuleList()
        self.attns = nn.ModuleList()
        for _ in range(max_ph_dim + 1):
            self.embeds.append(nn.Sequential(
                nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)))
            self.attns.append(nn.Sequential(
                nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)))

    @staticmethod
    def _make_non_decreasing(filt_tensor, face_tables):
        """f(face) <= f(coface) via max propagation up the (cached) face tables.
        Dimensions processed ascending. Autograd-aware (torch.max subgradients)."""
        adjusted = filt_tensor.clone()
        for sim_idx_t, face_idx_t in face_tables:
            face_vals = adjusted[face_idx_t]              # (S_d, F_d)
            max_face = face_vals.max(dim=1).values        # (S_d,)
            new_val = torch.max(adjusted[sim_idx_t], max_face)
            adjusted = adjusted.scatter(0, sim_idx_t, new_val)
        return adjusted

    def forward(self, filt_batch, device, num_nodes=None):
        """
        Args:
            filt_batch: list of B (GraphStruct, filtration_values[S])
            device, num_nodes: as before.
        Returns:
            graph_vec B x out_features, node_vec B x out_features x M (or None)
        """
        B = len(filt_batch)
        if not GUDHI_AVAILABLE:
            gv = torch.zeros(B, self.out_features, device=device)
            nv = torch.zeros(B, self.out_features, num_nodes,
                             device=device) if num_nodes else None
            return gv, nv

        # Phase 1 (GPU, no host sync): compute the non-decreasing-corrected
        # filtration for every graph. Then transfer ALL of them to the host in
        # a SINGLE sync (concatenate -> one .cpu()), instead of one sync per
        # graph -- this is what keeps the GPU pipeline from stalling.
        adj_list = []
        for st, filt_tensor in filt_batch:
            if st.S == 0:
                adj_list.append(None)
                continue
            _, _, _, _, face_tables = st.on(device)
            adj_list.append(self._make_non_decreasing(filt_tensor, face_tables))
        nonempty = [a for a in adj_list if a is not None]
        if nonempty:
            flat_vals = torch.cat(nonempty).detach().cpu().tolist()
        cursor = 0

        vectors, node_vectors = [], []
        for (st, filt_tensor), filt_adjusted in zip(filt_batch, adj_list):
            simplices_list = st.simplices_list
            simplex_to_idx = st.simplex_to_idx

            if st.S == 0:
                vectors.append(torch.zeros(self.out_features, device=device))
                if num_nodes is not None:
                    node_vectors.append(torch.zeros(self.out_features,
                                                    num_nodes, device=device))
                continue

            filt_vals_list = flat_vals[cursor:cursor + st.S]
            cursor += st.S

            st_tree = gudhi.SimplexTree()
            for i, sigma in enumerate(simplices_list):
                st_tree.insert(list(sigma), filtration=filt_vals_list[i])
            st_tree.make_filtration_non_decreasing()
            st_tree.persistence()
            pairs = st_tree.persistence_pairs()

            val_to_nodes = defaultdict(set)
            for idx, sigma in enumerate(simplices_list):
                key = (len(sigma), round(filt_vals_list[idx], 6))
                for v in sigma:
                    val_to_nodes[key].add(v)

            dim_data = {d: [] for d in range(self.max_ph_dim + 1)}
            for birth_simplex, death_simplex in pairs:
                if len(death_simplex) == 0:
                    continue
                dim = len(birth_simplex) - 1
                if dim > self.max_ph_dim:
                    continue
                birth_key = tuple(sorted(birth_simplex))
                death_key = tuple(sorted(death_simplex))
                if birth_key in simplex_to_idx and death_key in simplex_to_idx:
                    birth_val = filt_adjusted[simplex_to_idx[birth_key]]
                    death_val = filt_adjusted[simplex_to_idx[death_key]]
                    persistence = torch.clamp(death_val - birth_val, min=0)
                    bp = torch.stack([birth_val, persistence])
                    bv_r = round(filt_vals_list[simplex_to_idx[birth_key]], 6)
                    dv_r = round(filt_vals_list[simplex_to_idx[death_key]], 6)
                    involved = (val_to_nodes[(len(birth_simplex), bv_r)]
                                | val_to_nodes[(len(death_simplex), dv_r)])
                    dim_data[dim].append((bp, involved))

            graph_parts, node_parts = [], []
            M = num_nodes
            for d in range(self.max_ph_dim + 1):
                entries = dim_data[d]
                if len(entries) == 0:
                    graph_parts.append(torch.zeros(self.vec_dim, device=device))
                    if M is not None:
                        node_parts.append(torch.zeros(self.vec_dim, M, device=device))
                    continue
                bp = torch.stack([e[0] for e in entries])
                bp = (bp - bp.mean(dim=0)) / (bp.std(dim=0, correction=0) + 1e-8)
                embeds = self.embeds[d](bp)
                logits = self.attns[d](bp)
                weights = torch.softmax(logits, dim=0)
                graph_parts.append((weights * embeds).sum(dim=0))
                if M is not None:
                    N = len(entries)
                    # Build the node-involvement mask on the HOST (numpy) and
                    # transfer once, instead of writing element-by-element into
                    # a GPU tensor (each such write is a tiny device sync).
                    involve_np = np.zeros((N, M), dtype=np.float32)
                    for k, (_, ns) in enumerate(entries):
                        for n in ns:
                            if n < M:
                                involve_np[k, n] = 1.0
                    involve = torch.from_numpy(involve_np).to(device)
                    logits_exp = logits.expand(-1, M)
                    masked = logits_exp.masked_fill(involve == 0, float('-inf'))
                    node_w = torch.softmax(masked, dim=0).nan_to_num(0.0)
                    node_parts.append(torch.mm(embeds.t(), node_w))

            vectors.append(torch.cat(graph_parts))
            if M is not None:
                node_vectors.append(torch.cat(node_parts, dim=0))

        graph_out = torch.stack(vectors)
        node_out = torch.stack(node_vectors) if node_vectors else None
        return graph_out, node_out


class TopologyLayer(nn.Module):
    """Full topology branch for one layer: learned filtration -> differentiable
    PH -> broadcast (graph + node level) -> gated residual fusion."""
    def __init__(self, eqv_features, hidden_dim=64, max_ph_dim=1, num_stats=4):
        super().__init__()
        self.filtration = LearnedFiltration(eqv_features, hidden_dim)
        self.ph = DifferentiablePH(max_ph_dim, num_stats)

        topo_dim = self.ph.out_features
        self.topo_norm = nn.LayerNorm(topo_dim)
        self.node_norm = nn.LayerNorm(topo_dim)

        fuse_in = eqv_features + 3 * topo_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in, eqv_features, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(eqv_features, eqv_features, kernel_size=1, bias=True),
        )
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        nn.init.normal_(self.fusion[2].weight, std=0.01)
        nn.init.zeros_(self.fusion[2].bias)

        self.gate_conv = nn.Conv2d(fuse_in, eqv_features, kernel_size=1, bias=True)
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, 2.0)

    def forward(self, x_eqv, structs):
        B, d, M, _ = x_eqv.shape
        filt_batch = self.filtration(x_eqv, structs)
        graph_vec, node_vec = self.ph(filt_batch, device=x_eqv.device, num_nodes=M)

        graph_vec = self.topo_norm(graph_vec)
        node_vec = self.node_norm(node_vec.permute(0, 2, 1)).permute(0, 2, 1)

        graph_bc = graph_vec.unsqueeze(-1).unsqueeze(-1).expand(B, -1, M, M)
        node_row = node_vec.unsqueeze(-1).expand(B, -1, M, M)
        node_col = node_vec.unsqueeze(-2).expand(B, -1, M, M)

        combined = torch.cat([x_eqv, graph_bc, node_row, node_col], dim=1)
        gate = torch.sigmoid(self.gate_conv(combined))
        delta = self.fusion(combined)
        return x_eqv + gate * delta
