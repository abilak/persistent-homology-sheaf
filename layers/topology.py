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


def _simplex_keys(arr, M):
    """integer key of each row (sorted vertex tuple): big-endian base M, so
    ascending key order == lexicographic order."""
    key = np.zeros(len(arr), dtype=np.int64)
    for c in range(arr.shape[1]):
        key = key * M + arr[:, c].astype(np.int64)
    return key


def _clique_arrays(adj, max_dim):
    """Clique complex as per-dimension numpy arrays of sorted vertex tuples in
    lexicographic order (same enumeration as build_clique_complex)."""
    A = np.abs(np.asarray(adj)) > 1e-6
    np.fill_diagonal(A, False)
    M = A.shape[0]
    out = {0: np.arange(M, dtype=np.int64)[:, None]}
    iu, ju = np.nonzero(np.triu(A, 1))                 # row-major -> lexicographic
    out[1] = np.stack([iu, ju], 1).astype(np.int64) if len(iu) else np.zeros((0, 2), np.int64)
    prev = out[1]
    for k in range(2, max_dim + 1):
        if not len(prev):
            out[k] = np.zeros((0, k + 1), np.int64); prev = out[k]; continue
        # common neighbours of all vertices of each (k-1)-simplex, greater
        # than its last vertex: (n_prev, M) boolean, row-major nonzero keeps
        # lexicographic order
        common = np.ones((len(prev), M), dtype=bool)
        for c in range(prev.shape[1]):
            common &= A[prev[:, c]]
        common &= np.arange(M)[None, :] > prev[:, -1:]
        r, x = np.nonzero(common)
        out[k] = np.concatenate([prev[r], x[:, None]], 1).astype(np.int64) \
            if len(r) else np.zeros((0, k + 1), np.int64)
        prev = out[k]
    return out


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
    __slots__ = ("simplices_list", "simplex_to_idx", "S", "M", "dim_batches",
                 "simplex_dim", "tph",
                 "f_flat_cpu", "f_seg_cpu", "f_counts_cpu",
                 "face_tables_cpu", "_dev_cache")

    def __init__(self, adj, max_dim):
        M = adj.shape[0]
        self.M = M
        # Vectorized construction (numpy). Produces exactly the arrays of the
        # original per-simplex Python loops (test_struct_builder.py): simplices
        # in lexicographic order within each dimension, dimensions ascending.
        by_dim_arr = _clique_arrays(adj, max_dim)            # {k: (n_k, k+1)}
        dims_present = [k for k in sorted(by_dim_arr) if len(by_dim_arr[k])]
        simplices_list = []
        offsets = {}
        for k in sorted(by_dim_arr):
            offsets[k] = len(simplices_list)
            simplices_list.extend(map(tuple, by_dim_arr[k].tolist()))
        self.simplices_list = simplices_list
        self.simplex_to_idx = dict(zip(simplices_list, range(len(simplices_list))))
        self.S = len(simplices_list)

        # f(sigma) = rho(mean over sigma x sigma of X_ij): flat pair positions
        f_flat, f_seg, f_counts = [], [], []
        for k in sorted(by_dim_arr):
            arr = by_dim_arr[k]
            if not len(arr):
                continue
            n_k, w = arr.shape
            flat = (arr[:, :, None] * M + arr[:, None, :]).reshape(n_k, w * w)
            f_flat.append(flat.reshape(-1))
            f_seg.append(np.repeat(np.arange(n_k) + offsets[k], w * w))
            f_counts.append(np.full(n_k, w * w, dtype=np.float32))
        cat = lambda xs, dt: (np.concatenate(xs) if xs else np.zeros(0, dt))
        self.f_flat_cpu = torch.from_numpy(cat(f_flat, np.int64).astype(np.int64))
        self.f_seg_cpu = torch.from_numpy(cat(f_seg, np.int64).astype(np.int64))
        self.f_counts_cpu = torch.from_numpy(cat(f_counts, np.float32).astype(np.float32))

        # face tables: for each simplex of dim k, all proper faces in the
        # order (size 1 combinations, size 2, ...), looked up by integer key
        keys = {k: _simplex_keys(by_dim_arr[k], M) for k in by_dim_arr}
        face_tables = {}
        for k in sorted(by_dim_arr):
            arr = by_dim_arr[k]
            if k == 0 or not len(arr):
                continue
            cols = []
            for size in range(1, k + 1):
                for pos in combinations(range(k + 1), size):
                    fk = _simplex_keys(arr[:, list(pos)], M)
                    loc = np.searchsorted(keys[size - 1], fk)
                    cols.append(loc + offsets[size - 1])
            face_tables[k] = (torch.from_numpy(np.arange(len(arr)) + offsets[k]).long(),
                              torch.from_numpy(np.stack(cols, axis=1)).long())
        self.face_tables_cpu = face_tables
        self._dev_cache = {}
        # for gudhi.SimplexTree.insert_batch: per dimension, a (k+1, n_k)
        # vertex array and the positions of those simplices in simplices_list
        self.tph = None      # static arrays for the torch PH backend (lazy)
        self.simplex_dim = [len(sg) - 1 for sg in simplices_list]
        self.dim_batches = []
        for k in dims_present:
            arr = by_dim_arr[k]
            self.dim_batches.append((np.arange(len(arr), dtype=np.int64) + offsets[k],
                                     arr.T.astype(np.int32).copy()))

    def on(self, device):
        """Return (f_flat, f_seg, f_counts, face_tables{dim:(sim,face)})."""
        key = str(device)
        if key not in self._dev_cache:
            ft = {d: (a.to(device), b.to(device))
                  for d, (a, b) in self.face_tables_cpu.items()}
            self._dev_cache[key] = (
                self.f_flat_cpu.to(device), self.f_seg_cpu.to(device),
                self.f_counts_cpu.to(device), ft)
        return self._dev_cache[key]


_STRUCT_CACHE = {}
_STRUCT_CACHE_MAX = 200000


def get_graph_struct(adj, max_dim):
    """Cache lookup keyed by binary adjacency pattern AND the maximum simplex
    dimension (the same graph at D=2 and D=3 has different complexes; keying
    on the adjacency alone silently reused a complex of the wrong dimension
    whenever one process mixed dimensions)."""
    adj_bin = (np.abs(adj) > 1e-6)
    key = (adj_bin.shape[0], int(max_dim), adj_bin.astype(np.uint8).tobytes())
    st = _STRUCT_CACHE.get(key)
    if st is None:
        st = GraphStruct(adj, max_dim)
        if len(_STRUCT_CACHE) < _STRUCT_CACHE_MAX:
            _STRUCT_CACHE[key] = st
    return st


def build_graph_structs(adj_batch, max_dim):
    return [get_graph_struct(adj, max_dim) for adj in adj_batch]


# --------------------------------------------------------------------------- #
#  Per-batch plan: every index array the topology branch needs for a batch,
#  built ONCE on the host in numpy and moved to the device in a single pinned,
#  non-blocking copy. Shared by all topology layers and filtration heads of a
#  forward pass (memoized on the batch's struct identities).
# --------------------------------------------------------------------------- #
def _tph_static(st):
    """vertex / edge / triangle local tables of one graph for torch_ph."""
    if st.tph is None:
        v_idx, e_idx, e_end, t_idx, t_edges, t_verts = [], [], [], [], [], []
        eloc = {}
        for i, sg in enumerate(st.simplices_list):
            k = len(sg) - 1
            if k == 0:
                v_idx.append((sg[0], i))
            elif k == 1:
                eloc[tuple(sorted(sg))] = len(e_idx)
                e_idx.append(i); e_end.append(sorted(sg))
        tloc = {}
        for i, sg in enumerate(st.simplices_list):
            if len(sg) == 3:
                a, b, c = sorted(sg)
                tloc[(a, b, c)] = len(t_idx)
                t_idx.append(i); t_verts.append((a, b, c))
                t_edges.append((eloc[(a, b)], eloc[(a, c)], eloc[(b, c)]))
        q_idx, q_faces, q_verts = [], [], []
        for i, sg in enumerate(st.simplices_list):
            if len(sg) == 4:
                a, b, c, d = sorted(sg)
                q_idx.append(i); q_verts.append((a, b, c, d))
                # boundary order bcd, acd, abd, abc  (signs +, -, +, -)
                q_faces.append((tloc[(b, c, d)], tloc[(a, c, d)], tloc[(a, b, d)], tloc[(a, b, c)]))
        # cycle rank m - n + c (filtration-independent): number of positive
        # edges, hence of nonzero columns in the H1 cohomology reduction
        parent = list(range(st.M))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        comps = st.M
        for a, b in e_end:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb; comps -= 1
        vi = np.zeros(st.M, dtype=np.int64)
        for v, i in v_idx:
            vi[v] = i
        st.tph = dict(q=np.asarray(q_idx, np.int64),
                      qfaces=np.asarray(q_faces, np.int64).reshape(-1, 4),
                      qverts=np.asarray(q_verts, np.int64).reshape(-1, 4),
                      cyc=len(e_idx) - st.M + comps,
                      v=vi, e=np.asarray(e_idx, np.int64),
                      eend=np.asarray(e_end, np.int64).reshape(-1, 2),
                      t=np.asarray(t_idx, np.int64),
                      tedges=np.asarray(t_edges, np.int64).reshape(-1, 3),
                      tverts=np.asarray(t_verts, np.int64).reshape(-1, 3))
    return st.tph


class BatchPlan:
    TORCH_PH_MAX_ELEMS = 1 << 26     # per coboundary block; larger -> gudhi path

    def __init__(self, structs, device, reps=1, max_ph_dim=1, max_simplex_dim=None,
                 backend='auto', M_pad=None):
        B = len(structs)
        if max_simplex_dim is None:
            max_simplex_dim = max((max(st.simplex_dim) for st in structs if st.S), default=0)
        self.B, self.reps, self.device = B, reps, device
        # M: node dimension of the batch tensor. Without padding every graph
        # has M nodes; with padded batching graphs have st.M <= M and their
        # (i, j) positions are re-indexed into the padded M x M grid.
        self.M = M_pad if M_pad is not None else (structs[0].M if B else 0)
        self.same_M = all(st.M <= self.M for st in structs) if M_pad is not None \
            else all(st.M == self.M for st in structs)
        S = np.asarray([st.S for st in structs], np.int64)
        self.S = S
        off = np.concatenate([[0], np.cumsum(S)[:-1]]) if B else np.zeros(0, np.int64)
        S_tot = int(S.sum())
        self.S_tot = S_tot
        arrays = {}
        # learned filtration gather (one repetition; every head reuses it)
        if self.same_M and S_tot:
            MM = self.M * self.M
            def _flat(st):
                f = st.f_flat_cpu.numpy()
                if st.M == self.M:
                    return f
                return (f // st.M) * self.M + (f % st.M)
            arrays['f_flat'] = np.concatenate(
                [_flat(st) + b * MM for b, st in enumerate(structs) if st.S])
            arrays['f_seg'] = np.concatenate(
                [st.f_seg_cpu.numpy() + off[b] for b, st in enumerate(structs) if st.S])
            arrays['f_counts'] = np.concatenate(
                [st.f_counts_cpu.numpy().astype(np.int64) for st in structs if st.S])
        # face tables for all reps x graphs (item k = r * B + b)
        dims = sorted({d for st in structs for d in st.face_tables_cpu})
        self.face_dims = dims
        for d in dims:
            sims, faces = [], []
            for r in range(reps):
                for b, st in enumerate(structs):
                    if d in st.face_tables_cpu:
                        a_, f_ = st.face_tables_cpu[d]
                        o = r * S_tot + off[b]
                        sims.append(a_.numpy() + o); faces.append(f_.numpy() + o)
            arrays[f'sim{d}'] = np.concatenate(sims)
            arrays[f'face{d}'] = np.concatenate(faces)
        # torch PH backend eligibility (decided on host: no sync)
        T_max = max((len(_tph_static(st)['t']) for st in structs), default=0) \
            if max_simplex_dim >= 2 else 0
        Q_max = max((len(_tph_static(st)['q']) for st in structs), default=0) \
            if max_simplex_dim >= 3 else 0
        m_max = max((len(_tph_static(st)['e']) for st in structs), default=0)
        N_items = B * reps
        # dense coboundary blocks must fit: (items x cols x rows) int16
        # the reductions run in chunks of items, so only a SINGLE graph's
        # coboundary block has to fit (plus the batched H0 closure, which is
        # chunked internally as well)
        fits = (m_max * T_max <= self.TORCH_PH_MAX_ELEMS and
                T_max * Q_max <= self.TORCH_PH_MAX_ELEMS)
        self.use_torch_ph = (backend == 'torch' or (backend == 'auto' and device.type != 'cpu')) \
            and self.same_M and max_ph_dim <= 2 and max_simplex_dim <= 3 \
            and fits and S_tot > 0
        if backend == 'torch' and not self.use_torch_ph:
            pass    # unsupported config -> falls back to gudhi silently
        if self.use_torch_ph:
            arrays.update(self._torch_ph_arrays(structs, off, S_tot, reps, T_max, Q_max))
        # one pinned, non-blocking host->device copy for everything
        keys = list(arrays)
        flat = np.concatenate([arrays[k].reshape(-1) for k in keys]) if keys else np.zeros(0, np.int64)
        t = torch.from_numpy(flat.astype(np.int64))
        if device.type == 'cuda':
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        self.t = {}
        pos = 0
        for k in keys:
            a = arrays[k]
            self.t[k] = t[pos:pos + a.size].reshape(a.shape); pos += a.size
        if self.same_M and S_tot:
            self.f_counts = self.t['f_counts']
        if self.use_torch_ph:
            tp = dict(self._tph_meta)
            for k in ('vg', 'eg', 'ea', 'eb', 'tg', 'te', 'qg', 'qf', 'item'):
                if k in self.t:
                    tp[k] = self.t[k]
            for k in ('vmask', 'emask', 'tmask', 'qmask'):
                if k in self.t:
                    tp[k] = self.t[k].bool()
            N, n, m = tp['N'], tp['n'], tp['m']
            if m:
                inc = torch.zeros(N, m, n, device=device)
                bi = torch.arange(N, device=device)[:, None].expand(N, m)
                ei = torch.arange(m, device=device)[None, :].expand(N, m)
                w = tp['emask'].to(inc.dtype)
                inc.index_put_((bi, ei, tp['ea']), w, accumulate=True)
                inc.index_put_((bi, ei, tp['eb']), w, accumulate=True)
                tp['incE'] = inc.clamp(max=1)
            else:
                tp['incE'] = torch.zeros(N, 1, n, device=device)
            if tp['T']:
                T = tp['T']
                inc = torch.zeros(N, T, n, device=device)
                bi = torch.arange(N, device=device)[:, None].expand(N, T)
                ti = torch.arange(T, device=device)[None, :].expand(N, T)
                w = tp['tmask'].to(inc.dtype)
                for c in range(3):
                    inc.index_put_((bi, ti, self.t['tv'][:, :, c]), w, accumulate=True)
                tp['incT'] = inc.clamp(max=1)
            if tp['Q']:
                Qn = tp['Q']
                inc = torch.zeros(N, Qn, n, device=device)
                bi = torch.arange(N, device=device)[:, None].expand(N, Qn)
                qi = torch.arange(Qn, device=device)[None, :].expand(N, Qn)
                w = tp['qmask'].to(inc.dtype)
                for c in range(4):
                    inc.index_put_((bi, qi, self.t['qv'][:, :, c]), w, accumulate=True)
                tp['incQ'] = inc.clamp(max=1)
            self.torch_ph = tp

    def _torch_ph_arrays(self, structs, off, S_tot, reps, T_max, Q_max=0):
        B, n = len(structs), self.M
        m = max(len(_tph_static(st)['e']) for st in structs)
        N = B * reps
        vg = np.zeros((N, n), np.int64); vmask = np.zeros((N, n), np.int64)
        eg = np.zeros((N, max(m, 1)), np.int64); emask = np.zeros_like(eg)
        ea = np.zeros_like(eg); eb = np.zeros_like(eg)
        T = T_max
        tg = np.zeros((N, max(T, 1)), np.int64); tmask = np.zeros_like(tg)
        te = np.zeros((N, max(T, 1), 3), np.int64); tv = np.zeros_like(te)
        Q = Q_max
        qg = np.zeros((N, max(Q, 1)), np.int64); qmask = np.zeros_like(qg)
        qf = np.zeros((N, max(Q, 1), 4), np.int64); qv = np.zeros_like(qf)
        item = np.arange(N, dtype=np.int64)
        for r in range(reps):
            for b, st in enumerate(structs):
                k = r * B + b; o = r * S_tot + off[b]
                d = _tph_static(st)
                vg[k, :st.M] = d['v'] + o; vmask[k, :st.M] = 1
                me = len(d['e'])
                if me:
                    eg[k, :me] = d['e'] + o; emask[k, :me] = 1
                    ea[k, :me] = d['eend'][:, 0]; eb[k, :me] = d['eend'][:, 1]
                mt = len(d['t'])
                if mt and T:
                    tg[k, :mt] = d['t'] + o; tmask[k, :mt] = 1
                    te[k, :mt] = d['tedges']; tv[k, :mt] = d['tverts']
                mq = len(d['q'])
                if mq and Q:
                    qg[k, :mq] = d['q'] + o; qmask[k, :mq] = 1
                    qf[k, :mq] = d['qfaces']; qv[k, :mq] = d['qverts']
        K1 = max((_tph_static(st)['cyc'] for st in structs), default=0)
        self._tph_meta = dict(N=N, n=n, m=m, T=T, Q=Q, K1=K1, K2=T)
        out = dict(vg=vg, vmask=vmask, eg=eg[:, :m] if m else eg[:, :0],
                   emask=emask[:, :m] if m else emask[:, :0],
                   ea=ea[:, :m] if m else ea[:, :0], eb=eb[:, :m] if m else eb[:, :0],
                   item=item)
        if T:
            out.update(tg=tg, tmask=tmask, te=te, tv=tv)
        if Q:
            out.update(qg=qg, qmask=qmask, qf=qf, qv=qv)
        return out


_PLAN_CACHE = {}


def get_plan(structs, device, reps=1, max_ph_dim=1, max_simplex_dim=None, backend='auto',
             M_pad=None):
    key = (tuple(map(id, structs)), reps, str(device), max_ph_dim, max_simplex_dim, backend, M_pad)
    p = _PLAN_CACHE.get(key)
    if p is None:
        if len(_PLAN_CACHE) > 8:
            _PLAN_CACHE.clear()
        p = BatchPlan(structs, device, reps, max_ph_dim, max_simplex_dim, backend, M_pad)
        _PLAN_CACHE[key] = p
    return p



class LearnedFiltration(nn.Module):
    """
    Learned filtration f(sigma) = rho( mean_{(i,j) in sigma x sigma} X_ij ),
    rho a shared MLP. Uses cached per-graph gather/scatter indices so the
    per-forward cost is a single gather + segment-sum + batched MLP.
    """
    def __init__(self, in_features, hidden_dim=64, squash=False):
        super().__init__()
        # squash: f = tanh(rho(.)). tanh is strictly increasing, so it changes
        # no persistence PAIRING (only the values), commutes with the max
        # correction, and keeps filtration values in (-1, 1): prevents the
        # filtration blow-up seen on PROTEINS (std 0.75 -> 22).
        self.squash = squash
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

    def forward(self, pair_features, structs, plan=None):
        """
        Args:
            pair_features: B x d x M x M
            structs: list of B GraphStruct
        Returns:
            list of B (struct, filtration_values_tensor[S])

        The whole batch is processed with ONE gather, ONE segment-sum and ONE
        MLP call (instead of ~6 kernel launches per graph), which is what makes
        this cheap on GPU. Graphs are grouped by node count upstream, so every
        struct in the batch shares M and can index a flat (B*M*M, d) view.
        """
        B, d, M, _ = pair_features.shape
        device = pair_features.device

        if plan is not None and plan.same_M and plan.S_tot and plan.M == M:
            pf = pair_features.permute(0, 2, 3, 1).reshape(B * M * M, d)
            sums = torch.zeros(plan.S_tot, d, device=device, dtype=pf.dtype)
            sums = sums.index_add(0, plan.t['f_seg'], pf[plan.t['f_flat']])
            vals = self.mlp(sums / plan.f_counts.to(pf.dtype).unsqueeze(-1)).squeeze(-1)
            if self.squash:
                vals = torch.tanh(vals)
            out, cur = [], 0
            for st in structs:
                out.append((st, vals[cur:cur + st.S])); cur += st.S
            return out

        flat_list, seg_list, cnt_list, sizes = [], [], [], []
        seg_off = 0
        for b, st in enumerate(structs):
            if st.S == 0 or st.M != M:
                sizes.append(0 if st.S == 0 else -1)  # -1 => per-graph fallback
                continue
            f_flat, f_seg, f_counts, _ = st.on(device)
            flat_list.append(f_flat + b * M * M)
            seg_list.append(f_seg + seg_off)
            cnt_list.append(f_counts)
            seg_off += st.S
            sizes.append(st.S)

        if any(s == -1 for s in sizes):        # mismatched sizes: safe path
            return self._forward_per_graph(pair_features, structs)

        out = [None] * B
        if seg_off == 0:
            return [(st, torch.zeros(0, device=device)) for st in structs]

        flat_idx = torch.cat(flat_list)
        seg_idx = torch.cat(seg_list)
        counts = torch.cat(cnt_list)

        pf = pair_features.permute(0, 2, 3, 1).reshape(B * M * M, d)
        pair_vals = pf[flat_idx]                                   # (P_tot, d)
        sums = torch.zeros(seg_off, d, device=device, dtype=pair_vals.dtype)
        sums = sums.index_add(0, seg_idx, pair_vals)
        means = sums / counts.unsqueeze(-1)
        vals = self.mlp(means).squeeze(-1)                         # (S_tot,)
        if self.squash:
            vals = torch.tanh(vals)

        cursor = 0
        for b, st in enumerate(structs):
            s = sizes[b]
            if s == 0:
                out[b] = (st, torch.zeros(0, device=device))
            else:
                out[b] = (st, vals[cursor:cursor + s])
                cursor += s
        return out

    def _forward_per_graph(self, pair_features, structs):
        """Fallback for batches whose graphs differ in node count."""
        B, d, M, _ = pair_features.shape
        device = pair_features.device
        out = []
        for b, st in enumerate(structs):
            if st.S == 0:
                out.append((st, torch.zeros(0, device=device)))
                continue
            f_flat, f_seg, f_counts, _ = st.on(device)
            pf = pair_features[b].permute(1, 2, 0).reshape(st.M * st.M, d)
            pair_vals = pf[f_flat]
            sums = torch.zeros(st.S, d, device=device, dtype=pair_vals.dtype)
            sums = sums.index_add(0, f_seg, pair_vals)
            vals = self.mlp(sums / f_counts.unsqueeze(-1)).squeeze(-1)
            if self.squash:
                vals = torch.tanh(vals)
            out.append((st, vals))
        return out


class DifferentiablePH(nn.Module):
    """
    Differentiable persistent homology with learned attention-pooling
    vectorization. Uses gudhi for the combinatorial pairing while keeping
    gradient flow through the filtration values. Consumes cached GraphStructs.
    """
    def __init__(self, max_ph_dim=1, vec_dim=16, multiplicity=False,
                 scale_stats=False, essential=False):
        super().__init__()
        self.max_ph_dim = max_ph_dim
        self.vec_dim = vec_dim
        # essential: also vectorize ESSENTIAL classes (death = infinity), which
        # the base model discards. Each dimension gets a second block: attention
        # pooling over the essential births + log(1 + #essential). On clique
        # complexes of molecules (almost no triangles) every ring is an
        # essential H1 class, so without this the branch only sees H0.
        # The block is a function of the (invariant) persistence diagram, and
        # zero weights recover the base model, so invariance and every
        # expressivity result are preserved.
        self.essential = essential
        # multiplicity: append log(1 + #persistence-pairs) per dimension, so the
        # pooling is aware of feature COUNT (which the softmax average loses).
        self.multiplicity = multiplicity
        # scale_stats: the (birth, persistence) pairs are standardized per graph
        # and dimension before embedding, which stabilizes training but discards
        # the ABSOLUTE filtration scale. When enabled we append that scale back
        # explicitly as 4 permutation-invariant statistics
        # (mean/std of birth, mean/std of persistence) at the graph level.
        self.scale_stats = scale_stats
        extra = (1 if multiplicity else 0)
        self.ess_width = (vec_dim + 1) if essential else 0
        self.per_dim_node = vec_dim + extra + self.ess_width
        self.per_dim_graph = self.per_dim_node + (4 if scale_stats else 0)
        self.out_features = (max_ph_dim + 1) * self.per_dim_graph
        self.node_out_features = (max_ph_dim + 1) * self.per_dim_node
        self.embeds = nn.ModuleList()
        self.attns = nn.ModuleList()
        for _ in range(max_ph_dim + 1):
            self.embeds.append(nn.Sequential(
                nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)))
            self.attns.append(nn.Sequential(
                nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)))
        if essential:
            # essential classes carry a single finite coordinate: the birth
            self.ess_embeds = nn.ModuleList()
            self.ess_attns = nn.ModuleList()
            for _ in range(max_ph_dim + 1):
                self.ess_embeds.append(nn.Sequential(
                    nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)))
                self.ess_attns.append(nn.Sequential(
                    nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)))

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

    # ------------------------------------------------------------------ #
    #  Vectorized forward.
    #  Host side: gudhi + integer bookkeeping only (which simplices pair with
    #  which, and which nodes each pair involves). Device side: ONE gather of
    #  birth/death values for the whole batch, ONE embedding/attention MLP call
    #  per homological dimension, and segment softmax / segment sums via
    #  scatter ops. Replaces ~100 tiny tensor ops per graph (and their autograd
    #  nodes). Numerically identical to the per-graph reference
    #  (test_fast_vs_ref.py, float64).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _seg_softmax(logits, seg, n_seg):
        """softmax of logits (N,) within segments seg (N,) -> weights (N,)."""
        if logits.device.type == 'mps':   # no scatter_reduce on MPS: shift by the
            mx = (logits.detach().amax() if logits.numel() else   # global max
                  logits.new_zeros(())).expand(n_seg)
        else:
            mx = torch.full((n_seg,), float('-inf'), device=logits.device,
                            dtype=logits.dtype)
            mx = mx.scatter_reduce(0, seg, logits.detach(), reduce='amax', include_self=True)
        e = torch.exp(logits - mx[seg])
        den = torch.zeros(n_seg, device=logits.device, dtype=logits.dtype)
        den = den.index_add(0, seg, e)
        return e / den[seg]

    def _pool(self, q, seg, n_seg, embeds, attns, dims, P1,
              node_rows=None, node_cols=None, M=None):
        """Attention pooling of pair inputs q (N, c) grouped by seg (graph*P1+dim).
        Returns graph part (n_seg, v) and, if node_rows given, node part
        (n_seg*M, v) plus per-(seg,node) counts."""
        v = self.vec_dim
        dev, dt = q.device, q.dtype
        # every dimension's MLP on all rows, then select: no data-dependent
        # control flow (a `.any()` / `nonzero` here would sync the GPU)
        emb = torch.zeros(q.shape[0], v, device=dev, dtype=dt)
        logit = torch.zeros(q.shape[0], device=dev, dtype=dt)
        for d in range(P1):
            m = (dims == d)
            emb = torch.where(m[:, None], embeds[d](q), emb)
            logit = torch.where(m, attns[d](q).squeeze(-1), logit)
        w = self._seg_softmax(logit, seg, n_seg)
        gpool = torch.zeros(n_seg, v, device=dev, dtype=dt).index_add(
            0, seg, w.unsqueeze(-1) * emb)
        if node_rows is None:
            return gpool, None, None
        gid = seg[node_rows] * M + node_cols
        n_g = n_seg * M
        wn = self._seg_softmax(logit[node_rows], gid, n_g)
        npool = torch.zeros(n_g, v, device=dev, dtype=dt).index_add(
            0, gid, wn.unsqueeze(-1) * emb[node_rows])
        ncount = torch.zeros(n_g, device=dev, dtype=dt).index_add(
            0, gid, torch.ones_like(wn))
        return gpool, npool, ncount

    def forward(self, filt_batch, device, num_nodes=None, plan=None):
        B = len(filt_batch)
        P1 = self.max_ph_dim + 1
        M = num_nodes
        node_level = M is not None
        zeros_out = lambda: (torch.zeros(B, self.out_features, device=device),
                             torch.zeros(B, self.node_out_features, M, device=device)
                             if node_level else None)
        # Phase 1 (device): batched non-decreasing correction.
        offsets, cat_parts, off = [], [], 0
        for st, ft in filt_batch:
            offsets.append(None if st.S == 0 else off)
            if st.S:
                cat_parts.append(ft); off += st.S
        if not cat_parts:
            return zeros_out()
        vals_cat = torch.cat(cat_parts)
        if plan is not None and plan.reps * plan.B == B:
            tables = [(plan.t[f'sim{d}'], plan.t[f'face{d}']) for d in plan.face_dims]
        else:
            dim_sim, dim_face = defaultdict(list), defaultdict(list)
            for (st, _), o in zip(filt_batch, offsets):
                if o is None:
                    continue
                for dmn, (sim_t, face_t) in st.on(device)[3].items():
                    dim_sim[dmn].append(sim_t + o); dim_face[dmn].append(face_t + o)
            tables = [(torch.cat(dim_sim[dd]), torch.cat(dim_face[dd]))
                      for dd in sorted(dim_sim)]
        adj = self._make_non_decreasing(vals_cat, tables)

        if plan is not None and plan.use_torch_ph and plan.reps * plan.B == B:
            # GPU-resident persistence: no host synchronization anywhere
            from layers.torch_ph import torch_persistence
            blocks = torch_persistence(adj, plan, P1, self.essential, node_level, M)
            return self._vectorize_dense(blocks, B, P1, M, adj.dtype, device)
        else:
            if not GUDHI_AVAILABLE:
                return zeros_out()
            coo = self._gudhi_pairs(adj, filt_batch, offsets, P1, M, device)
        return self._vectorize(adj, coo, B, P1, M)

    @staticmethod
    def _masked_softmax(logits, mask, dim):
        """softmax over `dim` restricted to mask; all-masked slices -> 0.
        The max shift is detached (softmax is shift-invariant), so no
        gradient flows through it and no NaN appears for empty slices."""
        neg = torch.finfo(logits.dtype).min
        mx = torch.where(mask, logits, torch.full_like(logits, neg)).detach().amax(dim, keepdim=True)
        mx = torch.where(mx > neg, mx, torch.zeros_like(mx))
        e = torch.exp(torch.where(mask, logits - mx, torch.full_like(logits, neg))) * mask
        return e / e.sum(dim, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)

    def _dense_block(self, emb_mlp, attn_mlp, q, valid, inv, node_level):
        """q (N, R, c) inputs, valid (N, R), inv (N, R, n). Returns graph pool
        (N, v), count (N,), node pool (N, n, v), node count (N, n)."""
        emb = emb_mlp(q)                                    # (N, R, v)
        logit = attn_mlp(q).squeeze(-1)                     # (N, R)
        w = self._masked_softmax(logit, valid, dim=1)
        gpool = torch.einsum('nr,nrv->nv', w, emb)
        cnt = valid.sum(1).to(q.dtype)
        if not node_level:
            return gpool, cnt, None, None
        nm = inv & valid[:, :, None]                        # (N, R, n)
        wn = self._masked_softmax(logit[:, :, None].expand_as(nm), nm, dim=1)
        npool = torch.einsum('nru,nrv->nuv', wn, emb)       # (N, n, v)
        ncnt = nm.sum(1).to(q.dtype)                        # (N, n)
        return gpool, cnt, npool, ncnt

    def _vectorize_dense(self, blocks, B, P1, M, dt, device):
        node_level = M is not None
        g_dims, n_dims = [], []
        for d in range(P1):
            g_parts, n_parts = [], []
            fb = blocks.get(('fin', d))
            if fb is None:            # no candidate axis (e.g. no triangles)
                N = B
                gp = torch.zeros(N, self.vec_dim, device=device, dtype=dt)
                cnt = torch.zeros(N, device=device, dtype=dt)
                mean = std = torch.zeros(N, 2, device=device, dtype=dt)
                npool = torch.zeros(N, M, self.vec_dim, device=device, dtype=dt) if node_level else None
                ncnt = torch.zeros(N, M, device=device, dtype=dt) if node_level else None
            else:
                valid = fb['valid']
                bv, dv = fb['b'], fb['d']
                bp_raw = torch.stack([bv, torch.clamp(dv - bv, min=0)], dim=-1)   # (N,R,2)
                bp_raw = torch.where(valid[..., None], bp_raw, torch.zeros_like(bp_raw))
                vf = valid[..., None].to(dt)
                c = vf.sum(1).clamp_min(1)                                       # (N,1)
                mean = (bp_raw * vf).sum(1) / c                                  # (N,2)
                cen = (bp_raw - mean[:, None, :]) * vf
                var = (cen * cen).sum(1) / c
                posv = var > 0
                std = torch.where(posv, torch.sqrt(torch.where(posv, var, torch.ones_like(var))),
                                  torch.zeros_like(var))
                q = cen / (std[:, None, :] + STD_EPS)
                gp, cnt, npool, ncnt = self._dense_block(
                    self.embeds[d], self.attns[d], q, valid, fb['inv'], node_level)
            g_parts.append(gp)
            if self.multiplicity:
                g_parts.append(torch.log1p(cnt)[:, None])
            if self.scale_stats:
                g_parts += [mean, std]
            if node_level:
                n_parts.append(npool)
                if self.multiplicity:
                    n_parts.append(torch.log1p(ncnt)[..., None])
            if self.essential:
                eb = blocks.get(('ess', d))
                if eb is None:
                    N = B
                    egp = torch.zeros(N, self.vec_dim, device=device, dtype=dt)
                    ecnt = torch.zeros(N, device=device, dtype=dt)
                    enp = torch.zeros(N, M, self.vec_dim, device=device, dtype=dt) if node_level else None
                    encnt = torch.zeros(N, M, device=device, dtype=dt) if node_level else None
                else:
                    q = torch.where(eb['valid'], eb['b'], torch.zeros_like(eb['b']))[..., None]
                    egp, ecnt, enp, encnt = self._dense_block(
                        self.ess_embeds[d], self.ess_attns[d], q, eb['valid'], eb['inv'], node_level)
                g_parts += [egp, torch.log1p(ecnt)[:, None]]
                if node_level:
                    n_parts += [enp, torch.log1p(encnt)[..., None]]
            g_dims.append(torch.cat(g_parts, dim=1))                       # (N, per_dim_graph)
            if node_level:
                n_dims.append(torch.cat(n_parts, dim=2).permute(0, 2, 1))  # (N, per_dim_node, n)
        graph_out = torch.cat(g_dims, dim=1)
        node_out = torch.cat(n_dims, dim=1) if node_level else None
        return graph_out, node_out

    def _gudhi_pairs(self, adj, filt_batch, offsets, P1, M, device):
        """Host path: gudhi + integer bookkeeping (one device->host copy of
        the filtration, one host->device copy of all indices)."""
        node_level = M is not None
        host = adj.detach().cpu().numpy().astype(np.float64)
        host_list = host.tolist()
        f_seg, f_b, f_d = [], [], []
        e_seg, e_b = [], []
        fn_r, fn_c, en_r, en_c = [], [], [], []
        for bi, ((st, _), o) in enumerate(zip(filt_batch, offsets)):
            if o is None:
                continue
            vals = host[o:o + st.S]
            tree = gudhi.SimplexTree()
            for idx, verts in st.dim_batches:
                tree.insert_batch(verts, vals[idx])
            tree.make_filtration_non_decreasing()
            # min_persistence: drop pairs with persistence <= PERS_EPS (float-
            # robust zero-persistence test; see layers/torch_ph.PERS_EPS)
            tree.persistence(persistence_dim_max=self.essential,
                             min_persistence=PERS_EPS)
            pairs = tree.persistence_pairs()
            if node_level:
                vlist = host_list[o:o + st.S]
                val_to_nodes = defaultdict(set)
                for idx, sigma in enumerate(st.simplices_list):
                    val_to_nodes[(len(sigma), round(vlist[idx], 6))].update(sigma)
            s2i = st.simplex_to_idx
            for bs, ds in pairs:
                dim = len(bs) - 1
                if dim > self.max_ph_dim:
                    continue
                bk = tuple(sorted(bs))
                if bk not in s2i:
                    continue
                b_idx = s2i[bk]
                seg = bi * P1 + dim
                if len(ds) == 0:
                    if not self.essential:
                        continue
                    if node_level:
                        for n in val_to_nodes[(len(bs), round(vlist[b_idx], 6))]:
                            if n < M:
                                en_r.append(len(e_b)); en_c.append(n)
                    e_seg.append(seg); e_b.append(o + b_idx)
                    continue
                dk = tuple(sorted(ds))
                if dk not in s2i:
                    continue
                d_idx = s2i[dk]
                if node_level:
                    inv = (val_to_nodes[(len(bs), round(vlist[b_idx], 6))]
                           | val_to_nodes[(len(ds), round(vlist[d_idx], 6))])
                    for n in inv:
                        if n < M:
                            fn_r.append(len(f_b)); fn_c.append(n)
                f_seg.append(seg); f_b.append(o + b_idx); f_d.append(o + d_idx)
        lists = [f_seg, f_b, f_d, e_seg, e_b, fn_r, fn_c, en_r, en_c]
        lens = [len(a) for a in lists]
        packed = torch.as_tensor(
            np.fromiter((x for a in lists for x in a), dtype=np.int64,
                        count=sum(lens)), device=device)
        out = list(torch.split(packed, lens))
        if not node_level:
            out[5:] = [None] * 4
        return out

    def _vectorize(self, adj, coo, B, P1, M):
        """Shared device-side vectorization. Rows whose segment is the trash
        segment (index B*P1) and node entries in the trash column (index M)
        are computed but discarded, which lets the torch backend avoid any
        data-dependent (syncing) compaction."""
        f_seg, f_b, f_d, e_seg, e_b, fn_r, fn_c, en_r, en_c = coo
        node_level = M is not None
        device, dt = adj.device, adj.dtype
        n_seg = B * P1
        n_all = n_seg + 1                    # + trash segment
        Ms = (M + 1) if node_level else None  # + trash node column
        g_blocks, n_blocks = [], []

        # finite pairs -> [pooled(v), mult?, stats(4)?]
        seg_t = f_seg
        bv = adj[f_b]; dv = adj[f_d]
        bp_raw = torch.stack([bv, torch.clamp(dv - bv, min=0)], dim=1)
        ones = torch.ones(bp_raw.shape[0], device=device, dtype=dt)
        cnt = torch.zeros(n_all, device=device, dtype=dt).index_add(0, seg_t, ones)
        cden = cnt.clamp_min(1).unsqueeze(-1)
        mean = torch.zeros(n_all, 2, device=device, dtype=dt).index_add(0, seg_t, bp_raw) / cden
        cen = bp_raw - mean[seg_t]
        var = torch.zeros(n_all, 2, device=device, dtype=dt).index_add(0, seg_t, cen * cen) / cden
        # sqrt has an infinite derivative at 0 (empty / single-pair / constant
        # segments); evaluate it only where var > 0 so no NaN enters backward.
        posv = var > 0
        std = torch.where(posv, torch.sqrt(torch.where(posv, var, torch.ones_like(var))),
                          torch.zeros_like(var))
        q = cen / (std[seg_t] + STD_EPS)
        gpool, npool, ncnt = self._pool(
            q, seg_t, n_all, self.embeds, self.attns, seg_t % P1, P1,
            fn_r, fn_c, Ms)
        g_blocks.append(gpool)
        if self.multiplicity:
            g_blocks.append(torch.log1p(cnt).unsqueeze(-1))
        if self.scale_stats:
            g_blocks += [mean, std]
        if node_level:
            n_blocks.append(npool)
            if self.multiplicity:
                n_blocks.append(torch.log1p(ncnt).unsqueeze(-1))

        # essential classes -> [pooled(v), mult]
        if self.essential:
            eq = adj[e_b].unsqueeze(-1)
            ecnt = torch.zeros(n_all, device=device, dtype=dt).index_add(
                0, e_seg, torch.ones(e_b.shape[0], device=device, dtype=dt))
            egp, enp, encnt = self._pool(
                eq, e_seg, n_all, self.ess_embeds, self.ess_attns, e_seg % P1, P1,
                en_r, en_c, Ms)
            g_blocks += [egp, torch.log1p(ecnt).unsqueeze(-1)]
            if node_level:
                n_blocks += [enp, torch.log1p(encnt).unsqueeze(-1)]

        graph_out = torch.cat(g_blocks, dim=1)[:n_seg].reshape(B, P1 * self.per_dim_graph)
        node_out = None
        if node_level:
            node_out = (torch.cat(n_blocks, dim=1)            # (n_all*Ms, C)
                        .reshape(n_all, Ms, self.per_dim_node)[:n_seg, :M]
                        .reshape(B, P1, M, self.per_dim_node)
                        .permute(0, 1, 3, 2)
                        .reshape(B, P1 * self.per_dim_node, M))
        return graph_out, node_out


from layers.torch_ph import PERS_EPS  # noqa: E402
# epsilon of the per-graph standardization of pair inputs. Matches PERS_EPS:
# with a smaller value, near-degenerate diagrams (std ~ 1e-10) amplify float
# summation-order noise by up to 1/eps, which breaks invariance at ~1e-9.
STD_EPS = 1e-6


class InvertibleLayerNorm(nn.Module):
    """LayerNorm that also returns the statistics it divides out.

    Output is [LN(x) || mu(x) || log s(x)] with s = sqrt(Var(x) + eps), so
    x = mu * 1 + s * (LN(x) - beta) / gamma is recoverable exactly (gamma != 0).
    Plain LayerNorm identifies x with a*x + b*1 (a > 0); appending (mu, log s)
    makes the map injective, which is what lets the implemented fusion contain
    the idealized fusion of Section 3 (fusion receives an invertible function of
    the raw topological vector). Normalization is over the LAST dim.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        log_s = 0.5 * torch.log(var + self.eps)
        return torch.cat([self.ln(x), mu, log_s], dim=-1)


class TopologyLayer(nn.Module):
    """Full topology branch for one layer: learned filtration -> differentiable
    PH -> broadcast (graph + node level) -> gated residual fusion."""
    def __init__(self, eqv_features, hidden_dim=64, max_ph_dim=1, num_stats=4,
                 gate_bias=2.0, node_level=True, multiplicity=False,
                 scale_stats=False, gate_mode='conv', gate_init=0.0,
                 norm_stats=False, essential=False, filt_squash=False,
                 num_filtrations=1, ph_backend='auto'):
        super().__init__()
        self.node_level = node_level
        self.norm_stats = norm_stats
        self.gate_mode = gate_mode
        self.filtration = LearnedFiltration(eqv_features, hidden_dim,
                                            squash=filt_squash)
        # num_filtrations > 1: independent learned filtration heads (as in
        # TOGL), each an equivariant filtration generator; their diagrams are
        # vectorized by the shared pooling and concatenated. All heads go
        # through PH as ONE batch (B * m items). Zero weights on the extra
        # heads' channels recover the single-filtration model exactly.
        self.num_filtrations = num_filtrations
        # 'auto': GPU-resident PH (layers/torch_ph.py) on accelerators when the
        # complex is <= 2-dim, homology <= 1 and triangles are few; else gudhi.
        # 'torch' forces the torch path where supported; 'gudhi' disables it.
        self.ph_backend = ph_backend
        self.extra_filtrations = nn.ModuleList(
            LearnedFiltration(eqv_features, hidden_dim, squash=filt_squash)
            for _ in range(num_filtrations - 1))
        self.ph = DifferentiablePH(max_ph_dim, num_stats,
                                   multiplicity=multiplicity,
                                   scale_stats=scale_stats,
                                   essential=essential)

        topo_dim = self.ph.out_features            # per-head graph-level width
        node_dim = self.ph.node_out_features       # per-head node-level width
        # norm_stats: invertible normalization (LN output + its mean/log-std),
        # adds 2 channels per normalized vector
        norm_cls = InvertibleLayerNorm if norm_stats else nn.LayerNorm
        self.topo_norm = norm_cls(topo_dim)
        if node_level:
            self.node_norm = norm_cls(node_dim)
        if norm_stats:
            topo_dim, node_dim = topo_dim + 2, node_dim + 2
        # normalization is applied PER HEAD (shared affine), so the extra heads
        # never enter the first head's statistics: zeroing their fusion weights
        # recovers the single-filtration layer exactly.
        topo_dim, node_dim = topo_dim * num_filtrations, node_dim * num_filtrations

        # fuse x || T_graph  (+ t_row || t_col when node-level features are on)
        fuse_in = eqv_features + topo_dim + (2 * node_dim if node_level else 0)
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in, eqv_features, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(eqv_features, eqv_features, kernel_size=1, bias=True),
        )
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        nn.init.normal_(self.fusion[2].weight, std=0.01)
        nn.init.zeros_(self.fusion[2].bias)

        if gate_mode == 'scalar':
            # ReZero-style gate: ONE learnable scalar per layer, initialized to
            # `gate_init`. With gate_init = 0 the layer computes exactly
            # x_eqv, so the topology-augmented network's function at
            # initialization is IDENTICAL to the pure equivariant baseline's;
            # it can only move away from the baseline if the gradient on the
            # scalar says topology helps. dL/dalpha = <dL/dout, delta> is
            # nonzero, so the gate can still switch on.
            #
            # This does not touch the hypothesis class -- Theorem 3 and
            # Corollary 6 are EXISTENCE statements over parameters, and
            # alpha != 0 is reachable -- so all expressivity claims are
            # preserved. It also removes the fuse_in x eqv gate convolution,
            # which is a large share of the topology branch's parameters.
            self.gate_conv = None
            self.gate_scalar = nn.Parameter(torch.full((1,), float(gate_init)))
        else:
            self.gate_scalar = None
            self.gate_conv = nn.Conv2d(fuse_in, eqv_features, kernel_size=1,
                                       bias=True)
            nn.init.normal_(self.gate_conv.weight, std=0.01)
            # gate_bias controls how much topology contributes initially:
            # +2 -> sigmoid 0.88 (topo strongly on); <=0 -> starts near/at
            # baseline, so the model adds topology only where it helps.
            nn.init.constant_(self.gate_conv.bias, gate_bias)
        # Warm-start: while `frozen` is True (set from the training loop for
        # the first N epochs), the topology contribution is IDENTICALLY zero
        # regardless of gate value. This lets the equivariant backbone converge
        # first, then unfreezes so the gate can adapt topology on top of an
        # already-trained baseline.
        self.frozen = False
        # normalized graph-level topological vector of the last forward pass,
        # read by the model's optional topological readout head (invariant)
        self.graph_feat_dim = topo_dim
        self.last_graph_vec = None

    def forward(self, x_eqv, structs):
        B, d, M, _ = x_eqv.shape
        if self.frozen:                              # warm-start: skip topology
            self.last_graph_vec = None
            return x_eqv
        plan = get_plan(structs, x_eqv.device, reps=self.num_filtrations,
                        max_ph_dim=self.ph.max_ph_dim, backend=self.ph_backend, M_pad=M)
        filt_batch = self.filtration(x_eqv, structs, plan)
        for head in self.extra_filtrations:          # head-major: (m * B) items
            filt_batch = filt_batch + head(x_eqv, structs, plan)
        graph_vec, node_vec = self.ph(
            filt_batch, device=x_eqv.device,
            num_nodes=M if self.node_level else None, plan=plan)
        m = self.num_filtrations
        # (m*B, F) head-major -> per-head normalization -> (B, m*F')
        graph_vec = self.topo_norm(graph_vec).reshape(m, B, -1)
        graph_vec = graph_vec.permute(1, 0, 2).reshape(B, -1)
        self.last_graph_vec = graph_vec
        if self.node_level:
            nv = self.node_norm(node_vec.permute(0, 2, 1))          # (m*B, M, Fn')
            node_vec = (nv.reshape(m, B, M, -1).permute(1, 0, 3, 2)
                        .reshape(B, -1, M))

        # Factored fusion. The fused input [x_uv | g | t_u | t_v] is constant
        # in (u, v) for g, in v for t_u and in u for t_v, so a 1x1 conv on it
        # splits EXACTLY into a conv on x plus broadcast per-graph / per-node
        # products: W[x;g;t_u;t_v] = W_x x_uv + W_g g + W_r t_u + W_c t_v.
        # Same function and parameters as materializing the (B, C_in, M, M)
        # concatenation, at C_x/C_in of the dense cost. The gate conv reads the
        # same input, so both first layers share one factored product.
        convs = [self.fusion[0]] + ([self.gate_conv] if self.gate_conv is not None else [])
        W = torch.cat([c.weight[:, :, 0, 0] for c in convs], dim=0)   # (O, C_in)
        bias = torch.cat([c.bias for c in convs], dim=0)
        cx, cg = x_eqv.shape[1], graph_vec.shape[1]
        pre = torch.nn.functional.conv2d(
            x_eqv, W[:, :cx, None, None], bias)                        # (B, O, M, M)
        pre = pre + (graph_vec @ W[:, cx:cx + cg].t())[:, :, None, None]
        if self.node_level:
            cn = node_vec.shape[1]
            Wr, Wc = W[:, cx + cg:cx + cg + cn], W[:, cx + cg + cn:]
            pre = (pre + torch.einsum('bcm,oc->bom', node_vec, Wr)[:, :, :, None]
                   + torch.einsum('bcm,oc->bom', node_vec, Wc)[:, :, None, :])
        n_f = self.fusion[0].out_channels
        delta = self.fusion[2](self.fusion[1](pre[:, :n_f]))
        if self.gate_conv is not None:
            gate = torch.sigmoid(pre[:, n_f:])
        else:
            gate = self.gate_scalar          # raw scalar: 0 => exactly baseline
        return x_eqv + gate * delta
