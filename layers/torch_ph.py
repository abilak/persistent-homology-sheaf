"""
GPU-resident persistent homology for clique complexes up to dimension 2
(vertices, edges, triangles), homology in dimensions 0 and 1, including
essential classes. Pure tensor ops: no host synchronization, so the topology
branch never stalls the GPU.

Everything is computed in RANK space (a strict total order on each dimension:
filtration value, ties broken by index), which makes every birth/death simplex
an exact index; filtration VALUES are then gathered from the differentiable
filtration tensor, so gradients flow to the same simplices as with gudhi.
Diagrams (multisets of (birth, death) values) are independent of the tie-break
and agree with gudhi exactly; only which of several tied simplices receives
the gradient can differ, as it can between any two valid implementations.

  H0: with vertices ordered by rank, the class born at v dies at the first edge
      that connects v to an OLDER vertex, i.e. at min_{u older} bottleneck(v,u),
      where bottleneck(v,u) is the minimax edge rank over v-u paths (elder
      rule). Bottleneck ranks come from ~log2(n) min-max matrix squarings.
      The dying edges form the minimum spanning forest; all other edges are
      positive (each creates a 1-cycle).
  H1: finite pairs from the standard column reduction of the triangle
      boundary matrix over Z/p (default p = 11, gudhi's default field; the
      oriented boundary d[a,b,c] = [b,c] - [a,c] + [a,b]), columns in
      triangle-rank order, rows in edge-rank order, batched over graphs.
      Positive edges not killed are essential. (Over Z/2 non-orientable
      subcomplexes carry extra homology, so the field must match gudhi's.)
Zero-persistence pairs are dropped, as gudhi does.

The caller (BatchPlan) decides on the host -- from static graph sizes, with no
sync -- whether a batch uses this path or the gudhi path.
"""
import numpy as np
import torch


def _minmax_closure(R, rounds, max_elems=1 << 25):
    """R (N, n, n) edge ranks (inf = no edge). Repeated min-max squaring:
    R <- min(R, min_w max(R[:, u, w], R[:, w, v])). After k rounds it covers
    paths of up to 2^k edges."""
    N, n, _ = R.shape
    chunk = max(1, max_elems // max(1, n * n * n))
    for _ in range(rounds):
        parts = []
        for s in range(0, N, chunk):
            A = R[s:s + chunk]
            parts.append(torch.maximum(A[:, :, :, None], A[:, None, :, :]).amin(dim=2))
        R = torch.minimum(R, torch.cat(parts, 0))
    return R


def _stable_rank(values, valid):
    """rank of each entry within its row under (value, index); invalid -> last.
    Returns rank (N, K) long and order (N, K) long (order[r] = entry of rank r)."""
    v = torch.where(valid, values.detach(), torch.full_like(values, float('inf')))
    order = torch.argsort(v, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device)
                  .expand_as(order).contiguous())
    return rank, order


_CONST = {}


def _const(name, device, fn):
    k = (name, str(device))
    if k not in _CONST:
        _CONST[k] = fn().to(device)     # built once per device: no per-call H2D copy
    return _CONST[k]


def _key(x):
    """tie-merging key: value rounded to 1e-6 (a function of the value only,
    hence isomorphism-invariant; matches the gudhi path's round(x, 6) except
    at exact rounding boundaries)."""
    x = x.detach()
    return torch.round((x if x.device.type == 'mps' else x.double()) * 1e6)


def torch_persistence(adj, plan, P1, essential, node_level, M, p=11):
    """Returns dense pair blocks {(kind, dim): {b, d, valid, inv}} with a
    fixed candidate axis per item; consumed by DifferentiablePH._vectorize_dense."""
    tp = plan.torch_ph
    dev = adj.device
    N, n, m, T = tp['N'], tp['n'], tp['m'], tp['T']
    trash = N * P1
    INF = float('inf')

    vg, vmask = tp['vg'], tp['vmask']                    # (N, n)
    eg, emask, ea, eb = tp['eg'], tp['emask'], tp['ea'], tp['eb']   # (N, m)
    item = tp['item']                                    # (N,) item index
    vv = adj[vg]; ev = adj[eg]

    # ---- ranks -------------------------------------------------------------
    rank_v, _ = _stable_rank(vv, vmask)
    rank_e, order_e = _stable_rank(ev, emask)

    # ---- H0 via bottleneck ranks --------------------------------------------
    R = torch.full((N, n, n), INF, device=dev)
    if m:
        rv = torch.where(emask, rank_e.to(R.dtype), torch.full_like(ev, INF, dtype=R.dtype))
        bi = torch.arange(N, device=dev)[:, None].expand(N, m)
        R.index_put_((bi, ea, eb), rv); R.index_put_((bi, eb, ea), rv)
        rounds = max(1, int(np.ceil(np.log2(max(2, n - 1)))))
        R = _minmax_closure(R, rounds)
    older = (rank_v[:, None, :] < rank_v[:, :, None]) & vmask[:, None, :]   # [v, u]
    cand = torch.where(older, R, torch.full_like(R, INF))
    dr = cand.amin(dim=2)                                                     # (N, n)
    fin0 = vmask & torch.isfinite(dr)
    ess0 = vmask & ~torch.isfinite(dr)
    de = order_e.gather(1, torch.where(fin0, dr, torch.zeros_like(dr)).long()
                        .clamp(max=max(m - 1, 0))) if m else torch.zeros_like(vg)
    msf = torch.zeros(N, max(m, 1), dtype=torch.bool, device=dev)
    if m:
        msf = torch.zeros(N, m, dtype=torch.long, device=dev).scatter_add_(
            1, torch.where(fin0, de, torch.zeros_like(de)), fin0.long()) > 0
        msf = msf & emask
    d0 = ev.gather(1, de) if m else vv
    fin0 = fin0 & (d0.detach() != vv.detach())            # drop zero persistence

    # ---- H1 via batched Z/2 reduction of the triangle boundary -------------
    pos = emask & ~msf[:, :m] if m else emask
    killed = torch.zeros_like(pos)
    h1_rows = None
    if P1 > 1 and T and m:
        tg, tmask, te = tp['tg'], tp['tmask'], tp['te']              # (N,T), (N,T,3)
        tv = adj[tg]
        rank_t, order_t = _stable_rank(tv, tmask)
        te_sorted = te.gather(1, order_t[:, :, None].expand(N, T, 3))   # edges of j-th tri
        tmask_s = tmask.gather(1, order_t)
        erank = rank_e.gather(1, te_sorted.reshape(N, -1)).reshape(N, T, 3)
        # oriented boundary over Z/p: edges (ab, ac, bc) get (+1, -1, +1)
        sign = _const(f'sign{p}', dev, lambda: torch.tensor([1, p - 1, 1], dtype=torch.long))
        C = torch.zeros(N, T, m, dtype=torch.long, device=dev)
        C.scatter_(2, erank, (sign[None, None, :] * tmask_s[:, :, None].long()).expand(N, T, 3).contiguous())
        inv = _const(f'inv{p}', dev, lambda: torch.tensor([0] + [pow(a, p - 2, p) for a in range(1, p)]))
        pivot = torch.full((N, m), -1, dtype=torch.long, device=dev)
        ar = torch.arange(1, m + 1, device=dev)
        bidx = torch.arange(N, device=dev)
        lows = torch.full((N, T), -1, dtype=torch.long, device=dev)
        lowf = lambda c: ((c != 0) * ar).amax(dim=1) - 1
        for j in range(T):
            col = C[:, j]
            for _ in range(j):          # each addition lowers `low`: <= j steps
                low = lowf(col)
                lc = low.clamp(min=0)
                piv = pivot.gather(1, lc[:, None])[:, 0]
                hit = (low >= 0) & (piv >= 0)
                pc = C[bidx, piv.clamp(min=0)]                       # pivot column
                a = col.gather(1, lc[:, None])[:, 0]
                b = pc.gather(1, lc[:, None])[:, 0]
                f = (a * inv[b]) % p                                 # eliminate row `low`
                col = torch.where(hit[:, None], (col - f[:, None] * pc) % p, col)
            low = lowf(col)
            C[:, j] = col
            lows[:, j] = low
            has = low >= 0
            lc = low.clamp(min=0)[:, None]
            pivot.scatter_(1, lc, torch.where(has[:, None], torch.full_like(lc, j),
                                              pivot.gather(1, lc)))
        f1 = lows >= 0
        e_of = order_e.gather(1, lows.clamp(min=0))                      # edge local idx
        t_of = order_t                                                    # tri local idx
        b1 = ev.gather(1, e_of); d1 = tv.gather(1, t_of)
        f1 = f1 & (b1.detach() != d1.detach())
        killed = torch.zeros(N, m, dtype=torch.long, device=dev).scatter_add_(
            1, e_of, (lows >= 0).long()) > 0
        killed = killed & emask
        h1_rows = (f1, e_of, t_of, tv)
    ess1 = pos & ~killed if m else pos

    # ---- dense blocks -------------------------------------------------------
    # Each pair type lives on a fixed candidate axis per item (vertices for H0,
    # triangles for finite H1, edges for essential H1), with a validity mask.
    # Consumers pool with masked softmax + batched matmul: no scatter, no
    # compaction, no host sync.
    kv, ke = _key(vv), _key(ev)
    incE = tp['incE']
    def verts_of_vkey(k):                  # k (N, R) -> (N, R, n) bool
        return (kv[:, None, :] == k[:, :, None]) & vmask[:, None, :]
    def verts_of_ekey(k):
        match = ((ke[:, None, :] == k[:, :, None]) & emask[:, None, :]).to(incE.dtype)
        return torch.bmm(match, incE) > 0

    blocks = {}   # (kind, dim) -> dict(b=birth value, d=death value or None, valid, inv)
    inv0 = None
    if node_level:
        inv0 = verts_of_vkey(kv)
        if m:
            inv0 = inv0 | verts_of_ekey(ke.gather(1, de))
    blocks[('fin', 0)] = dict(b=vv, d=(ev.gather(1, de) if m else vv), valid=fin0, inv=inv0)
    if essential:
        blocks[('ess', 0)] = dict(b=vv, valid=ess0,
                                  inv=verts_of_vkey(kv) if node_level else None)
    if P1 > 1:
        if h1_rows is not None:
            f1, e_of, t_of, tv = h1_rows
            inv1 = None
            if node_level:
                kt = _key(tv)
                tm = ((kt[:, None, :] == kt.gather(1, t_of)[:, :, None])
                      & tp['tmask'][:, None, :]).to(incE.dtype)
                inv1 = verts_of_ekey(ke.gather(1, e_of)) | (torch.bmm(tm, tp['incT']) > 0)
            blocks[('fin', 1)] = dict(b=ev.gather(1, e_of), d=tv.gather(1, t_of),
                                      valid=f1, inv=inv1)
        if essential and m:
            blocks[('ess', 1)] = dict(b=ev, valid=ess1,
                                      inv=verts_of_ekey(ke) if node_level else None)
    return blocks
