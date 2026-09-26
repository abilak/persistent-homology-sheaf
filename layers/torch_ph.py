"""
GPU-resident persistent homology of clique complexes (vertices, edges,
triangles, tetrahedra), homology in dimensions 0, 1, 2, including essential
classes. Pure tensor ops, no gudhi, no host computation.

Everything is computed in RANK space (a strict total order within each
dimension: filtration value, ties broken by index), so every birth / death
simplex is an exact index; values are then gathered from the differentiable
filtration tensor, so gradients reach the same simplices as with gudhi.
Diagrams (multisets of (birth, death) values) do not depend on the tie-break
and agree with gudhi exactly (test_torch_ph.py).

  H0  Elder rule via bottleneck distances: the class born at v dies at
      min_{u older than v} bottleneck(v, u), the minimax edge rank over v-u
      paths, obtained from ~log2(n) batched min-max squarings. The dying edges
      form the minimum spanning forest; all other edges are positive.
  H1  Persistent COHOMOLOGY with clearing (de Silva, Morozov, Vejdemo-Johansson
      2011; the scheme Ripser uses): reduce the anti-transposed coboundary
      delta_1 (columns = edges in decreasing rank, rows = triangles in
      decreasing rank), with the columns of negative edges (the spanning
      forest, known from H0) cleared. At most the cycle rank m - n + c columns
      are nonzero, far fewer than the number of triangles on dense graphs.
      Pairs and essential classes equal those of persistent homology.
  H2  Same, one dimension up: columns = triangles not paired in H1, rows =
      tetrahedra.
Reductions are over Z/p (default p = 11, gudhi's default field; over Z/2
non-orientable subcomplexes have different homology) and PARALLEL across
columns (as in OpenPH): every column whose low collides with that of an
earlier column eliminates it using the leftmost such column, simultaneously.
Only earlier columns are ever added to later ones, so the result is a valid
reduction and its pairing is the persistence pairing. If K columns are
nonzero, the reduction has finished after at most K(K-1)/2 + 1 steps (the
first j columns are final after sum_{i<j} i steps), so when the host-known
bound is small a fixed number of steps is run with no synchronization.
Otherwise the SAME steps are run sparsely: after one dense pass for the lows,
each step touches only the colliding columns (typically a few dozen out of
thousands), one sync per step; the pivots and updates are identical, so the
result is bit-for-bit that of the dense steps.
Zero-persistence pairs are dropped, as gudhi does.
"""
import numpy as np
import torch

# Pairs with persistence <= PERS_EPS count as zero-persistence and are dropped
# (both backends). An exact `death == birth` test is not robust: the max-
# correction makes birth == death exactly in exact arithmetic for many pairs
# (ubiquitous with categorical node labels), but float summation order differs
# between isomorphic inputs, so persistence comes out as 0 or 1e-16 depending
# on the vertex labelling, and a pair would be dropped for one labelling and
# counted for the other. With a tolerance, the counted diagram is an
# isomorphism invariant in floating point as well (away from persistence
# == PERS_EPS itself).
PERS_EPS = 1e-6
FIXED_STEPS_MAX = 64      # use a sync-free fixed schedule up to this many steps
CHECK_EVERY = 4           # otherwise check convergence every CHECK_EVERY steps

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


def _minmax_closure(R, rounds, max_elems=1 << 25):
    """R (N, n, n) edge ranks (inf = no edge). Repeated min-max squaring:
    R <- min(R, min_w max(R[:, u, w], R[:, w, v])); k rounds cover paths of up
    to 2^k edges."""
    N, n, _ = R.shape
    chunk = max(1, max_elems // max(1, n * n * n))
    for _ in range(rounds):
        parts = []
        for s in range(0, N, chunk):
            A = R[s:s + chunk]
            parts.append(torch.maximum(A[:, :, :, None], A[:, None, :, :]).amin(dim=2))
        R = torch.minimum(R, torch.cat(parts, 0))
    return R


SQUARING_MAX_N = 128      # dense min-max squaring (sync-free) up to this many nodes


def _bottleneck_relax(ea, eb, rank_e, emask, N, n, check_every=8):
    """All-pairs bottleneck edge ranks by relaxation over the edge list (for
    large sparse graphs, where squaring's O(n^3 log n) is prohibitive):
    R[v, b] <- min(R[v, b], max(R[v, a], rank(a, b))), both edge directions.
    O(n m) per step; converges after (longest minimax path in edges) steps;
    convergence is checked every `check_every` steps (one scalar sync each)."""
    dev = ea.device
    INF = float('inf')
    R = torch.full((N, n, n), INF, device=dev)
    R.diagonal(dim1=1, dim2=2).fill_(-1.0)          # empty path: below every rank
    r = torch.where(emask, rank_e.to(R.dtype), torch.full_like(rank_e, INF, dtype=R.dtype))
    m = ea.shape[1]
    ia = ea[:, None, :].expand(N, n, m); ib = eb[:, None, :].expand(N, n, m)
    while True:
        prev = R
        for _ in range(check_every):
            cb = torch.maximum(R.gather(2, ia), r[:, None, :])     # reach b via a
            ca = torch.maximum(R.gather(2, ib), r[:, None, :])     # reach a via b
            R = R.scatter_reduce(2, ib, cb, reduce='amin')
            R = R.scatter_reduce(2, ia, ca, reduce='amin')
        if bool((R == prev).all()):
            break
    R = R.clone()
    R.diagonal(dim1=1, dim2=2).fill_(INF)
    return R


def _stable_rank(values, valid):
    """rank (N, K) and order (N, K) (order[r] = entry of rank r) under
    (value, index); invalid entries rank last."""
    v = torch.where(valid, values.detach(), torch.full_like(values, float('inf')))
    order = torch.argsort(v, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device)
                  .expand_as(order).contiguous())
    return rank, order


def _reduce_parallel(Mc, p, k_bound):
    """Parallel column reduction over Z/p of Mc (N, K, R) (values in [0, p)).
    Returns low (N, K): index of the lowest nonzero row, -1 for zero columns.
    k_bound: host-known upper bound on the number of nonzero columns."""
    N, K, R = Mc.shape
    dev = Mc.device
    if K == 0 or R == 0:
        return torch.full((N, K), -1, dtype=torch.long, device=dev), Mc
    inv = _const(f'inv{p}', dev, lambda: torch.tensor(
        [0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=torch.long))
    rows = torch.arange(1, R + 1, device=dev)
    colidx = torch.arange(K, device=dev)
    ridx = torch.arange(R, device=dev)

    def lows(M):
        return ((M != 0) * rows).amax(dim=2) - 1                        # (N, K)

    def step(M):
        low = lows(M)
        lc = low.clamp(min=0)
        # leftmost column holding each low (dense: no scatter_reduce)
        hold = (low[:, :, None] == ridx[None, None, :])                  # (N, K, R)
        first = torch.where(hold, colidx[None, :, None],
                            torch.full_like(colidx, K)[None, :, None]).amin(dim=1)   # (N, R)
        piv = first.gather(1, lc)                                          # (N, K)
        coll = (low >= 0) & (piv < colidx[None, :])
        pc = M.gather(1, piv.clamp(max=K - 1)[:, :, None].expand(N, K, R))
        a = M.gather(2, lc[:, :, None])[:, :, 0].long()
        b = pc.gather(2, lc[:, :, None])[:, :, 0].long()
        f = ((a * inv[b]) % p).to(M.dtype)
        M = torch.where(coll[:, :, None], torch.remainder(M - f[:, :, None] * pc, p), M)
        return M, coll

    steps = k_bound * (k_bound - 1) // 2 + 1
    if steps <= FIXED_STEPS_MAX:
        for _ in range(steps):                      # sync-free, provably enough
            Mc, _ = step(Mc)
    elif dev.type != 'mps':                         # (no scatter_reduce on MPS)
        return _reduce_sparse(Mc, p, inv)
    else:
        while True:                                 # one scalar sync per CHECK_EVERY
            for _ in range(CHECK_EVERY):
                Mc, coll = step(Mc)
            if not bool(coll.any()):
                break
    return lows(Mc), Mc


def _last_nonzero(M):
    """index of the lowest nonzero entry along the last dim; -1 if none."""
    rows = torch.arange(1, M.shape[-1] + 1, device=M.device, dtype=torch.int32)
    return ((M != 0) * rows).amax(dim=-1).long() - 1


def _reduce_sparse(Mc, p, inv):
    """The parallel reduction of _reduce_parallel with sparse steps: every
    colliding column c (low shared with an earlier column) is reduced by the
    leftmost column holding that low, exactly as in the dense step, but only
    the colliding columns are gathered, updated and scattered back, and only
    their lows recomputed: O(N K + C R) per step instead of several O(N K R)
    passes. Pivots never collide in the step they are used (they are the
    leftmost holders of their low), so the updates are conflict-free and the
    result equals the dense reduction's."""
    N, K, R = Mc.shape
    dev = Mc.device
    M = Mc.reshape(N * K, R)
    low = _last_nonzero(M).reshape(N, K)
    colidx = torch.arange(K, device=dev)[None, :].expand(N, K)
    base = (torch.arange(N, device=dev) * K)[:, None]
    while True:
        # leftmost column holding each low (slot 0 collects the zero columns)
        first = torch.full((N, R + 1), K, dtype=torch.long, device=dev)
        first.scatter_reduce_(1, low + 1, colidx, reduce='amin')
        piv = first.gather(1, low + 1)
        coll = (low >= 0) & (piv < colidx)
        idx = coll.reshape(-1).nonzero().squeeze(1)          # one sync per step
        if idx.numel() == 0:
            break
        pidx = (base + piv).reshape(-1)[idx]
        col, pcol = M[idx], M[pidx]                          # (C, R)
        L = low.reshape(-1)[idx]
        ar = torch.arange(idx.numel(), device=dev)
        f = ((col[ar, L].long() * inv[pcol[ar, L].long()]) % p).to(M.dtype)
        new = torch.remainder(col - f[:, None] * pcol, p)
        M[idx] = new
        low.view(-1)[idx] = _last_nonzero(new)
    return low, M.reshape(N, K, R)


def _reduce_chunked(build, N, K, R, p, k_bound, max_elems):
    """Build and reduce the (N, K, R) block in chunks of items so that each
    chunk has <= max_elems entries (memory), concatenating the lows."""
    c = max(1, max_elems // max(1, K * R))
    lows = []
    for s in range(0, N, c):
        low, _ = _reduce_parallel(build(s, min(N, s + c)), p, k_bound)
        lows.append(low)
    return torch.cat(lows, 0)


MAX_BLOCK_ELEMS = 1 << 26


def _coboundary(n_items, cols_rank, n_cols, rows_rank, n_rows, faces, face_mask, signs):
    """Anti-transposed coboundary block (N, n_cols, n_rows) over Z/p:
    column c = (n_cols-1) - rank(face), row r = (n_rows-1) - rank(coface).
    faces: (N, n_rows_simplices, k) local face ids of each coface simplex;
    signs: (k,) boundary coefficients mod p."""
    N = n_items
    dev = faces.device
    Rk = rows_rank                                   # (N, Q) rank of each coface
    Q, k = faces.shape[1], faces.shape[2]
    col = (n_cols - 1) - cols_rank.gather(1, faces.reshape(N, -1)).reshape(N, Q, k)
    row = ((n_rows - 1) - Rk)[:, :, None].expand(N, Q, k)
    val = (signs[None, None, :] * face_mask[:, :, None].long()).expand(N, Q, k)
    M = torch.zeros(N, n_cols * n_rows, dtype=torch.int16, device=dev)
    M.scatter_(1, (col * n_rows + row).reshape(N, -1), val.reshape(N, -1).to(torch.int16))
    return M.reshape(N, n_cols, n_rows)


def torch_persistence(adj, plan, P1, essential, node_level, M, p=11):
    """Returns dense pair blocks {(kind, dim): {b, d, valid, inv}} on fixed
    candidate axes per item; consumed by DifferentiablePH._vectorize_dense."""
    tp = plan.torch_ph
    dev = adj.device
    N, n, m, T, Qn = tp['N'], tp['n'], tp['m'], tp['T'], tp.get('Q', 0)
    INF = float('inf')
    vg, vmask = tp['vg'], tp['vmask']
    eg, emask, ea, eb = tp['eg'], tp['emask'], tp['ea'], tp['eb']
    vv = adj[vg]; ev = adj[eg]
    rank_v, _ = _stable_rank(vv, vmask)
    rank_e, order_e = _stable_rank(ev, emask)
    blocks = {}

    # ---------------- H0 -----------------------------------------------------
    if m and n > SQUARING_MAX_N:
        R = _bottleneck_relax(ea, eb, rank_e, emask, N, n)
    else:
        R = torch.full((N, n, n), INF, device=dev)
        if m:
            rv = torch.where(emask, rank_e.to(R.dtype), torch.full_like(ev, INF, dtype=R.dtype))
            bi = torch.arange(N, device=dev)[:, None].expand(N, m)
            R.index_put_((bi, ea, eb), rv); R.index_put_((bi, eb, ea), rv)
            R = _minmax_closure(R, max(1, int(np.ceil(np.log2(max(2, n - 1))))))
    older = (rank_v[:, None, :] < rank_v[:, :, None]) & vmask[:, None, :]
    dr = torch.where(older, R, torch.full_like(R, INF)).amin(dim=2)          # (N, n)
    fin0 = vmask & torch.isfinite(dr)
    ess0 = vmask & ~torch.isfinite(dr)
    if m:
        de = order_e.gather(1, torch.where(fin0, dr, torch.zeros_like(dr)).long().clamp(max=m - 1))
        msf = torch.zeros(N, m, dtype=torch.long, device=dev).scatter_add_(
            1, torch.where(fin0, de, torch.zeros_like(de)), fin0.long()) > 0
        msf = msf & emask
        d0 = ev.gather(1, de)
    else:
        de = torch.zeros_like(vg); msf = torch.zeros(N, 0, dtype=torch.bool, device=dev); d0 = vv
    fin0 = fin0 & ((d0.detach() - vv.detach()) > PERS_EPS)
    blocks[('fin', 0)] = dict(b=vv, d=d0, valid=fin0, _de=de)
    if essential:
        blocks[('ess', 0)] = dict(b=vv, valid=ess0)

    # ---------------- H1 (cohomology over edges x triangles) ----------------
    pos_e = emask & ~msf                            # positive edges
    tri_paired = None
    if P1 > 1 and m:
        if T:
            tg, tmask, te = tp['tg'], tp['tmask'], tp['te']
            tv = adj[tg]
            rank_t, order_t = _stable_rank(tv, tmask)
            sign3 = _const(f's3_{p}', dev, lambda: torch.tensor([1, p - 1, 1]))
            # clearing: columns of negative (spanning-forest) edges
            col_pos = pos_e.gather(1, order_e.flip(1))          # column c <-> edge rank m-1-c
            def build1(a, b):
                Mc = _coboundary(b - a, rank_e[a:b], m, rank_t[a:b], T, te[a:b], tmask[a:b], sign3)
                return Mc * col_pos[a:b, :, None].to(Mc.dtype)
            low = _reduce_chunked(build1, N, m, T, p, tp['K1'], MAX_BLOCK_ELEMS)
            has = low >= 0
            e_col = order_e.flip(1)                               # edge local idx of column
            t_of = order_t.gather(1, ((T - 1) - low).clamp(min=0, max=T - 1))
            b1, d1 = ev.gather(1, e_col), tv.gather(1, t_of)
            f1 = has & col_pos & ((d1.detach() - b1.detach()) > PERS_EPS)
            blocks[('fin', 1)] = dict(b=b1, d=d1, valid=f1, _e=e_col, _t=t_of, _tv=tv)
            killed_cols = has
            tri_paired = torch.zeros(N, T, dtype=torch.long, device=dev).scatter_add_(
                1, t_of, has.long()) > 0
            ess1_col = col_pos & ~killed_cols
            if essential:
                blocks[('ess', 1)] = dict(b=ev.gather(1, e_col), valid=ess1_col, _e=e_col)
        elif essential:
            blocks[('ess', 1)] = dict(b=ev, valid=pos_e, _e=None)

    # ---------------- H2 (cohomology over triangles x tetrahedra) -----------
    if P1 > 2 and T:
        tg, tmask = tp['tg'], tp['tmask']
        tv = adj[tg]
        rank_t, order_t = _stable_rank(tv, tmask)
        pos_t = tmask & ~(tri_paired if tri_paired is not None
                          else torch.zeros_like(tmask))
        if Qn:
            qg, qmask, qf = tp['qg'], tp['qmask'], tp['qf']
            qv = adj[qg]
            rank_q, order_q = _stable_rank(qv, qmask)
            sign4 = _const(f's4_{p}', dev, lambda: torch.tensor([1, p - 1, 1, p - 1]))
            col_pos = pos_t.gather(1, order_t.flip(1))
            def build2(a, b):
                Mc = _coboundary(b - a, rank_t[a:b], T, rank_q[a:b], Qn, qf[a:b], qmask[a:b], sign4)
                return Mc * col_pos[a:b, :, None].to(Mc.dtype)
            low = _reduce_chunked(build2, N, T, Qn, p, tp['K2'], MAX_BLOCK_ELEMS)
            has = low >= 0
            t_col = order_t.flip(1)
            q_of = order_q.gather(1, ((Qn - 1) - low).clamp(min=0, max=Qn - 1))
            b2, d2 = tv.gather(1, t_col), qv.gather(1, q_of)
            f2 = has & col_pos & ((d2.detach() - b2.detach()) > PERS_EPS)
            blocks[('fin', 2)] = dict(b=b2, d=d2, valid=f2, _t2=t_col, _q=q_of, _qv=qv, _tv=tv)
            if essential:
                blocks[('ess', 2)] = dict(b=tv.gather(1, t_col), valid=col_pos & ~has,
                                          _t2=t_col, _tv=tv)
        elif essential:
            blocks[('ess', 2)] = dict(b=tv, valid=pos_t, _t2=None, _tv=tv)

    # ---------------- tie-merged involved nodes --------------------------------
    if node_level:
        kv, ke = _key(vv), _key(ev)
        incE = tp['incE']
        def V(k):                                   # vertices of value-key k (N,R)
            return (kv[:, None, :] == k[:, :, None]) & vmask[:, None, :]
        def E(k):
            mt = ((ke[:, None, :] == k[:, :, None]) & emask[:, None, :]).to(incE.dtype)
            return torch.bmm(mt, incE) > 0
        def Tri(k, tv):
            kt = _key(tv)
            mt = ((kt[:, None, :] == k[:, :, None]) & tp['tmask'][:, None, :]).to(incE.dtype)
            return torch.bmm(mt, tp['incT']) > 0
        def Tet(k, qv):
            kq = _key(qv)
            mt = ((kq[:, None, :] == k[:, :, None]) & tp['qmask'][:, None, :]).to(incE.dtype)
            return torch.bmm(mt, tp['incQ']) > 0
        b = blocks[('fin', 0)]
        inv = V(kv)
        if m:
            inv = inv | E(ke.gather(1, b['_de']))
        b['inv'] = inv
        if ('ess', 0) in blocks:
            blocks[('ess', 0)]['inv'] = V(kv)
        if ('fin', 1) in blocks:
            b = blocks[('fin', 1)]
            b['inv'] = E(ke.gather(1, b['_e'])) | Tri(_key(b['_tv']).gather(1, b['_t']), b['_tv'])
        if ('ess', 1) in blocks:
            b = blocks[('ess', 1)]
            b['inv'] = E(ke if b['_e'] is None else ke.gather(1, b['_e']))
        if ('fin', 2) in blocks:
            b = blocks[('fin', 2)]
            b['inv'] = (Tri(_key(b['_tv']).gather(1, b['_t2']), b['_tv'])
                        | Tet(_key(b['_qv']).gather(1, b['_q']), b['_qv']))
        if ('ess', 2) in blocks:
            b = blocks[('ess', 2)]
            kt = _key(b['_tv'])
            b['inv'] = Tri(kt if b['_t2'] is None else kt.gather(1, b['_t2']), b['_tv'])
    else:
        for b in blocks.values():
            b['inv'] = None
    return blocks
