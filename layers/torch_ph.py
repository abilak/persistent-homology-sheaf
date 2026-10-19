# persistent homology of clique complexes with tensor ops, gives the same diagrams as gudhi
import numpy as np
import torch

PERS_EPS = 1e-6
FIXED_STEPS_MAX = 64
CHECK_EVERY = 4
SQUARING_MAX_N = 128
MAX_BLOCK_ELEMS = 1 << 26

_consts = {}

def take(x, idx):
    return x.index_select(0, idx.reshape(-1)).reshape(*idx.shape, *x.shape[1:])

def const(name, device, fn):
    k = (name, str(device))
    if(k not in _consts):
        _consts[k] = fn().to(device)
    return _consts[k]

def tie_key(x):
    return torch.round(x.detach().double() * 1e6)

def mm_closure(R, rounds, maxel=1 << 25):
    N, n, _ = R.shape
    chunk = max(1, maxel // max(1, n * n * n))
    for _ in range(rounds):
        parts = []
        for s in range(0, N, chunk):
            A = R[s:s + chunk]
            parts.append(torch.maximum(A[:, :, :, None], A[:, None, :, :]).amin(dim=2))
        R = torch.minimum(R, torch.cat(parts, 0))
    return R

def relax(ea, eb, rank_e, emask, N, n, every=8):
    INF = float('inf')
    R = torch.full((N, n, n), INF, device=ea.device)
    R.diagonal(dim1=1, dim2=2).fill_(-1.0)
    r = torch.where(emask, rank_e.to(R.dtype), torch.full_like(rank_e, INF, dtype=R.dtype))
    m = ea.shape[1]
    ia, ib = ea[:, None, :].expand(N, n, m), eb[:, None, :].expand(N, n, m)
    while(True):
        prev = R
        for _ in range(every):
            cb = torch.maximum(R.gather(2, ia), r[:, None, :])
            ca = torch.maximum(R.gather(2, ib), r[:, None, :])
            R = R.scatter_reduce(2, ib, cb, reduce='amin')
            R = R.scatter_reduce(2, ia, ca, reduce='amin')
        if(bool((R == prev).all())):
            break
    R = R.clone()
    R.diagonal(dim1=1, dim2=2).fill_(INF)
    return R

def srank(values, valid):
    v = torch.where(valid, values.detach(), torch.full_like(values, float('inf')))
    order = torch.argsort(v, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order).contiguous())
    return rank, order

def last_nz(M):
    rows = torch.arange(1, M.shape[-1] + 1, device=M.device, dtype=torch.int32)
    return ((M != 0) * rows).amax(dim=-1).long() - 1

def reduce_par(Mc, p, k_bound):
    N, K, R = Mc.shape
    dev = Mc.device
    if(K == 0 or R == 0):
        return torch.full((N, K), -1, dtype=torch.long, device=dev), Mc
    inv = const(f'inv{p}', dev, lambda: torch.tensor([0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=torch.long))
    steps = k_bound * (k_bound - 1) // 2 + 1
    if(steps > FIXED_STEPS_MAX):
        return reduce_sp(Mc, p, inv)

    rows = torch.arange(1, R + 1, device=dev)
    colidx = torch.arange(K, device=dev)
    ridx = torch.arange(R, device=dev)

    def lows(M):
        return ((M != 0) * rows).amax(dim=2) - 1

    for _ in range(steps):
        low = lows(Mc)
        lc = low.clamp(min=0)
        hold = low[:, :, None] == ridx[None, None, :]
        first = torch.where(hold, colidx[None, :, None], torch.full_like(colidx, K)[None, :, None]).amin(dim=1)
        piv = first.gather(1, lc)
        coll = (low >= 0) & (piv < colidx[None, :])
        pc = Mc.gather(1, piv.clamp(max=K - 1)[:, :, None].expand(N, K, R))
        a = Mc.gather(2, lc[:, :, None])[:, :, 0].long()
        b = pc.gather(2, lc[:, :, None])[:, :, 0].long()
        f = ((a * inv[b]) % p).to(Mc.dtype)
        Mc = torch.where(coll[:, :, None], torch.remainder(Mc - f[:, :, None] * pc, p), Mc)
    return lows(Mc), Mc

def reduce_sp(Mc, p, inv):
    N, K, R = Mc.shape
    dev = Mc.device
    NK = N * K
    M = torch.cat([Mc.reshape(NK, R), torch.zeros(1, R, dtype=Mc.dtype, device=dev)], 0)
    low = torch.cat([last_nz(M[:NK]), torch.full((1,), -1, dtype=torch.long, device=dev)])
    colidx = torch.arange(K, device=dev)[None, :].expand(N, K)
    base = (torch.arange(N, device=dev) * K)[:, None]
    fids = torch.arange(NK, device=dev)

    def colls():
        lo = low[:NK].reshape(N, K)
        first = torch.full((N, R + 1), K, dtype=torch.long, device=dev)
        first.scatter_reduce_(1, lo + 1, colidx, reduce='amin')
        piv = first.gather(1, lo + 1)
        return ((lo >= 0) & (piv < colidx)).reshape(-1), (base + piv).reshape(-1)

    def update(idx, pidx):
        col, pcol = M[idx], M[pidx]
        L = low[idx].clamp(min=0)
        ar = torch.arange(idx.numel(), device=dev)
        f = ((col[ar, L].long() * inv[pcol[ar, L].long()]) % p).to(M.dtype)
        new = torch.remainder(col - f[:, None] * pcol, p)
        M[idx] = new
        low[idx] = last_nz(new)
        low[NK] = -1

    coll, pflat = colls()
    idx = coll.nonzero().squeeze(1)
    cap = idx.numel()
    if(cap == 0):
        return low[:NK].reshape(N, K), M[:NK].reshape(N, K, R)
    update(idx, pflat[idx])
    slots = torch.arange(cap, device=dev)
    while(True):
        for _ in range(CHECK_EVERY):
            coll, pflat = colls()
            pos = torch.cumsum(coll.long(), 0) - 1
            tgt = torch.where(coll & (pos < cap), pos, torch.full_like(pos, cap))
            sel = torch.full((cap + 1,), NK, dtype=torch.long, device=dev)
            sel.scatter_(0, tgt, fids)
            sel = sel[:cap]
            valid = slots < coll.sum()
            sel = torch.where(valid, sel, torch.full_like(sel, NK))
            psel = torch.where(valid, pflat[sel.clamp(max=NK - 1)], torch.full_like(sel, NK))
            update(sel, psel)
        if(not bool(colls()[0].any())):
            break
    return low[:NK].reshape(N, K), M[:NK].reshape(N, K, R)

def reduce_chunks(build, N, K, R, p, k_bound, maxel):
    c = max(1, maxel // max(1, K * R))
    return torch.cat([reduce_par(build(s, min(N, s + c)), p, k_bound)[0] for s in range(0, N, c)], 0)

def cobd(N, crank, n_cols, rrank, n_rows, faces, fmask, signs):
    Q, k = faces.shape[1], faces.shape[2]
    col = (n_cols - 1) - crank.gather(1, faces.reshape(N, -1)).reshape(N, Q, k)
    row = ((n_rows - 1) - rrank)[:, :, None].expand(N, Q, k)
    val = (signs[None, None, :] * fmask[:, :, None].long()).expand(N, Q, k)
    M = torch.zeros(N, n_cols * n_rows, dtype=torch.int16, device=faces.device)
    M.scatter_(1, (col * n_rows + row).reshape(N, -1), val.reshape(N, -1).to(torch.int16))
    return M.reshape(N, n_cols, n_rows)

def torch_pers(adj, plan, P1, use_ess, node_lvl, p=11):
    tp = plan['tph']
    dev = adj.device
    N, n, m, T, Qn = tp['N'], tp['n'], tp['m'], tp['T'], tp.get('Q', 0)
    INF = float('inf')
    vmask, emask, ea, eb = tp['vmask'], tp['emask'], tp['ea'], tp['eb']
    vv, ev = take(adj, tp['vg']), take(adj, tp['eg'])
    rank_v, _ = srank(vv, vmask)
    rank_e, order_e = srank(ev, emask)
    blocks = {}

    # H0
    if(m and n > SQUARING_MAX_N):
        R = relax(ea, eb, rank_e, emask, N, n)
    else:
        R = torch.full((N, n, n), INF, device=dev)
        if(m):
            rv = torch.where(emask, rank_e.to(R.dtype), torch.full_like(ev, INF, dtype=R.dtype))
            bi = torch.arange(N, device=dev)[:, None].expand(N, m)
            R.index_put_((bi, ea, eb), rv)
            R.index_put_((bi, eb, ea), rv)
            R = mm_closure(R, max(1, int(np.ceil(np.log2(max(2, n - 1))))))
    older = (rank_v[:, None, :] < rank_v[:, :, None]) & vmask[:, None, :]
    dr = torch.where(older, R, torch.full_like(R, INF)).amin(dim=2)
    fin0 = vmask & torch.isfinite(dr)
    ess0 = vmask & ~torch.isfinite(dr)
    if(m):
        de = order_e.gather(1, torch.where(fin0, dr, torch.zeros_like(dr)).long().clamp(max=m - 1))
        msf = torch.zeros(N, m, dtype=torch.long, device=dev).scatter_add_(1, torch.where(fin0, de, torch.zeros_like(de)), fin0.long()) > 0
        msf = msf & emask
        d0 = ev.gather(1, de)
    else:
        de, msf, d0 = torch.zeros_like(tp['vg']), torch.zeros(N, 0, dtype=torch.bool, device=dev), vv
    fin0 = fin0 & ((d0.detach() - vv.detach()) > PERS_EPS)
    blocks[('fin', 0)] = dict(b=vv, d=d0, valid=fin0, _de=de)
    if(use_ess):
        blocks[('ess', 0)] = dict(b=vv, valid=ess0)

    # H1, cohomology over edges x triangles
    pos_e = emask & ~msf
    tpaired = None
    if(P1 > 1 and m):
        if(T):
            tmask, te = tp['tmask'], tp['te']
            tv = take(adj, tp['tg'])
            rank_t, order_t = srank(tv, tmask)
            sign3 = const(f's3_{p}', dev, lambda: torch.tensor([1, p - 1, 1]))
            col_pos = pos_e.gather(1, order_e.flip(1))   # clearing

            def build1(a, b):
                Mc = cobd(b - a, rank_e[a:b], m, rank_t[a:b], T, te[a:b], tmask[a:b], sign3)
                return Mc * col_pos[a:b, :, None].to(Mc.dtype)
            low = reduce_chunks(build1, N, m, T, p, tp['K1'], MAX_BLOCK_ELEMS)
            has = low >= 0
            e_col = order_e.flip(1)
            t_of = order_t.gather(1, ((T - 1) - low).clamp(min=0, max=T - 1))
            b1, d1 = ev.gather(1, e_col), tv.gather(1, t_of)
            f1 = has & col_pos & ((d1.detach() - b1.detach()) > PERS_EPS)
            blocks[('fin', 1)] = dict(b=b1, d=d1, valid=f1, _e=e_col, _t=t_of, _tv=tv)
            tpaired = torch.zeros(N, T, dtype=torch.long, device=dev).scatter_add_(1, t_of, has.long()) > 0
            if(use_ess):
                blocks[('ess', 1)] = dict(b=ev.gather(1, e_col), valid=col_pos & ~has, _e=e_col)
        elif(use_ess):
            blocks[('ess', 1)] = dict(b=ev, valid=pos_e, _e=None)

    # H2, triangles x tetrahedra
    if(P1 > 2 and T):
        tmask = tp['tmask']
        tv = take(adj, tp['tg'])
        rank_t, order_t = srank(tv, tmask)
        pos_t = tmask & ~(tpaired if tpaired is not None else torch.zeros_like(tmask))
        if(Qn):
            qmask, qf = tp['qmask'], tp['qf']
            qv = take(adj, tp['qg'])
            rank_q, order_q = srank(qv, qmask)
            sign4 = const(f's4_{p}', dev, lambda: torch.tensor([1, p - 1, 1, p - 1]))
            col_pos = pos_t.gather(1, order_t.flip(1))

            def build2(a, b):
                Mc = cobd(b - a, rank_t[a:b], T, rank_q[a:b], Qn, qf[a:b], qmask[a:b], sign4)
                return Mc * col_pos[a:b, :, None].to(Mc.dtype)
            low = reduce_chunks(build2, N, T, Qn, p, tp['K2'], MAX_BLOCK_ELEMS)
            has = low >= 0
            t_col = order_t.flip(1)
            q_of = order_q.gather(1, ((Qn - 1) - low).clamp(min=0, max=Qn - 1))
            b2, d2 = tv.gather(1, t_col), qv.gather(1, q_of)
            f2 = has & col_pos & ((d2.detach() - b2.detach()) > PERS_EPS)
            blocks[('fin', 2)] = dict(b=b2, d=d2, valid=f2, _t2=t_col, _q=q_of, _qv=qv, _tv=tv)
            if(use_ess):
                blocks[('ess', 2)] = dict(b=tv.gather(1, t_col), valid=col_pos & ~has, _t2=t_col, _tv=tv)
        elif(use_ess):
            blocks[('ess', 2)] = dict(b=tv, valid=pos_t, _t2=None, _tv=tv)

    if(not node_lvl):
        for b in blocks.values():
            b['inv'] = None
        return blocks

    kv, ke = tie_key(vv), tie_key(ev)
    inc_e = tp['inc_e']

    def V(k):
        return (kv[:, None, :] == k[:, :, None]) & vmask[:, None, :]

    def hit(keys, mask, inc, k):
        mt = ((keys[:, None, :] == k[:, :, None]) & mask[:, None, :]).to(inc_e.dtype)
        return torch.bmm(mt, inc) > 0

    b = blocks[('fin', 0)]
    if(m):
        b['inv'] = V(kv) | hit(ke, emask, inc_e, ke.gather(1, b['_de']))
    else:
        b['inv'] = V(kv)
    if(('ess', 0) in blocks):
        blocks[('ess', 0)]['inv'] = V(kv)
    if(('fin', 1) in blocks):
        b = blocks[('fin', 1)]
        kt = tie_key(b['_tv'])
        b['inv'] = hit(ke, emask, inc_e, ke.gather(1, b['_e'])) | hit(kt, tp['tmask'], tp['inc_t'], kt.gather(1, b['_t']))
    if(('ess', 1) in blocks):
        b = blocks[('ess', 1)]
        b['inv'] = hit(ke, emask, inc_e, ke if b['_e'] is None else ke.gather(1, b['_e']))
    if(('fin', 2) in blocks):
        b = blocks[('fin', 2)]
        kt, kq = tie_key(b['_tv']), tie_key(b['_qv'])
        b['inv'] = hit(kt, tp['tmask'], tp['inc_t'], kt.gather(1, b['_t2'])) | hit(kq, tp['qmask'], tp['inc_q'], kq.gather(1, b['_q']))
    if(('ess', 2) in blocks):
        b = blocks[('ess', 2)]
        kt = tie_key(b['_tv'])
        b['inv'] = hit(kt, tp['tmask'], tp['inc_t'], kt if b['_t2'] is None else kt.gather(1, b['_t2']))
    return blocks

def static_blocks(adj, plan, P1, use_ess, node_lvl):
    # static control, no homology
    tp = plan['tph']
    dev = adj.device
    m = tp['m']
    ar = torch.arange(tp['n'], device=dev)
    vv, vmask = take(adj, tp['vg']), tp['vmask']
    inv0 = ((ar[:, None] == ar[None, :])[None] & vmask[:, :, None]) if node_lvl else None
    blocks = {('fin', 0): dict(b=vv, d=vv + 1.0, valid=vmask, inv=inv0)}
    if(P1 > 1 and m):
        ev, emask = take(adj, tp['eg']), tp['emask']
        inv1 = (((tp['ea'][:, :, None] == ar) | (tp['eb'][:, :, None] == ar)) & emask[:, :, None]) if node_lvl else None
        blocks[('fin', 1)] = dict(b=ev, d=ev + 1.0, valid=emask, inv=inv1)
    if(P1 > 2 and tp.get('T', 0)):
        tv, tmask = take(adj, tp['tg']), tp['tmask']
        inv2 = ((plan['t']['tv'][:, :, :, None] == ar).any(2) & tmask[:, :, None]) if node_lvl else None
        blocks[('fin', 2)] = dict(b=tv, d=tv + 1.0, valid=tmask, inv=inv2)
    if(use_ess):
        for d in range(P1):
            fb = blocks.get(('fin', d))
            if(fb is not None):
                blocks[('ess', d)] = dict(b=fb['b'], valid=torch.zeros_like(fb['valid']), inv=None if fb['inv'] is None else torch.zeros_like(fb['inv']))
    return blocks
