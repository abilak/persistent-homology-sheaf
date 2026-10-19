from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

import gudhi

from layers.torch_ph import PERS_EPS, torch_pers, static_blocks

STD_EPS = 1e-6
TORCH_PH_MAX_ELEMS = 1 << 26  # bigger blocks go to gudhi

def skeys(arr, M):
    key = np.zeros(len(arr), dtype=np.int64)
    for c in range(arr.shape[1]):
        key = key * M + arr[:, c].astype(np.int64)
    return key

def cliques(adj, max_dim):
    A = np.abs(np.asarray(adj)) > 1e-6
    np.fill_diagonal(A, False)
    M = A.shape[0]
    out = {0: np.arange(M, dtype=np.int64)[:, None]}
    iu, ju = np.nonzero(np.triu(A, 1))
    out[1] = np.stack([iu, ju], 1).astype(np.int64) if len(iu) else np.zeros((0, 2), np.int64)
    prev = out[1]
    for k in range(2, max_dim + 1):
        if(not len(prev)):
            out[k] = prev = np.zeros((0, k + 1), np.int64)
            continue
        common = np.ones((len(prev), M), dtype=bool)
        for c in range(prev.shape[1]):
            common &= A[prev[:, c]]
        common &= np.arange(M)[None, :] > prev[:, -1:]
        r, x = np.nonzero(common)
        if(len(r)):
            out[k] = np.concatenate([prev[r], x[:, None]], 1).astype(np.int64)
        else:
            out[k] = np.zeros((0, k + 1), np.int64)
        prev = out[k]
    return out

def graph_struct(adj, max_dim):
    M = adj.shape[0]
    by_dim = cliques(adj, max_dim)
    offsets, simps = {}, []
    for k in sorted(by_dim):
        offsets[k] = len(simps)
        simps.extend(map(tuple, by_dim[k].tolist()))

    flat, seg, cnt = [], [], []
    for k in sorted(by_dim):
        arr = by_dim[k]
        if(not len(arr)):
            continue
        n_k, w = arr.shape
        flat.append((arr[:, :, None] * M + arr[:, None, :]).reshape(-1))
        seg.append(np.repeat(np.arange(n_k) + offsets[k], w * w))
        cnt.append(np.full(n_k, w * w, dtype=np.int64))

    keys = {k: skeys(by_dim[k], M) for k in by_dim}
    faces = {}
    for k in sorted(by_dim):
        arr = by_dim[k]
        if(k == 0 or not len(arr)):
            continue
        cols = []
        for size in range(1, k + 1):
            for pos in combinations(range(k + 1), size):
                loc = np.searchsorted(keys[size - 1], skeys(arr[:, list(pos)], M))
                cols.append(loc + offsets[size - 1])
        faces[k] = (np.arange(len(arr)) + offsets[k], np.stack(cols, axis=1))

    gudhi_batches = []
    for k in sorted(by_dim):
        if(len(by_dim[k]) == 0):
            continue
        idx = np.arange(len(by_dim[k]), dtype=np.int64) + offsets[k]
        gudhi_batches.append((idx, by_dim[k].T.astype(np.int32).copy()))

    z = np.zeros(0, np.int64)
    return dict(M=M, S=len(simps), simplices=simps, index=dict(zip(simps, range(len(simps)))), dims=[len(s) - 1 for s in simps], f_flat=np.concatenate(flat) if flat else z, f_seg=np.concatenate(seg) if seg else z, f_counts=np.concatenate(cnt) if cnt else z, faces=faces, gudhi_batches=gudhi_batches, tph=None, betti=None)

_structs = {}

def get_struct(adj, max_dim):
    a = np.abs(adj) > 1e-6
    key = (a.shape[0], int(max_dim), a.astype(np.uint8).tobytes())
    st = _structs.get(key)
    if(st is None):
        st = graph_struct(adj, max_dim)
        if(len(_structs) < 200000):
            _structs[key] = st
    return st

def build_structs(adjs, max_dim):
    return [get_struct(adj, max_dim) for adj in adjs]

def betti_vec(st, max_dim):
    if(st['betti'] is None or len(st['betti']) != max_dim + 1):
        tree = gudhi.SimplexTree()
        for s in st['simplices']:
            if(len(s) - 1 <= max_dim):
                tree.insert(list(s), filtration=0.0)
        tree.compute_persistence(persistence_dim_max=True)
        b = list(tree.betti_numbers())[:max_dim + 1]
        st['betti'] = tuple(b + [0] * (max_dim + 1 - len(b)))
    return st['betti']

def ph_tables(st):
    if(st['tph'] is not None):
        return st['tph']
    v_idx, e_idx, e_end, eloc = [], [], [], {}
    for(i, s) in enumerate(st['simplices']):
        if(len(s) == 1):
            v_idx.append((s[0], i))
        elif(len(s) == 2):
            eloc[tuple(sorted(s))] = len(e_idx)
            e_idx.append(i)
            e_end.append(sorted(s))
    t_idx, t_edges, t_verts, tloc = [], [], [], {}
    for(i, s) in enumerate(st['simplices']):
        if(len(s) == 3):
            a, b, c = sorted(s)
            tloc[(a, b, c)] = len(t_idx)
            t_idx.append(i)
            t_verts.append((a, b, c))
            t_edges.append((eloc[(a, b)], eloc[(a, c)], eloc[(b, c)]))
    q_idx, q_faces, q_verts = [], [], []
    for(i, s) in enumerate(st['simplices']):
        if(len(s) == 4):
            a, b, c, d = sorted(s)
            q_idx.append(i)
            q_verts.append((a, b, c, d))
            q_faces.append((tloc[(b, c, d)], tloc[(a, c, d)], tloc[(a, b, d)], tloc[(a, b, c)]))

    # cycle rank
    parent = list(range(st['M']))
    def find(x):
        while(parent[x] != x):
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    comps = st['M']
    for(a, b) in e_end:
        ra, rb = find(a), find(b)
        if(ra != rb):
            parent[ra] = rb
            comps -= 1
    vi = np.zeros(st['M'], dtype=np.int64)
    for(v, i) in v_idx:
        vi[v] = i
    st['tph'] = dict(v=vi, e=np.asarray(e_idx, np.int64), eend=np.asarray(e_end, np.int64).reshape(-1, 2), t=np.asarray(t_idx, np.int64), tedges=np.asarray(t_edges, np.int64).reshape(-1, 3), tverts=np.asarray(t_verts, np.int64).reshape(-1, 3), q=np.asarray(q_idx, np.int64), qfaces=np.asarray(q_faces, np.int64).reshape(-1, 4), qverts=np.asarray(q_verts, np.int64).reshape(-1, 4), cyc=len(e_idx) - st['M'] + comps)
    return st['tph']

def ph_arrays(structs, off, n, T, Q):
    B = len(structs)
    m = max(len(ph_tables(st)['e']) for st in structs)
    vg, vmask = np.zeros((B, n), np.int64), np.zeros((B, n), np.int64)
    eg = np.zeros((B, max(m, 1)), np.int64)
    emask, ea, eb = np.zeros_like(eg), np.zeros_like(eg), np.zeros_like(eg)
    tg = np.zeros((B, max(T, 1)), np.int64)
    tmask = np.zeros_like(tg)
    te = np.zeros((B, max(T, 1), 3), np.int64)
    tv = np.zeros_like(te)
    qg = np.zeros((B, max(Q, 1)), np.int64)
    qmask = np.zeros_like(qg)
    qf = np.zeros((B, max(Q, 1), 4), np.int64)
    qv = np.zeros_like(qf)
    for(b, st) in enumerate(structs):
        o, d = off[b], ph_tables(st)
        vg[b, :st['M']] = d['v'] + o
        vmask[b, :st['M']] = 1
        me, mt, mq = len(d['e']), len(d['t']), len(d['q'])
        if(me):
            eg[b, :me] = d['e'] + o
            emask[b, :me] = 1
            ea[b, :me], eb[b, :me] = d['eend'][:, 0], d['eend'][:, 1]
        if(mt and T):
            tg[b, :mt] = d['t'] + o
            tmask[b, :mt] = 1
            te[b, :mt], tv[b, :mt] = d['tedges'], d['tverts']
        if(mq and Q):
            qg[b, :mq] = d['q'] + o
            qmask[b, :mq] = 1
            qf[b, :mq], qv[b, :mq] = d['qfaces'], d['qverts']
    K1 = max((ph_tables(st)['cyc'] for st in structs), default=0)
    meta = dict(N=B, n=n, m=m, T=T, Q=Q, K1=K1, K2=T)
    out = dict(vg=vg, vmask=vmask, eg=eg[:, :m], emask=emask[:, :m], ea=ea[:, :m], eb=eb[:, :m], item=np.arange(B, dtype=np.int64))
    if(T):
        out.update(tg=tg, tmask=tmask, te=te, tv=tv)
    if(Q):
        out.update(qg=qg, qmask=qmask, qf=qf, qv=qv)
    return out, meta

def incid(N, rows, n, mask, verts, device):
    inc = torch.zeros(N, rows, n, device=device)
    bi = torch.arange(N, device=device)[:, None].expand(N, rows)
    ri = torch.arange(rows, device=device)[None, :].expand(N, rows)
    w = mask.to(inc.dtype)
    for v in verts:
        inc.index_put_((bi, ri, v), w, accumulate=True)
    return inc.clamp(max=1)

def make_plan(structs, device, pdim=1, sdim=None, backend='auto', M_pad=None):
    B = len(structs)
    if(sdim is None):
        sdim = max((max(st['dims']) for st in structs if st['S']), default=0)
    if(M_pad is not None):
        M = M_pad
        same_M = all(st['M'] <= M for st in structs)
    else:
        M = structs[0]['M'] if B else 0
        same_M = all(st['M'] == M for st in structs)
    S = np.asarray([st['S'] for st in structs], np.int64)
    off = np.concatenate([[0], np.cumsum(S)[:-1]]) if B else np.zeros(0, np.int64)
    S_tot = int(S.sum())

    arrays = {}
    if(same_M and S_tot):
        def flat(st):
            if(st['M'] == M):
                return st['f_flat']
            return (st['f_flat'] // st['M']) * M + (st['f_flat'] % st['M'])
        arrays['f_flat'] = np.concatenate([flat(st) + b * M * M for b, st in enumerate(structs) if st['S']])
        arrays['f_seg'] = np.concatenate([st['f_seg'] + off[b] for b, st in enumerate(structs) if st['S']])
        arrays['f_counts'] = np.concatenate([st['f_counts'] for st in structs if st['S']])
    fdims = sorted({d for st in structs for d in st['faces']})
    for d in fdims:
        sims, faces = [], []
        for(b, st) in enumerate(structs):
            if(d in st['faces']):
                sims.append(st['faces'][d][0] + off[b])
                faces.append(st['faces'][d][1] + off[b])
        arrays[f'sim{d}'] = np.concatenate(sims)
        arrays[f'face{d}'] = np.concatenate(faces)

    T = max((len(ph_tables(st)['t']) for st in structs), default=0) if sdim >= 2 else 0
    Q = max((len(ph_tables(st)['q']) for st in structs), default=0) if sdim >= 3 else 0
    m_max = max((len(ph_tables(st)['e']) for st in structs), default=0)
    fits = m_max * T <= TORCH_PH_MAX_ELEMS and T * Q <= TORCH_PH_MAX_ELEMS
    use_torch = (backend == 'torch' or (backend == 'auto' and device.type != 'cpu')) and same_M and pdim <= 2 and sdim <= 3 and fits and S_tot > 0
    if(use_torch):
        tarr, meta = ph_arrays(structs, off, M, T, Q)
        arrays.update(tarr)

    names = list(arrays)
    cat = np.concatenate([arrays[k].reshape(-1) for k in names]) if names else np.zeros(0, np.int64)
    flat_t = torch.from_numpy(cat.astype(np.int64))
    flat_t = flat_t.pin_memory().to(device, non_blocking=True) if device.type == 'cuda' else flat_t.to(device)
    t, pos = {}, 0
    for k in names:
        t[k] = flat_t[pos:pos + arrays[k].size].reshape(arrays[k].shape)
        pos += arrays[k].size

    plan = dict(M=M, same_M=same_M, S_tot=S_tot, t=t, face_dims=fdims, use_torch=use_torch, tph=None)
    if(use_torch):
        tp = dict(meta)
        for k in ('vg', 'eg', 'ea', 'eb', 'tg', 'te', 'qg', 'qf', 'item'):
            if(k in t):
                tp[k] = t[k]
        for k in ('vmask', 'emask', 'tmask', 'qmask'):
            if(k in t):
                tp[k] = t[k].bool()
        N, n = tp['N'], tp['n']
        if(tp['m']):
            tp['inc_e'] = incid(N, tp['m'], n, tp['emask'], [tp['ea'], tp['eb']], device)
        else:
            tp['inc_e'] = torch.zeros(N, 1, n, device=device)
        if(T):
            tp['inc_t'] = incid(N, T, n, tp['tmask'], [t['tv'][:, :, c] for c in range(3)], device)
        if(Q):
            tp['inc_q'] = incid(N, Q, n, tp['qmask'], [t['qv'][:, :, c] for c in range(4)], device)
        plan['tph'] = tp
    return plan

_plans = {}

def get_plan(structs, device, pdim=1, sdim=None, backend='auto', M_pad=None):
    key = (tuple(map(id, structs)), str(device), pdim, sdim, backend, M_pad)
    if(key in _plans):
        return _plans[key][0]
    if(len(_plans) > 8):
        _plans.clear()
    p = make_plan(structs, device, pdim, sdim, backend, M_pad)
    _plans[key] = (p, list(structs))
    return p

class LearnedFiltration(nn.Module):
    def __init__(self, in_f, hid=64, squash=False):
        super().__init__()
        self.squash = squash
        self.mlp = nn.Sequential(nn.Linear(in_f, hid), nn.ReLU(), nn.Linear(hid, 1))
        for m in self.mlp:
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, structs, plan=None):
        # f(sigma) = mlp(mean of x_ij over sigma x sigma)
        B, d, M, _ = x.shape
        if(plan is None):
            plan = get_plan(structs, x.device, M_pad=M)
        if(not plan['S_tot']):
            return [(st, torch.zeros(0, device=x.device)) for st in structs]
        t = plan['t']
        pf = x.permute(0, 2, 3, 1).reshape(B * M * M, d)
        sums = torch.zeros(plan['S_tot'], d, device=x.device, dtype=pf.dtype)
        sums = sums.index_add(0, t['f_seg'], pf.index_select(0, t['f_flat']))
        vals = self.mlp(sums / t['f_counts'].to(pf.dtype).unsqueeze(-1)).squeeze(-1)
        if(self.squash):
            vals = torch.tanh(vals)
        out, cur = [], 0
        for st in structs:
            out.append((st, vals[cur:cur + st['S']]))
            cur += st['S']
        return out

def masked_sm(logits, mask, dim):
    neg = torch.finfo(logits.dtype).min
    mx = torch.where(mask, logits, torch.full_like(logits, neg)).detach().amax(dim, keepdim=True)
    mx = torch.where(mx > neg, mx, torch.zeros_like(mx))
    e = torch.exp(torch.where(mask, logits - mx, torch.full_like(logits, neg))) * mask
    return e / e.sum(dim, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)

def seg_sm(logits, seg, n_seg):
    mx = torch.full((n_seg,), float('-inf'), device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(0, seg, logits.detach(), reduce='amax', include_self=True)
    e = torch.exp(logits - mx[seg])
    den = torch.zeros(n_seg, device=logits.device, dtype=logits.dtype).index_add(0, seg, e)
    return e / den[seg]

def safe_std(var):
    pos = var > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, var, torch.ones_like(var))), torch.zeros_like(var))

class DifferentiablePH(nn.Module):
    def __init__(self, pdim=1, vec_dim=16, use_ess=False, static=False):
        super().__init__()
        self.pdim = pdim
        self.vec_dim = vec_dim
        self.use_ess = use_ess
        self.static = static
        self.per_dim = vec_dim + 1 + ((vec_dim + 1) if use_ess else 0)
        self.out_f = self.nout_f = (pdim + 1) * self.per_dim
        self.embeds, self.attns = nn.ModuleList(), nn.ModuleList()
        for _ in range(pdim + 1):
            self.embeds.append(nn.Sequential(nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)))
            self.attns.append(nn.Sequential(nn.Linear(2, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)))
        if(use_ess):
            self.ess_embeds, self.ess_attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(pdim + 1):
                self.ess_embeds.append(nn.Sequential(nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, vec_dim)))
                self.ess_attns.append(nn.Sequential(nn.Linear(1, vec_dim), nn.ReLU(), nn.Linear(vec_dim, 1)))

    @staticmethod
    def nondec(vals, tables):
        # f(face) <= f(coface), one dimension at a time
        out = vals.clone()
        for(sim, face) in tables:
            new = torch.max(out[sim], out[face].max(dim=1).values)
            out = out.scatter(0, sim, new)
        return out

    def forward(self, filt_batch, device, nnodes=None, plan=None):
        B, P1, M = len(filt_batch), self.pdim + 1, nnodes
        if(plan is None):
            plan = get_plan([st for st, _ in filt_batch], device, self.pdim, M_pad=M)
        offsets, parts, off = [], [], 0
        for(st, ft) in filt_batch:
            offsets.append(off if st['S'] else None)
            if(st['S']):
                parts.append(ft)
                off += st['S']
        if(not parts):
            return (torch.zeros(B, self.out_f, device=device), torch.zeros(B, self.nout_f, M, device=device) if M is not None else None)
        t = plan['t']
        tables = [(t['sim%d' % d], t['face%d' % d]) for d in plan['face_dims']]
        vals = self.nondec(torch.cat(parts), tables)

        if(plan['use_torch']):
            fn = static_blocks if self.static else torch_pers
            return self.vec_dense(fn(vals, plan, P1, self.use_ess, M is not None), B, P1, M, vals.dtype, device)
        if(self.static):
            raise RuntimeError('static control needs the torch backend')
        return self.vec(vals, self.gpairs(vals, filt_batch, offsets, P1, M, device), B, P1, M)

    def dblock(self, emb_mlp, attn_mlp, q, valid, inv, node_lvl):
        emb = emb_mlp(q)
        logit = attn_mlp(q).squeeze(-1)
        w = masked_sm(logit, valid, dim=1)
        gpool = torch.einsum('nr,nrv->nv', w, emb)
        cnt = valid.sum(1).to(q.dtype)
        if(not node_lvl):
            return gpool, cnt, None, None
        nm = inv & valid[:, :, None]
        wn = masked_sm(logit[:, :, None].expand_as(nm), nm, dim=1)
        return gpool, cnt, torch.einsum('nru,nrv->nuv', wn, emb), nm.sum(1).to(q.dtype)

    def vec_dense(self, blocks, B, P1, M, dt, device):
        node_lvl = M is not None
        v = self.vec_dim

        def empty():
            gp = torch.zeros(B, v, device=device, dtype=dt)
            cnt = torch.zeros(B, device=device, dtype=dt)
            if(not node_lvl):
                return gp, cnt, None, None
            return gp, cnt, torch.zeros(B, M, v, device=device, dtype=dt), torch.zeros(B, M, device=device, dtype=dt)

        g_dims, n_dims = [], []
        for d in range(P1):
            fb = blocks.get(('fin', d))
            if(fb is None):
                gp, cnt, npool, ncnt = empty()
            else:
                valid = fb['valid']
                bp = torch.stack([fb['b'], torch.clamp(fb['d'] - fb['b'], min=0)], dim=-1)
                bp = torch.where(valid[..., None], bp, torch.zeros_like(bp))
                vf = valid[..., None].to(dt)
                c = vf.sum(1).clamp_min(1)
                mean = (bp * vf).sum(1) / c
                cen = (bp - mean[:, None, :]) * vf
                std = safe_std((cen * cen).sum(1) / c)
                q = cen / (std[:, None, :] + STD_EPS)
                gp, cnt, npool, ncnt = self.dblock(self.embeds[d], self.attns[d], q, valid, fb['inv'], node_lvl)
            g = [gp, torch.log1p(cnt)[:, None]]
            n = [npool, torch.log1p(ncnt)[..., None]] if node_lvl else []
            if(self.use_ess):
                eb = blocks.get(('ess', d))
                if(eb is None):
                    egp, ecnt, enp, encnt = empty()
                else:
                    q = torch.where(eb['valid'], eb['b'], torch.zeros_like(eb['b']))[..., None]
                    egp, ecnt, enp, encnt = self.dblock(self.ess_embeds[d], self.ess_attns[d], q, eb['valid'], eb['inv'], node_lvl)
                g += [egp, torch.log1p(ecnt)[:, None]]
                if(node_lvl):
                    n += [enp, torch.log1p(encnt)[..., None]]
            g_dims.append(torch.cat(g, dim=1))
            if(node_lvl):
                n_dims.append(torch.cat(n, dim=2).permute(0, 2, 1))
        return torch.cat(g_dims, dim=1), (torch.cat(n_dims, dim=1) if node_lvl else None)

    def gpairs(self, vals, filt_batch, offsets, P1, M, device):
        node_lvl = M is not None
        host = vals.detach().cpu().numpy().astype(np.float64)
        hl = host.tolist()
        f_seg, f_b, f_d, e_seg, e_b = [], [], [], [], []
        fn_r, fn_c, en_r, en_c = [], [], [], []
        for(bi, ((st, _), o)) in enumerate(zip(filt_batch, offsets)):
            if(o is None):
                continue
            tree = gudhi.SimplexTree()
            for(idx, verts) in st['gudhi_batches']:
                tree.insert_batch(verts, host[o:o + st['S']][idx])
            tree.make_filtration_non_decreasing()
            tree.persistence(persistence_dim_max=self.use_ess, min_persistence=PERS_EPS)
            if(node_lvl):
                # TODO slow on cpu
                vl = hl[o:o + st['S']]
                by_val = defaultdict(set)
                for(i, s) in enumerate(st['simplices']):
                    by_val[(len(s), round(vl[i], 6))].update(s)
            s2i = st['index']
            for(bs, ds) in tree.persistence_pairs():
                dim = len(bs) - 1
                if(dim > self.pdim):
                    continue
                b_idx = s2i[tuple(sorted(bs))]
                seg = bi * P1 + dim
                if(len(ds) == 0):
                    if(not self.use_ess):
                        continue
                    if(node_lvl):
                        for u in by_val[(len(bs), round(vl[b_idx], 6))]:
                            if(u < M):
                                en_r.append(len(e_b))
                                en_c.append(u)
                    e_seg.append(seg)
                    e_b.append(o + b_idx)
                    continue
                d_idx = s2i[tuple(sorted(ds))]
                if(node_lvl):
                    for u in by_val[(len(bs), round(vl[b_idx], 6))] | by_val[(len(ds), round(vl[d_idx], 6))]:
                        if(u < M):
                            fn_r.append(len(f_b))
                            fn_c.append(u)
                f_seg.append(seg)
                f_b.append(o + b_idx)
                f_d.append(o + d_idx)
        lists = [f_seg, f_b, f_d, e_seg, e_b, fn_r, fn_c, en_r, en_c]
        lens = [len(a) for a in lists]
        packed = torch.as_tensor(np.fromiter((x for a in lists for x in a), dtype=np.int64, count=sum(lens)), device=device)
        out = list(torch.split(packed, lens))
        if(not node_lvl):
            out[5:] = [None] * 4
        return out

    def pool(self, q, seg, n_seg, embeds, attns, P1, nrows=None, ncols=None, M=None):
        v, dev, dt = self.vec_dim, q.device, q.dtype
        dims = seg % P1
        emb = torch.zeros(q.shape[0], v, device=dev, dtype=dt)
        logit = torch.zeros(q.shape[0], device=dev, dtype=dt)
        for d in range(P1):
            m = dims == d
            emb = torch.where(m[:, None], embeds[d](q), emb)
            logit = torch.where(m, attns[d](q).squeeze(-1), logit)
        w = seg_sm(logit, seg, n_seg)
        gpool = torch.zeros(n_seg, v, device=dev, dtype=dt).index_add(0, seg, w.unsqueeze(-1) * emb)
        if(nrows is None):
            return gpool, None, None
        gid = seg[nrows] * M + ncols
        wn = seg_sm(logit[nrows], gid, n_seg * M)
        npool = torch.zeros(n_seg * M, v, device=dev, dtype=dt).index_add(0, gid, wn.unsqueeze(-1) * emb[nrows])
        ncount = torch.zeros(n_seg * M, device=dev, dtype=dt).index_add(0, gid, torch.ones_like(wn))
        return gpool, npool, ncount

    def vec(self, vals, coo, B, P1, M):
        f_seg, f_b, f_d, e_seg, e_b, fn_r, fn_c, en_r, en_c = coo
        node_lvl = M is not None
        dev, dt = vals.device, vals.dtype
        n_seg = B * P1

        bv, dv = vals[f_b], vals[f_d]
        bp = torch.stack([bv, torch.clamp(dv - bv, min=0)], dim=1)
        cnt = torch.zeros(n_seg, device=dev, dtype=dt).index_add(0, f_seg, torch.ones(bp.shape[0], device=dev, dtype=dt))
        cden = cnt.clamp_min(1).unsqueeze(-1)
        mean = torch.zeros(n_seg, 2, device=dev, dtype=dt).index_add(0, f_seg, bp) / cden
        cen = bp - mean[f_seg]
        std = safe_std(torch.zeros(n_seg, 2, device=dev, dtype=dt).index_add(0, f_seg, cen * cen) / cden)
        q = cen / (std[f_seg] + STD_EPS)
        gpool, npool, ncnt = self.pool(q, f_seg, n_seg, self.embeds, self.attns, P1, fn_r, fn_c, M)
        g = [gpool, torch.log1p(cnt).unsqueeze(-1)]
        n = [npool, torch.log1p(ncnt).unsqueeze(-1)] if node_lvl else []

        if(self.use_ess):
            ecnt = torch.zeros(n_seg, device=dev, dtype=dt).index_add(0, e_seg, torch.ones(e_b.shape[0], device=dev, dtype=dt))
            egp, enp, encnt = self.pool(vals[e_b].unsqueeze(-1), e_seg, n_seg, self.ess_embeds, self.ess_attns, P1, en_r, en_c, M)
            g += [egp, torch.log1p(ecnt).unsqueeze(-1)]
            if(node_lvl):
                n += [enp, torch.log1p(encnt).unsqueeze(-1)]

        gout = torch.cat(g, dim=1).reshape(B, P1 * self.per_dim)
        nout = None
        if(node_lvl):
            nout = torch.cat(n, dim=1).reshape(B, P1, M, self.per_dim).permute(0, 1, 3, 2).reshape(B, P1 * self.per_dim, M)
        return gout, nout

class InvertibleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return torch.cat([self.ln(x), mu, 0.5 * torch.log(var + self.eps)], dim=-1)

class TopologyLayer(nn.Module):
    def __init__(self, n_eqv, hid=64, pdim=1, nstat=4, norm_st=False, use_ess=False, squash=False, static=False, ph_backend='auto'):
        super().__init__()
        self.ph_backend = 'torch' if static else ph_backend
        self.filtration = LearnedFiltration(n_eqv, hid, squash=squash)
        self.ph = DifferentiablePH(pdim, nstat, use_ess=use_ess, static=static)

        td, nd = self.ph.out_f, self.ph.nout_f
        norm = InvertibleLayerNorm if norm_st else nn.LayerNorm
        self.topo_norm = norm(td)
        self.node_norm = norm(nd)
        if(norm_st):
            td, nd = td + 2, nd + 2
        self.gdim = td

        fuse_in = n_eqv + td + 2 * nd
        self.fusion = nn.Sequential(nn.Conv2d(fuse_in, n_eqv, kernel_size=1, bias=True), nn.ReLU(), nn.Conv2d(n_eqv, n_eqv, kernel_size=1, bias=True))
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        nn.init.normal_(self.fusion[2].weight, std=0.01)
        nn.init.zeros_(self.fusion[2].bias)
        self.gate_conv = nn.Conv2d(fuse_in, n_eqv, kernel_size=1, bias=True)
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, 2.0)
        self.last_g = None

    def forward(self, x, structs):
        B, d, M, _ = x.shape
        plan = get_plan(structs, x.device, pdim=self.ph.pdim, backend=self.ph_backend, M_pad=M)
        gvec, nvec = self.ph(self.filtration(x, structs, plan), device=x.device, nnodes=M, plan=plan)
        gvec = self.topo_norm(gvec)
        self.last_g = gvec
        nvec = self.node_norm(nvec.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # 1x1 conv on [x_uv, g, t_u, t_v] without building it
        W = torch.cat([self.fusion[0].weight[:, :, 0, 0], self.gate_conv.weight[:, :, 0, 0]], dim=0)
        bias = torch.cat([self.fusion[0].bias, self.gate_conv.bias], dim=0)
        cx, cg, cn = x.shape[1], gvec.shape[1], nvec.shape[1]
        pre = torch.nn.functional.conv2d(x, W[:, :cx, None, None], bias)
        pre = pre + (gvec @ W[:, cx:cx + cg].t())[:, :, None, None]
        w_row, w_col = W[:, cx + cg:cx + cg + cn], W[:, cx + cg + cn:]
        pre = pre + torch.einsum('bcm,oc->bom', nvec, w_row)[:, :, :, None] + torch.einsum('bcm,oc->bom', nvec, w_col)[:, :, None, :]
        nf = self.fusion[0].out_channels
        delta = self.fusion[2](self.fusion[1](pre[:, :nf]))
        return x + torch.sigmoid(pre[:, nf:]) * delta
