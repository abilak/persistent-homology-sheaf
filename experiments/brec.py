"""
python experiments/brec.py --model topo --data data/brec_v3.npy --sample-num 400 --epochs 20 --out results/brec/topo.json
python experiments/brec.py --compare results/brec/gin.json results/brec/ppgn.json results/brec/topo.json
"""
import os, sys, json, time, glob, shutil, subprocess
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
os.chdir(ROOT)
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from sr_classification import cfg, norm_input
from models.base_model import build_model, get_device

DEV = get_device()
DATA = 'data/brec_v3.npy'

PARTS = {"Basic": (0, 60), "Regular": (60, 160), "Extension": (160, 260), "CFI": (260, 360), "4-Vertex_Condition": (360, 380), "Distance_Regular": (380, 400)}
OUTPUT_DIM = 16
THRESHOLD = 72.34  # T^2 critical value from the BREC paper
COV_EPS = 1e-3

MODEL_TYPES = {"ppgn": "baseline", "topo": "topo", "mlp": "mlp", "gcn": "gcn", "gin": "gin", "gsn": "gsn"}

def fetch(dst=DATA):
    tmp = os.path.join('results', '_brec_repo')
    if(not os.path.isdir(tmp)):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/GraphPKU/BREC.git", tmp], check=True)
    z = glob.glob(os.path.join(tmp, "**", "BREC_data_all.zip"), recursive=True)
    if(z):
        subprocess.run(["unzip", "-o", "-q", z[0], "-d", os.path.join(tmp, "data")], check=True)
    found = glob.glob(os.path.join(tmp, "**", "brec_v3.npy"), recursive=True)
    if(not found):
        sys.exit("brec_v3.npy not in the BREC repo, put it at %s by hand" % dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(found[0], dst)
    print('copied BREC data to', dst)

def load_graphs(path):
    # graph6 bytes, two per pair
    return [nx.convert_node_labels_to_integers(nx.from_graph6_bytes(bytes(it))) for it in np.load(path, allow_pickle=True)]

def same_pair(a1, a2, b1, b2):
    # some dumps store instances with the two graphs swapped
    na1, na2 = a1.number_of_nodes(), a2.number_of_nodes()
    nb1, nb2 = b1.number_of_nodes(), b2.number_of_nodes()
    if(na1 == nb1 and na2 == nb2 and nx.is_isomorphic(a1, b1) and nx.is_isomorphic(a2, b2)):
        return True
    return na1 == nb2 and na2 == nb1 and nx.is_isomorphic(a1, b2) and nx.is_isomorphic(a2, b1)

# brec_v3.npy has permuted copies of each pair back to back, this finds how many
def pair_stride(graphs, max_scan=2048):
    total = len(graphs) // 2
    R = 1
    while(R < min(max_scan, total)):
        if(not same_pair(graphs[2 * R], graphs[2 * R + 1], graphs[0], graphs[1])):
            break
        R += 1
    return R

def perm_batch(g, num, rng):
    n = g.number_of_nodes()
    xs = [norm_input(g, rng.permutation(n)) for _ in range(num)]
    return torch.stack(xs, 0).to(DEV)

def cov(X):
    Xc = X - X.mean(0, keepdim=True)
    return (Xc.t() @ Xc) / max(1, X.shape[0] - 1)

def hotelling(X, Y, eps=COV_EPS):
    nx_, ny = X.shape[0], Y.shape[0]
    diff = X.mean(0) - Y.mean(0)
    Sp = ((nx_ - 1) * cov(X) + (ny - 1) * cov(Y)) / max(1, nx_ + ny - 2)
    Sp = Sp + eps * torch.eye(X.shape[1], device=Sp.device, dtype=Sp.dtype)
    T2 = (nx_ * ny) / (nx_ + ny) * (diff @ torch.linalg.pinv(Sp) @ diff)
    return float(T2)

def eval_pair(name, g1, g2, args, rng):
    mt = MODEL_TYPES[name]
    c = cfg(OUTPUT_DIM, mt)
    model = build_model(c)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    S, bs = args.sample_num, args.batch

    def embed(g, num, grad):
        outs = []
        for i in range(0, num, bs):
            b = perm_batch(g, min(bs, num - i), rng)
            if(grad):
                outs.append(model(b))
            else:
                with torch.no_grad():
                    outs.append(model(b))
        return torch.cat(outs, 0)

    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        e1 = F.normalize(embed(g1, bs, True), dim=1)
        e2 = F.normalize(embed(g2, bs, True), dim=1)
        between = (e1.mean(0) - e2.mean(0)).pow(2).sum()
        within = e1.var(0).sum() + e2.var(0).sum()
        loss = -between + 0.5 * within
        loss.backward()
        opt.step()

    model.eval()
    E1 = embed(g1, S, False)
    E2 = embed(g2, S, False)
    E1b = embed(g1, S, False)
    pooled = torch.cat([E1, E2], 0)
    mu = pooled.mean(0, keepdim=True)
    sd = pooled.std(0, keepdim=True).clamp_min(1e-8)
    major = hotelling((E1 - mu) / sd, (E2 - mu) / sd)
    rel = hotelling((E1 - mu) / sd, (E1b - mu) / sd)
    dist_ok = bool(major > THRESHOLD and rel < THRESHOLD)
    return {"major": major, "reliability": rel, "distinguished": dist_ok}

def category(pid):
    for(k, (lo, hi)) in PARTS.items():
        if(lo <= pid < hi):
            return k
    return "Unknown"

def bucket(n):
    if(n <= 16):
        return "n<=16"
    if(n <= 40):
        return "17-40"
    return "n>40"

def compare(paths):
    models = {}
    for p in paths:
        d = json.load(open(p))
        models[d["model"]] = {r["pair"]: r for r in d["records"]}
    names = list(models)
    common = sorted(set.intersection(*[set(r) for r in models.values()]))
    print("models:", names)
    print("pairs done by all:", len(common))
    for m in names:
        d = sum(models[m][p]["distinguished"] for p in common)
        print("%s: %d / %d (%.1f%%)" % (m, d, len(common), 100 * d / max(1, len(common))))
    if(not ("ppgn" in models and "topo" in models)):
        return

    ppgn, topo = models["ppgn"], models["topo"]
    hard = [p for p in common if not ppgn[p]["distinguished"]]
    gained = [p for p in hard if topo[p]["distinguished"]]
    lost = []
    for p in common:
        if(ppgn[p]["distinguished"] and not topo[p]["distinguished"]):
            lost.append(p)
    print("\nppgn fails %d pairs, topo gets %d of them" % (len(hard), len(gained)))
    print("ppgn gets but topo doesn't: %d" % len(lost))
    if(len(gained) > 40):
        print("gained:", gained[:40], "...")
    elif(gained):
        print("gained:", gained)

    buckets = {}
    for p in common:
        buckets.setdefault(bucket(ppgn[p].get("n", 0)), []).append(p)
    print("\nby size")
    for b in sorted(buckets):
        ps = buckets[b]
        row = b
        for m in names:
            row += "  %s %d/%d" % (m, sum(models[m][p]['distinguished'] for p in ps), len(ps))
        g = sum(1 for p in ps if topo[p]["distinguished"] and not ppgn[p]["distinguished"])
        print(row + "  topo>ppgn %d" % g)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_TYPES), default="topo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--sample-num", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-nodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs='+', default=None)
    args = ap.parse_args()
    if(args.compare):
        return compare(args.compare)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    path = args.data or DATA
    if(path == DATA and not os.path.exists(path)):
        fetch()
    graphs = load_graphs(path)
    print("loaded %d graphs from %s" % (len(graphs), path))
    stride = pair_stride(graphs)
    n_pairs = len(graphs) // 2 // stride
    if(stride > 1):
        print("%d permuted copies per pair, using the first of each, %d pairs" % (stride, n_pairs))
    print("device", DEV, "model", args.model, "sample_num", args.sample_num, "epochs", args.epochs)

    if(args.pairs is None):
        pids = list(range(n_pairs))
    elif('-' in args.pairs):
        a, b = args.pairs.split('-')
        pids = list(range(int(a), min(int(b), n_pairs)))
    else:
        pids = [int(x) for x in args.pairs.split(',')]

    per_cat = {}
    records = []
    skipped = 0
    t0 = time.time()
    for pid in pids:
        sp = pid * stride
        g1, g2 = graphs[2 * sp], graphs[2 * sp + 1]
        n = max(g1.number_of_nodes(), g2.number_of_nodes())
        if(args.max_nodes and n > args.max_nodes):
            skipped += 1
            continue
        r = eval_pair(args.model, g1, g2, args, rng)
        cat = category(pid)
        if(cat not in per_cat):
            per_cat[cat] = [0, 0]
        per_cat[cat][0] += int(r["distinguished"])
        per_cat[cat][1] += 1
        records.append({"pair": pid, "category": cat, "n": n, **r})
        print("pair %d %s n=%d major %.2f reliab %.2f %s" % (pid, cat, n, r['major'], r['reliability'], 'yes' if r['distinguished'] else 'no'))

    print("\n%s, %.0fs" % (args.model, time.time() - t0))
    tot_d = 0
    tot_s = 0
    for k in PARTS:
        if(k not in per_cat):
            continue
        d, s = per_cat[k]
        tot_d += d
        tot_s += s
        print("%s: %d / %d" % (k, d, s))
    print("total: %d / %d (%.1f%%)" % (tot_d, tot_s, 100 * tot_d / max(1, tot_s)))
    if(skipped):
        print("(skipped %d pairs with more than %d nodes)" % (skipped, args.max_nodes))

    if(args.out):
        cfg_ = {k: v for k, v in vars(args).items() if k != 'compare'}
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({"model": args.model, "config": cfg_, "per_category": per_cat, "total": [tot_d, tot_s], "skipped": skipped, "records": records}, f, indent=2)
        print('Wrote', args.out)

if(__name__ == '__main__'):
    main()
