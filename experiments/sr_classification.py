import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from easydict import EasyDict

from srg import rook_graph, shrikhande_graph, triangular_graph, chang_graphs, paley_graph, latin_square_L3_5, cone
from models.base_model import build_model, get_device

DEV = get_device()
DEFAULT_OUT = 'results/expr/sr_classification.json'

D, P = 3, 2
MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']

def csl(n, s):
    # circular skip links
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
        g.add_edge(i, (i + s) % n)
    return g

def cfg(ncls, mtype):
    arch = dict(block_features=[64, 64], depth_of_mlp=2, use_topology=False, topo_hidden_dim=32, topo_max_ph_dim=P, topo_num_stats=16, topo_max_simplex_dim=D)
    if(mtype == 'topo'):
        arch.update(use_topology=True, topo_multiplicity=True, topo_norm_stats=True, topo_essential=True, topo_filt_squash=True, topo_readout=True)
    elif(mtype in ('gcn', 'gin', 'mlp', 'gsn')):
        arch['baseline_type'] = mtype
    return EasyDict(architecture=arch, node_labels=0, num_classes=ncls)

def norm_input(g, perm):
    n = g.number_of_nodes()
    A = nx.to_numpy_array(g, nodelist=range(n)).astype(np.float32)
    A = A[np.ix_(perm, perm)]
    deg = np.sqrt(A.sum(0))
    dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    A_norm = (dinv[:, None] * A) * dinv[None, :]
    return torch.from_numpy(A_norm).unsqueeze(0)

def make_ds(graphs, n_copies, seed):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for(label, g) in enumerate(graphs):
        for _ in range(n_copies):
            perm = rng.permutation(g.number_of_nodes())
            xs.append(norm_input(g, perm))
            ys.append(label)
    X = torch.cat(xs, 0).unsqueeze(1).to(DEV)
    y = torch.tensor(ys, device=DEV)
    return X, y

def run(mtype, graphs, n_copies=120, epochs=150, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = len(graphs)
    X, y = make_ds(graphs, n_copies, seed)
    idx = np.random.permutation(X.shape[0])
    ntr = int(0.8 * X.shape[0])
    tr, te = idx[:ntr], idx[ntr:]

    c = cfg(K, mtype)
    model = build_model(c)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, ytr = X[tr], y[tr]
    best_tr = best_te = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            logits = model(Xtr[b])
            loss = F.cross_entropy(logits, ytr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tra = (model(X[tr]).argmax(1) == y[tr]).float().mean().item()
            tea = (model(X[te]).argmax(1) == y[te]).float().mean().item()
        best_tr = max(best_tr, tra)
        best_te = max(best_te, tea)
    return best_tr, best_te, 1.0 / K

def get_tasks():
    rook, shri = rook_graph(4), shrikhande_graph()
    return {"CSL (10-way, N=41, S in {2,3,4,5,6,9,11,12,13,16})": [csl(41, s) for s in (2, 3, 4, 5, 6, 9, 11, 12, 13, 16)], "Rook(4,4) vs Shrikhande [srg(16,6,2,2)]": [rook, shri], "T(8) vs Chang1/2/3 [srg(28,12,6,4)]": [triangular_graph(8)] + chang_graphs(), "Paley(25) vs L3(5) [srg(25,12,5,6)]": [paley_graph(25), latin_square_L3_5()], "cone(Rook(4,4)) vs cone(Shrikhande) [D=2 witness]": [cone(rook), cone(shri)]}

ALIASES = {"SR16": "Rook(4,4) vs", "SR28": "T(8)", "SR25": "Paley(25)", "CONE": "cone("}

def flag(name, default):
    for a in sys.argv[1:]:
        if(a.startswith('--%s=' % name)):
            return a.split('=', 1)[1]
    return default

if(__name__ == '__main__'):
    tasks = get_tasks()
    names = [a for a in sys.argv[1:] if not a.startswith('-')]
    if(names):
        keys = []
        for nm in names:
            pattern = ALIASES.get(nm, nm)
            keys += [k for k in tasks if pattern in k]
        tasks = {k: tasks[k] for k in keys}
    n_seeds = int(flag('seeds', 1))
    n_epochs = int(flag('epochs', 150))
    n_copies = int(flag('copies', 120))
    D = int(flag('D', D))
    P = int(flag('P', P))
    models = flag('models', ','.join(MODELS)).split(',')
    out = flag('out', DEFAULT_OUT)

    print("%d seeds, %d epochs, %d copies per class\n" % (n_seeds, n_epochs, n_copies))
    res = {}
    for(task, graphs) in tasks.items():
        print("=" * 82)
        print("%s   (%d-way, chance = %.1f%%)" % (task, len(graphs), 100 / len(graphs)))
        print("=" * 82)
        if(n_seeds == 1):
            print("%-12s%-14s%-14s" % ('model', 'train acc', 'test acc'))
        else:
            print("%-12s%-22s%-22s" % ('model', 'train (mean/best)', 'test (mean/best)'))
        res[task] = {}
        for m in models:
            t0 = time.time()
            trs, tes = [], []
            for s in range(n_seeds):
                tr, te, chance = run(m, graphs, n_copies=n_copies, epochs=n_epochs, seed=s)
                trs.append(tr)
                tes.append(te)
            res[task][m] = dict(train_mean=float(np.mean(trs)), train_best=float(np.max(trs)), test_mean=float(np.mean(tes)), test_best=float(np.max(tes)), train_seeds=trs, test_seeds=tes, chance=chance)
            dt = time.time() - t0
            if(n_seeds == 1):
                print("%-12s%6.1f%%       %6.1f%%      (%.0fs)" % (m, trs[0] * 100, tes[0] * 100, dt))
            else:
                print("%-12s%5.1f/%5.1f%%      %5.1f/%5.1f%%      (%.0fs)" % (m, np.mean(trs) * 100, np.max(trs) * 100, np.mean(tes) * 100, np.max(tes) * 100, dt))
        print()

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    if(out == DEFAULT_OUT):
        to_save = res
    else:
        to_save = dict(D=D, P=P, results=res)
    with open(out, 'w') as f:
        json.dump(to_save, f, indent=2)
    print('saved', out)
