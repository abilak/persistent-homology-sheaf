"""
Expressivity as ACCURACY (the strongest single rebuttal experiment).

We turn the 3-WL-equivalent strongly-regular graph families into supervised
classification tasks: each non-isomorphic SRG in a family is a class; we
generate many random node-permutations of each as examples and train models to
classify them. Because every model here is permutation-invariant, it produces
identical logits for all permuted copies of a class -- so it can achieve better
than chance *training* accuracy only if it can DISTINGUISH the non-isomorphic
graphs. This is not overfitting-vs-generalization: a 3-WL-bounded model assigns
identical representations to these graphs and is therefore capped at chance on
the training set no matter how long it trains. It is a hard expressivity
ceiling, reported as accuracy.

Expected:  MLP / GCN / GIN (1-WL) and 2-IGN / PPGN (3-WL) -> chance;
           PPGN+PH -> 100%.  (GSN reported for context.)

Tasks:
  * Rook(4,4) vs Shrikhande      srg(16,6,2,2)   binary   (needs 4-cliques)
  * T(8) vs 3 Chang graphs       srg(28,12,6,4)  4-way
  * Paley(25) vs L3(5)           srg(25,12,5,6)  binary
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from easydict import EasyDict
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from srg_graphs import (rook_graph, shrikhande_graph, triangular_graph,
                        chang_graphs, paley_graph, latin_square_L3_5)
import networkx as nx


def csl(N, S):
    """Circular Skip Links graph on N vertices with skip S. 4-regular.
    Every CSL(N, S) is 1-WL-equivalent to every other CSL(N, S') --
    message-passing GNNs (GCN, GIN, MLP-on-node-features) cannot
    distinguish them (10% chance on 10-class CSL). 3-WL suffices.
    Standard CSL benchmark: N = 41, S in {2, 3, 4, 5, 6, 9, 11, 12, 13, 16}."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        G.add_edge(i, (i + 1) % N)
        G.add_edge(i, (i + S) % N)
    return G


def csl_dataset(N=41, skips=(2, 3, 4, 5, 6, 9, 11, 12, 13, 16)):
    return [csl(N, S) for S in skips]
from models.base_model import BaseModel
from models.baseline_models import BaselineModel
from utils.device import get_device

DEV = get_device()


SAFE_GATE = False   # set by --safe: topo uses a scalar gate initialized to 0,
                    # so PPGN+PH starts EXACTLY at the PPGN baseline and only
                    # engages topology if the gradient says it helps. This makes
                    # a single config behave well both where topology is needed
                    # (SR pairs) and where the baseline already suffices (CSL).


def cfg(num_classes, model_type):
    arch = dict(block_features=[64, 64], depth_of_mlp=2, new_suffix=True,
                use_topology=False, topo_hidden_dim=32, topo_max_ph_dim=2,
                topo_num_stats=16, topo_max_simplex_dim=3)  # dim3 => 4-cliques
    if model_type == 'topo':
        arch['use_topology'] = True
        if SAFE_GATE:
            arch['topo_gate_mode'] = 'scalar'
            arch['topo_gate_init'] = 0.0
    elif model_type in ('gcn', 'gin', 'mlp', 'gsn'):
        arch['baseline_type'] = model_type
    return EasyDict(dict(architecture=arch, node_labels=0,
                         num_classes=num_classes))


def norm_adj_input(G, perm):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)).astype(np.float32)
    A = A[np.ix_(perm, perm)]
    deg = np.sqrt(A.sum(0))
    dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    norm = (dinv[:, None] * A) * dinv[None, :]
    return torch.from_numpy(norm).unsqueeze(0)          # (1,1,n,n)


def make_dataset(graphs, n_copies, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, G in enumerate(graphs):
        n = G.number_of_nodes()
        for _ in range(n_copies):
            X.append(norm_adj_input(G, rng.permutation(n)))
            y.append(label)
    X = torch.cat(X, 0).unsqueeze(1) if X[0].dim() == 3 else torch.cat(X, 0)
    return X.to(DEV), torch.tensor(y, device=DEV)


def run(model_type, graphs, n_copies=120, epochs=150, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(graphs)
    X, y = make_dataset(graphs, n_copies, seed)
    N = X.shape[0]
    idx = np.random.permutation(N)
    ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]
    conf = cfg(K, model_type)
    model = (BaseModel(conf) if model_type in ('baseline', 'topo')
             else BaselineModel(conf)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, ytr = X[tr], y[tr]
    best_te = 0.0; best_tr = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tra = (model(X[tr]).argmax(1) == y[tr]).float().mean().item()
            tea = (model(X[te]).argmax(1) == y[te]).float().mean().item()
        best_tr = max(best_tr, tra); best_te = max(best_te, tea)
    return best_tr, best_te, 1.0 / K


TASKS = {
    # CSL: separates 1-WL from 3-WL. GCN/GIN/MLP (1-WL) -> chance 10%.
    # PPGN and PPGN+PH (both >=3-WL) -> 100%. This is the standard expressivity
    # benchmark from Murphy et al. 2019 / Chen et al. 2019.
    "CSL (10-way, N=41, S in {2,3,4,5,6,9,11,12,13,16})":
        csl_dataset(),
    # SRG pairs: separate 3-WL from PPGN+PH. PPGN -> chance, PPGN+PH -> 100%.
    "Rook(4,4) vs Shrikhande [srg(16,6,2,2)]":
        [rook_graph(4), shrikhande_graph()],
    "T(8) vs Chang1/2/3 [srg(28,12,6,4)]":
        [triangular_graph(8)] + chang_graphs(),
    "Paley(25) vs L3(5) [srg(25,12,5,6)]":
        [paley_graph(25), latin_square_L3_5()],
}
MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']


def _select_tasks(names):
    """Optional CLI filter: python srg_classification.py CSL SR16."""
    if not names:
        return TASKS
    aliases = {"CSL": "CSL", "SR16": "Rook", "SR28": "T(8)",
               "SR25": "Paley(25)", "SR49": "Paley(49)"}
    keys = []
    for name in names:
        needle = aliases.get(name, name)
        keys += [k for k in TASKS if needle in k]
    return {k: TASKS[k] for k in keys}


def _parse_int_flag(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return int(a.split("=", 1)[1])
    return default


if __name__ == '__main__':
    if '--safe' in sys.argv:
        SAFE_GATE = True
    tasks = _select_tasks([a for a in sys.argv[1:] if not a.startswith('-')])
    N_SEEDS = _parse_int_flag("seeds", 1)
    N_EPOCHS = _parse_int_flag("epochs", 150)
    N_COPIES = _parse_int_flag("copies", 120)
    print(f"# {N_SEEDS} seed(s), {N_EPOCHS} epochs, {N_COPIES} copies/class\n")
    results = {}
    for task, graphs in tasks.items():
        print("=" * 82)
        print(f"{task}   ({len(graphs)}-way, chance = {100/len(graphs):.1f}%)")
        print("=" * 82)
        header = f"{'model':<12}{'train (mean/best)':<22}{'test (mean/best)':<22}"
        if N_SEEDS == 1:
            header = f"{'model':<12}{'train acc':<14}{'test acc':<14}"
        print(header)
        results[task] = {}
        for m in MODELS:
            t0 = time.time()
            trs, tes = [], []
            for s in range(N_SEEDS):
                tr, te, chance = run(m, graphs, n_copies=N_COPIES,
                                     epochs=N_EPOCHS, seed=s)
                trs.append(tr); tes.append(te)
            results[task][m] = dict(train_mean=float(np.mean(trs)),
                                    train_best=float(np.max(trs)),
                                    test_mean=float(np.mean(tes)),
                                    test_best=float(np.max(tes)),
                                    train_seeds=trs, test_seeds=tes,
                                    chance=chance)
            flag = "  <-- breaks WL bound" if (m == 'topo' and max(trs) > 0.9) else ""
            if N_SEEDS == 1:
                print(f"{m:<12}{trs[0]*100:>6.1f}%       {tes[0]*100:>6.1f}%      "
                      f"({time.time()-t0:.0f}s){flag}")
            else:
                print(f"{m:<12}"
                      f"{np.mean(trs)*100:>5.1f}/{np.max(trs)*100:>5.1f}%      "
                      f"{np.mean(tes)*100:>5.1f}/{np.max(tes)*100:>5.1f}%      "
                      f"({time.time()-t0:.0f}s){flag}")
        print()
    with open('rebuttal_results/srg_classification.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("saved rebuttal_results/srg_classification.json")
