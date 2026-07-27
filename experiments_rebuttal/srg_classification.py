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
from models.base_model import BaseModel
from models.baseline_models import BaselineModel
from utils.device import get_device

DEV = get_device()


def cfg(num_classes, model_type):
    arch = dict(block_features=[64, 64], depth_of_mlp=2, new_suffix=True,
                use_topology=False, topo_hidden_dim=32, topo_max_ph_dim=2,
                topo_num_stats=16, topo_max_simplex_dim=3)  # dim3 => 4-cliques
    if model_type == 'topo':
        arch['use_topology'] = True
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
    "Rook(4,4) vs Shrikhande [srg(16,6,2,2)]":
        [rook_graph(4), shrikhande_graph()],
    "T(8) vs Chang1/2/3 [srg(28,12,6,4)]":
        [triangular_graph(8)] + chang_graphs(),
    "Paley(25) vs L3(5) [srg(25,12,5,6)]":
        [paley_graph(25), latin_square_L3_5()],
}
MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']

if __name__ == '__main__':
    results = {}
    for task, graphs in TASKS.items():
        print("=" * 82)
        print(f"{task}   ({len(graphs)}-way, chance = {100/len(graphs):.1f}%)")
        print("=" * 82)
        print(f"{'model':<12}{'train acc':<14}{'test acc':<14}")
        results[task] = {}
        for m in MODELS:
            t0 = time.time()
            tr, te, chance = run(m, graphs)
            results[task][m] = dict(train=tr, test=te, chance=chance)
            flag = "  <-- breaks 3-WL" if (m == 'topo' and tr > 0.9) else ""
            print(f"{m:<12}{tr*100:>6.1f}%       {te*100:>6.1f}%      "
                  f"({time.time()-t0:.0f}s){flag}")
        print()
    with open('rebuttal_results/srg_classification.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("saved rebuttal_results/srg_classification.json")
