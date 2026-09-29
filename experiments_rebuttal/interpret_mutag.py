"""
What does the learned filtration encode? (MUTAG, final PPGN+PH model)

  python experiments_rebuttal/interpret_mutag.py            # train on fold 1, analyze
  python experiments_rebuttal/interpret_mutag.py --no-train # reuse the saved model

1. Trains PPGN+PH (the paper's configuration, runner.FULL + padded batching)
   on fold 1 of MUTAG and keeps the weights of the best validation epoch.
2. Reads out the first topology layer's learned filtration f(sigma) (after
   tanh, before the max-correction) on every vertex and edge of all 188
   molecules, and reports
     - mean f of ring bonds vs other bonds, overall and within C-C bonds only
       (controls for atom type), with the AUROC of f for ring membership
       (probability that a random ring bond gets a higher value than a random
       non-ring bond) and a molecule-level bootstrap CI;
     - Pearson correlation of vertex values with degree;
     - mean vertex value per atom type;
     - the same statistics for the UNTRAINED model (same seed), as a control
       that the pattern is learned rather than built into the architecture.
3. Saves figures of three example molecules (vertices and bonds coloured by
   the learned filtration, ring bonds outlined) with their persistence
   diagrams (finite pairs and essential classes), plus summary plots.

Outputs: rebuttal_results/interpret_mutag/{stats.json, *.png}
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import run_job, runner
import data_loader.data_helper as H
from layers.topology import build_graph_structs

# --seed=N trains an independent model (training seed N) into its own directory
SEED = next((int(s.split("=")[1]) for s in sys.argv if s.startswith("--seed=")), 0)
OUT = "rebuttal_results/interpret_mutag" + (f"_seed{SEED}" if SEED else "")
CKPT = os.path.join(OUT, "model.pt")
os.makedirs(OUT, exist_ok=True)


def config():
    cfg = run_job.build_config("MUTAG", "topo", dict(runner.FULL, **runner.FAST))
    cfg.num_fold = 1
    return cfg


def train(epochs=None):
    from data_loader.data_generator import DataGenerator
    from models.model_wrapper import ModelWrapper
    from trainers.trainer import Trainer
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = config()
    if epochs:
        cfg.num_epochs = epochs
    data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(cfg.num_epochs):
        tr.train_epoch(ep)
        va, _ = tr.validate(ep)
        if va > best:
            best, best_ep = float(va), ep
            best_state = {k: v.detach().cpu().clone() for k, v in mw.model.state_dict().items()}
    torch.save(dict(state_dict=best_state, best_val=best, best_epoch=best_ep), CKPT)
    print(f"trained: best fold-1 validation accuracy {best:.4f} at epoch {best_ep}")


def load_model(trained=True):
    from models.base_model import BaseModel
    torch.manual_seed(SEED)
    model = BaseModel(config())
    info = {}
    if trained:
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        info = dict(best_val=ck["best_val"], best_epoch=ck["best_epoch"])
    return model.eval(), info


@torch.no_grad()
def layer0_filtration(model, g):
    """learned f(sigma) of the first topology layer for one graph (numpy C x n x n)"""
    x = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)).unsqueeze(0)
    structs = build_graph_structs([g[0]], max_dim=2)
    x_half = model.reg_blocks[0](x)
    st, vals = model.topo_layers[0].filtration(x_half, structs)[0]
    return {tuple(sorted(s)): float(vals[i]) for i, s in enumerate(st.simplices_list)}


def molecule(g):
    A = (np.abs(g[0]) > 1e-6).astype(int); np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    lab = g[1:].diagonal(axis1=1, axis2=2)          # (labels, n)
    atype = lab.argmax(0) if lab.shape[0] else np.zeros(A.shape[0], int)
    ring = set()
    for cyc in nx.cycle_basis(G):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            ring.add(tuple(sorted((a, b))))
    return G, atype, ring


def diagram(f):
    """persistence of the max-corrected filtration, with the birth/death simplices"""
    import gudhi
    st = gudhi.SimplexTree()
    for s in f:                                  # gudhi's insert lowers existing faces to the new
        st.insert(list(s))                       # value, so assign the learned values afterwards
    for s, v in f.items():
        st.assign_filtration(list(s), v)
    st.make_filtration_non_decreasing()
    st.compute_persistence(persistence_dim_max=True)
    out = []
    for b, d in st.persistence_pairs():
        fb = st.filtration(b); fd = st.filtration(d) if d else float("inf")
        out.append((len(b) - 1, fb, fd, tuple(sorted(b)), tuple(sorted(d)) if d else None))
    return out


def class_stats(graphs, labels, per_mol):
    """per-molecule diagram summaries vs the mutagenicity label"""
    y = np.asarray(labels)
    rows, births = [], []
    for (gi, G, atype, ring, f) in per_mol:
        dg = diagram(f)
        fin0 = [(d - b, s) for dim, b, d, s, _ in dg if dim == 0 and np.isfinite(d) and d - b > 1e-6]
        ess1 = sum(1 for dim, b, d, *_ in dg if dim == 1 and not np.isfinite(d))
        births += [elem(atype[s[0]]) for _, s in fin0]
        rows.append((len(fin0), max([p for p, _ in fin0], default=0.0), ess1))
    R = np.array(rows, float)
    out = {}
    for j, name in enumerate(["finite_H0_pairs", "max_H0_persistence", "essential_H1_rings"]):
        a, b = R[y == 1, j], R[y != 1, j]
        out[name] = dict(mean_mutagenic=round(float(a.mean()), 3), mean_nonmutagenic=round(float(b.mean()), 3),
                         auroc_mutagenic=round(auroc(a, b), 3))
    out["finite_H0_birth_element"] = {e: births.count(e) for e in sorted(set(births))}
    out["finite_H0_born_at_heteroatom"] = round(1 - births.count("C") / max(len(births), 1), 3)
    return out


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(pos = neg) via ranks"""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# MUTAG atom counts over all 188 molecules: C 2395, O 593, N 345, I 23, Cl 12, Br 2, F 1 (all distinct),
# so the element of each integer label is recovered from its count
MUTAG_COUNTS = {2395: "C", 593: "O", 345: "N", 23: "I", 12: "Cl", 2: "Br", 1: "F"}
ELEM = {}


def elem(a):
    return ELEM.get(int(a), str(int(a)))


def analyze(model, graphs, carbon):
    per_mol, nodes, edges = [], [], []
    for gi, g in enumerate(graphs):
        G, atype, ring = molecule(g)
        f = layer0_filtration(model, g)
        for v in G.nodes():
            nodes.append((gi, f[(v,)], G.degree(v), int(atype[v])))
        for a, b in G.edges():
            e = tuple(sorted((a, b)))
            cc = atype[a] == carbon and atype[b] == carbon
            edges.append((gi, f[e], e in ring, cc, *sorted((int(atype[a]), int(atype[b])))))
        per_mol.append((gi, G, atype, ring, f))
    nodes = np.array(nodes, dtype=float); edges = np.array(edges, dtype=float)
    ef, er, ecc, emol = edges[:, 1], edges[:, 2] > 0, edges[:, 3] > 0, edges[:, 0].astype(int)

    def ring_stats(mask):
        pos, neg = ef[mask & er], ef[mask & ~er]
        rng = np.random.default_rng(0); mols = np.unique(emol[mask]); boots = []
        for _ in range(1000):                        # molecule-level bootstrap
            pick = rng.choice(mols, len(mols))
            sel = np.concatenate([np.where((emol == m) & mask)[0] for m in pick])
            p, n = ef[sel][er[sel]], ef[sel][~er[sel]]
            if len(p) and len(n):
                boots.append(auroc(p, n))
        return dict(ring_mean=round(float(pos.mean()), 4), nonring_mean=round(float(neg.mean()), 4),
                    ratio=round(float(pos.mean() / neg.mean()), 2) if neg.mean() != 0 else None,
                    auroc=round(auroc(pos, neg), 4),
                    auroc_ci95=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                    n_ring=int(len(pos)), n_nonring=int(len(neg)))
    btype = ["-".join(sorted((elem(x), elem(y)))) for x, y in edges[:, 4:6].astype(int)]
    types = {t: [float(v) for v, u in zip(ef, btype) if u == t] for t in set(btype)}
    no = np.array(types.get("N-O", [])); rest = np.array([v for v, u in zip(ef, btype) if u != "N-O"])
    stats = dict(
        bond_type_mean={t: dict(n=len(v), mean=round(float(np.mean(v)), 4))
                        for t, v in sorted(types.items(), key=lambda kv: -len(kv[1]))},
        p_NO_below_other_bond=round(1 - auroc(no, rest), 4) if len(no) and len(rest) else None,
        all_bonds=ring_stats(np.ones_like(er)),
        carbon_carbon_bonds=ring_stats(ecc),
        vertex_degree_pearson=round(float(np.corrcoef(nodes[:, 1], nodes[:, 2])[0, 1]), 4),
        vertex_mean_by_element={elem(a):
                              round(float(nodes[nodes[:, 3] == a, 1].mean()), 4)
                              for a in np.unique(nodes[:, 3])},
    )
    return stats, per_mol, nodes, edges


def figures(per_mol, nodes, edges, carbon, labels):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gudhi
    ef, er = edges[:, 1], edges[:, 2] > 0
    plt.figure(figsize=(4, 3))
    plt.hist([ef[~er], ef[er]], bins=30, label=["other bonds", "ring bonds"], density=True, alpha=.8)
    plt.xlabel("learned filtration value"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/bond_filtration_hist.png", dpi=160); plt.close()
    rng = np.random.default_rng(0)
    plt.figure(figsize=(4, 3))
    plt.scatter(nodes[:, 2] + rng.uniform(-.12, .12, len(nodes)), nodes[:, 1], s=5, alpha=.3)
    plt.xlabel("vertex degree"); plt.ylabel("learned filtration value"); plt.tight_layout()
    plt.savefig(f"{OUT}/vertex_filtration_vs_degree.png", dpi=160); plt.close()
    # three example molecules with rings, of increasing size
    with_rings = sorted([m for m in per_mol if m[3]], key=lambda m: m[1].number_of_nodes())
    picks = [with_rings[0], with_rings[len(with_rings) // 2], with_rings[-1]]
    for gi, G, atype, ring, f in picks:
        pos = nx.kamada_kawai_layout(G)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        allv = [f[(v,)] for v in G.nodes()] + [f[tuple(sorted(e))] for e in G.edges()]
        vmin, vmax = min(allv), max(allv)
        ecol = [f[tuple(sorted(e))] for e in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=[e for e in G.edges() if tuple(sorted(e)) in ring],
                               width=7, edge_color="k", alpha=.25)
        eh = nx.draw_networkx_edges(G, pos, ax=ax[0], edge_color=ecol, edge_cmap=plt.cm.viridis,
                                    edge_vmin=vmin, edge_vmax=vmax, width=3)
        nh = nx.draw_networkx_nodes(G, pos, ax=ax[0], node_color=[f[(v,)] for v in G.nodes()],
                                    cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, node_size=160)
        nx.draw_networkx_labels(G, pos, {v: elem(atype[v]) for v in G.nodes()},
                                font_size=7, font_color="w", ax=ax[0])
        fig.colorbar(nh, ax=ax[0], label="learned filtration")
        ax[0].set_title(f"molecule {gi} (class {int(labels[gi])})"); ax[0].axis("off")
        dg = [(dim, (b, d)) for dim, b, d, *_ in diagram(f)]
        top = max(allv) + 0.1 * (vmax - vmin + 1e-9)
        for dim, col, mk in [(0, "tab:blue", "o"), (1, "tab:orange", "s")]:
            fin = np.array([[b, d] for dd, (b, d) in dg if dd == dim and np.isfinite(d)])
            ess = np.array([b for dd, (b, d) in dg if dd == dim and not np.isfinite(d)])
            if len(fin):
                ax[1].scatter(fin[:, 0], fin[:, 1], c=col, marker=mk, label=f"$H_{dim}$ finite", s=28)
            if len(ess):
                ax[1].scatter(ess, np.full(len(ess), top), marker=mk, facecolors="none", edgecolors=col,
                              label=f"$H_{dim}$ essential", s=40)
        lim = [vmin - .05, top + .05]
        ax[1].plot(lim, lim, "k--", lw=.6); ax[1].axhline(top, color="gray", lw=.4)
        ax[1].set_xlabel("birth"); ax[1].set_ylabel("death (top line: essential)")
        ax[1].set_title("persistence diagram"); ax[1].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{OUT}/molecule_{gi}.png", dpi=160); plt.close()


def main_figure(per_mol, labels, gi=130):
    """compact two-panel figure for the main text: one molecule and its annotated diagram"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = [x for x in per_mol if x[0] == gi]
    if not m:
        return
    _, G, atype, ring, f = m[0]
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    allv = [f[(v,)] for v in G.nodes()] + [f[tuple(sorted(e))] for e in G.edges()]
    vmin, vmax = min(allv), max(allv)
    nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=[e for e in G.edges() if tuple(sorted(e)) in ring],
                           width=6, edge_color="k", alpha=.18)
    nx.draw_networkx_edges(G, pos, ax=ax[0], edge_color=[f[tuple(sorted(e))] for e in G.edges()],
                           edge_cmap=plt.cm.viridis, edge_vmin=vmin, edge_vmax=vmax, width=2.5)
    nh = nx.draw_networkx_nodes(G, pos, ax=ax[0], node_color=[f[(v,)] for v in G.nodes()],
                                cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, node_size=150)
    nx.draw_networkx_labels(G, pos, {v: elem(atype[v]) for v in G.nodes()}, font_size=6.5,
                            font_color="w", ax=ax[0])
    cb = fig.colorbar(nh, ax=ax[0], fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6)
    cb.set_label("learned filtration", fontsize=7)
    ax[0].axis("off")
    dg = diagram(f)
    top = vmax + 0.12 * (vmax - vmin)
    grp = {}
    for dim, b, d, s, _ in dg:
        if dim > 1 or (np.isfinite(d) and d - b <= 1e-6):
            continue
        lab = ("$H_0$ " + (("comp. born at " if np.isfinite(d) else "essential, born at ") + elem(atype[s[0]]))) if dim == 0 \
            else "$H_1$ " + ("essential (ring)" if not np.isfinite(d) else "finite")
        grp.setdefault(lab, []).append((b, d if np.isfinite(d) else top))
    style = {0: ("o", "tab:blue"), 1: ("s", "tab:orange")}
    mk = ["o", "^", "D", "v", "P"]
    for k, (lab, pts) in enumerate(sorted(grp.items())):
        pts = np.array(pts); dim = 1 if lab.startswith("$H_1") else 0
        ess = "essential" in lab
        ax[1].scatter(pts[:, 0], pts[:, 1], marker="s" if dim else mk[k % len(mk)], s=34,
                      facecolors="none" if ess else style[dim][1], edgecolors=style[dim][1] if ess else "k",
                      linewidths=.8, label=f"{lab} ($\\times${len(pts)})")
    lim = [vmin - .03, top + .03]
    ax[1].plot(lim, lim, "k--", lw=.5); ax[1].axhline(top, color="gray", lw=.4)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("birth", fontsize=7); ax[1].set_ylabel("death (top line: essential)", fontsize=7)
    ax[1].tick_params(labelsize=6); ax[1].legend(fontsize=5.5, loc="lower right", frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/main_molecule_{gi}.pdf"); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="default: the dataset schedule (200)")
    args = ap.parse_args()
    if not args.no_train or not os.path.exists(CKPT):
        train(args.epochs)
    graphs, labels = H.load_dataset("MUTAG")
    lab = np.concatenate([g[1:].diagonal(axis1=1, axis2=2).argmax(0) for g in graphs])
    carbon = int(np.bincount(lab).argmax())          # most frequent label = carbon
    cnt = np.bincount(lab)
    if sorted(cnt[cnt > 0]) == sorted(MUTAG_COUNTS):
        ELEM.update({a: MUTAG_COUNTS[int(c)] for a, c in enumerate(cnt) if c})
    model, info = load_model(trained=True)
    stats, per_mol, nodes, edges = analyze(model, graphs, carbon)
    init_model, _ = load_model(trained=False)
    stats["by_class"] = class_stats(graphs, labels, per_mol)
    init_stats, init_mol, *_ = analyze(init_model, graphs, carbon)
    init_stats["by_class"] = class_stats(graphs, labels, init_mol)
    res = dict(model=info, carbon_label=carbon, trained=stats, untrained_control=init_stats)
    json.dump(res, open(f"{OUT}/stats.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    figures(per_mol, nodes, edges, carbon, labels)
    main_figure(per_mol, labels)
    print(f"figures and stats in {OUT}/")


if __name__ == "__main__":
    main()
