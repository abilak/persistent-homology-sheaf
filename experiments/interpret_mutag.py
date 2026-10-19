import os, sys, json, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
import networkx as nx
from scipy.stats import rankdata
import run_job, runner as plans
import data_loader.data_helper as H
from layers.topology import build_structs

# Atom counts are all different, so we can tell the element from the count
MUTAG_COUNTS = {2395: "C", 593: "O", 345: "N", 23: "I", 12: "Cl", 2: "Br", 1: "F"}
ELEM = {}

def elem(a):
    return ELEM.get(int(a), str(int(a)))

def config():
    cfg = run_job.build_config("MUTAG", "topo", dict(plans.FULL, **plans.FAST))
    cfg.num_fold = 1
    return cfg

def train(args, ckpt):
    from data_loader.data_generator import DataGenerator
    from models.base_model import build_model
    from trainers.trainer import Trainer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = config()
    if(args.epochs):
        cfg.num_epochs = args.epochs
    data = DataGenerator(cfg)
    net = build_model(cfg)
    trainer = Trainer(net, data, cfg)

    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(cfg.num_epochs):
        trainer.train_epoch()
        val_acc, _ = trainer.validate()
        if(val_acc > best_val):
            best_val = float(val_acc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    last_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if(args.final):
        state, weights = last_state, "final"
    else:
        state, weights = best_state, "best"
    torch.save(dict(state_dict=state, best_val=best_val, best_epoch=best_epoch, final_val=float(val_acc), final_epoch=cfg.num_epochs - 1, weights=weights), ckpt)
    print("trained: best fold-1 val acc {:.4f} at epoch {}".format(best_val, best_epoch))

def load_model(seed, ckpt=None):
    from models.base_model import BaseModel
    torch.manual_seed(seed)
    model = BaseModel(config())
    info = {}
    if(ckpt):
        ck = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        for k in ("weights", "best_val", "best_epoch", "final_val", "final_epoch"):
            if(k in ck):
                info[k] = ck[k]
    return model.eval(), info

@torch.no_grad()
def filt0(model, g):
    x = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)).unsqueeze(0)
    structs = build_structs([g[0]], max_dim=2)
    h = model.reg_blocks[0](x)
    st, vals = model.topo_layers[0].filtration(h, structs)[0]
    return {tuple(sorted(s)): float(vals[i]) for i, s in enumerate(st['simplices'])}

def molecule(g):
    A = (np.abs(g[0]) > 1e-6).astype(int)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    lab = g[1:].diagonal(axis1=1, axis2=2)
    if(lab.shape[0]):
        atype = lab.argmax(0)
    else:
        atype = np.zeros(A.shape[0], int)
    ring = set()
    for cyc in nx.cycle_basis(G):
        for(a, b) in zip(cyc, cyc[1:] + cyc[:1]):
            ring.add(tuple(sorted((a, b))))
    return G, atype, ring

def diagram(f):
    import gudhi
    st = gudhi.SimplexTree()
    # insert() changes the faces too, assign after
    for s in f:
        st.insert(list(s))
    for(s, v) in f.items():
        st.assign_filtration(list(s), v)
    st.make_filtration_non_decreasing()
    st.compute_persistence(persistence_dim_max=True)
    out = []
    for(b, d) in st.persistence_pairs():
        fb = st.filtration(b)
        if(d):
            fd = st.filtration(d)
            out.append((len(b) - 1, fb, fd, tuple(sorted(b)), tuple(sorted(d))))
        else:
            out.append((len(b) - 1, fb, float("inf"), tuple(sorted(b)), None))
    return out

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = rankdata(np.concatenate([pos, neg]))
    n_pos, n_neg = len(pos), len(neg)
    return float((r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def class_stats(labels, per_mol):
    y = np.asarray(labels)
    rows = []
    births, pers = [], []
    ess_het = 0
    for(gi, G, atype, ring, f) in per_mol:
        dg = diagram(f)
        fin0 = []
        ess1 = 0
        for(dim, b, d, s, _) in dg:
            if(dim == 0 and np.isfinite(d) and d - b > 1e-6):
                fin0.append((d - b, s))
            if(dim == 1 and not np.isfinite(d)):
                ess1 += 1
            if(dim == 0 and not np.isfinite(d) and elem(atype[s[0]]) != "C"):
                ess_het += 1
        for(p, s) in fin0:
            births.append(elem(atype[s[0]]))
            pers.append(p)
        max_pers = max([p for p, _ in fin0], default=0.0)
        rows.append((len(fin0), max_pers, ess1))
    R = np.array(rows, float)

    out = {}
    for(j, name) in enumerate(["finite_H0_pairs", "max_H0_persistence", "essential_H1_rings"]):
        a, b = R[y == 1, j], R[y != 1, j]
        out[name] = dict(mean_mutagenic=round(float(a.mean()), 3), mean_nonmutagenic=round(float(b.mean()), 3), auroc_mutagenic=round(auroc(a, b), 3))
    out["finite_H0_birth_element"] = {e: births.count(e) for e in sorted(set(births))}
    out["finite_H0_born_at_heteroatom"] = round(1 - births.count("C") / max(len(births), 1), 3)

    het = np.array([p for p, e in zip(pers, births) if e != "C"])
    car = np.array([p for p, e in zip(pers, births) if e == "C"])
    out["H0_persistence_mean_heteroatom_born"] = round(float(het.mean()), 4) if len(het) else None
    out["H0_persistence_mean_carbon_born"] = round(float(car.mean()), 4) if len(car) else None
    out["H0_persistence_share_heteroatom_born"] = round(float(het.sum() / (het.sum() + car.sum())), 3)
    out["essential_H0_born_at_heteroatom"] = "%d/%d" % (ess_het, len(per_mol))
    return out

def analyze(model, graphs, carbon):
    per_mol, nodes, edges = [], [], []
    for(gi, g) in enumerate(graphs):
        G, atype, ring = molecule(g)
        f = filt0(model, g)
        for v in G.nodes():
            nodes.append((gi, f[(v,)], G.degree(v), int(atype[v])))
        for(a, b) in G.edges():
            e = tuple(sorted((a, b)))
            cc = atype[a] == carbon and atype[b] == carbon
            t1, t2 = sorted((int(atype[a]), int(atype[b])))
            edges.append((gi, f[e], e in ring, cc, t1, t2))
        per_mol.append((gi, G, atype, ring, f))
    nodes = np.array(nodes, dtype=float)
    edges = np.array(edges, dtype=float)
    ef = edges[:, 1]
    er = edges[:, 2] > 0
    ecc = edges[:, 3] > 0
    emol = edges[:, 0].astype(int)

    def ring_stats(mask):
        pos, neg = ef[mask & er], ef[mask & ~er]
        rng = np.random.default_rng(0)
        mols = np.unique(emol[mask])
        boots = []
        # bootstrap over molecules
        for _ in range(1000):
            pick = rng.choice(mols, len(mols))
            sel = np.concatenate([np.where((emol == m) & mask)[0] for m in pick])
            p, n = ef[sel][er[sel]], ef[sel][~er[sel]]
            if(len(p) and len(n)):
                boots.append(auroc(p, n))
        if(neg.mean() != 0):
            ratio = round(float(pos.mean() / neg.mean()), 2)
        else:
            ratio = None
        ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        return dict(ring_mean=round(float(pos.mean()), 4), nonring_mean=round(float(neg.mean()), 4), ratio=ratio, auroc=round(auroc(pos, neg), 4), auroc_ci95=ci, n_ring=int(len(pos)), n_nonring=int(len(neg)))

    btype = ["-".join(sorted((elem(x), elem(y)))) for x, y in edges[:, 4:6].astype(int)]
    types = {}
    for t in set(btype):
        types[t] = [float(v) for v, u in zip(ef, btype) if u == t]
    no = np.array(types.get("N-O", []))
    rest = np.array([v for v, u in zip(ef, btype) if u != "N-O"])

    bond_mean = {}
    for(t, v) in sorted(types.items(), key=lambda kv: -len(kv[1])):
        bond_mean[t] = dict(n=len(v), mean=round(float(np.mean(v)), 4))
    vmean = {}
    for a in np.unique(nodes[:, 3]):
        vmean[elem(a)] = round(float(nodes[nodes[:, 3] == a, 1].mean()), 4)

    stats = dict(bond_type_mean=bond_mean, p_NO_below_other_bond=round(1 - auroc(no, rest), 4) if len(no) and len(rest) else None, all_bonds=ring_stats(np.ones_like(er)), carbon_carbon_bonds=ring_stats(ecc), vertex_degree_pearson=round(float(np.corrcoef(nodes[:, 1], nodes[:, 2])[0, 1]), 4), vertex_mean_by_element=vmean)
    return stats, per_mol, nodes, edges

def paper_style():
    import matplotlib
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})

def figures(out, per_mol, nodes, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paper_style()
    ef, er = edges[:, 1], edges[:, 2] > 0

    plt.figure(figsize=(4, 3))
    plt.hist([ef[~er], ef[er]], bins=30, label=["other bonds", "ring bonds"], density=True, alpha=.8)
    plt.xlabel("learned filtration value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/bond_filtration_hist.png", dpi=160)
    plt.close()

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-.12, .12, len(nodes))
    plt.figure(figsize=(4, 3))
    plt.scatter(nodes[:, 2] + jitter, nodes[:, 1], s=5, alpha=.3)
    plt.xlabel("vertex degree")
    plt.ylabel("learned filtration value")
    plt.tight_layout()
    plt.savefig(f"{out}/vertex_filtration_vs_degree.png", dpi=160)
    plt.close()

    ringed = sorted([m for m in per_mol if m[3]], key=lambda m: m[1].number_of_nodes())
    picked = [ringed[0], ringed[len(ringed) // 2], ringed[-1]]
    for(gi, G, atype, ring, f) in picked:
        pos = nx.kamada_kawai_layout(G)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        allv = [f[(v,)] for v in G.nodes()] + [f[tuple(sorted(e))] for e in G.edges()]
        vmin, vmax = min(allv), max(allv)
        ring_e = [e for e in G.edges() if tuple(sorted(e)) in ring]
        nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=ring_e, width=7, edge_color="k", alpha=.25)
        nx.draw_networkx_edges(G, pos, ax=ax[0], edge_color=[f[tuple(sorted(e))] for e in G.edges()], edge_cmap=plt.cm.viridis, edge_vmin=vmin, edge_vmax=vmax, width=3)
        nh = nx.draw_networkx_nodes(G, pos, ax=ax[0], node_color=[f[(v,)] for v in G.nodes()], cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, node_size=160)
        nx.draw_networkx_labels(G, pos, {v: elem(atype[v]) for v in G.nodes()}, font_size=7, font_color="w", ax=ax[0])
        fig.colorbar(nh, ax=ax[0], label="learned filtration")
        ax[0].axis("off")

        dg = [(dim, b, d) for dim, b, d, *_ in diagram(f)]
        top = max(allv) + 0.1 * (vmax - vmin + 1e-9)
        for(dim, col, mk) in [(0, "tab:blue", "o"), (1, "tab:orange", "s")]:
            fin = np.array([[b, d] for k, b, d in dg if k == dim and np.isfinite(d)])
            ess = np.array([b for k, b, d in dg if k == dim and not np.isfinite(d)])
            if(len(fin)):
                ax[1].scatter(fin[:, 0], fin[:, 1], c=col, marker=mk, label=f"$H_{dim}$ finite", s=28)
            if(len(ess)):
                ax[1].scatter(ess, np.full(len(ess), top), marker=mk, facecolors="none", edgecolors=col, label=f"$H_{dim}$ essential", s=40)
        lim = [vmin - .05, top + .05]
        ax[1].plot(lim, lim, "k--", lw=.6)
        ax[1].axhline(top, color="gray", lw=.4)
        ax[1].set_xlabel("birth")
        ax[1].set_ylabel("death (top line: essential)")
        ax[1].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{out}/molecule_{gi}.png", dpi=160)
        plt.close()

def main_fig(out, per_mol, gi=130):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paper_style()
    m = [x for x in per_mol if x[0] == gi]
    if(not m):
        return
    _, G, atype, ring, f = m[0]
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
    allv = [f[(v,)] for v in G.nodes()] + [f[tuple(sorted(e))] for e in G.edges()]
    vmin, vmax = min(allv), max(allv)
    nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=[e for e in G.edges() if tuple(sorted(e)) in ring], width=4, edge_color="k", alpha=.18)
    nx.draw_networkx_edges(G, pos, ax=ax[0], edge_color=[f[tuple(sorted(e))] for e in G.edges()], edge_cmap=plt.cm.viridis, edge_vmin=vmin, edge_vmax=vmax, width=1.6)
    nh = nx.draw_networkx_nodes(G, pos, ax=ax[0], node_color=[f[(v,)] for v in G.nodes()], cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, node_size=58)
    nx.draw_networkx_labels(G, pos, {v: elem(atype[v]) for v in G.nodes()}, font_size=4, font_color="w", ax=ax[0])
    cb = fig.colorbar(nh, ax=ax[0], fraction=0.05, pad=0.01)
    cb.ax.tick_params(labelsize=4.5, length=2)
    cb.set_label("learned filtration", fontsize=5, labelpad=1)
    ax[0].axis("off")

    top = vmax + 0.12 * (vmax - vmin)
    grp = {}
    for(dim, b, d, s, _) in diagram(f):
        if(dim > 1 or (np.isfinite(d) and d - b <= 1e-6)):
            continue
        if(dim == 0):
            if(np.isfinite(d)):
                lab = "$H_0$ comp. born at " + elem(atype[s[0]])
            else:
                lab = "$H_0$ essential, born at " + elem(atype[s[0]])
        elif(np.isfinite(d)):
            lab = "$H_1$ finite"
        else:
            lab = "$H_1$ essential (ring)"
        death = d if np.isfinite(d) else top
        grp.setdefault(lab, []).append((b, death))
    colors = {0: "tab:blue", 1: "tab:orange"}
    markers = ["o", "^", "D", "v", "P"]
    for(k, (lab, pts)) in enumerate(sorted(grp.items())):
        pts = np.array(pts)
        dim = 1 if lab.startswith("$H_1") else 0
        ess = "essential" in lab
        ax[1].scatter(pts[:, 0], pts[:, 1], marker="s" if dim else markers[k % len(markers)], s=14, facecolors="none" if ess else colors[dim], edgecolors=colors[dim] if ess else "k", linewidths=.6, label=f"{lab} ($\\times${len(pts)})")
    lim = [vmin - .03, top + .03]
    ax[1].plot(lim, lim, "k--", lw=.5)
    ax[1].axhline(top, color="gray", lw=.4)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("birth", fontsize=5, labelpad=1); ax[1].set_ylabel("death (top: essential)", fontsize=5, labelpad=1)
    ax[1].tick_params(labelsize=4.5, length=2, pad=1)
    ax[1].legend(fontsize=3.8, loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    plt.tight_layout(pad=0.2, w_pad=0.4)
    plt.savefig(f"{out}/main_molecule_{gi}.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="reuse model.pt if it's there")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="keep last epoch weights instead of best val")
    args = parser.parse_args()

    out = "results/interpret_mutag"
    if(args.seed):
        out += f"_seed{args.seed}"
    if(args.final):
        out += "_final"
    ckpt = os.path.join(out, "model.pt")
    os.makedirs(out, exist_ok=True)
    if(not args.no_train or not os.path.exists(ckpt)):
        train(args, ckpt)

    graphs, labels = H.load_dataset("MUTAG")
    lab = np.concatenate([g[1:].diagonal(axis1=1, axis2=2).argmax(0) for g in graphs])
    cnt = np.bincount(lab)
    carbon = int(cnt.argmax())
    if(sorted(cnt[cnt > 0]) == sorted(MUTAG_COUNTS)):
        for(a, c) in enumerate(cnt):
            if(c):
                ELEM[a] = MUTAG_COUNTS[int(c)]

    model, info = load_model(args.seed, ckpt)
    stats, per_mol, nodes, edges = analyze(model, graphs, carbon)
    model0, _ = load_model(args.seed)
    stats["by_class"] = class_stats(labels, per_mol)
    init_stats, mol0, _, _ = analyze(model0, graphs, carbon)
    init_stats["by_class"] = class_stats(labels, mol0)

    res = dict(model=info, carbon_label=carbon, trained=stats, untrained_control=init_stats)
    with open(f"{out}/stats.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    figures(out, per_mol, nodes, edges)
    main_fig(out, per_mol)
    print("figures and stats in %s/" % out)

if(__name__ == "__main__"):
    main()
