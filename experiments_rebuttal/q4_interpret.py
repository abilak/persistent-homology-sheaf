"""
Q4: Do the learned filtration functions permit an interpretation?

We load a trained MUTAG topology model and read out its layer-0 learned
filtration f(sigma) = rho(mean_{(i,j) in sigmaxsigma} X^{(1/2)}_{ij}) for every
vertex and edge of each molecule. We then ask whether these learned values track
interpretable chemical/structural quantities:
  * node filtration vs node degree (Pearson r)
  * node filtration vs atom type (per-type means)
  * edge filtration vs ring membership (in-cycle vs not)
and we plot the learned filtration on individual small molecules plus their
persistence diagrams.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data_loader.data_helper as helper
from models.model_wrapper import ModelWrapper
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config

OUT = 'rebuttal_results/q4'
os.makedirs(OUT, exist_ok=True)
# In this MUTAG encoding, node-label index 2 is the dominant type (2395 of
# ~3371 atoms) = carbon; the remaining indices are the heteroatoms/substituents
# (N, O, halogens). We label carbon explicitly and the rest by index to avoid
# asserting an unverified atom mapping; the interpretable findings (ring bonds,
# degree) do not depend on the naming.
ATOM = {2: 'C'}
def atom_name(i):
    return ATOM.get(int(i), f'L{int(i)}')


def load_model():
    cfg = build_config('MUTAG', 'topo')
    mw = ModelWrapper(cfg, None)
    ckpt = torch.load('rebuttal_results/mutag_topo_model.tar', map_location='cpu')
    mw.model.load_state_dict(ckpt['state_dict'])
    mw.model.eval()
    return mw.model, ckpt.get('best_val'), cfg


def graph_from_tensor(g):
    """g: C x M x M numpy. Returns nx graph, atom-type per node."""
    A = (np.abs(g[0]) > 1e-6).astype(int)
    np.fill_diagonal(A, 0)
    M = A.shape[0]
    G = nx.from_numpy_array(A)
    atoms = g[1:, :, :]                          # label channels
    atype = atoms.diagonal(axis1=1, axis2=2).argmax(0) if atoms.shape[0] else \
        np.zeros(M, int)
    return G, atype


@torch.no_grad()
def layer0_filtration(model, g_tensor):
    """Return dict simplex->learned layer-0 filtration value for one graph."""
    x = torch.from_numpy(g_tensor).unsqueeze(0).float()
    from layers.topology import build_graph_structs
    structs = build_graph_structs([g_tensor[0]], max_dim=2)
    x0 = model.reg_blocks[0](x)
    st, vals = model.topo_layers[0].filtration(x0, structs)[0]
    return {tuple(sorted(s)): float(vals[i])
            for i, s in enumerate(st.simplices_list)}


def main():
    model, best_val, cfg = load_model()
    print(f"loaded model best_val={best_val}")
    graphs, labels = helper.load_dataset('MUTAG')

    node_filt_all, node_deg_all, node_atom_all = [], [], []
    edge_filt_all, edge_inring_all = [], []
    per_graph = []
    for gi in range(len(graphs)):
        g = graphs[gi]
        G, atype = graph_from_tensor(g)
        filt = layer0_filtration(model, g)
        cyc_edges = set()
        for cyc in nx.cycle_basis(G):
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                cyc_edges.add(tuple(sorted((a, b))))
        nf = {v: filt.get((v,), np.nan) for v in G.nodes()}
        for v in G.nodes():
            node_filt_all.append(nf[v]); node_deg_all.append(G.degree(v))
            node_atom_all.append(int(atype[v]))
        for (a, b) in G.edges():
            e = tuple(sorted((a, b)))
            if e in filt:
                edge_filt_all.append(filt[e])
                edge_inring_all.append(e in cyc_edges)
        per_graph.append((gi, G, atype, nf, filt, cyc_edges, int(labels[gi])))

    nf = np.array(node_filt_all); nd = np.array(node_deg_all, float)
    na = np.array(node_atom_all)
    ef = np.array(edge_filt_all); er = np.array(edge_inring_all)
    r_deg = float(np.corrcoef(nf, nd)[0, 1])
    atom_means = {atom_name(a): (float(nf[na == a].mean()), int((na == a).sum()))
                  for a in sorted(set(na)) if (na == a).sum() > 0}
    ring_mean = float(ef[er].mean()) if er.any() else float('nan')
    nonring_mean = float(ef[~er].mean()) if (~er).any() else float('nan')

    res = dict(best_val=best_val, n_graphs=len(graphs),
               node_filt_vs_degree_pearson=round(r_deg, 4),
               atom_type_mean_filtration=atom_means,
               edge_filtration_ring_mean=round(ring_mean, 4),
               edge_filtration_nonring_mean=round(nonring_mean, 4),
               node_filt_mean=round(float(nf.mean()), 4),
               node_filt_std=round(float(nf.std()), 4))
    with open(f'{OUT}/correlations.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- plot: filtration vs degree ----
    plt.figure(figsize=(4, 3))
    plt.scatter(nd + np.random.uniform(-.12, .12, len(nd)), nf, s=6, alpha=.3)
    plt.xlabel('node degree'); plt.ylabel('learned filtration f(v)')
    plt.title(f'Node filtration vs degree (r={r_deg:.2f})')
    plt.tight_layout(); plt.savefig(f'{OUT}/filtration_vs_degree.png', dpi=140)
    plt.close()

    # ---- plot: filtration by atom type ----
    plt.figure(figsize=(4, 3))
    keys = list(atom_means.keys())
    plt.bar(keys, [atom_means[k][0] for k in keys])
    plt.ylabel('mean learned filtration'); plt.xlabel('atom type')
    plt.title('Learned filtration by atom type')
    plt.tight_layout(); plt.savefig(f'{OUT}/filtration_by_atom.png', dpi=140)
    plt.close()

    # ---- example molecules: smallest few with rings ----
    sizes = sorted(per_graph, key=lambda t: t[1].number_of_nodes())
    picks = [t for t in sizes if t[5]][:3]  # have rings
    for gi, G, atype, nfv, filt, cyc, lab in picks:
        vals = np.array([nfv[v] for v in G.nodes()])
        pos = nx.spring_layout(G, seed=0)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ec = ['crimson' if tuple(sorted(e)) in cyc else '#888' for e in G.edges()]
        nc = nx.draw_networkx_nodes(G, pos, node_color=vals, cmap='viridis',
                                    node_size=320, ax=ax[0])
        nx.draw_networkx_edges(G, pos, edge_color=ec, width=2, ax=ax[0])
        nx.draw_networkx_labels(G, pos, {v: atom_name(atype[v]) for v in G.nodes()},
                                font_size=8, font_color='white', ax=ax[0])
        fig.colorbar(nc, ax=ax[0], label='learned filtration f(v)')
        ax[0].set_title(f'MUTAG #{gi} (label={lab}); red=ring bond')
        ax[0].axis('off')
        # persistence diagram (layer-0 filtration on clique complex)
        import gudhi
        st = gudhi.SimplexTree()
        for s, v in filt.items():
            st.insert(list(s), filtration=v)
        st.make_filtration_non_decreasing()
        st.compute_persistence()
        for dim, col in [(0, 'tab:blue'), (1, 'tab:orange')]:
            pts = np.array([[b, d] for (dd, (b, d)) in st.persistence()
                            if dd == dim and d != float('inf')])
            if len(pts):
                ax[1].scatter(pts[:, 0], pts[:, 1], c=col, label=f'H{dim}', s=30)
        lims = [min(vals) - .1, max(vals) + .1]
        ax[1].plot(lims, lims, 'k--', lw=.6)
        ax[1].set_xlabel('birth'); ax[1].set_ylabel('death')
        ax[1].set_title('Persistence diagram'); ax[1].legend()
        plt.tight_layout(); plt.savefig(f'{OUT}/molecule_{gi}.png', dpi=140)
        plt.close()
    print(f"wrote figures to {OUT}/")


if __name__ == '__main__':
    main()
