import os
import pickle
import numpy as np

NUM_LABELS = {'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38, 'PROTEINS': 3, 'PTC': 22, 'ENZYMES': 3, 'ZINC': 32, 'MOLHIV': 98}
NUM_CLASSES = {'IMDBBINARY': 2, 'IMDBMULTI': 3, 'MUTAG': 2, 'NCI1': 2, 'NCI109': 2, 'PROTEINS': 2, 'PTC': 2, 'ENZYMES': 6, 'ZINC': 1, 'MOLHIV': 2}
LEARNING_RATES = {'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0005, 'NCI1': 0.0001, 'NCI109': 0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001, 'ENZYMES': 0.001, 'ZINC': 0.001, 'MOLHIV': 0.001}
DECAY_RATES = {'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 0.5, 'NCI1': 0.75, 'NCI109': 0.75, 'PROTEINS': 0.5, 'PTC': 1.0, 'ENZYMES': 0.5, 'ZINC': 0.75, 'MOLHIV': 0.75}
EPOCHS = {'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 200, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 180, 'ENZYMES': 100, 'ZINC': 300, 'MOLHIV': 100}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, 'data')

ZINC_ATOM_TYPES = 28
ZINC_BOND_TYPES = 4

def load_dataset(ds_name):
    path = os.path.join(DATA, 'benchmark_graphs', ds_name, ds_name + '.txt')
    nl = NUM_LABELS[ds_name]
    graphs, labels = [], []
    with open(path) as f:
        ng = int(f.readline().rstrip().split(' ')[0])
        for _ in range(ng):
            meta = f.readline().rstrip().split(' ')
            n = int(meta[0])
            labels.append(int(meta[1]))
            curr_graph = np.zeros((n, n, nl + 1), dtype=np.float32)
            for j in range(n):
                vertex = f.readline().rstrip().split(' ')
                if(nl != 0):
                    curr_graph[j, j, int(vertex[0]) + 1] = 1.
                for k in range(2, len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            graphs.append(normalize_graph(curr_graph))
    graphs = np.array(graphs, dtype=object)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    return graphs, np.array(labels)

def normalize_graph(g):
    # D^-1/2 A D^-1/2, same as the original ppgn code
    adj = g[:, :, 0]
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg != 0)
    D = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(D, adj), D), axis=2)
    spread = np.multiply(np.ones(g.shape, dtype=np.float32), norm_adj)
    lab = np.append(np.zeros((g.shape[0], g.shape[1], 1)), g[:, :, 1:], axis=2)
    return np.add(spread, lab)

def get_train_val_indexes(fold, ds_name):
    d = os.path.join(DATA, 'benchmark_graphs', ds_name, '10fold_idx')
    with open(os.path.join(d, 'train_idx-%d.txt' % fold)) as f:
        train_idx = [int(l.rstrip()) for l in f]
    with open(os.path.join(d, 'test_idx-%d.txt' % fold)) as f:
        test_idx = [int(l.rstrip()) for l in f]
    return train_idx, test_idx

def load_zinc():
    out = []
    C = 1 + ZINC_ATOM_TYPES + ZINC_BOND_TYPES
    for split in ('train', 'val', 'test'):
        with open(os.path.join(DATA, 'ZINC', 'ZINC_%s.p' % split), 'rb') as f:
            data = pickle.load(f)
        graphs = np.empty(len(data), dtype=object)
        labels = np.zeros((len(data), 1), dtype=np.float32)
        for(i, g) in enumerate(data):
            n = len(g['atoms'])
            x = np.zeros((C, n, n), dtype=np.float32)
            src, dst = g['edge_index']
            x[0, src, dst] = 1.0
            x[1 + np.asarray(g['atoms']), np.arange(n), np.arange(n)] = 1.0
            bonds = np.asarray(g['bonds'])
            ok = (bonds >= 1) & (bonds <= ZINC_BOND_TYPES)
            x[1 + ZINC_ATOM_TYPES + bonds[ok] - 1, np.asarray(src)[ok], np.asarray(dst)[ok]] = 1.0
            graphs[i] = x
            labels[i, 0] = g['y']
        out += [graphs, labels]
    return tuple(out)

def densify(g):
    # molhiv is way too big dense
    n, C, src, dst, atom_ch, bond_ch = g
    x = np.zeros((C, n, n), dtype=np.float32)
    x[0, src, dst] = 1.0
    d = np.arange(n)
    for c in range(atom_ch.shape[1]):
        x[atom_ch[:, c], d, d] = 1.0
    for c in range(bond_ch.shape[1]):
        x[bond_ch[:, c], src, dst] = 1.0
    return x

def nnodes(g):
    return g[0] if isinstance(g, tuple) else g.shape[1]

def load_molhiv():
    with open(os.path.join(DATA, 'MOLHIV', 'molhiv.p'), 'rb') as f:
        d = pickle.load(f)
    gs = np.empty(len(d['graphs']), dtype=object)
    ys = np.zeros(len(d['graphs']), dtype=np.int64)
    for(i, g) in enumerate(d['graphs']):
        gs[i] = (g['n'], d['C'], g['src'].astype(np.int64), g['dst'].astype(np.int64), g['atom_ch'].astype(np.int64), g['bond_ch'].astype(np.int64))
        ys[i] = g['y']
    out = []
    for s in ('train', 'valid', 'test'):
        idx = d['split'][s]
        out += [gs[idx], ys[idx]]
    return tuple(out)

def group_same_size(graphs, labels):
    sizes = [g.shape[1] for g in graphs]
    order = np.argsort(sizes)
    graphs, labels = graphs[order], labels[order]
    r_graphs, r_labels, cur = [], [], []
    start, size = 0, graphs[0].shape[1]
    for i in range(len(graphs)):
        if(graphs[i].shape[1] != size):
            r_graphs.append(np.concatenate(cur, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start, cur, size = i, [], graphs[i].shape[1]
        cur.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(cur, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels

def split_to_batches(graphs, labels, size):
    r_graphs, r_labels = [], []
    for k in range(len(graphs)):
        cuts = list(range(size, graphs[k].shape[0], size))
        r_graphs += np.split(graphs[k], cuts)
        r_labels += np.split(labels[k], cuts)
    g, l = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
    g[:] = r_graphs
    l[:] = r_labels
    return g, l
