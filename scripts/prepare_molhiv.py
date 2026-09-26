"""
Download ogbg-molhiv (OGB raw CSV release; no `ogb` package needed) and write
a compact pickle for data_loader.data_helper.load_molhiv:

    data/MOLHIV/molhiv.p = dict(
        graphs=[dict(n, src, dst, atom_ch (n, k), bond_ch (E, l), y)],
        split=dict(train=..., valid=..., test=...),      # OGB scaffold split
        C=<number of input channels>)

Channels: 0 = adjacency; then one-hot atom features and one-hot bond
features, restricted to the values that occur in the dataset (columns with a
single value are dropped). Undirected edges are stored in both directions, as
OGB does for molhiv (add_inverse_edge). Dense (C, n, n) tensors are built per
batch, so the full dataset never has to be held densely in memory.

Usage:  python scripts/prepare_molhiv.py
"""
import os, gzip, ssl, shutil, pickle, zipfile, urllib.request
import numpy as np
import certifi

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(project_dir, "data", "MOLHIV")
URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"
os.makedirs(OUT, exist_ok=True)
raw = os.path.join(OUT, "hiv")
if not os.path.exists(os.path.join(raw, "raw", "edge.csv.gz")):
    zpath = os.path.join(OUT, "hiv.zip")
    ctx = ssl.create_default_context(cafile=certifi.where())
    print("downloading", URL)
    with urllib.request.urlopen(URL, context=ctx) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    zipfile.ZipFile(zpath).extractall(OUT)
    os.remove(zpath)


def load(name, dtype=np.int64):
    a = np.loadtxt(gzip.open(os.path.join(raw, name)), delimiter=",", dtype=dtype)
    return a.reshape(len(a), -1) if a.ndim == 1 else a


nf = load("raw/node-feat.csv.gz")
ef = load("raw/edge-feat.csv.gz")
ed = load("raw/edge.csv.gz")
nn = load("raw/num-node-list.csv.gz").reshape(-1)
ne = load("raw/num-edge-list.csv.gz").reshape(-1)
y = load("raw/graph-label.csv.gz").reshape(-1)
split = {s: load(f"split/scaffold/{s}.csv.gz").reshape(-1) for s in ("train", "valid", "test")}


def vocab(cols):
    """per column: value -> channel offset, dropping single-valued columns"""
    maps, off = [], 0
    for c in range(cols.shape[1]):
        vals = np.unique(cols[:, c])
        if len(vals) < 2:
            maps.append(None); continue
        lut = np.full(vals.max() + 1, -1, dtype=np.int64)
        lut[vals] = np.arange(len(vals)) + off
        maps.append(lut); off += len(vals)
    return maps, off


atom_maps, n_atom = vocab(nf)
bond_maps, n_bond = vocab(ef)
C = 1 + n_atom + n_bond
atom_ch = np.stack([1 + m[nf[:, c]] for c, m in enumerate(atom_maps) if m is not None], 1)
bond_ch = np.stack([1 + n_atom + m[ef[:, c]] for c, m in enumerate(bond_maps) if m is not None], 1)

graphs, no, eo = [], 0, 0
for g in range(len(nn)):
    n, e = int(nn[g]), int(ne[g])
    s, d = ed[eo:eo + e, 0], ed[eo:eo + e, 1]          # local node indices
    b = bond_ch[eo:eo + e]
    graphs.append(dict(n=n,
                       src=np.concatenate([s, d]).astype(np.int32),
                       dst=np.concatenate([d, s]).astype(np.int32),
                       atom_ch=atom_ch[no:no + n].astype(np.int16),
                       bond_ch=np.concatenate([b, b]).astype(np.int16),
                       y=int(y[g])))
    no += n; eo += e
assert no == len(nf) and eo == len(ed)
with open(os.path.join(OUT, "molhiv.p"), "wb") as f:
    pickle.dump(dict(graphs=graphs, split=split, C=C), f)
print(f"{len(graphs)} graphs, C = {C} channels, split sizes "
      + ", ".join(f"{k} {len(v)}" for k, v in split.items()) + " -> data/MOLHIV/molhiv.p")
