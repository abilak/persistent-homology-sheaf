import os
import sys
import gzip
import ssl
import shutil
import pickle
import zipfile
import urllib.request
import numpy as np
import certifi
import torch  # the raw zinc pickles have torch tensors in them

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ddir = os.path.join(ROOT, "data")

TU_URL = "https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?dl=1"
ZINC_URL = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
ZINC_IDX = "https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/{}.index"
HIV_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"

def ssl_ctx():
    # certifi bc the python.org mac builds don't see the system certs
    return ssl.create_default_context(cafile=certifi.where())

def download(url, path):
    print("Downloading", url)
    with urllib.request.urlopen(url, context=ssl_ctx()) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path

def get_tu():
    os.makedirs(ddir, exist_ok=True)
    zp = download(TU_URL, os.path.join(ddir, "benchmark_graphs.zip"))
    zipfile.ZipFile(zp).extractall(ddir)
    os.unlink(zp)

def prep_zinc():
    out_dir = os.path.join(ddir, "ZINC")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if(not os.path.exists(os.path.join(raw_dir, "molecules", "train.pickle"))):
        zpath = download(ZINC_URL, os.path.join(raw_dir, "molecules.zip"))
        zipfile.ZipFile(zpath).extractall(raw_dir)
        os.remove(zpath)

    ctx = ssl_ctx()
    for split in ("train", "val", "test"):
        with open(os.path.join(raw_dir, "molecules", split + ".pickle"), "rb") as f:
            mols = pickle.load(f)
        with urllib.request.urlopen(ZINC_IDX.format(split), context=ctx) as r:
            txt = r.read().decode().strip()
        idx = [int(i) for i in txt.split(",")]

        rows = []
        for i in idx:
            mol = mols[i]
            adj = mol["bond_type"]
            ei = (adj > 0).nonzero().t()
            bonds = adj[ei[0], ei[1]].view(-1).tolist()
            row = dict(atoms=mol["atom_type"].view(-1).tolist(), edge_index=ei.numpy().copy(), bonds=bonds, y=float(mol["logP_SA_cycle_normalized"]))
            rows.append(row)
        with open(os.path.join(out_dir, "ZINC_{}.p".format(split)), "wb") as f:
            pickle.dump(rows, f)
        print("wrote %d %s graphs to data/ZINC/ZINC_%s.p" % (len(rows), split, split))

def load_csv(raw, name):
    a = np.loadtxt(gzip.open(os.path.join(raw, name)), delimiter=",", dtype=np.int64)
    if(a.ndim == 1):
        a = a.reshape(len(a), -1)
    return a

def vocab(cols):
    maps = []
    off = 0
    for c in range(cols.shape[1]):
        vals = np.unique(cols[:, c])
        if(len(vals) < 2):
            maps.append(None)
            continue
        lut = np.full(vals.max() + 1, -1, dtype=np.int64)
        lut[vals] = np.arange(len(vals)) + off
        maps.append(lut)
        off += len(vals)
    return maps, off

def prep_molhiv():
    out_dir = os.path.join(ddir, "MOLHIV")
    raw = os.path.join(out_dir, "hiv")
    os.makedirs(out_dir, exist_ok=True)
    if(not os.path.exists(os.path.join(raw, "raw", "edge.csv.gz"))):
        zpath = download(HIV_URL, os.path.join(out_dir, "hiv.zip"))
        zipfile.ZipFile(zpath).extractall(out_dir)
        os.remove(zpath)

    nfeat = load_csv(raw, "raw/node-feat.csv.gz")
    efeat = load_csv(raw, "raw/edge-feat.csv.gz")
    edges = load_csv(raw, "raw/edge.csv.gz")
    nnodes = load_csv(raw, "raw/num-node-list.csv.gz").reshape(-1)
    ne = load_csv(raw, "raw/num-edge-list.csv.gz").reshape(-1)
    y = load_csv(raw, "raw/graph-label.csv.gz").reshape(-1)
    split = {s: load_csv(raw, f"split/scaffold/{s}.csv.gz").reshape(-1) for s in ("train", "valid", "test")}

    amaps, n_atom = vocab(nfeat)
    bmaps, n_bond = vocab(efeat)
    C = 1 + n_atom + n_bond
    atom_ch = []
    for(c, m) in enumerate(amaps):
        if(m is not None):
            atom_ch.append(1 + m[nfeat[:, c]])
    atom_ch = np.stack(atom_ch, 1)
    bond_ch = []
    for(c, m) in enumerate(bmaps):
        if(m is not None):
            bond_ch.append(1 + n_atom + m[efeat[:, c]])
    bond_ch = np.stack(bond_ch, 1)

    graphs = []
    noff, eoff = 0, 0
    for g in range(len(nnodes)):
        n, e = int(nnodes[g]), int(ne[g])
        src = edges[eoff:eoff + e, 0]
        dst = edges[eoff:eoff + e, 1]
        b = bond_ch[eoff:eoff + e]
        graphs.append(dict(n=n, src=np.concatenate([src, dst]).astype(np.int32), dst=np.concatenate([dst, src]).astype(np.int32), atom_ch=atom_ch[noff:noff + n].astype(np.int16), bond_ch=np.concatenate([b, b]).astype(np.int16), y=int(y[g])))
        noff += n
        eoff += e
    assert noff == len(nfeat) and eoff == len(edges)

    with open(os.path.join(out_dir, "molhiv.p"), "wb") as f:
        pickle.dump(dict(graphs=graphs, split=split, C=C), f)
    sizes = ", ".join(f"{k} {len(v)}" for k, v in split.items())
    print(f"{len(graphs)} graphs, C = {C}, split sizes {sizes}, saved data/MOLHIV/molhiv.p")

if(__name__ == "__main__"):
    what = sys.argv[1] if len(sys.argv) > 1 else "tu"
    if(what == "tu"):
        get_tu()
    elif(what == "zinc"):
        prep_zinc()
    elif(what == "molhiv"):
        prep_molhiv()
    else:
        print("usage: python utils/get_data.py [tu|zinc|molhiv]")
        sys.exit(1)
