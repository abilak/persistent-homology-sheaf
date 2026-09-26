"""
Download ZINC-12k (the standard 10k / 1k / 1k subset of Dwivedi et al.,
"Benchmarking GNNs"; identical to torch_geometric.datasets.ZINC(subset=True))
and write compact pickles for data_loader.data_helper.load_zinc:

    data/ZINC/ZINC_{train,val,test}.p
    each: list of {'atoms': [n], 'edge_index': (2, E), 'bonds': [E], 'y': float}

Same raw files and subset indices that PyTorch Geometric uses; no PyG needed.
Dense (C, n, n) tensors are built at load time (1 adjacency + 28 atom-type +
4 bond-type channels).

Usage:  python scripts/prepare_zinc.py
"""
import os, io, sys, ssl, shutil, pickle, zipfile, urllib.request
import certifi
import numpy as np
import torch    # the raw pickles contain torch tensors

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(project_dir, "data", "ZINC")
RAW = os.path.join(OUT, "raw")
URL = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
IDX = "https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/{}.index"
os.makedirs(RAW, exist_ok=True)
CTX = ssl.create_default_context(cafile=certifi.where())


def fetch(url):
    return urllib.request.urlopen(url, context=CTX)

zpath = os.path.join(RAW, "molecules.zip")
if not os.path.exists(os.path.join(RAW, "molecules", "train.pickle")):
    print("downloading", URL)
    with fetch(URL) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    zipfile.ZipFile(zpath).extractall(RAW)
    os.remove(zpath)

for split in ("train", "val", "test"):
    with open(os.path.join(RAW, "molecules", f"{split}.pickle"), "rb") as f:
        mols = pickle.load(f)
    with fetch(IDX.format(split)) as r:
        idx = [int(x) for x in r.read().decode().strip().split(",")]
    rows = []
    for i in idx:
        m = mols[i]
        adj = m["bond_type"]
        ei = (adj > 0).nonzero().t()
        rows.append(dict(atoms=m["atom_type"].view(-1).tolist(),
                         edge_index=ei.numpy().copy(),
                         bonds=adj[ei[0], ei[1]].view(-1).tolist(),
                         y=float(m["logP_SA_cycle_normalized"])))
    with open(os.path.join(OUT, f"ZINC_{split}.p"), "wb") as f:
        pickle.dump(rows, f)
    print(f"{split}: {len(rows)} graphs -> data/ZINC/ZINC_{split}.p")
