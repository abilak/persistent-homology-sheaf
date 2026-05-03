"""
Download ZINC-10k (subset=True) from PyTorch Geometric and convert to
pickle format compatible with this project's data loader.

Output: data/ZINC/ZINC_train.p, data/ZINC/ZINC_val.p, data/ZINC/ZINC_test.p

Each pickle file is a list of dicts with keys:
  - 'graph': np.ndarray of shape (num_nodes, num_nodes, num_channels) in HWC format
             Channels: [adj(1), node_onehot(28), edge_onehot(4)] = 33 total
  - 'y': float (constrained solubility)

Usage:
    python scripts/prepare_zinc.py
"""

import os
import sys
import pickle
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)

try:
    from torch_geometric.datasets import ZINC
except ImportError:
    print("torch_geometric is required. Install with:")
    print("  pip install torch-geometric")
    sys.exit(1)

NUM_ATOM_TYPES = 28  # ZINC node labels are integers 0-27
NUM_BOND_TYPES = 4   # ZINC edge_attr values are 1, 2, 3 (single/double/triple bonds) — we use 4 slots

OUTPUT_DIR = os.path.join(project_dir, "data", "ZINC")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_split(split_name):
    """Convert one split (train/val/test) to pickle format."""
    dataset = ZINC(root=os.path.join(OUTPUT_DIR, "raw"), subset=True, split=split_name)
    data_list = []

    for g in dataset:
        num_nodes = g.num_nodes

        # Adjacency channel
        adj = np.zeros((num_nodes, num_nodes, 1), dtype=np.float32)
        edge_index = g.edge_index.numpy()
        adj[edge_index[0], edge_index[1], 0] = 1.0

        # Node features: one-hot atom type on diagonal
        node_feat = np.zeros((num_nodes, num_nodes, NUM_ATOM_TYPES), dtype=np.float32)
        for i in range(num_nodes):
            node_feat[i, i, g.x[i].item()] = 1.0

        # Edge features: one-hot bond type
        edge_feat = np.zeros((num_nodes, num_nodes, NUM_BOND_TYPES), dtype=np.float32)
        for idx in range(g.edge_attr.shape[0]):
            src, dst = edge_index[0, idx], edge_index[1, idx]
            bond_type = g.edge_attr[idx].item()  # 1-indexed in ZINC
            if 1 <= bond_type <= NUM_BOND_TYPES:
                edge_feat[src, dst, bond_type - 1] = 1.0

        # Stack all channels: adj(1) + node(28) + edge(4) = 33
        graph = np.concatenate([adj, node_feat, edge_feat], axis=2)

        data_list.append({
            'graph': graph,
            'y': g.y.item()
        })

    output_path = os.path.join(OUTPUT_DIR, f"ZINC_{split_name}.p")
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)

    print(f"{split_name}: {len(data_list)} graphs -> {output_path}")


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        convert_split(split)
    print("Done.")
