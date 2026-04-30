"""
Generate k-fold cross-validation split files for a dataset.

Usage:
    python utils/generate_kfold_splits.py --dataset NCI1 --k 7

This overwrites the existing 10fold_idx/ directory with k new splits.
"""
import os
import sys
import argparse
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_helper import load_dataset


def generate_kfold_splits(dataset_name, k, seed=100):
    result = load_dataset(dataset_name)
    graphs = result[0]
    n = len(graphs)

    np.random.seed(seed)
    indices = np.random.permutation(n)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1  # distribute remainder

    out_dir = os.path.join("data", "benchmark_graphs", dataset_name, "10fold_idx")
    os.makedirs(out_dir, exist_ok=True)

    # Remove old split files
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    current = 0
    for fold in range(1, k + 1):
        test_idx = indices[current:current + fold_sizes[fold - 1]]
        train_idx = np.concatenate([indices[:current], indices[current + fold_sizes[fold - 1]:]])
        current += fold_sizes[fold - 1]

        with open(os.path.join(out_dir, f"test_idx-{fold}.txt"), 'w') as f_out:
            for idx in sorted(test_idx):
                f_out.write(f"{idx}\n")

        with open(os.path.join(out_dir, f"train_idx-{fold}.txt"), 'w') as f_out:
            for idx in sorted(train_idx):
                f_out.write(f"{idx}\n")

    print(f"Generated {k}-fold splits for {dataset_name} (n={n}) in {out_dir}/")
    print(f"Fold sizes: {fold_sizes.tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    generate_kfold_splits(args.dataset, args.k, args.seed)
