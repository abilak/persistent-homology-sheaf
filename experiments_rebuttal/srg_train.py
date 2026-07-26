"""
Stronger SRG separation experiment (addresses 'SRG uses random initialization').
Instead of relying on a single random init, we TRAIN both the baseline 2-IGN and
the topology-augmented model to tell the two graphs of each pair apart (a binary
task, labels 0/1), from many random seeds, and report the success rate.

Expectation from theory:
  * baseline (3-WL) produces IDENTICAL representations for a 3-WL-equivalent
    pair, so no amount of training can separate them -> ~0% success, loss stuck.
  * topology-augmented model CAN separate (Corollary 6), so gradient descent
    finds separating parameters -> ~100% success from every seed.
This converts the existence proof into a training-robust, init-robust result.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from srg_graphs import all_pairs
from srg_analysis import graph_to_input, make_config
from models.base_model import BaseModel

SEEDS = 10
STEPS = 150


def train_separate(G1, G2, use_topology, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.cat([graph_to_input(G1), graph_to_input(G2)], dim=0)  # (2,1,n,n)
    y = torch.tensor([0, 1])
    model = BaseModel(make_config(use_topology))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(STEPS):
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x)
    separated = bool(out.argmax(1)[0].item() != out.argmax(1)[1].item())
    margin = float((out[0] - out[1]).abs().max())
    return separated, margin, float(loss)


def main():
    rows = []
    print(f"Train-to-separate over {SEEDS} seeds, {STEPS} steps each")
    print(f"{'pair':<26}{'baseline sep%':<15}{'topo sep%':<12}{'topo margin(med)':<16}")
    for name, (G1, G2, params) in all_pairs().items():
        b = [train_separate(G1, G2, False, s) for s in range(SEEDS)]
        t = [train_separate(G1, G2, True, s) for s in range(SEEDS)]
        b_sep = 100 * np.mean([r[0] for r in b])
        t_sep = 100 * np.mean([r[0] for r in t])
        t_margin = float(np.median([r[1] for r in t]))
        rows.append(dict(pair=name, baseline_sep_pct=b_sep, topo_sep_pct=t_sep,
                         topo_margin_median=t_margin,
                         baseline_loss_med=float(np.median([r[2] for r in b]))))
        print(f"{name:<26}{b_sep:<15.0f}{t_sep:<12.0f}{t_margin:<16.2e}")
    json.dump(rows, open('experiments_rebuttal/srg_train_results.json', 'w'),
              indent=2, default=str)
    print(f"\nBaseline mean sep rate: {np.mean([r['baseline_sep_pct'] for r in rows]):.0f}%"
          f"   Topo mean sep rate: {np.mean([r['topo_sep_pct'] for r in rows]):.0f}%")


if __name__ == '__main__':
    main()
