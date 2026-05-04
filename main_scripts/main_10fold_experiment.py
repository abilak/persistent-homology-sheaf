import os
import sys
import torch
import numpy as np
from datetime import datetime

"""
How To:
Example for running from command line:
python <path_to>/ProvablyPowerfulGraphNetworks/main_scripts/main_10fold_experiment.py --config=configs/10fold_config.json --dataset_name=COLLAB
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args


def _write_ablation_summary(summary_dir, all_ablation_results, num_folds):
    """Write aggregated ablation summary across all folds."""
    import os
    summary_path = os.path.join(summary_dir, "ablation_summary.txt")

    modes = ['full', 'no_node_features', 'no_gate']
    mode_accs = {m: [] for m in modes}
    mode_losses = {m: [] for m in modes}

    for fold_results in all_ablation_results:
        for mode in modes:
            if mode in fold_results:
                mode_accs[mode].append(fold_results[mode]['accuracy'])
                mode_losses[mode].append(fold_results[mode]['loss'])

    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDY — AGGREGATED SUMMARY ({} folds)\n".format(num_folds))
        f.write("=" * 70 + "\n\n")

        f.write("  Ablation Descriptions:\n")
        f.write("    'full':              Full topology model (T_graph || t_u || t_v, gated residual)\n")
        f.write("                         X^(l+1) = X^(l+1/2) + σ(gate) * ψ(X^(l+1/2) || T_graph || t_i || t_j)\n\n")
        f.write("    'no_node_features':  Drop node-level PH features t_u, t_v (zero them out)\n")
        f.write("                         X^(l+1) = X^(l+1/2) + σ(gate) * ψ(X^(l+1/2) || T_graph || 0 || 0)\n")
        f.write("                         Rationale: Section 4.5 identifies node features as a deviation\n")
        f.write("                         from the idealized architecture. This ablation measures their\n")
        f.write("                         empirical contribution to justify the design choice.\n\n")
        f.write("    'no_gate':           Drop gated residual, use plain additive fusion\n")
        f.write("                         X^(l+1) = X^(l+1/2) + ψ(X^(l+1/2) || T_graph || t_i || t_j)\n")
        f.write("                         Rationale: Section 3 does not include gating. This tests\n")
        f.write("                         whether learned gating provides empirical benefit over\n")
        f.write("                         a simpler additive residual connection.\n\n")

        # Per-fold results
        f.write("  Per-Fold Accuracy:\n")
        f.write("    {:>6s}".format("Fold"))
        for mode in modes:
            f.write(" {:>18s}".format(mode))
        f.write("\n")
        f.write("    {:>6s}".format("-" * 6))
        for _ in modes:
            f.write(" {:>18s}".format("-" * 18))
        f.write("\n")

        for i in range(len(all_ablation_results)):
            f.write("    {:>6d}".format(i + 1))
            for mode in modes:
                if i < len(mode_accs[mode]):
                    f.write(" {:>18.4f}".format(mode_accs[mode][i]))
                else:
                    f.write(" {:>18s}".format("N/A"))
            f.write("\n")

        # Aggregated statistics
        f.write("\n  Aggregated Results:\n")
        f.write("    {:<22s} {:>10s} {:>10s} {:>10s} {:>10s}\n".format(
            "Mode", "Mean Acc", "Std Acc", "Mean Loss", "Std Loss"))
        f.write("    {:<22s} {:>10s} {:>10s} {:>10s} {:>10s}\n".format(
            "-" * 22, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
        for mode in modes:
            accs = mode_accs[mode]
            losses = mode_losses[mode]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=0)
                mean_loss = np.mean(losses)
                std_loss = np.std(losses, ddof=0)
                f.write("    {:<22s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
                    mode, mean_acc, std_acc, mean_loss, std_loss))

        # Delta from full model
        f.write("\n  Impact of Ablations (vs Full Model):\n")
        f.write("    {:<22s} {:>12s} {:>12s} {:>12s}\n".format(
            "Ablation", "Mean Δ Acc", "Mean Δ %", "Significance"))
        f.write("    {:<22s} {:>12s} {:>12s} {:>12s}\n".format(
            "-" * 22, "-" * 12, "-" * 12, "-" * 12))
        full_mean = np.mean(mode_accs['full']) if mode_accs['full'] else 0
        for mode in ['no_node_features', 'no_gate']:
            accs = mode_accs[mode]
            if accs and mode_accs['full']:
                # Compute per-fold deltas
                deltas = [accs[i] - mode_accs['full'][i] for i in range(min(len(accs), len(mode_accs['full'])))]
                mean_delta = np.mean(deltas)
                mean_delta_pct = mean_delta * 100
                # Simple significance: all folds agree on direction?
                all_neg = all(d < 0 for d in deltas)
                all_pos = all(d > 0 for d in deltas)
                sig = "consistent ↓" if all_neg else "consistent ↑" if all_pos else "mixed"
                f.write("    {:<22s} {:>+12.4f} {:>+12.2f}% {:>12s}\n".format(
                    mode, mean_delta, mean_delta_pct, sig))

        # Interpretation
        f.write("\n  Interpretation:\n")
        for mode in ['no_node_features', 'no_gate']:
            accs = mode_accs[mode]
            if accs and mode_accs['full']:
                deltas = [accs[i] - mode_accs['full'][i] for i in range(min(len(accs), len(mode_accs['full'])))]
                mean_delta = np.mean(deltas)
                if mode == 'no_node_features':
                    if mean_delta < -0.01:
                        f.write("    Node-level features (t_u, t_v): BENEFICIAL (+{:.1f}% accuracy).\n".format(-mean_delta * 100))
                        f.write("      → Empirically justifies Section 4.5 deviation from idealized arch.\n")
                    elif mean_delta > 0.01:
                        f.write("    Node-level features (t_u, t_v): HARMFUL ({:.1f}% accuracy loss when included).\n".format(mean_delta * 100))
                        f.write("      → Suggests node features may cause overfitting on this dataset.\n")
                    else:
                        f.write("    Node-level features (t_u, t_v): NEGLIGIBLE impact ({:+.2f}%).\n".format(mean_delta * 100))
                        f.write("      → Graph-level topology T_graph carries most information.\n")
                elif mode == 'no_gate':
                    if mean_delta < -0.01:
                        f.write("    Gated residual: BENEFICIAL (+{:.1f}% accuracy).\n".format(-mean_delta * 100))
                        f.write("      → Learned gating helps modulate topology contribution.\n")
                    elif mean_delta > 0.01:
                        f.write("    Gated residual: HARMFUL ({:.1f}% accuracy loss when included).\n".format(mean_delta * 100))
                        f.write("      → Plain additive residual sufficient for this dataset.\n")
                    else:
                        f.write("    Gated residual: NEGLIGIBLE impact ({:+.2f}%).\n".format(mean_delta * 100))
                        f.write("      → Both fusion strategies perform similarly.\n")

        f.write("\n" + "=" * 70 + "\n")

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY ({} folds)".format(num_folds))
    print("=" * 70)
    print("  {:<22s} {:>10s} {:>10s} {:>10s}".format("Mode", "Mean Acc", "Std", "Δ vs Full"))
    print("  {:<22s} {:>10s} {:>10s} {:>10s}".format("-" * 22, "-" * 10, "-" * 10, "-" * 10))
    full_mean = np.mean(mode_accs['full']) if mode_accs['full'] else 0
    for mode in modes:
        accs = mode_accs[mode]
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=0)
            delta_str = "—" if mode == 'full' else "{:+.2f}%".format((mean_acc - full_mean) * 100)
            print("  {:<22s} {:>10.4f} {:>10.4f} {:>10s}".format(
                mode, mean_acc, std_acc, delta_str))
    print("=" * 70)
    print("  Detailed report: {}".format(summary_path))


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, args.dataset_name, use_topology=args.use_topology)
        config.use_checkpoint = getattr(args, 'use_checkpoint', False)

        # If resuming, reuse the existing experiment directory
        if args.resume_dir:
            config.use_checkpoint = True
            config.summary_dir = os.path.join(args.resume_dir, "summary/")
            config.checkpoint_dir = os.path.join(args.resume_dir, "checkpoint/")

    except Exception as e:
        print("missing or invalid arguments {}".format(e))
        exit(0)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # TODO uncomment only for CUDA error debugging
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    torch.manual_seed(100)
    np.random.seed(100)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)
    num_folds = getattr(config, 'num_folds', 10)
    base_checkpoint_dir = config.checkpoint_dir
    all_ablation_results = []
    for exp in range(1, config.num_exp+1):
        for fold in range(1, num_folds+1):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # per-fold checkpoint dir so folds don't clobber or resume from each other
            config.checkpoint_dir = os.path.join(base_checkpoint_dir, "fold_{}".format(fold))
            create_dirs([config.checkpoint_dir])
            # create your data generator
            config.num_fold = fold
            data = DataGenerator(config)
            # create an instance of the model you want
            model_wrapper = ModelWrapper(config, data)
            # create trainer and pass all the previous components to it
            trainer = Trainer(model_wrapper, data, config)
            # here you train your model
            trainer.train()
            # Run ablation study at end of fold (topology models only)
            if getattr(config.architecture, 'use_topology', False):
                abl_results = trainer.evaluate_ablations(fold_num=fold)
                if abl_results:
                    all_ablation_results.append(abl_results)

    doc_utils.summary_10fold_results(config.summary_dir)

    # Write aggregated ablation summary across all folds
    if all_ablation_results:
        _write_ablation_summary(config.summary_dir, all_ablation_results, num_folds)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Runtime: {}'.format(datetime.now() - start))
