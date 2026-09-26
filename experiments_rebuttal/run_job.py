"""
Run ONE (dataset, model_type, seed, fold) job under the standard PPGN/GIN
10-fold protocol and write a result JSON. Metric = best epoch validation
accuracy (the paper's reported quantity). Also records wall-clock timing so we
can compare training cost across architectures (Q3).

model_type in {baseline, topo, gcn, gin, mlp}.
Usage: python run_job.py DATASET MODEL SEED FOLD OUT.json [EPOCHS]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from easydict import EasyDict
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
import utils.config as C

# per-dataset architecture (block width, batch); topo/lr/decay/epochs come from
# utils.config constants so every model shares the dataset's training schedule.
ARCH = {
    'MUTAG':      dict(block=[64, 64], batch=20),
    'PTC':        dict(block=[32, 32], batch=16),
    'NCI1':       dict(block=[64, 64], batch=64),   # larger batch: better GPU util
    'NCI109':     dict(block=[64, 64], batch=64),
    'PROTEINS':   dict(block=[64, 64], batch=16),
    'IMDBBINARY': dict(block=[64, 64], batch=32),
    'IMDBMULTI':  dict(block=[64, 64], batch=32),
    'ENZYMES':    dict(block=[64, 64], batch=16),
    # regression (fixed 10k/1k/1k split, MAE); "fold" is unused, "seed" varies
    'ZINC':       dict(block=[64, 64], batch=64),
    # ogbg-molhiv: fixed OGB scaffold split, ROC-AUC; "fold" unused
    'MOLHIV':     dict(block=[64, 64], batch=64),
}


def build_config(dataset, model_type, overrides=None):
    a = ARCH[dataset]
    arch = dict(block_features=a['block'], depth_of_mlp=2, new_suffix=True,
                use_topology=False,
                topo_hidden_dim=32, topo_max_ph_dim=1, topo_num_stats=32,
                topo_max_simplex_dim=2)
    if model_type == 'topo':
        arch['use_topology'] = True
    elif model_type in ('gcn', 'gin', 'mlp', 'gsn'):
        arch['baseline_type'] = model_type
    cfg = EasyDict(dict(
        dataset_name=dataset, max_to_keep=1, gpu='0', num_exp=1,
        exp_name='10fold_cross_validation', val_exist=True,
        architecture=arch,
        hyperparams=dict(batch_size=a['batch'], optimizer='adam'),
    ))
    # fill schedule/labels/classes exactly like process_config
    cfg.num_classes = C.NUM_CLASSES[dataset]
    cfg.node_labels = C.NUM_LABELS[dataset]
    cfg.num_epochs = C.CHOSEN_EPOCH[dataset]
    cfg.hyperparams.learning_rate = C.LEARNING_RATES[dataset]
    cfg.hyperparams.decay_rate = C.DECAY_RATES[dataset]
    cfg.num_fold = None
    # optional fair hyperparameter overrides (applied to BOTH baseline & topo)
    if overrides:
        for k, v in overrides.items():
            if k in ('learning_rate', 'decay_rate', 'batch_size'):
                cfg.hyperparams[k] = v
            elif k == 'block_width':
                cfg.architecture.block_features = [v] * len(a['block'])
            else:  # architecture knobs (topo_hidden_dim, topo_num_stats, ...)
                cfg.architecture[k] = v
    return cfg


def run(dataset, model_type, seed, fold, epochs=None, overrides=None):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = build_config(dataset, model_type, overrides)
    if epochs is not None:
        cfg.num_epochs = epochs
    cfg.num_fold = fold
    data = DataGenerator(cfg)
    mw = ModelWrapper(cfg, data)
    n_params = sum(p.numel() for p in mw.model.parameters())
    tr = Trainer(mw, data, cfg)

    best_val, best_ep = -1.0, -1
    val_curve = []
    epoch_times = []
    if dataset == 'ZINC':
        return _run_regression(cfg, data, mw, tr, n_params, dataset, model_type,
                               seed, fold, overrides)
    if dataset == 'MOLHIV':
        return _run_auc(cfg, data, mw, tr, n_params, dataset, model_type,
                        seed, fold, overrides)
    t_start = time.time()
    for ep in range(cfg.num_epochs):
        t0 = time.time()
        tr.train_epoch(ep)
        epoch_times.append(time.time() - t0)
        val_acc, _ = tr.validate(ep)
        val_curve.append(round(float(val_acc), 5))
        if val_acc > best_val:
            best_val, best_ep = float(val_acc), ep
    total = time.time() - t_start
    return dict(
        dataset=dataset, model=model_type, seed=int(seed), fold=int(fold),
        epochs=cfg.num_epochs, n_params=int(n_params),
        best_val_acc=round(best_val, 5), best_epoch=int(best_ep),
        sec_per_train_epoch=round(float(np.mean(epoch_times)), 4),
        total_sec=round(total, 1), val_size=int(data.val_size),
        train_size=int(data.train_size), val_curve=val_curve,
        overrides=overrides or {},
    )


def _mae(mw, data, which):
    """MAE (in target units) over the val or test split, no autograd; one
    host sync per call."""
    data.initialize(which)
    n_it = data.num_iterations_val if which == 'val' else data.num_iterations_test
    size = data.val_size if which == 'val' else data.test_size
    mw.eval()
    tot = None
    with torch.no_grad():
        for _ in range(n_it):
            g, y = data.next_batch()
            _, d = mw.run_model_get_loss_and_results(g, y)
            tot = d if tot is None else tot + d
    return float((tot.sum().cpu() * float(np.asarray(data.labels_std).reshape(-1)[0])) / size)


def _run_regression(cfg, data, mw, tr, n_params, dataset, model_type, seed, fold, overrides):
    """ZINC: train for num_epochs; after every epoch record val and test MAE;
    report the test MAE at the epoch of lowest val MAE (standard protocol)."""
    val_curve, test_curve, epoch_times = [], [], []
    t_start = time.time()
    for ep in range(cfg.num_epochs):
        t0 = time.time()
        tr.train_epoch(ep)
        epoch_times.append(time.time() - t0)
        val_curve.append(round(_mae(mw, data, 'val'), 6))
        test_curve.append(round(_mae(mw, data, 'test'), 6))
    best = int(np.argmin(val_curve))
    return dict(
        dataset=dataset, model=model_type, seed=int(seed), fold=int(fold),
        epochs=cfg.num_epochs, n_params=int(n_params), task='regression',
        best_val_mae=val_curve[best], test_mae_at_best_val=test_curve[best], best_epoch=best,
        sec_per_train_epoch=round(float(np.mean(epoch_times)), 4),
        total_sec=round(time.time() - t_start, 1),
        val_curve=val_curve, test_curve=test_curve, overrides=overrides or {},
    )


def roc_auc(y, score):
    """ROC-AUC via the Mann-Whitney U statistic (average ranks for ties);
    equals sklearn.metrics.roc_auc_score."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); r = rankdata(score)
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _auc(mw, data, which):
    data.initialize(which)
    n_it = data.num_iterations_val if which == 'val' else data.num_iterations_test
    mw.eval()
    ys, ps = [], []
    with torch.no_grad():
        for _ in range(n_it):
            g, y = data.next_batch()
            ps.append(torch.softmax(mw.model(g), dim=1)[:, 1])
            ys.append(y)
    return roc_auc(torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy())


def _run_auc(cfg, data, mw, tr, n_params, dataset, model_type, seed, fold, overrides):
    """ogbg-molhiv: after every epoch record val and test ROC-AUC; report the
    test ROC-AUC at the epoch of highest val ROC-AUC (OGB protocol)."""
    val_curve, test_curve, epoch_times = [], [], []
    t_start = time.time()
    for ep in range(cfg.num_epochs):
        t0 = time.time()
        tr.train_epoch(ep)
        epoch_times.append(time.time() - t0)
        val_curve.append(round(_auc(mw, data, 'val'), 6))
        test_curve.append(round(_auc(mw, data, 'test'), 6))
    best = int(np.argmax(val_curve))
    return dict(
        dataset=dataset, model=model_type, seed=int(seed), fold=int(fold),
        epochs=cfg.num_epochs, n_params=int(n_params), task='auc',
        best_val_auc=val_curve[best], test_auc_at_best_val=test_curve[best], best_epoch=best,
        sec_per_train_epoch=round(float(np.mean(epoch_times)), 4),
        total_sec=round(time.time() - t_start, 1),
        val_curve=val_curve, test_curve=test_curve, overrides=overrides or {},
    )


if __name__ == '__main__':
    dataset, model_type = sys.argv[1], sys.argv[2]
    seed, fold = int(sys.argv[3]), int(sys.argv[4])
    out = sys.argv[5]
    epochs = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != '-' else None
    overrides = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
    res = run(dataset, model_type, seed, fold, epochs, overrides)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # atomic write: a job killed mid-dump must not leave a truncated file that
    # the resumable runner would then treat as "already done"
    tmp = out + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(res, f)
    os.replace(tmp, out)
    if res.get('task') == 'auc':
        print(f"DONE {dataset}/{model_type}/seed{seed} best_val_AUC={res['best_val_auc']:.4f} "
              f"test_AUC={res['test_auc_at_best_val']:.4f}@{res['best_epoch']} "
              f"{res['sec_per_train_epoch']:.2f}s/ep total={res['total_sec']:.0f}s")
        sys.exit(0)
    if res.get('task') == 'regression':
        print(f"DONE {dataset}/{model_type}/seed{seed} best_val_MAE={res['best_val_mae']:.4f} "
              f"test_MAE={res['test_mae_at_best_val']:.4f}@{res['best_epoch']} "
              f"{res['sec_per_train_epoch']:.2f}s/ep total={res['total_sec']:.0f}s")
        sys.exit(0)
    print(f"DONE {dataset}/{model_type}/seed{seed}/fold{fold} "
          f"best_val={res['best_val_acc']:.4f}@{res['best_epoch']} "
          f"{res['sec_per_train_epoch']:.2f}s/ep total={res['total_sec']:.0f}s")
