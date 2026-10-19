import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
from easydict import EasyDict
from scipy.stats import rankdata

from data_loader.data_generator import DataGenerator
from models.base_model import build_model
from trainers.trainer import Trainer
import data_loader.data_helper as C

ARCH = {'MUTAG': ([64, 64], 20), 'PTC': ([32, 32], 16), 'NCI1': ([64, 64], 64), 'NCI109': ([64, 64], 64), 'PROTEINS': ([64, 64], 16), 'IMDBBINARY': ([64, 64], 32), 'IMDBMULTI': ([64, 64], 32), 'ENZYMES': ([64, 64], 16), 'ZINC': ([64, 64], 64), 'MOLHIV': ([64, 64], 64)}
MP_MODELS = ('gcn', 'gin', 'mlp', 'gsn')

def build_config(dataset, model, ovr=None):
    blocks, bs = ARCH[dataset]
    arch = dict(block_features=list(blocks), depth_of_mlp=2, use_topology=(model == 'topo'), topo_hidden_dim=32, topo_max_ph_dim=1, topo_num_stats=32, topo_max_simplex_dim=2)
    if(model in MP_MODELS):
        arch['baseline_type'] = model

    hp = dict(batch_size=bs, learning_rate=C.LEARNING_RATES[dataset], decay_rate=C.DECAY_RATES[dataset])
    cfg = EasyDict(dataset_name=dataset, num_fold=None, architecture=arch, num_classes=C.NUM_CLASSES[dataset], node_labels=C.NUM_LABELS[dataset], num_epochs=C.EPOCHS[dataset], hyperparams=hp)

    if(ovr is None):
        ovr = {}
    for(k, v) in ovr.items():
        if(k in ('learning_rate', 'decay_rate', 'batch_size', 'micro_batch')):
            cfg.hyperparams[k] = v
        elif(k == 'num_epochs'):
            cfg.num_epochs = int(v)
        elif(k == 'structure_only'):
            cfg.architecture.structure_only = bool(v)
            if(v):
                cfg.node_labels = 0
        elif(k == 'block_width'):
            cfg.architecture.block_features = [v] * len(blocks)
        else:
            cfg.architecture[k] = v
    return cfg

def mae(tr, data, split):
    data.initialize(split)
    if(split == 'val'):
        n_it, size = data.num_iterations_val, data.val_size
    else:
        n_it, size = data.num_iterations_test, data.test_size
    tr.model.eval()
    total = None
    with torch.no_grad():
        for _ in range(n_it):
            graphs, labels = data.next_batch()
            _, dist = tr.loss(graphs, labels)
            if(total is None):
                total = dist
            else:
                total = total + dist
    std = float(np.asarray(data.labels_std).reshape(-1)[0])
    return float((total.sum().cpu() * std) / size)

def roc_auc(y, score):
    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    npos = y.sum()
    nneg = (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def auc(tr, data, split):
    data.initialize(split)
    n_it = data.num_iterations_val if split == 'val' else data.num_iterations_test
    tr.model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for _ in range(n_it):
            graphs, labels = data.next_batch()
            probs.append(torch.softmax(tr.model(graphs), dim=1)[:, 1])
            ys.append(labels)
    ys = torch.cat(ys).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    return roc_auc(ys, probs)

def run(dataset, model, seed, fold, epochs=None, ovr=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(dataset, model, ovr)
    if(epochs is not None):
        cfg.num_epochs = epochs
    cfg.num_fold = fold
    data = DataGenerator(cfg)
    net = build_model(cfg)
    n_params = sum(p.numel() for p in net.parameters())
    tr = Trainer(net, data, cfg)

    res = dict(dataset=dataset, model=model, seed=int(seed), fold=int(fold), epochs=cfg.num_epochs, n_params=int(n_params))
    t_start = time.time()
    times = []

    if(dataset in ('ZINC', 'MOLHIV')):
        metric = mae if dataset == 'ZINC' else auc
        val_curve = []
        test_curve = []
        for ep in range(cfg.num_epochs):
            t0 = time.time()
            tr.train_epoch()
            times.append(time.time() - t0)
            val_curve.append(round(metric(tr, data, 'val'), 6))
            test_curve.append(round(metric(tr, data, 'test'), 6))
        if(dataset == 'ZINC'):
            best = int(np.argmin(val_curve))
            res.update(task='regression', best_val_mae=val_curve[best], test_mae_at_best_val=test_curve[best])
        else:
            best = int(np.argmax(val_curve))
            res.update(task='auc', best_val_auc=val_curve[best], test_auc_at_best_val=test_curve[best])
        res['best_epoch'] = best
        res['sec_per_train_epoch'] = round(float(np.mean(times)), 4)
        res['total_sec'] = round(time.time() - t_start, 1)
        res['val_curve'] = val_curve
        res['test_curve'] = test_curve
        res['overrides'] = ovr or {}
        return res

    best_val = -1.0
    best_ep = -1
    val_curve = []
    for ep in range(cfg.num_epochs):
        t0 = time.time()
        tr.train_epoch()
        times.append(time.time() - t0)
        acc, _ = tr.validate()
        val_curve.append(round(float(acc), 5))
        if(acc > best_val):
            best_val = float(acc)
            best_ep = ep
    res.update(best_val_acc=round(best_val, 5), best_epoch=int(best_ep), sec_per_train_epoch=round(float(np.mean(times)), 4), total_sec=round(time.time() - t_start, 1), val_size=int(data.val_size), train_size=int(data.train_size), val_curve=val_curve, overrides=ovr or {})
    return res

if(__name__ == '__main__'):
    ds, model = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3])
    fold = int(sys.argv[4])
    out = sys.argv[5]
    epochs = None
    if(len(sys.argv) > 6 and sys.argv[6] != '-'):
        epochs = int(sys.argv[6])
    ov = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
    res = run(ds, model, seed, fold, epochs, ov)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out + '.tmp', 'w') as f:
        json.dump(res, f)
    os.replace(out + '.tmp', out)

    if('test_mae_at_best_val' in res):
        print('done %s/%s/seed%d  val mae %.4f  test mae %.4f @ %d  %.2fs/ep' % (ds, model, seed, res['best_val_mae'], res['test_mae_at_best_val'], res['best_epoch'], res['sec_per_train_epoch']))
    elif('test_auc_at_best_val' in res):
        print('done %s/%s/seed%d  val auc %.4f  test auc %.4f @ %d  %.2fs/ep' % (ds, model, seed, res['best_val_auc'], res['test_auc_at_best_val'], res['best_epoch'], res['sec_per_train_epoch']))
    else:
        print(f"done {ds}/{model}/seed{seed}/fold{fold}  best val {res['best_val_acc']:.4f} @ {res['best_epoch']}  {res['sec_per_train_epoch']:.2f}s/ep  total {res['total_sec']:.0f}s")
