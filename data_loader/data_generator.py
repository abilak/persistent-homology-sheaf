import numpy as np
import torch

import data_loader.data_helper as helper
from models.base_model import get_device

DEVICE = get_device()
PAD_WASTE = 1.5

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.hyperparams.batch_size
        ds = config.dataset_name
        self.has_test = ds in ('ZINC', 'MOLHIV')
        self.labels_dtype = torch.float32 if ds == 'ZINC' else torch.long
        self.padded = bool(config.architecture.get('padded_batching', False)) or ds == 'MOLHIV'

        if(ds == 'MOLHIV'):
            tr_g, tr_y, va_g, va_y, te_g, te_y = helper.load_molhiv()
        elif(ds == 'ZINC'):
            tr_g, tr_y, va_g, va_y, te_g, te_y = helper.load_zinc()
            mean, std = tr_y.mean(axis=0), tr_y.std(axis=0)
            tr_y, va_y, te_y = (tr_y - mean) / std, (va_y - mean) / std, (te_y - mean) / std
            self.labels_std = std
        else:
            graphs, labels = helper.load_dataset(ds)
            tr, te = helper.get_train_val_indexes(config.num_fold, ds)
            tr_g, tr_y, va_g, va_y = graphs[tr], labels[tr], graphs[te], labels[te]
            te_g = te_y = None
        # FIXME breaks on molhiv (tuples), never needed it
        if(config.architecture.get('structure_only', False)):
            tr_g, va_g, te_g = adj_only(tr_g), adj_only(va_g), adj_only(te_g)

        self.train_graphs, self.train_labels = tr_g, tr_y
        self.val_graphs, self.val_labels = va_g, va_y
        self.test_graphs, self.test_labels = te_g, te_y
        self.train_size, self.val_size = len(tr_g), len(va_g)
        if(self.has_test):
            self.test_size = len(te_g)
        self.train_blocks = None

        self.eval_batches = {}
        splits = [('val', va_g, va_y)]
        if(self.has_test):
            splits.append(('test', te_g, te_y))
        for(name, gs, ys) in splits:
            if(self.padded):
                chunks = self.padded_chunks(gs, False)
                self.eval_batches[name] = (gs, ys, chunks)
                n = len(chunks)
            else:
                g, l = helper.group_same_size(gs, ys)
                self.eval_batches[name] = list(zip(*helper.split_to_batches(g, l, self.batch_size)))
                n = len(self.eval_batches[name])
            setattr(self, 'num_iterations_' + name, n)

    def initialize(self, what):
        if(what == 'train'):
            self.reshuffle_data()
        elif(self.padded):
            self.iter = pad_iter(*self.eval_batches[what])
        else:
            self.iter = iter(self.eval_batches[what])

    def reshuffle_data(self):
        if(self.padded):
            chunks = self.padded_chunks(self.train_graphs, True)
            self.num_iterations_train = len(chunks)
            self.iter = pad_iter(self.train_graphs, self.train_labels, chunks)
            return
        if(self.train_blocks is None):
            g, l = helper.group_same_size(self.train_graphs, self.train_labels)
            self.train_blocks = [np.ascontiguousarray(x, dtype=np.float32) for x in g]
            self.train_block_labels = [np.asarray(x) for x in l]
        batches = []
        for(bi, block) in enumerate(self.train_blocks):
            perm = np.random.permutation(block.shape[0])
            for s in range(0, block.shape[0], self.batch_size):
                batches.append((bi, perm[s:s + self.batch_size]))
        order = np.random.permutation(len(batches))
        batches = [batches[i] for i in order]
        self.num_iterations_train = len(batches)
        self.iter = ((self.train_blocks[bi][idx], self.train_block_labels[bi][idx]) for bi, idx in batches)

    def padded_chunks(self, graphs, shuffle):
        sizes = np.asarray([helper.nnodes(g) for g in graphs])
        tie = np.random.permutation(len(graphs)) if shuffle else np.arange(len(graphs))
        order = np.lexsort((tie, sizes))
        # don't let the padding blow up the n^3 cost
        chunks, cur, real = [], [], 0.0
        for i in order:
            n3 = float(sizes[i]) ** 3
            if(cur and (len(cur) >= self.batch_size or (len(cur) + 1) * n3 > PAD_WASTE * (real + n3))):
                chunks.append(np.asarray(cur))
                cur, real = [], 0.0
            cur.append(i)
            real += n3
        if(cur):
            chunks.append(np.asarray(cur))
        if(shuffle):
            chunks = [chunks[i] for i in np.random.permutation(len(chunks))]
        return chunks

    def next_batch(self):
        item = next(self.iter)
        graphs, labels = item[0], item[1]
        n_real = item[2] if len(item) == 3 else None
        if(not torch.is_tensor(graphs)):
            host = torch.from_numpy(np.ascontiguousarray(graphs, dtype=np.float32))
            hadj = host[:, 0].numpy()
            if(DEVICE.type == 'cuda'):
                host = host.pin_memory()
            graphs = host.to(DEVICE, non_blocking=True)
            graphs._host_adj = hadj
            if(n_real is not None):
                graphs._n_real = n_real
                m = torch.from_numpy(np.arange(host.shape[-1])[None, :] < n_real[:, None])
                if(DEVICE.type == 'cuda'):
                    m = m.pin_memory()
                graphs._node_mask = m.to(DEVICE, non_blocking=True)
        if(not torch.is_tensor(labels)):
            labels = torch.as_tensor(np.asarray(labels), dtype=self.labels_dtype).to(DEVICE, non_blocking=True)
        return graphs, labels

def adj_only(gs):
    if(gs is None):
        return None
    out = np.empty(len(gs), dtype=object)
    for(i, g) in enumerate(gs):
        out[i] = np.ascontiguousarray(g[:1])
    return out

def pad_batch(gs):
    sizes = np.array([helper.nnodes(g) for g in gs], dtype=np.int64)
    n = sizes.max()
    if(isinstance(gs[0], tuple)):
        out = np.zeros((len(gs), gs[0][1], n, n), dtype=np.float32)
    else:
        out = np.zeros((len(gs), gs[0].shape[0], n, n), dtype=np.result_type(gs[0].dtype, np.float32))
    for(k, g) in enumerate(gs):
        s = sizes[k]
        out[k, :, :s, :s] = helper.densify(g) if isinstance(g, tuple) else g
    return out, sizes

def pad_iter(graphs, labels, chunks):
    labels = np.asarray(labels)
    for idx in chunks:
        g, n = pad_batch([graphs[i] for i in idx])
        yield g, labels[idx], n
