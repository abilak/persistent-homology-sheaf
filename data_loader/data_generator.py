import data_loader.data_helper as helper
import utils.config
import numpy as np
import torch

from utils.device import get_device
DEVICE = get_device()


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.hyperparams.batch_size
        # regression datasets (fixed train/val/test split, MAE): QM9 and ZINC
        self.is_qm9 = self.config.dataset_name in ('QM9', 'ZINC')
        self.labels_dtype = torch.float32 if self.is_qm9 else torch.long
        # padded_batching: batch graphs of SIMILAR size (sorted, consecutive)
        # padded to the batch max with a node mask, instead of only graphs of
        # IDENTICAL size. ~2-3x fewer, fuller batches per epoch; the PPGN/topo
        # model masks padding exactly, so each graph's output is unchanged.
        self.padded = bool(getattr(config.architecture, 'padded_batching', False))

        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.config.dataset_name == 'ZINC':
            self.load_zinc_data()
        elif self.is_qm9:
            self.load_qm9_data()
        else:
            self.load_data_benchmark()

        self.split_val_test_to_batches()

    # load ZINC-12k (standard split); targets normalized by train mean / std,
    # reported MAE is multiplied back by labels_std (same as QM9)
    def load_zinc_data(self):
        tr_g, tr_y, va_g, va_y, te_g, te_y = helper.load_zinc()
        mean, std = tr_y.mean(axis=0), tr_y.std(axis=0)
        self.train_graphs, self.train_labels = tr_g, (tr_y - mean) / std
        self.val_graphs, self.val_labels = va_g, (va_y - mean) / std
        self.test_graphs, self.test_labels = te_g, (te_y - mean) / std
        self.train_size, self.val_size, self.test_size = len(tr_g), len(va_g), len(te_g)
        self.labels_std = std

    # load QM9 data set
    def load_qm9_data(self):
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = \
            helper.load_qm9(self.config.target_param)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std

    # load data for a benchmark graph (COLLAB, NCI1, NCI109, MUTAG, PTC, IMDBBINARY, IMDBMULTI, PROTEINS)
    def load_data_benchmark(self):
        graphs, labels = helper.load_dataset(self.config.dataset_name)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        item = next(self.iter)
        n_real = None
        if len(item) == 3:
            graphs, labels, n_real = item
        else:
            graphs, labels = item
        host_adj = None
        if not torch.is_tensor(graphs):
            host = torch.from_numpy(np.ascontiguousarray(graphs, dtype=np.float32))
            host_adj = host[:, 0].numpy()
            if DEVICE.type == 'cuda':
                host = host.pin_memory()         # truly async H2D copy
            graphs = host.to(DEVICE, non_blocking=True)
            # host copy of the adjacency for the topology branch, so it never
            # has to copy the batch back from the device (a sync per forward)
            graphs._host_adj = host_adj
            if n_real is not None:
                graphs._n_real = n_real
                m = torch.from_numpy(np.arange(host.shape[-1])[None, :] < n_real[:, None])
                if DEVICE.type == 'cuda':
                    m = m.pin_memory()
                graphs._node_mask = m.to(DEVICE, non_blocking=True)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(np.asarray(labels), dtype=self.labels_dtype
                                     ).to(DEVICE, non_blocking=True)
        return graphs, labels

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            if self.padded:
                self.iter = iter(self._val_padded)
            else:
                self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif what_set == 'test':
            if self.padded:
                self.iter = iter(self._test_padded)
            else:
                self.iter = zip(self.test_graphs_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def _ensure_train_blocks(self):
        """Group the training graphs by node count ONCE.

        The grouping is a property of the dataset, not of the epoch, but the
        original implementation recomputed it (sorting + concatenating every
        graph) on every call to reshuffle_data -- ~1.4 s/epoch on NCI1, which
        was comparable to the epoch's compute. We build the contiguous
        same-size blocks a single time and then only permute indices per epoch.
        """
        if getattr(self, '_train_blocks', None) is not None:
            return
        graphs, labels = helper.group_same_size(self.train_graphs,
                                                self.train_labels)
        self._train_blocks = [np.ascontiguousarray(g, dtype=np.float32)
                              for g in graphs]
        self._train_block_labels = [np.asarray(l) for l in labels]

    PAD_WASTE = 1.5

    @staticmethod
    def _pad_batch(graph_list):
        n = max(g.shape[1] for g in graph_list)
        C = graph_list[0].shape[0]
        out = np.zeros((len(graph_list), C, n, n), dtype=np.result_type(graph_list[0].dtype, np.float32))
        sizes = np.zeros(len(graph_list), dtype=np.int64)
        for k, g in enumerate(graph_list):
            s = g.shape[1]; out[k, :, :s, :s] = g; sizes[k] = s
        return out, sizes

    def _padded_batches(self, graphs, labels, shuffle):
        sizes = np.asarray([g.shape[1] for g in graphs])
        tie = np.random.permutation(len(graphs)) if shuffle else np.arange(len(graphs))
        order = np.lexsort((tie, sizes))               # by size, random within size
        # consecutive chunks of <= batch_size, closed early when padding would
        # push the batch's O(n^3) matmul cost above PAD_WASTE x its real cost
        # (sizes ascend, so each new graph is the batch max)
        chunks, cur, real = [], [], 0.0
        for i in order:
            n3 = float(sizes[i]) ** 3
            if cur and (len(cur) >= self.batch_size or
                        (len(cur) + 1) * n3 > self.PAD_WASTE * (real + n3)):
                chunks.append(np.asarray(cur)); cur, real = [], 0.0
            cur.append(i); real += n3
        if cur:
            chunks.append(np.asarray(cur))
        if shuffle:
            chunks = [chunks[i] for i in np.random.permutation(len(chunks))]
        out = []
        for idx in chunks:
            g, n = self._pad_batch([graphs[i] for i in idx])
            out.append((g, np.asarray(labels)[idx], n))
        return out

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs.

        Same semantics as before (shuffle within each same-size group, split
        into batches, then shuffle batch order) but batches are described by
        (block, row indices) and materialized lazily in next_batch, so no full
        copy of the training set is made per epoch.
        """
        if self.padded:
            batches = self._padded_batches(self.train_graphs, self.train_labels, True)
            self.num_iterations_train = len(batches)
            self.iter = iter(batches)
            return
        self._ensure_train_blocks()
        batches = []
        for bi, block in enumerate(self._train_blocks):
            n = block.shape[0]
            perm = np.random.permutation(n)
            for s in range(0, n, self.batch_size):
                batches.append((bi, perm[s:s + self.batch_size]))
        order = np.random.permutation(len(batches))
        batches = [batches[i] for i in order]
        self.num_iterations_train = len(batches)
        self.iter = self._iter_batches(batches)

    def _iter_batches(self, batches):
        for bi, idx in batches:
            yield self._train_blocks[bi][idx], self._train_block_labels[bi][idx]

    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels = helper.group_same_size(self.val_graphs, self.val_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches = graphs, labels
        if self.padded:
            self._val_padded = self._padded_batches(self.val_graphs, self.val_labels, False)
            self.num_iterations_val = len(self._val_padded)

        if self.is_qm9:
            # Benchmark graphs have no test sets
            graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
            graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches = graphs, labels
            if self.padded:
                self._test_padded = self._padded_batches(self.test_graphs, self.test_labels, False)
                self.num_iterations_test = len(self._test_padded)


if __name__ == '__main__':
    config = utils.config.process_config('../configs/10fold_config.json')
    data = DataGenerator(config)
    data.initialize('train')


