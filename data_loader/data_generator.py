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
        self.is_qm9 = self.config.dataset_name == 'QM9'
        self.labels_dtype = torch.float32 if self.is_qm9 else torch.long

        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.is_qm9:
            self.load_qm9_data()
        else:
            self.load_data_benchmark()

        self.split_val_test_to_batches()

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
        graphs, labels = next(self.iter)
        if not torch.is_tensor(graphs):
            graphs = torch.from_numpy(np.ascontiguousarray(
                graphs, dtype=np.float32)).to(DEVICE, non_blocking=True)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(np.asarray(labels), dtype=self.labels_dtype
                                     ).to(DEVICE, non_blocking=True)
        return graphs, labels

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif what_set == 'test':
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

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs.

        Same semantics as before (shuffle within each same-size group, split
        into batches, then shuffle batch order) but batches are described by
        (block, row indices) and materialized lazily in next_batch, so no full
        copy of the training set is made per epoch.
        """
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

        if self.is_qm9:
            # Benchmark graphs have no test sets
            graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
            graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches = graphs, labels


if __name__ == '__main__':
    config = utils.config.process_config('../configs/10fold_config.json')
    data = DataGenerator(config)
    data.initialize('train')


