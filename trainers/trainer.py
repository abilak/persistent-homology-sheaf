# import tensorflow as tf
from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from utils import doc_utils


class Trainer(object):
    def __init__(self, model_wrapper, data, config):
        self.is_QM9 = config.dataset_name == 'QM9'
        self.best_val_loss = np.inf
        self.best_epoch = -1
        self.cur_epoch = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_checkpoint = getattr(config, 'use_checkpoint', False)

        self.model_wrapper = model_wrapper
        self.config = config
        self.data_loader = data

        self.optimizer = None
        if self.config.hyperparams.optimizer == 'momentum':
            self.optimizer = torch.optim.SGD(self.model_wrapper.model.parameters(),
                                             lr=self.config.hyperparams.learning_rate,
                                             momentum=self.config.hyperparams.momentum)
        elif self.config.hyperparams.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model_wrapper.model.parameters(),
                                              lr=self.config.hyperparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=config.hyperparams.decay_rate)

    def train(self):
        """
        Trains for the num of epochs in the config.
        :return:
        """
        # Resume from checkpoint if available
        if self.use_checkpoint:
            self._try_resume()

        for cur_epoch in range(self.cur_epoch, self.config.num_epochs, 1):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            self.cur_epoch = cur_epoch
            # validation step
            if self.config.val_exist:
                val_acc, val_loss = self.validate(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, val_acc, val_loss, cur_epoch, self.config)

                # save checkpoint every epoch (classification + QM9)
                if self.use_checkpoint:
                    self._save_checkpoint(cur_epoch, val_loss)

        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            if not self.is_QM9:
                doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)

        # Final gate summary
        self._log_final_gate_summary()

    def _save_checkpoint(self, epoch, val_loss):
        """Save last checkpoint every epoch, and best when val loss improves."""
        # Always save last
        self.model_wrapper.save(best=False, epoch=epoch, optimizer=self.optimizer,
                                best_val_loss=self.best_val_loss, best_epoch=self.best_epoch)
        # Save best if improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.model_wrapper.save(best=True, epoch=epoch, optimizer=self.optimizer,
                                    best_val_loss=self.best_val_loss, best_epoch=self.best_epoch)
            print("New best validation loss: {:.4f} at epoch {}".format(val_loss, epoch))

    def _try_resume(self):
        """Resume from last.tar checkpoint if it exists."""
        import os
        last_path = os.path.join(self.config.checkpoint_dir, 'last.tar')
        if os.path.exists(last_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(last_path, map_location=self.device)
            self.model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.cur_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', np.inf)
            self.best_epoch = checkpoint.get('best_epoch', -1)
            print("Resumed from epoch {}".format(self.cur_epoch))
        else:
            print("No checkpoint found, starting from scratch.")

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param num_epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize('train')
        self.model_wrapper.train()

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct_labels_or_distances = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct_labels_or_distances = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct_labels_or_distances += correct_labels_or_distances

        tt.close()
        self.scheduler.step()

        # Log mean gate activation per topology layer
        self._log_gate_values(num_epoch)

        # Log topology diagnostics at epoch 1 and final epoch
        if num_epoch == 0 or num_epoch == self.config.num_epochs - 1:
            self._log_topology_diagnostics(num_epoch)

        loss_per_epoch = total_loss/self.data_loader.train_size
        if not self.is_QM9:
            acc_per_epoch = total_correct_labels_or_distances/self.data_loader.train_size
            print("\t\tEpoch-{}  loss:{:.4f} -- acc:{:.4f}\n".format(num_epoch, loss_per_epoch, acc_per_epoch))
            return acc_per_epoch, loss_per_epoch
        else:
            dist_per_epoch = (total_correct_labels_or_distances * self.data_loader.labels_std)/self.data_loader.train_size
            print("\t\tEpoch-{}  loss:{:.4f} -- mean_distances:\n{}\n".format(num_epoch, loss_per_epoch, dist_per_epoch))
            return dist_per_epoch, loss_per_epoch

    def train_step(self):
        """

        :return: tuple of (loss, num_correct_labels or distances_array)
        """
        graphs, labels = self.data_loader.next_batch()
        loss, correct_labels_or_distances = self.model_wrapper.run_model_get_loss_and_results(graphs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_wrapper.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.cpu().item(), correct_labels_or_distances

    def _log_gate_values(self, epoch):
        """Log mean sigmoid(gate) per topology layer and running average."""
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return

        # Initialize accumulator on first call
        if not hasattr(self, '_gate_history'):
            self._gate_history = [[] for _ in model.topo_layers]

        gate_values = []
        for i, topo_layer in enumerate(model.topo_layers):
            gate_mean = getattr(topo_layer, 'last_gate_mean', None)
            if gate_mean is not None:
                gate_values.append(gate_mean)
                self._gate_history[i].append(gate_mean)

        if gate_values:
            avg_across_layers = sum(gate_values) / len(gate_values)
            print("\t\tGate σ(γ) [epoch {}]: avg={:.4f}".format(epoch, avg_across_layers))
            # Write to gate log file
            import os
            gate_log = os.path.join(self.config.summary_dir, "gate_log.txt")
            with open(gate_log, 'a') as f:
                f.write("{},{:.4f},{}\n".format(
                    epoch, avg_across_layers,
                    ",".join(["{:.4f}".format(v) for v in gate_values])))

    def _log_final_gate_summary(self):
        """Print final gate summary at end of training."""
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return
        if not hasattr(self, '_gate_history'):
            return

        print("\n\t\t===== Final Gate Summary =====")
        all_values = []
        for i, history in enumerate(self._gate_history):
            if history:
                layer_avg = sum(history) / len(history)
                all_values.extend(history)
                print("\t\tLayer {}: mean σ(γ) = {:.4f} (over {} epochs)".format(i, layer_avg, len(history)))
        if all_values:
            overall_avg = sum(all_values) / len(all_values)
            print("\t\tOverall: mean σ(γ) = {:.4f}".format(overall_avg))
        print("\t\t==============================\n")

        # Also log per-graph gate values on test set
        self._log_test_per_graph_gates()
        self._save_representative_filtrations()

    def _log_topology_diagnostics(self, epoch):
        """Log persistence pair counts, tie fractions, and filtration stats."""
        import os
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return

        diag_path = os.path.join(self.config.summary_dir, "topo_diagnostics.txt")
        with open(diag_path, 'a') as f:
            f.write("=== Epoch {} ===\n".format(epoch))
            for i, topo_layer in enumerate(model.topo_layers):
                ph = topo_layer.ph
                if ph.last_pair_counts is not None:
                    # Average pair counts across batch
                    dims = range(ph.max_ph_dim + 1)
                    avg_counts = {d: np.mean([pc[d] for pc in ph.last_pair_counts]) for d in dims}
                    f.write("  Layer {} pair counts: {}\n".format(i, 
                        {d: "{:.1f}".format(v) for d, v in avg_counts.items()}))
                if ph.last_tie_fractions is not None:
                    avg_tie = np.mean(ph.last_tie_fractions)
                    f.write("  Layer {} tie fraction: {:.4f}\n".format(i, avg_tie))
                if ph.last_filt_mean is not None:
                    f.write("  Layer {} filt mean={:.4f}, std={:.4f}\n".format(
                        i, ph.last_filt_mean, ph.last_filt_std))
            f.write("\n")
        print("\t\tTopology diagnostics logged for epoch {}".format(epoch))

    def _log_test_per_graph_gates(self):
        """Log per-graph gate values during test set evaluation."""
        import os
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return

        # Run a pass over test data collecting per-graph gates
        self.data_loader.initialize('test' if not self.config.val_exist else 'val')
        self.model_wrapper.eval()

        per_graph_gates = []
        num_iters = (self.data_loader.num_iterations_test if not self.config.val_exist
                     else self.data_loader.num_iterations_val)

        with torch.no_grad():
            for _ in range(min(num_iters, 10)):  # cap at 10 batches
                graph, label = self.data_loader.next_batch()
                self.model_wrapper.run_model_get_loss_and_results(graph, label)
                # Collect gate values from each topology layer
                for topo_layer in model.topo_layers:
                    gate_mean = getattr(topo_layer, 'last_gate_mean', None)
                    if gate_mean is not None:
                        per_graph_gates.append(gate_mean)

        if per_graph_gates:
            gate_path = os.path.join(self.config.summary_dir, "test_per_graph_gates.txt")
            with open(gate_path, 'w') as f:
                f.write("per_batch_gate_means\n")
                for g in per_graph_gates:
                    f.write("{:.4f}\n".format(g))
            print("\t\tPer-graph gate values saved ({} entries)".format(len(per_graph_gates)))

    def _save_representative_filtrations(self):
        """Save actual filtration values for 3-5 representative test graphs."""
        import os, json
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return

        self.data_loader.initialize('test' if not self.config.val_exist else 'val')
        self.model_wrapper.eval()

        # Get one batch and save filtration info from PH layers
        with torch.no_grad():
            graph, label = self.data_loader.next_batch()
            self.model_wrapper.run_model_get_loss_and_results(graph, label)

        # Collect from the first topo layer's PH
        ph = model.topo_layers[0].ph
        if ph.last_pair_counts is None:
            return

        filt_path = os.path.join(self.config.summary_dir, "representative_filtrations.txt")
        with open(filt_path, 'w') as f:
            f.write("Representative filtration stats (first batch, first layer)\n")
            f.write("Pair counts per graph: {}\n".format(ph.last_pair_counts[:5]))
            f.write("Tie fractions per graph: {}\n".format(
                ["{:.4f}".format(t) for t in ph.last_tie_fractions[:5]]))
            f.write("Batch filt mean={:.4f}, std={:.4f}\n".format(
                ph.last_filt_mean, ph.last_filt_std))
        print("\t\tRepresentative filtrations saved.")

    def evaluate_ablations(self, fold_num=None):
        """
        Run ablation evaluation at end of fold using the fully-trained model.
        Tests the same trained weights under modified forward passes:
          1. 'no_node_features': zeros out t_u, t_v (node-level PH features)
          2. 'no_gate': removes gating, uses plain additive fusion

        Returns dict of ablation results and writes detailed report to summary dir.
        """
        import os
        model = self.model_wrapper.model
        if not getattr(model, 'use_topology', False) or model.topo_layers is None:
            return None

        ablation_modes = [
            ('full', None),
            ('no_node_features', 'no_node_features'),
            ('no_gate', 'no_gate'),
        ]

        results = {}
        for mode_name, mode_flag in ablation_modes:
            # Evaluate on validation set (used as test in 10-fold)
            self.data_loader.initialize('val')
            self.model_wrapper.eval()

            total_loss = 0.
            total_correct = 0.
            total_samples = 0
            per_batch_details = []

            with torch.no_grad():
                for _ in range(self.data_loader.num_iterations_val):
                    graph, label = self.data_loader.next_batch()
                    loss, correct = self.model_wrapper.run_model_get_loss_and_results(
                        graph, label, ablation_mode=mode_flag)
                    batch_size = label.shape[0]
                    total_loss += loss.cpu().item()
                    total_correct += correct
                    total_samples += batch_size
                    per_batch_details.append({
                        'batch_size': batch_size,
                        'batch_correct': correct,
                        'batch_loss': loss.cpu().item(),
                    })

            acc = total_correct / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

            # Collect gate statistics for this mode
            gate_means = []
            for topo_layer in model.topo_layers:
                gm = getattr(topo_layer, 'last_gate_mean', None)
                if gm is not None:
                    gate_means.append(gm)

            results[mode_name] = {
                'accuracy': acc,
                'loss': avg_loss,
                'total_correct': total_correct,
                'total_samples': total_samples,
                'num_batches': len(per_batch_details),
                'gate_means': gate_means,
                'per_batch_details': per_batch_details,
            }

        # Write detailed ablation report
        report_path = os.path.join(self.config.summary_dir, "ablation_report.txt")
        append_mode = 'a' if os.path.exists(report_path) else 'w'
        with open(report_path, append_mode) as f:
            f.write("=" * 70 + "\n")
            f.write("ABLATION STUDY — Fold {}\n".format(fold_num if fold_num else "?"))
            f.write("=" * 70 + "\n\n")

            # Model info
            n_params = sum(p.numel() for p in model.parameters())
            n_topo_params = sum(p.numel() for tl in model.topo_layers for p in tl.parameters())
            n_base_params = n_params - n_topo_params
            f.write("  Model Parameters:\n")
            f.write("    Total:              {}\n".format(n_params))
            f.write("    Base (equivariant): {}\n".format(n_base_params))
            f.write("    Topology branch:    {}\n".format(n_topo_params))
            f.write("    Topology layers:    {}\n\n".format(len(model.topo_layers)))

            # Architecture details
            f.write("  Architecture Config:\n")
            arch = self.config.architecture
            f.write("    block_features:      {}\n".format(arch.block_features))
            f.write("    depth_of_mlp:        {}\n".format(arch.depth_of_mlp))
            f.write("    use_topology:        {}\n".format(arch.use_topology))
            f.write("    topo_hidden_dim:     {}\n".format(arch.topo_hidden_dim))
            f.write("    topo_max_ph_dim:     {}\n".format(arch.topo_max_ph_dim))
            f.write("    topo_num_stats:      {}\n".format(arch.topo_num_stats))
            f.write("    topo_max_simplex_dim:{}\n\n".format(arch.topo_max_simplex_dim))

            # Ablation descriptions
            f.write("  Ablation Descriptions:\n")
            f.write("    'full':              Full model (T_graph + t_u + t_v, gated residual)\n")
            f.write("    'no_node_features':  Zero out t_u and t_v; keep only T_graph\n")
            f.write("                         Tests: Do node-level PH features contribute?\n")
            f.write("                         Section 4.5 deviation: node features not in\n")
            f.write("                         idealized architecture but theoretically justified.\n")
            f.write("    'no_gate':           Remove sigmoid gate; plain additive fusion\n")
            f.write("                         X^(l+1) = X^(l+1/2) + psi(X || T)\n")
            f.write("                         Tests: Does learned gating help over plain residual?\n")
            f.write("                         Section 3 deviation: gate not in base formulation.\n\n")

            # Results table
            f.write("  Results:\n")
            f.write("    {:<22s} {:>10s} {:>10s} {:>8s} {:>8s}\n".format(
                "Mode", "Accuracy", "Loss", "Correct", "Total"))
            f.write("    {:<22s} {:>10s} {:>10s} {:>8s} {:>8s}\n".format(
                "-" * 22, "-" * 10, "-" * 10, "-" * 8, "-" * 8))
            for mode_name in ['full', 'no_node_features', 'no_gate']:
                r = results[mode_name]
                f.write("    {:<22s} {:>10.4f} {:>10.4f} {:>8d} {:>8d}\n".format(
                    mode_name, r['accuracy'], r['loss'],
                    int(r['total_correct']), r['total_samples']))

            # Delta analysis
            f.write("\n  Delta from Full Model:\n")
            full_acc = results['full']['accuracy']
            for mode_name in ['no_node_features', 'no_gate']:
                r = results[mode_name]
                delta = r['accuracy'] - full_acc
                f.write("    {:<22s}: {:+.4f} ({:+.2f}%)\n".format(
                    mode_name, delta, delta * 100))

            # Gate statistics
            f.write("\n  Gate Statistics (last batch, per topology layer):\n")
            for mode_name in ['full', 'no_node_features', 'no_gate']:
                r = results[mode_name]
                gate_str = ", ".join(["{:.4f}".format(g) for g in r['gate_means']])
                f.write("    {:<22s}: [{}]\n".format(mode_name, gate_str))

            # Per-batch breakdown for full model
            f.write("\n  Per-Batch Breakdown (full model):\n")
            f.write("    {:>6s} {:>8s} {:>8s} {:>10s}\n".format(
                "Batch", "Size", "Correct", "Loss"))
            for i, bd in enumerate(results['full']['per_batch_details']):
                f.write("    {:>6d} {:>8d} {:>8d} {:>10.4f}\n".format(
                    i + 1, bd['batch_size'], int(bd['batch_correct']), bd['batch_loss']))

            f.write("\n")

        # Print summary to stdout
        print("\n\t\t===== ABLATION RESULTS (Fold {}) =====".format(fold_num))
        print("\t\t{:<22s} {:>10s} {:>8s}".format("Mode", "Accuracy", "Delta"))
        print("\t\t{:<22s} {:>10s} {:>8s}".format("-" * 22, "-" * 10, "-" * 8))
        full_acc = results['full']['accuracy']
        for mode_name in ['full', 'no_node_features', 'no_gate']:
            r = results[mode_name]
            delta = r['accuracy'] - full_acc
            delta_str = "{:+.2f}%".format(delta * 100) if mode_name != 'full' else "—"
            print("\t\t{:<22s} {:>10.4f} {:>8s}".format(mode_name, r['accuracy'], delta_str))
        print("\t\t======================================\n")

        return results

    def validate(self, epoch):
        """
        Perform forward pass on the model with the validation set
        :param epoch: Epoch number
        :return: (val_acc, val_loss) for benchmark graphs, (val_dists, val_loss) for QM9
        """
        # initialize dataset
        self.data_loader.initialize('val')
        self.model_wrapper.eval()

        # initialize tqdm
        # tt = tqdm(range(self.data_loader.num_iterations_val), total=self.data_loader.num_iterations_val,
        #           desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct_or_dist = 0.

        # Iterate over batches
        # for cur_it in tt:
        for cur_it in range(self.data_loader.num_iterations_val):
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            # label = np.expand_dims(label, 0)
            loss, correct_or_dist = self.model_wrapper.run_model_get_loss_and_results(graph, label)

            # update metrics returned from train_step func
            total_loss += loss.cpu().item()
            total_correct_or_dist += correct_or_dist

        # tt.close()

        val_loss = total_loss/self.data_loader.val_size
        if self.is_QM9:
            val_dists = (total_correct_or_dist*self.data_loader.labels_std)/self.data_loader.val_size
            print("\t\tVal-{}  loss:{:.4f} -- mean_distances:\n{}\n".format(epoch, val_loss, val_dists))

            # save best model by validation loss to be used for test set
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("New best validation score achieved.")
                self.model_wrapper.save(best=True, epoch=self.cur_epoch, optimizer=self.optimizer)
                self.best_epoch = epoch
            return val_dists, val_loss
        else:
            val_acc = total_correct_or_dist/self.data_loader.val_size
            # print("\t\tVal-{}  loss:{:.4f} -- acc:{:.4f}\n".format(epoch, val_loss, val_acc))
            return val_acc, val_loss

    def test(self, load_best_model=False):
        """
        Perform forward pass on the model for the test set
        :param load_best_model: Boolean. True for loading the best model saved, based on validation loss
        :return: (test_dists, test_loss)
        """
        # load best saved model
        if load_best_model:
            _optimizer_state_dict, _epoch = self.model_wrapper.load(best=True)

        # initialize dataset
        self.data_loader.initialize('test')
        self.model_wrapper.eval()

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Test-{}-".format(self.best_epoch))

        total_loss = 0.
        total_dists = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            # label = np.expand_dims(label, 0)
            loss, dists = self.model_wrapper.run_model_get_loss_and_results(graph, label)
            # update metrics returned from train_step func
            total_loss += loss.cpu().item()
            total_dists += dists

        test_loss = total_loss/self.data_loader.test_size
        test_dists = (total_dists*self.data_loader.labels_std) / self.data_loader.test_size
        print("\t\tTest-{}  loss:{:.4f} -- mean_distances:\n{}\n".format(self.best_epoch, test_loss, test_dists))

        tt.close()

        return test_dists, test_loss
