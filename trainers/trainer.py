import torch
import torch.nn.functional as F

from models.base_model import get_device

def to_host(x):
    if(torch.is_tensor(x)):
        x = x.detach().cpu()
        return float(x) if x.numel() == 1 else x.numpy()
    return x

def slice_batch(graphs, a, b):
    g = graphs[a:b]
    for attr in ('_host_adj', '_n_real', '_node_mask'):
        v = getattr(graphs, attr, None)
        if(v is not None):
            setattr(g, attr, v[a:b])
    return g

class Trainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data_loader = data
        self.config = config
        self.regression = config.dataset_name == 'ZINC'
        self.device = get_device()
        fused = {'fused': True} if self.device.type == 'cuda' else {}
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate, **fused)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=config.hyperparams.decay_rate)
        # accumulate grads over chunks of the batch, it's exact since the loss is a sum
        self.micro_batch = config.hyperparams.get('micro_batch', 0)

    def loss(self, graphs, labels):
        scores = self.model(graphs)
        if(self.regression):
            diffs = (scores - labels).abs().sum(dim=0)
            return diffs.sum(), diffs.detach()
        loss = F.cross_entropy(scores, labels, reduction='sum')
        return loss, torch.eq(torch.argmax(scores, dim=1), labels).sum()

    def train_epoch(self):
        self.data_loader.initialize('train')
        self.model.train()
        tot_loss, tot_metric = 0., 0.
        for _ in range(self.data_loader.num_iterations_train):
            loss, metric = self.train_step()
            tot_loss += loss
            tot_metric += metric
        self.scheduler.step()
        n = self.data_loader.train_size
        return to_host(tot_metric) / n, float(tot_loss) / n

    def train_step(self):
        graphs, labels = self.data_loader.next_batch()
        mb = self.micro_batch
        self.optimizer.zero_grad()
        if(mb and graphs.shape[0] > mb):
            loss, metric = 0.0, 0.0
            for s in range(0, graphs.shape[0], mb):
                l, r = self.loss(slice_batch(graphs, s, s + mb), labels[s:s + mb])
                l.backward()
                loss = loss + l.detach()
                metric = metric + r
        else:
            loss, metric = self.loss(graphs, labels)
            loss.backward()
        self.optimizer.step()
        return loss.detach(), metric

    def validate(self):
        self.data_loader.initialize('val')
        self.model.eval()
        tot_loss, tot_correct = 0., 0.
        with torch.no_grad():
            for _ in range(self.data_loader.num_iterations_val):
                graphs, labels = self.data_loader.next_batch()
                loss, correct = self.loss(graphs, labels)
                tot_loss += loss.detach()
                tot_correct += correct
        n = self.data_loader.val_size
        return to_host(tot_correct) / n, float(tot_loss) / n
