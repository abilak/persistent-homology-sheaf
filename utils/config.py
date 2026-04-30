import json
from easydict import EasyDict
import os
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")

NUM_LABELS = {'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'DD':89, 'QM9': 18}
NUM_CLASSES = {'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2, 'DD':2, 'QM9': 12}
LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0005, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001, 'DD': 0.0001}
DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 0.5, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0, 'DD': 0.5}
CHOSEN_EPOCH = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 200, 'NCI1': 200, 'NCI109': 200, 'PROTEINS': 100, 'PTC': 180, 'DD': 150}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file, dataset_name, use_topology=None):
    config = get_config_from_json(json_file)
    if dataset_name != '':
        config.dataset_name = dataset_name
    config.num_classes = NUM_CLASSES[config.dataset_name]
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join(EXPERIMENTS_ROOT, config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join(EXPERIMENTS_ROOT, config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
        config.hyperparams.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    config.distributed_fold = None  # specific for distrib 10fold - override to use as a flag

    # Topology defaults (for configs that don't specify these)
    arch = config.architecture
    if not hasattr(arch, 'use_topology'):
        arch.use_topology = False
    if not hasattr(arch, 'topo_hidden_dim'):
        arch.topo_hidden_dim = 64
    if not hasattr(arch, 'topo_max_ph_dim'):
        arch.topo_max_ph_dim = 1
    if not hasattr(arch, 'topo_num_stats'):
        arch.topo_num_stats = 4
    if not hasattr(arch, 'topo_max_simplex_dim'):
        arch.topo_max_simplex_dim = 2

    # CLI override
    if use_topology is not None:
        arch.use_topology = use_topology

    return config


if __name__ == '__main__':
    config = process_config('../configs/10fold_config.json')
    print(config.values())
