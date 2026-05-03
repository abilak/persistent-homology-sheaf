import os
import sys
import torch
import numpy as np
from datetime import datetime

"""
How To:
Example for running from command line:
    python main_scripts/main_zinc_experiment.py --config=configs/zinc_config.json

First prepare the data:
    python scripts/prepare_zinc.py
"""
# Change working directory to project's main directory, and add it to path
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


def main():
    try:
        args = get_args()
        config = process_config(args.config, dataset_name='ZINC', use_topology=args.use_topology)
    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    torch.manual_seed(100)
    np.random.seed(100)

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    print(config.architecture)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)

    data = DataGenerator(config)
    model_wrapper = ModelWrapper(config, data)
    trainer = Trainer(model_wrapper, data, config)

    # train
    trainer.train()

    # test with best model
    test_dists, test_loss = trainer.test(load_best_model=True)
    print("Test MAE: {:.4f}".format(test_dists.item()))

    doc_utils.summary_qm9_results(config.summary_dir, test_dists, test_loss, trainer.best_epoch)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Runtime: {}'.format(datetime.now() - start))
