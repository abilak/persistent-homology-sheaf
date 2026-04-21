import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--dataset_name',
        metavar='D',
        default='',
        help='The dataset name (overrides config file value)')
    argparser.add_argument(
        '--use_topology',
        action='store_true',
        default=None,
        help='Enable persistent homology topology branch (overrides config)')
    argparser.add_argument(
        '--use_checkpoint',
        action='store_true',
        default=False,
        help='Save checkpoints every epoch and resume from last if available')
    argparser.add_argument(
        '--resume_dir',
        default=None,
        help='Path to existing experiment dir to resume from (e.g. experiments/10fold_...)')
    args = argparser.parse_args()
    return args
