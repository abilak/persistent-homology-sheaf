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
    args = argparser.parse_args()
    return args
