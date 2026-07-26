import os
import torch


def get_device():
    """Device selection. Honors env CCD_DEVICE (e.g. 'cuda', 'cuda:1', 'mps',
    'cpu') for the GPU server; otherwise prefers CUDA, then CPU."""
    override = os.environ.get('CCD_DEVICE')
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
