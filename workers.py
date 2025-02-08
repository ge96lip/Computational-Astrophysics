# workers.py
import numpy as np

def pairwise_worker(args):
    """
    Worker function for multiprocessing.
    Each worker computes pairwise separations for a block of the dataset.
    """
    ri, rj = args
    dx = ri[:, 0][:, None] - rj[:, 0][None, :]
    dy = ri[:, 1][:, None] - rj[:, 1][None, :]
    dz = ri[:, 2][:, None] - rj[:, 2][None, :]
    return dx, dy, dz