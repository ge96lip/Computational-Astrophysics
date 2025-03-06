
import numpy as np 
import numexpr as ne

from workers import pairwise_worker

def getPairwiseSeparations_numpy(ri, rj):
    """
    Optimized NumPy version avoiding large temporary arrays and improving memory locality.
    """
    dx = np.subtract.outer(ri[:, 0], rj[:, 0])
    dy = np.subtract.outer(ri[:, 1], rj[:, 1])
    dz = np.subtract.outer(ri[:, 2], rj[:, 2])
    return dx, dy, dz
def getPairwiseSeparations_rolling(ri, rj, previous_dx=None, previous_dy=None, previous_dz=None):
    """
    Rolling window approach to reuse previous calculations.
    """
    if previous_dx is None or previous_dy is None or previous_dz is None:
        return getPairwiseSeparations_numpy(ri, rj)

    # Shift by one and recompute only new elements
    dx_new = np.roll(previous_dx, shift=-1, axis=1)
    dy_new = np.roll(previous_dy, shift=-1, axis=1)
    dz_new = np.roll(previous_dz, shift=-1, axis=1)

    dx_new[:, -1] = ri[:, 0] - rj[-1, 0]
    dy_new[:, -1] = ri[:, 1] - rj[-1, 1]
    dz_new[:, -1] = ri[:, 2] - rj[-1, 2]

    return dx_new, dy_new, dz_new

def getPairwiseSeparations_numexpr(ri, rj):
    """
    Optimized version using NumExpr to leverage CPU parallelism.
    """
    M, N = ri.shape[0], rj.shape[0]
    
    # Explicit reshaping to avoid broadcasting issues
    ri_x = ri[:, 0].reshape(M, 1)
    ri_y = ri[:, 1].reshape(M, 1)
    ri_z = ri[:, 2].reshape(M, 1)
    
    rj_x = rj[:, 0].reshape(1, N)
    rj_y = rj[:, 1].reshape(1, N)
    rj_z = rj[:, 2].reshape(1, N)

    # Allocate memory for output
    dx = np.empty((M, N), dtype=np.float64)
    dy = np.empty((M, N), dtype=np.float64)
    dz = np.empty((M, N), dtype=np.float64)

    # Use NumExpr for fast evaluation
    dx[:] = ne.evaluate("ri_x - rj_x")
    dy[:] = ne.evaluate("ri_y - rj_y")
    dz[:] = ne.evaluate("ri_z - rj_z")

    return dx, dy, dz
def getPairwiseSeparations_fully_vectorized(ri, rj):
    """
    Fully vectorized pairwise separations using NumPy advanced indexing
    to minimize function calls and improve vectorization efficiency.
    """
    return ri[:, None, :] - rj[None, :, :]
def getPairwiseSeparations_lists(ri, rj):
    """
    List-based approach that avoids NumPy overhead for small datasets.
    """
    dx = [[rix - rjx for rjx in rj[:, 0]] for rix in ri[:, 0]]
    dy = [[riy - rjy for rjy in rj[:, 1]] for riy in ri[:, 1]]
    dz = [[riz - rjz for rjz in rj[:, 2]] for riz in ri[:, 2]]

    return np.array(dx), np.array(dy), np.array(dz)

def getPairwiseSeparations_tuples(ri, rj):
    """
    Tuple-based approach that avoids unnecessary allocations.
    """
    dx = tuple(tuple(rix - rjx for rjx in rj[:, 0]) for rix in ri[:, 0])
    dy = tuple(tuple(riy - rjy for rjy in rj[:, 1]) for riy in ri[:, 1])
    dz = tuple(tuple(riz - rjz for rjz in rj[:, 2]) for riz in ri[:, 2])

    return np.array(dx), np.array(dy), np.array(dz)



def getPairwiseSeparations_blockwise(ri, rj, block_size=512):
    """
    Optimized block-wise pairwise separations computation for better cache locality.
    """
    M, N = ri.shape[0], rj.shape[0]

    dx = np.empty((M, N), dtype=np.float64)
    dy = np.empty((M, N), dtype=np.float64)
    dz = np.empty((M, N), dtype=np.float64)

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            i_end = min(i + block_size, M)
            j_end = min(j + block_size, N)

            dx[i:i_end, j:j_end] = ri[i:i_end, 0][:, None] - rj[j:j_end, 0][None, :]
            dy[i:i_end, j:j_end] = ri[i:i_end, 1][:, None] - rj[j:j_end, 1][None, :]
            dz[i:i_end, j:j_end] = ri[i:i_end, 2][:, None] - rj[j:j_end, 2][None, :]
    return dx, dy, dz
def getPairwiseSeparations_parallel(ri, rj, num_workers=8):
    """
    Parallelized version of pairwise separation using multiprocessing.
    """
    M = ri.shape[0]
    block_size = M // num_workers

    # Convert NumPy arrays to lists before passing them (avoiding pickle issues)
    ri_list = ri.tolist()
    rj_list = rj.tolist()

    blocks = [(np.array(ri_list[i:i+block_size]), np.array(rj_list)) for i in range(0, M, block_size)]

    with mp.Pool(num_workers) as pool:
        results = pool.map(pairwise_worker, blocks)

    # Merge results
    dx = np.vstack([res[0] for res in results])
    dy = np.vstack([res[1] for res in results])
    dz = np.vstack([res[2] for res in results])

    return dx, dy, dz

def optimizedGetPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between 2 sets of coordinates using efficient NumPy broadcasting.
    """
    return (ri[:, None, :] - rj[None, :, :]).transpose(2, 0, 1)