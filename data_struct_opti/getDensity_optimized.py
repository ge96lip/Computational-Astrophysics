import numpy as np
from scipy.spatial import cKDTree
from data_struct_opti.sph_optimized import getPairwiseSeparations_inplace, optimizedW, getPairwiseSeparations, W

def getDensity(r, pos, m, h, kernel = W):
    """
    Get Density at sampling locations from SPH particle distribution
    """
    # Use in-place optimized function
    dx, dy, dz = getPairwiseSeparations_inplace(r, pos) # getPairwiseSeparations(r, pos) 
    # rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))
    #rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((M, 1))
    rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((r.shape[0], 1))
    return rho
def optimizedGetDensity(r, pos, m, h, kernel = optimizedW):
    """
    Optimized: Get Density at sampling locations from SPH particle distribution
    """
    # Build a KD-tree for efficient neighbor search
    tree = cKDTree(pos)
    rho = np.zeros(r.shape[0])

    # Iterate over each sampling location
    for i, ri in enumerate(r):
        # Find neighbors within the smoothing length h
        neighbors_idx = tree.query_ball_point(ri, h)

        # Calculate pairwise separations only for neighbors
        dx, dy, dz = getPairwiseSeparations(r[i:i+1], pos[neighbors_idx])

        # Calculate density using the smoothing kernel
        rho[i] = np.sum(m[neighbors_idx] * kernel(dx, dy, dz, h))

    return rho.reshape(-1, 1)
#@profile
def optimizedGetDensity_fast(r, pos, m, h, kernel=optimizedW):
    """
    Faster optimized version of getDensity() using batch KD-tree queries and vectorization.
    """
    # Build KD-tree for neighbor search
    tree = cKDTree(pos)

    # Query all points at once (batch operation)
    neighbors_idx_list = tree.query_ball_tree(tree, h)

    # Prepare density array
    rho = np.zeros(r.shape[0])

    # Process all particles in a vectorized manner
    for i, neighbors_idx in enumerate(neighbors_idx_list):
        dx, dy, dz = getPairwiseSeparations(r[i:i+1], pos[neighbors_idx])
        rho[i] = np.sum(m[neighbors_idx] * kernel(dx, dy, dz, h))

    return rho.reshape(-1, 1)