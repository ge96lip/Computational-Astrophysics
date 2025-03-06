import numpy as np
import pytest
from data_struct_opti.memory_view import W, optimizedW, getPairwiseSeparations

def testW(): 
    """
    Get Density at sampling locations from SPH particle distribution
    """
    N = 400
    r = np.random.randn(N, 3)
    dx, dy, dz = getPairwiseSeparations(r, r)
    h = 0.1 

    # Calculate the pairwise distances
    pairwise_r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Compute weights
    w = W(dx, dy, dz, h)
    w_optimized = optimizedW(dx, dy, dz, h)

    # Apply combined filtering conditions
    mask = (w > 1e-12) & (pairwise_r <= h)
    assert np.allclose(
        w[mask],
        w_optimized[mask],
        atol=1e-6
    )