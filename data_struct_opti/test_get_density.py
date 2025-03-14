import numpy as np
import pytest
from data_struct_opti.getDensity_optimized import getDensity, optimizedGetDensity, optimizedGetDensity_fast
from data_struct_opti.sph_optimized import optimizedW

def test_get_density():
    """
    Compare rho values from original and optimized getDensity methods.
    """

    # Parameters for test
    N = 400
    M = 2
    R = 0.75
    h = 0.1
    np.random.seed(42)
    m = np.full(N, M / N)  # Mass of each particle as an array
    pos = np.random.randn(N, 3)
    r = pos  # Sampling locations are the particle positions

    # Compute rho using the original and optimized getDensity
    rho_original = getDensity(r, pos, m, h, optimizedW)
    rho_optimized = optimizedGetDensity(r, pos, m, h)
    rho_optimized_fast = optimizedGetDensity_fast(r, pos, m, h)

    # Check that the values are close
    assert np.allclose(rho_original, rho_optimized, rho_optimized_fast,  atol=1e-6), "Rho values differ between methods!"