
import numpy as np
import dask.array as da
from delayed_dask import getAcc as getAcc_dask
from sph import getAcc as getAcc_sph
from scipy.special import gamma

def test_daskAcc():
    """
    SPH simulation using Dask arrays for parallel computation.
    """
    # Simulation parameters
    N = 400            # Number of particles
    M_val = 2          # Total star mass
    R = 0.75           # Star radius
    h = 0.1            # Smoothing length
    k = 0.1            # Equation-of-state constant
    n = 1              # Polytropic index
    nu = 1             # Viscosity (damping)

    # Set random seed and compute external force constant lambda
    np.random.seed(42)
    lmbda = 2 * k * (1 + n) * np.pi**(-3/(2*n)) * ((M_val * gamma(5/2+n) / (R**3)) / gamma(1+n))**(1/n) / R**2
    m = M_val / N  # Particle mass
    
    # Generate initial conditions (as NumPy arrays)
    pos_np = np.random.randn(N, 3)
    vel_np = np.zeros((N, 3))

    acc_dask = getAcc_dask(pos_np, vel_np, m, h, k, n, lmbda, nu)

    # calculate initial gravitational accelerations
    acc_original = getAcc_sph(pos_np, vel_np, m, h, k, n, lmbda, nu )

    # Check that the values are close
    assert np.allclose(acc_dask, acc_original, atol=1e-6), "values differ between methods!"