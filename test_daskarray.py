from dask.distributed import Client
import numpy as np
import pytest
import dask.array as da
from dask_array import getAcc as getAcc_dask
from dask_array_v2 import getAcc as getAcc_dask_v2
from sph import getAcc as getAcc_sph
from scipy.special import gamma

def test_daskAcc():
    """
    SPH simulation using Dask arrays for parallel computation.
    """
    # Simulation parameters
    N = 5000            # Number of particles
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
    
    # For small arrays, use one chunk to reduce scheduling overhead.
    chunk_size = N
    pos = da.from_array(pos_np, chunks=(chunk_size, 3))
    vel = da.from_array(vel_np, chunks=(chunk_size, 3))

    # Compute initial accelerations (force immediate computation for use in the loop)
    acc_dask = getAcc_dask(pos, vel, m, h, k, n, lmbda, nu).compute()

    M         = 2      # star mass
    pos = np.random.randn(N, 3)  # Initial positions
    vel = np.zeros_like(pos) 
   
    # acc_chunk = getAcc_chunk(pos, vel, M, h, k, n, lmbda, nu)
    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)

    # calculate initial gravitational accelerations
    acc_original = getAcc_sph( pos, vel, m, h, k, n, lmbda, nu )

    # Check that the values are close
    assert np.allclose(acc_dask, acc_original, atol=1e-6), "Rho values differ between methods!"