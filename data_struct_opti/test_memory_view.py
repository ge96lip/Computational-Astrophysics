from scipy.special import gamma
import numpy as np
import pytest
from data_struct_opti.memory_view import getAcc as getAcc_mem
from data_struct_opti.sph_optimized import getAcc as getAcc_opt

def test_memory_view():
    """
    Compare rho values from original and optimized getDensity methods.
    """

    N = 400
    t = 0
    tEnd = 12
    dt = 0.04
    M = 2
    R = 0.75
    h = 0.1
    k = 0.1
    n = 1
    nu = 1
    plotRealTime = False  # Disable real-time plotting for profiling

    # Generate Initial Conditions
    np.random.seed(42)
    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2
    #m = M/N
    m = np.full(N, M / N)
    pos = np.random.randn(N, 3)
    vel = np.zeros(pos.shape)

    # Compute rho using the original and optimized getDensity
    acc_mem =getAcc_mem(pos, vel, m, h, k, n, lmbda, nu)
    acc_opt = getAcc_opt(pos, vel, m, h, k, n, lmbda, nu)

    # Check that the values are close
    assert np.allclose(acc_mem, acc_opt,  atol=1e-6), "Rho values differ between methods!"