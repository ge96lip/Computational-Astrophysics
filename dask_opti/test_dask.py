from dask.distributed import Client
import numpy as np
import pytest
import dask.array as da
from dask_array import getAcc as getAcc_dask, getPairwiseSeparations_chunked as getPairwiseSeparations_da, W as W_da, getPressure, gradW as gradW_da, getDensity as getDensity_da
from sph import getAcc as getAcc_sph, getDensity, getPairwiseSeparations, W, gradW
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

    np.random.seed(42)
    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * ((M_val * gamma(5/2+n) / (R**3)) / gamma(1+n))**(1/n) / R**2
    m = M_val / N  # Mass per particle
    
    # Convert to Dask arrays with a specified chunk size.
    chunk_size = 2050

    pos_np = np.random.standard_normal((N, 3))
    vel_np = np.zeros_like(pos_np)

    pos_da = da.from_array(pos_np, chunks=(chunk_size, 3))
    vel_da = da.from_array(vel_np, chunks=(chunk_size, 3))
        
    rho = getDensity( pos_np, pos_np, m, h )
	
	# Get the pressures
    P = getPressure(rho, k, n)
    
    dx_sph, dy_sph, dz_sph = getPairwiseSeparations(pos_np, pos_np)
    dx_dask, dy_dask, dz_dask = getPairwiseSeparations_da(pos_da, pos_da)
    dx_dask, dy_dask, dz_dask = dx_dask.compute(), dy_dask.compute(), dz_dask.compute()
    """
    print("Max diff in dx:", np.max(np.abs(dx_sph - dx_dask)))
    print("Max diff in dy:", np.max(np.abs(dy_sph - dy_dask)))
    print("Max diff in dz:", np.max(np.abs(dz_sph - dz_dask)))"""
    dWx_sph, dWy_sph, dWz_sph = gradW(dx_sph, dy_sph, dz_sph, h)
    dWx_dask, dWy_dask, dWz_dask = gradW(dx_dask, dy_dask, dz_dask, h)

    print("Max diff in dWx:", np.max(np.abs(dWx_sph - dWx_dask)))
    print("Max diff in dWy:", np.max(np.abs(dWy_sph - dWy_dask)))
    print("Max diff in dWz:", np.max(np.abs(dWz_sph - dWz_dask)))
    
    rho_sph = getDensity(pos_np, pos_np, m, h)
    rho_dask = getDensity(pos_da, pos_da, m, h).compute()

    # Compute pressure term (make sure you use the correct variable name for pressure)
    P_sph = getPressure(rho_sph, k, n)
    P_dask = getPressure(rho_dask, k, n)

    # Compute acceleration components using Dask
    term_dask = (P_dask / rho_dask**2) + (P_dask.T / rho_dask.T**2)

    ax_dask = - da.sum(m * term_dask * dWx_dask, axis=1).reshape((N, 1))
    ay_dask = - da.sum(m * term_dask * dWy_dask, axis=1).reshape((N, 1))
    az_dask = - da.sum(m * term_dask * dWz_dask, axis=1).reshape((N, 1))

    # Convert the Dask acceleration components to NumPy
    ax_dask = ax_dask.compute()
    ay_dask = ay_dask.compute()
    az_dask = az_dask.compute()

    # Pack the acceleration components together
    a_dask = np.hstack([ax_dask, ay_dask, az_dask])
    
    # Compute acceleration using NumPy
    ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx_sph, 1).reshape((N,1))
    ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy_sph, 1).reshape((N,1))
    az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz_sph, 1).reshape((N,1))
	
	# pack together the acceleration components
    a = np.hstack((ax,ay,az))
    diff_before = np.max(np.abs(a - a_dask))
	# Add external potential force
    a -= lmbda * pos_np
    a -= nu * vel_np
    
    a_dask -= lmbda * pos_np
    a_dask -= nu * vel_np
    diff_after = np.max(np.abs(a - a_dask))

    print("Max diff before external force:", diff_before)
    print("Max diff after external force:", diff_after)
    
    # Compute the max difference
    max_diff_a = np.max(np.abs(a - a_dask))  # Now both are NumPy arrays

    print("Max diff in density:", np.max(np.abs(rho_sph - rho_dask)))
    print("Max diff in a:", max_diff_a)
    print("Max diff in pos:", np.max(np.abs(pos_np - pos_da.compute())))
    print("Max diff in vel:", np.max(np.abs(vel_np - vel_da.compute())))
    
    acc_dask = getAcc_dask(pos_da, vel_da, m, h, k, n, lmbda, nu)

    acc_dask = acc_dask.compute()
    
    acc = getAcc_sph( pos_np, vel_np, m, h, k, n, lmbda, nu )
    assert np.allclose(acc_dask, acc, atol=1e-6), "values differ between methods!"
    #return acc_dask, acc

"""if __name__ == "__main__":
    acc_dask, acc = test_daskAcc()
    diff = acc_dask - acc 
    print("Type of diff:", type(diff))
    print("Type of acc_dask:", type(diff))  # To infer if it's NumPy or Dask
    print("Shape of diff:", diff.shape)  # Check if expected shape
    print("Max absolute difference:", np.max(np.abs(diff)))
    print("Mean absolute difference:", np.mean(np.abs(diff)))
    print("Standard deviation of differences:", np.std(diff))
    print("First few differences:\n", diff[:5])
    assert np.allclose(acc_dask, acc, atol=1e-6), "values differ between methods!"""