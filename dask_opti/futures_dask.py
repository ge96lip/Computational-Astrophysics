import sys
import multiprocessing
import time
from matplotlib import pyplot as plt
import numpy as np
from dask.distributed import Client, wait
import math
from sph import getAcc as getAcc_sph  # if needed elsewhere

# For gamma calculations
gamma = math.gamma

def W(x, y, z, h):
    """
    Gaussian smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    return (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)

def gradW(x, y, z, h):
    """
    Gradient of the Gaussian smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-r**2 / h**2) / h**5 / (np.pi)**(3/2)
    return n * x, n * y, n * z

def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between two sets of coordinates.
    """
    M = ri.shape[0]
    N = rj.shape[0]
    dx = ri[:, 0].reshape((M, 1)) - rj[:, 0].reshape((N, 1)).T
    dy = ri[:, 1].reshape((M, 1)) - rj[:, 1].reshape((N, 1)).T
    dz = ri[:, 2].reshape((M, 1)) - rj[:, 2].reshape((N, 1)).T
    return dx, dy, dz

def getDensity_chunk(r_chunk, pos, m, h):
    """
    Compute density for a chunk of sampling positions.
    """
    dx, dy, dz = getPairwiseSeparations(r_chunk, pos)
    w_vals = W(dx, dy, dz, h)
    return np.sum(m * w_vals, axis=1, keepdims=True)

def getDensity(r, pos, m, h, num_chunks=10):
    """
    Compute the density at the sampling locations `r` by splitting the work into chunks.
    Avoid unnecessary broadcasting of large arrays.
    """
    M = r.shape[0]
    chunk_size = int(np.ceil(M / num_chunks))
    futures = []
    for i in range(0, M, chunk_size):
        r_chunk = r[i:i+chunk_size]
        future = client.submit(getDensity_chunk, r_chunk, pos, m, h)  # Remove scatter
        futures.append(future)
    
    wait(futures)
    results = client.gather(futures)
    return np.vstack(results)

def getPressure(rho, k, n):
    """
    Equation of state: P = k * rho^(1+1/n)
    """
    return k * rho**(1 + 1/n)

def compute_acc_chunk(i_start, i_end, m, P, rho, dWx, dWy, dWz):
    """
    Compute the acceleration contribution for particles [i_start, i_end) by summing over all interactions.
    """
    A = P[i_start:i_end, 0] / (rho[i_start:i_end, 0]**2)
    B = P[:, 0] / (rho[:, 0]**2)
    terms = m * (A[:, None] + B[None, :])
    ax_chunk = -np.sum(terms * dWx[i_start:i_end, :], axis=1, keepdims=True)
    ay_chunk = -np.sum(terms * dWy[i_start:i_end, :], axis=1, keepdims=True)
    az_chunk = -np.sum(terms * dWz[i_start:i_end, :], axis=1, keepdims=True)
    return ax_chunk, ay_chunk, az_chunk

#@profile
def getAcc(pos, vel, m, h, k, n, lmbda, nu, num_chunks=10):
    """
    Calculate the acceleration on each SPH particle using parallel summation.
    Scatter large arrays so that they are not repeatedly embedded in the task graph.
    """
    N = pos.shape[0]
    # Compute densities and pressures
    rho = getDensity(pos, pos, m, h, num_chunks=100)
    P = getPressure(rho, k, n)
    
    # Compute full pairwise separations and kernel gradients (NxN arrays)
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    # Scatter and replicate large arrays for resilience
    futures = {}
    for name, arr in zip(["P", "rho", "dWx", "dWy", "dWz"], [P, rho, dWx, dWy, dWz]):
        futures[name] = client.scatter(arr, broadcast=True)
        client.replicate([futures[name]])
        
    # Break the acceleration summation into chunks over the i-index
    acc_futures = []
    chunk_size = int(np.ceil(N / num_chunks))
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        future = client.submit(
            compute_acc_chunk, i, i_end, m,
            futures["P"], futures["rho"],
            futures["dWx"], futures["dWy"], futures["dWz"]
        )
        acc_futures.append(future)
    wait(acc_futures)
    results = client.gather(acc_futures)
    
    # Combine chunked results into full acceleration arrays
    ax = np.vstack([r[0] for r in results])
    ay = np.vstack([r[1] for r in results])
    az = np.vstack([r[2] for r in results])
    a = np.hstack((ax, ay, az))
    
    # Add contributions from external potential and viscosity
    a -= lmbda * pos
    a -= nu * vel
    return a

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # On macOS and Windows, set the start method to "spawn"
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)

    # Start the Dask client with multiple workers for better resilience
    client = Client(n_workers=2, threads_per_worker=1, dashboard_address=":8790")
    print("Dask dashboard available at:", client.dashboard_link)

    # Simulation parameters
    N = 1000                # Number of particles
    M = 2 / N               # Mass per particle
    h = 0.1                 # Smoothing length
    pos = np.random.randn(N, 3)  # Initial positions
    vel = np.zeros_like(pos)     # Initial velocities
    n = 1                   # Polytropic index
    R = 0.75                # Star radius
    k = 0.1                 # Equation-of-state constant
    tEnd = 12
    dt = 0.04
    t = 0
    
    # Compute the external force constant LAMBDA
    LAMBDA = 2 * k * (1 + n) * np.pi**(-3 / (2 * n)) * \
             (M * gamma(5/2 + n) / R**3 / gamma(1 + n))**(1/n) / R**2
    num_chunks = 10

    # Initial acceleration
    acc = getAcc(pos, vel, M, h, k, n, LAMBDA, 1, num_chunks)
    Nt = int(np.ceil(tEnd / dt))
    
    # Prepare figure for plotting (if desired)
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = LAMBDA / (4 * k) * (R**2 - rlin**2)
    
    # Main simulation loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2
        # Drift
        pos += vel * dt
        # Update acceleration
        acc = getAcc(pos, vel, M, h, k, n, LAMBDA, 1, num_chunks=num_chunks)
        # (1/2) kick
        vel += acc * dt/2
        t += dt
        # Optionally get density for plotting
        rho = getDensity(pos, pos, M, h)
        
    # Close the Dask client
    client.close()