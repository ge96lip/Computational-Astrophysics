import sys
import multiprocessing
import time
import numpy as np
from dask.distributed import Client, wait
import math
from sph import getAcc as getAcc_sph
gamma = math.gamma

def W(x, y, z, h):
    """
    Gaussian smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)
    return w

def gradW(x, y, z, h):
    """
    Gradient of the Gaussian smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-r**2 / h**2) / h**5 / (np.pi)**(3/2)
    wx = n * x
    wy = n * y
    wz = n * z
    return wx, wy, wz

def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between two sets of coordinates.
    """
    M = ri.shape[0]
    N = rj.shape[0]
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    return dx, dy, dz

def getDensity_chunk(r_chunk, pos, m, h):
    """
    Compute density for a chunk of sampling positions.
    """
    dx, dy, dz = getPairwiseSeparations(r_chunk, pos)
    w_vals = W(dx, dy, dz, h)
    rho_chunk = np.sum(m * w_vals, axis=1, keepdims=True)
    return rho_chunk

def getDensity(r, pos, m, h, num_chunks=10):
    """
    Compute the density at the sampling locations `r` by splitting the work into chunks.
    """
    M = r.shape[0]
    chunk_size = int(np.ceil(M / num_chunks))
    futures = []
    for i in range(0, M, chunk_size):
        r_chunk = r[i:i+chunk_size]
        future = client.submit(getDensity_chunk, r_chunk, pos, m, h)
        futures.append(future)
    wait(futures)
    results = client.gather(futures)
    rho = np.vstack(results)
    return rho

def getPressure(rho, k, n):
    """
    Equation of state: P = k * rho^(1+1/n)
    """
    P = k * rho**(1 + 1/n)
    return P

def compute_acc_chunk(i_start, i_end, m, P, rho, dWx, dWy, dWz):
    """
    Compute the acceleration contribution for particles [i_start, i_end) by summing over all interactions.
    This function is designed to be submitted as a Dask task.
    """
    # For each particle i in the chunk, we need to compute:
    #   a_i = - sum_j m * (P_i/rho_i^2 + P_j/rho_j^2) * gradW(i,j)
    # We vectorize over the j-index:
    A = P[i_start:i_end, 0] / (rho[i_start:i_end, 0]**2)  # shape: (chunk,)
    B = P[:, 0] / (rho[:, 0]**2)                           # shape: (N,)
    terms = m * (A[:, np.newaxis] + B[np.newaxis, :])       # shape: (chunk, N)
    ax_chunk = -np.sum(terms * dWx[i_start:i_end, :], axis=1, keepdims=True)
    ay_chunk = -np.sum(terms * dWy[i_start:i_end, :], axis=1, keepdims=True)
    az_chunk = -np.sum(terms * dWz[i_start:i_end, :], axis=1, keepdims=True)
    return ax_chunk, ay_chunk, az_chunk

#@profile
def getAcc(pos, vel, m, h, k, n, lmbda, nu, num_chunks=10):
    """
    Calculate the acceleration on each SPH particle using parallel summation.
    """
    N = pos.shape[0]
    # Compute densities at the particle positions (parallelized)
    rho = getDensity(pos, pos, m, h)
    # Compute pressures from the equation of state
    P = getPressure(rho, k, n)
    # Compute full pairwise separations and gradients (NxN arrays)
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    # Break the acceleration summation into chunks over the i-index
    futures = []
    chunk_size = int(np.ceil(N / num_chunks))
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        future = client.submit(compute_acc_chunk, i, i_end, m, P, rho, dWx, dWy, dWz)
        futures.append(future)
    wait(futures)
    results = client.gather(futures)
    
    # Combine the chunked results into full acceleration arrays
    ax = np.vstack([r[0] for r in results])
    ay = np.vstack([r[1] for r in results])
    az = np.vstack([r[2] for r in results])
    a = np.hstack((ax, ay, az))
    
    # Add contributions from the external potential and viscosity
    a -= lmbda * pos
    a -= nu * vel
    return a

if __name__ == '__main__':
    # On macOS and Windows, protect the entry point of the program.
    from multiprocessing import freeze_support
    freeze_support()
    
    # Fix multiprocessing issues on macOS by setting the start method to "spawn"
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)

    # Start the Dask Client with a dashboard (adjust port if necessary)
    try:
        client = Client(dashboard_address=":8790")
        print("Dask dashboard available at:", client.dashboard_link)
    except Exception as e:
        print(f"Error starting Dask Client: {e}")
        print("Falling back to default settings.")
        client = Client()  # Fallback to default settings

    # Simulation parameters
    N = 1000               # Number of particles
    M = 2 / N              # Mass per particle
    h = 0.1                # Smoothing length
    pos = np.random.randn(N, 3)  # Initial positions
    vel = np.zeros_like(pos)     # Initial velocities
    n = 1                  # Polytropic index
    R = 0.75               # Star radius
    k = 0.1                # Equation-of-state constant

    # Compute the external force constant LAMBDA
    LAMBDA = 2 * k * (1 + n) * np.pi**(-3 / (2 * n)) * \
             (M * gamma(5/2 + n) / R**3 / gamma(1 + n))**(1/n) / R**2

    def run_simulation(num_chunks):
        """
        Run the simulation for a given number of chunks and return the elapsed time.
        """
        start_time = time.time()
        acc = getAcc(pos, vel, M, h, k, 1, LAMBDA, n, num_chunks=num_chunks)
        return time.time() - start_time, acc

    # Try different chunk sizes to test performance scaling
    chunk_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    acc_results = {}
    for chunks in chunk_sizes:
        print(f"\nRunning simulation with {chunks} chunk{'s' if chunks > 1 else ''}...")
        elapsed, acc = run_simulation(chunks)
        results[chunks] = elapsed
        acc_results[chunks] = acc
        print(f"Chunks: {chunks}, Time: {elapsed:.3f}s")

    # Close the Dask client
    client.close()
    acc_original = getAcc_sph( pos, vel, M, h, k, n, LAMBDA, 1 )

    assert np.allclose(acc, acc_original, atol=1e-6), "Values differ between methods!"
    
    print("\nSimulation complete. Results:", results)