import time
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Original Code: 
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH

This code is an optimized version of the original code.
"""

plt.ioff()

def W(x, y, z, h):
    """
    Gaussian smoothing kernel (3D) using Dask arrays.
    
    Parameters:
      x, y, z : Dask arrays of coordinates (can be matrices)
      h       : smoothing length (float)
    
    Returns:
      w       : the evaluated kernel as a Dask array
    """
    # Use dask versions of sqrt and exp so the computation is lazy/parallel.
    r = da.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * da.sqrt(np.pi)))**3 * da.exp(-r**2 / h**2)
    return w

def gradW(x, y, z, h):
    """
    Gradient of the Gaussian smoothing kernel (3D) using Dask arrays.
    
    Returns:
      wx, wy, wz : gradients in the x, y, z directions (Dask arrays)
    """
    r = da.sqrt(x**2 + y**2 + z**2)
    n = -2 * da.exp(-r**2 / h**2) / (h**5 * np.pi**(3/2))
    wx = n * x
    wy = n * y
    wz = n * z
    return wx, wy, wz
def pairwise_sep_block(ri_chunk, rj):
    """
    Compute pairwise separations for a chunk of `ri` against the full `rj`.
    
    Parameters:
      ri_chunk : a subset of `ri` (chunked along M dimension)
      rj       : full `rj` array
    
    Returns:
      dx, dy, dz : chunks of M_chunk x N separations
    """
    dx = ri_chunk[:, 0, None] - rj[:, 0]  # Broadcasting for chunked computation
    dy = ri_chunk[:, 1, None] - rj[:, 1]
    dz = ri_chunk[:, 2, None] - rj[:, 2]
    
    return da.stack([dx, dy, dz], axis=0)
  
def getPairwiseSeparations_da(ri, rj):
    result = da.map_blocks(
        pairwise_sep_block, ri, rj,
        dtype=ri.dtype, drop_axis=0, new_axis=0
    )

    # Let Dask infer the correct shape dynamically
    dx, dy, dz = result[:3]  # Extract first three dimensions

    return dx, dy, dz
  
def getPairwiseSeparations_chunked(ri, rj):
    """
    Get pairwise separations between two sets of positions using Dask arrays with chunking.
    
    Parameters:
      ri : an M x 3 Dask array (first set of points)
      rj : an N x 3 Dask array (second set of points)
      
    Returns:
      dx, dy, dz : M x N Dask arrays of separations in each coordinate.
    """
    # Use map_blocks to apply the function to each chunk
    dx = da.map_blocks(lambda ri_chunk, rj_chunk: getPairwiseSeparations(ri_chunk, rj_chunk)[0], ri, rj, dtype=float)
    dy = da.map_blocks(lambda ri_chunk, rj_chunk: getPairwiseSeparations(ri_chunk, rj_chunk)[1], ri, rj, dtype=float)
    dz = da.map_blocks(lambda ri_chunk, rj_chunk: getPairwiseSeparations(ri_chunk, rj_chunk)[2], ri, rj, dtype=float)
    
    return dx, dy, dz
  
def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between two sets of positions using Dask arrays.
    
    Parameters:
      ri : an M x 3 Dask array (first set of points)
      rj : an N x 3 Dask array (second set of points)
      
    Returns:
      dx, dy, dz : M x N Dask arrays of separations in each coordinate.
    """
    M = ri.shape[0]
    N = rj.shape[0]
    
    # Reshape so that broadcasting will work as desired
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

def getDensity(r, pos, m, h):
    """
    Compute density at sample locations r from SPH particle positions.
    
    Parameters:
      r   : an M x 3 Dask array of sampling locations
      pos : an N x 3 Dask array of particle positions
      m   : particle mass (float)
      h   : smoothing length
      
    Returns:
      density : an M x 1 Dask array of computed densities
    """
    dx, dy, dz = getPairwiseSeparations(r, pos)
    density = da.sum(m * W(dx, dy, dz, h), axis=1).reshape((r.shape[0], 1))
    return density

def getPressure(rho, k, n):
    """
    Equation of state: compute pressure from density.
    
    Parameters:
      rho : density (Dask array)
      k   : equation-of-state constant
      n   : polytropic index
      
    Returns:
      Pressure (Dask array)
    """
    P = k * rho**(1+1/n)
    return P
  
#@profile
def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    """
    Calculate the acceleration on each SPH particle using Dask arrays.
    
    Parameters:
      pos   : N x 3 Dask array of positions
      vel   : N x 3 Dask array of velocities
      m     : particle mass
      h     : smoothing length
      k     : equation-of-state constant
      n     : polytropic index
      lmbda : external force constant
      nu    : viscosity coefficient
      
    Returns:
      a : N x 3 Dask array of accelerations
    """
    N = pos.shape[0]
    # Compute density at particle positions
    rho = getDensity(pos, pos, m, h)
    P = getPressure(rho, k, n)
    
    # Compute pairwise separations and gradients of the kernel
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    #dx, dy, dz = da.map_overlap(getPairwiseSeparations, pos, pos, depth=1, boundary='reflect')
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    # Compute the symmetric pressure term.
    term = (P / rho**2) + (P.T / rho.T**2)
    
    # Compute acceleration components
    ax = - da.sum(m * term * dWx, axis=1).reshape((N, 1))
    ay = - da.sum(m * term * dWy, axis=1).reshape((N, 1))
    az = - da.sum(m * term * dWz, axis=1).reshape((N, 1))
    
    # Pack the acceleration components together
    a = da.hstack([ax, ay, az])
    
    # Add external potential and viscous damping
    a = (a - lmbda * pos) - nu * vel

    return a
  
def getAcc_wrapper(pos_chunk, vel_chunk, m, h, k, n, lmbda, nu):
    return getAcc(pos_chunk, vel_chunk, m, h, k, n, lmbda, nu)
  
def main():
    """
    SPH simulation using Dask arrays for parallel computation.
    """
    #print(f"Started execution of dask_array.py")
    #start_time = time.time() 
    # Simulation parameters
    N = 1000            # Number of particles
    t = 0              # Current simulation time
    tEnd = 12          # End time
    dt = 0.04          # Timestep
    M_val = 2          # Total star mass
    R = 0.75           # Star radius
    h = 0.1            # Smoothing length
    k = 0.1            # Equation-of-state constant
    n = 1              # Polytropic index
    nu = 1             # Viscosity (damping)
    plotRealTime = False  # Set to True to see live plots
    
    # Set random seed and compute external force constant lambda
    np.random.seed(42)
    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * ((M_val * gamma(5/2+n) / (R**3)) / gamma(1+n))**(1/n) / R**2
    m = M_val / N  # Mass per particle
    
    # Convert to Dask arrays with a specified chunk size.
    chunk_size = 2050

    pos = da.random.standard_normal((N, 3), chunks=(chunk_size, 3))
    vel = da.zeros((N, 3), chunks=(chunk_size, 3))
    
    # Compute initial accelerations (force immediate computation for use in the loop)
    #acc = da.map_overlap(getAcc, pos, vel, m, h, k, n, lmbda, nu, depth=1, boundary='reflect')
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu).compute()
    #print("type acc after first compute: ", type(acc))
    Nt = int(np.ceil(tEnd / dt))
    #print("Number of timesteps:", Nt)
    # Prepare figure for plotting
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)
    
    # Main simulation loop
    for i in range(Nt):
        vel = vel + (dt / 2) * acc
        pos = pos + dt * vel

        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)  # Acceleration as a Dask array
        #acc = da.map_overlap(getAcc, pos, vel, m, h, k, n, lmbda, nu, depth=1, boundary='reflect')
        
        vel = vel + (dt / 2) * acc

        # Compute all at once
        pos, vel, acc = da.compute(pos, vel, acc)
        # Increment time
        t += dt
        
        #if i % 100 == 0:
         #   print(f"Completed {i} timesteps")
            
        if plotRealTime:
            ax1.cla()
            # Color by density (with some simple normalization)
            # For plotting, compute density at particle positions
            rho = getDensity(da.from_array(pos, chunks=(chunk_size, 3)),
                         da.from_array(pos, chunks=(chunk_size, 3)), m, h).compute()
        
            cval = np.minimum((rho - 3) / 3, 1).flatten()
            ax1.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])
            ax1.set_facecolor((0.1, 0.1, 0.1))
            
            ax2.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)
            ax2.plot(rlin, rho_analytic, color='gray', linewidth=2)
            rho_radial = getDensity(da.from_array(rr, chunks=(50, 3)),
                                     da.from_array(pos, chunks=(chunk_size, 3)), m, h).compute()
            ax2.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
    
    if plotRealTime:
        ax2.set_xlabel('radius')
        ax2.set_ylabel('density')
        plt.savefig('sph.png', dpi=240)
        plt.show()
        
    #end_time = time.time()
    #print(f"Execution time of dask_array.py: {end_time - start_time} seconds")
    return 0

if __name__ == "__main__":
    main()