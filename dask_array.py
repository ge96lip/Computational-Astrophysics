import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from scipy.special import gamma

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
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    # Compute the symmetric pressure term.
    # Note: P and rho are of shape (N,1) so we use their transposes to broadcast.
    term = (P / rho**2) + (P.T / (rho.T**2))
    ax = - da.sum(m * term * dWx, axis=1).reshape((N, 1))
    ay = - da.sum(m * term * dWy, axis=1).reshape((N, 1))
    az = - da.sum(m * term * dWz, axis=1).reshape((N, 1))
    
    # Pack the acceleration components together
    a = da.hstack([ax, ay, az])
    
    # Add external potential and viscous damping
    a = a - lmbda * pos - nu * vel
    return a

def main():
    """
    SPH simulation using Dask arrays for parallel computation.
    """
    # Simulation parameters
    N = 10000            # Number of particles
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
    
    # Generate initial conditions (NumPy arrays)
    pos_np = np.random.randn(N, 3)
    vel_np = np.zeros((N, 3))
    
    # Convert to Dask arrays with a specified chunk size.
    chunk_size = 2500
    pos = da.from_array(pos_np, chunks=(chunk_size, 3))
    vel = da.from_array(vel_np, chunks=(chunk_size, 3))
    
    # Compute initial accelerations (force immediate computation for use in the loop)
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu).compute()
    
    Nt = int(np.ceil(tEnd / dt))
    
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
        # (1/2) kick: update velocity half-step
        vel = (vel + (dt / 2) * da.from_array(acc, chunks=(chunk_size, 3))).compute()
        
        # Drift: update positions
        pos = (pos + dt * da.from_array(vel, chunks=(chunk_size, 3))).compute()
        
        # Update accelerations using the new positions and velocities.
        # (Convert back to dask arrays before calling getAcc.)
        pos_dask = da.from_array(pos, chunks=(chunk_size, 3))
        vel_dask = da.from_array(vel, chunks=(chunk_size, 3))
        
        acc = getAcc(pos_dask, vel_dask, m, h, k, n, lmbda, nu).compute()
        
        # (1/2) kick: complete velocity update
        vel = (da.from_array(vel, chunks=(chunk_size, 3)) + (dt / 2) * da.from_array(acc, chunks=(chunk_size, 3))).compute()
        
        # Increment time
        t += dt
        
        # For plotting, compute density at particle positions
        rho = getDensity(da.from_array(pos, chunks=(chunk_size, 3)),
                         da.from_array(pos, chunks=(chunk_size, 3)), m, h).compute()
        
        if plotRealTime:
            ax1.cla()
            # Color by density (with some simple normalization)
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
    
    return 0

if __name__ == "__main__":
    main()