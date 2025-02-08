from matplotlib import pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.spatial import cKDTree
import numexpr as ne
import multiprocessing as mp
from workers import pairwise_worker 
mp.set_start_method("fork", force=True)

def W(x, y, z, h):
    """
    Gausssian Smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)
    return w

def optimizedW(x, y, z, h):
    """
    Gaussian Smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    mask = r <= h  # Only consider distances within h
    w = np.zeros_like(r)
    w[mask] = (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r[mask]**2 / h**2)
    return w

def gradW(x, y, z, h):
    """
    Gradient of the Gausssian Smoothing kernel (3D)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-r**2 / h**2) / h**5 / (np.pi)**(3/2)
    wx = n * x
    wy = n * y
    wz = n * z
    return wx, wy, wz

def optimizedGetPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between 2 sets of coordinates using efficient NumPy broadcasting.
    """
    return (ri[:, None, :] - rj[None, :, :]).transpose(2, 0, 1)

def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between 2 sets of coordinates
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


def getPairwiseSeparations_numpy(ri, rj):
    """
    Optimized NumPy version avoiding large temporary arrays and improving memory locality.
    """
    dx = np.subtract.outer(ri[:, 0], rj[:, 0])
    dy = np.subtract.outer(ri[:, 1], rj[:, 1])
    dz = np.subtract.outer(ri[:, 2], rj[:, 2])
    return dx, dy, dz

def getPairwiseSeparations_numexpr(ri, rj):
    """
    Optimized version using NumExpr to leverage CPU parallelism.
    """
    M, N = ri.shape[0], rj.shape[0]
    
    # Explicit reshaping to avoid broadcasting issues
    ri_x = ri[:, 0].reshape(M, 1)
    ri_y = ri[:, 1].reshape(M, 1)
    ri_z = ri[:, 2].reshape(M, 1)
    
    rj_x = rj[:, 0].reshape(1, N)
    rj_y = rj[:, 1].reshape(1, N)
    rj_z = rj[:, 2].reshape(1, N)

    # Allocate memory for output
    dx = np.empty((M, N), dtype=np.float64)
    dy = np.empty((M, N), dtype=np.float64)
    dz = np.empty((M, N), dtype=np.float64)

    # Use NumExpr for fast evaluation
    dx[:] = ne.evaluate("ri_x - rj_x")
    dy[:] = ne.evaluate("ri_y - rj_y")
    dz[:] = ne.evaluate("ri_z - rj_z")

    return dx, dy, dz
def getPairwiseSeparations_fully_vectorized(ri, rj):
    """
    Fully vectorized pairwise separations using NumPy advanced indexing
    to minimize function calls and improve vectorization efficiency.
    """
    return ri[:, None, :] - rj[None, :, :]
def getPairwiseSeparations_lists(ri, rj):
    """
    List-based approach that avoids NumPy overhead for small datasets.
    """
    dx = [[rix - rjx for rjx in rj[:, 0]] for rix in ri[:, 0]]
    dy = [[riy - rjy for rjy in rj[:, 1]] for riy in ri[:, 1]]
    dz = [[riz - rjz for rjz in rj[:, 2]] for riz in ri[:, 2]]

    return np.array(dx), np.array(dy), np.array(dz)

def getPairwiseSeparations_tuples(ri, rj):
    """
    Tuple-based approach that avoids unnecessary allocations.
    """
    dx = tuple(tuple(rix - rjx for rjx in rj[:, 0]) for rix in ri[:, 0])
    dy = tuple(tuple(riy - rjy for rjy in rj[:, 1]) for riy in ri[:, 1])
    dz = tuple(tuple(riz - rjz for rjz in rj[:, 2]) for riz in ri[:, 2])

    return np.array(dx), np.array(dy), np.array(dz)
def getPairwiseSeparations_inplace(ri, rj, dx, dy, dz):
    """
    In-place computation to minimize memory allocations.
    """
    ri = ri.astype(np.float32)
    rj = rj.astype(np.float32)
    np.subtract(ri[:, 0][:, None], rj[:, 0][None, :], out=dx)
    np.subtract(ri[:, 1][:, None], rj[:, 1][None, :], out=dy)
    np.subtract(ri[:, 2][:, None], rj[:, 2][None, :], out=dz)

    return dx, dy, dz
def getPairwiseSeparations_rolling(ri, rj, previous_dx=None, previous_dy=None, previous_dz=None):
    """
    Rolling window approach to reuse previous calculations.
    """
    if previous_dx is None or previous_dy is None or previous_dz is None:
        return getPairwiseSeparations_numpy(ri, rj)

    # Shift by one and recompute only new elements
    dx_new = np.roll(previous_dx, shift=-1, axis=1)
    dy_new = np.roll(previous_dy, shift=-1, axis=1)
    dz_new = np.roll(previous_dz, shift=-1, axis=1)

    dx_new[:, -1] = ri[:, 0] - rj[-1, 0]
    dy_new[:, -1] = ri[:, 1] - rj[-1, 1]
    dz_new[:, -1] = ri[:, 2] - rj[-1, 2]

    return dx_new, dy_new, dz_new

def getPairwiseSeparations_blockwise(ri, rj, block_size=512):
    """
    Optimized block-wise pairwise separations computation for better cache locality.
    """
    M, N = ri.shape[0], rj.shape[0]

    dx = np.empty((M, N), dtype=np.float64)
    dy = np.empty((M, N), dtype=np.float64)
    dz = np.empty((M, N), dtype=np.float64)

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            i_end = min(i + block_size, M)
            j_end = min(j + block_size, N)

            dx[i:i_end, j:j_end] = ri[i:i_end, 0][:, None] - rj[j:j_end, 0][None, :]
            dy[i:i_end, j:j_end] = ri[i:i_end, 1][:, None] - rj[j:j_end, 1][None, :]
            dz[i:i_end, j:j_end] = ri[i:i_end, 2][:, None] - rj[j:j_end, 2][None, :]
    return dx, dy, dz
def getPairwiseSeparations_parallel(ri, rj, num_workers=8):
    """
    Parallelized version of pairwise separation using multiprocessing.
    """
    M = ri.shape[0]
    block_size = M // num_workers

    # Convert NumPy arrays to lists before passing them (avoiding pickle issues)
    ri_list = ri.tolist()
    rj_list = rj.tolist()

    blocks = [(np.array(ri_list[i:i+block_size]), np.array(rj_list)) for i in range(0, M, block_size)]

    with mp.Pool(num_workers) as pool:
        results = pool.map(pairwise_worker, blocks)

    # Merge results
    dx = np.vstack([res[0] for res in results])
    dy = np.vstack([res[1] for res in results])
    dz = np.vstack([res[2] for res in results])

    return dx, dy, dz
#@profile
def getDensity(r, pos, m, h, kernel = W):
    """
    Get Density at sampling locations from SPH particle distribution
    """
    M, N = pos.shape[0], pos.shape[0]
    
    # Use float32 instead of float64 for reduced memory usage and better cache performance
    dx = np.empty((M, N), dtype=np.float32)
    dy = np.empty((M, N), dtype=np.float32)
    dz = np.empty((M, N), dtype=np.float32)

    # Use in-place optimized function
    dx, dy, dz = getPairwiseSeparations_inplace(r, pos, dx, dy, dz)
    # rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))
    #rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((M, 1))
    rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((r.shape[0], 1))
    return rho
#@profile
def optimizedGetDensity(r, pos, m, h, kernel = optimizedW):
    """
    Optimized: Get Density at sampling locations from SPH particle distribution
    """
    # Build a KD-tree for efficient neighbor search
    tree = cKDTree(pos)
    rho = np.zeros(r.shape[0])

    # Iterate over each sampling location
    for i, ri in enumerate(r):
        # Find neighbors within the smoothing length h
        neighbors_idx = tree.query_ball_point(ri, h)

        # Calculate pairwise separations only for neighbors
        dx, dy, dz = getPairwiseSeparations(r[i:i+1], pos[neighbors_idx])

        # Calculate density using the smoothing kernel
        rho[i] = np.sum(m[neighbors_idx] * kernel(dx, dy, dz, h))

    return rho.reshape(-1, 1)
#@profile
def optimizedGetDensity_fast(r, pos, m, h, kernel=optimizedW):
    """
    Faster optimized version of getDensity() using batch KD-tree queries and vectorization.
    """
    # Build KD-tree for neighbor search
    tree = cKDTree(pos)

    # Query all points at once (batch operation)
    neighbors_idx_list = tree.query_ball_tree(tree, h)

    # Prepare density array
    rho = np.zeros(r.shape[0])

    # Process all particles in a vectorized manner
    for i, neighbors_idx in enumerate(neighbors_idx_list):
        dx, dy, dz = getPairwiseSeparations(r[i:i+1], pos[neighbors_idx])
        rho[i] = np.sum(m[neighbors_idx] * kernel(dx, dy, dz, h))

    return rho.reshape(-1, 1)

def getPressure(rho, k, n):
    """
    Equation of State
    """
    P = k * rho**(1+1/n)
    return P

#@profile 
def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    """
    Calculate the acceleration on each SPH particle
    """
    N = pos.shape[0]
    rho = getDensity(pos, pos, m, h, optimizedW)
    #rho = optimizedGetDensity(pos, pos, m, h)
    P = getPressure(rho, k, n)
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    ax = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWz, 1).reshape((N, 1))
    a = np.hstack((ax, ay, az))
    a -= lmbda * pos
    a -= nu * vel
    return a


def main():
    """ SPH simulation """

    # Simulation parameters
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
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
    Nt = int(np.ceil(tEnd/dt))
    
    # prepear figure: 
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
    rr = np.zeros((100,3))
    rlin = np.linspace(0,1,100)
    rr[:,0] =rlin
    rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)

    # Simulation Main Loop
    for _ in range(Nt):
        vel += acc * dt/2
        pos += vel * dt
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
        vel += acc * dt/2
        t += dt

        if plotRealTime:
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten() 
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            
            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            rho_radial = getDensity( rr, pos, m, h )
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)


    if plotRealTime: 
        # add labels/legend
        plt.sca(ax2)
        plt.xlabel('radius')
        plt.ylabel('density')
        
        # Save figure
        plt.savefig('sph.png',dpi=240)
        plt.show()
        
    return 0


if __name__ == "__main__":
    main()