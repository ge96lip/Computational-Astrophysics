from matplotlib import pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.spatial import cKDTree

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

#@profile
def getDensity(r, pos, m, h, kernel = optimizedW):
    """
    Get Density at sampling locations from SPH particle distribution
    """
    M = r.shape[0]
    dx, dy, dz = getPairwiseSeparations(r, pos)
    # rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))
    rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((M, 1))
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
@profile
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
    # rho = getDensity(pos, pos, m, h)
    rho = optimizedGetDensity(pos, pos, m, h)
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