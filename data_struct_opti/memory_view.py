from matplotlib import pyplot as plt
import numpy as np
from scipy.special import gamma

import numexpr as ne
import multiprocessing as mp
from workers import pairwise_worker 
mp.set_start_method("fork", force=True)


def W(x, y, z, h):
    """
    Gausssian Smoothing kernel (3D)
    """
    h = np.asarray(h)
    r = np.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)
    return w

def optimizedW(x, y, z, h):
    """
    Gaussian Smoothing kernel (3D)
    """
    h_np = np.asarray(h)
    r = np.sqrt(x**2 + y**2 + z**2)
    mask = r <= h  # Only consider distances within h
    w = np.zeros_like(r)
    w[mask] = (1.0 / (h_np * np.sqrt(np.pi)))**3 * np.exp(-r[mask]**2 / h_np**2)
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
def gradW_float32_ne(x, y, z, h, N):
    """Optimized gradient computation using numexpr."""
    h_np = np.asarray(h)  # Ensure `h` is a NumPy array

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    pi = np.pi
    r2 = ne.evaluate("x_np**2 + y_np**2 + z_np**2")
    exp_term = ne.evaluate("exp(-r2 / h_np**2)")
    scale = ne.evaluate("-2 * exp_term / (h_np**5 * (pi**(3/2)))")

    wx = ne.evaluate("scale * x_np")
    wy = ne.evaluate("scale * y_np")
    wz = ne.evaluate("scale * z_np")

    return wx, wy, wz
def gradW_float32_inplace(x, y, z, h, N):
    """
    Fully optimized in-place gradient computation using float32.
    - Removes unnecessary temporary arrays.
    - Uses in-place operations to minimize memory overhead.
    - Leverages NumPy’s out= parameter for efficiency.
    """
    h_np = np.asarray(h)
    # Allocate output arrays inside function
    wx = np.empty((N, N), dtype=np.float32)
    wy = np.empty((N, N), dtype=np.float32)
    wz = np.empty((N, N), dtype=np.float32)

    # Compute squared distances directly in-place
    r2 = np.empty((N, N), dtype=np.float32)
    np.multiply(x, x, out=r2)
    np.add(r2, y*y, out=r2)
    np.add(r2, z*z, out=r2)  # Now r2 contains x² + y² + z²

    # Compute exponential term in-place
    np.divide(r2, -h_np**2, out=r2)
    np.exp(r2, out=r2)  # Now r2 contains exp(-r² / h²)

    # Compute the scaling factor in-place
    np.multiply(r2, -2 / (h_np**5 * (np.pi)**(3/2)), out=r2)

    # Compute wx, wy, wz in-place
    np.multiply(r2, x, out=wx)
    np.multiply(r2, y, out=wy)
    np.multiply(r2, z, out=wz)

    return wx, wy, wz

def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between 2 sets of coordinates
    """
    #ri = ri.astype(np.float32)
    ri = np.asarray(ri).astype(np.float32) 
    #rj = rj.astype(np.float32)
    rj = np.asarray(rj).astype(np.float32) 
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

def getPairwiseSeparations_inplace(ri, rj):
    """
    In-place computation to minimize memory allocations.
    """
    M, N = rj.shape[0], rj.shape[0]

    dx = np.empty((M, N), dtype=np.float32)
    dy = np.empty((M, N), dtype=np.float32)
    dz = np.empty((M, N), dtype=np.float32)
    #ri = ri.astype(np.float32)
    ri = np.asarray(ri).astype(np.float32) 
    #rj = rj.astype(np.float32)
    rj = np.asarray(rj).astype(np.float32) 
    np.subtract(ri[:, 0][:, None], rj[:, 0][None, :], out=dx)
    np.subtract(ri[:, 1][:, None], rj[:, 1][None, :], out=dy)
    np.subtract(ri[:, 2][:, None], rj[:, 2][None, :], out=dz)

    return dx, dy, dz

#@profile
def getDensity(r, pos, m, h, kernel = W):
    """
    Get Density at sampling locations from SPH particle distribution
    """
    # Use in-place optimized function
    dx, dy, dz = getPairwiseSeparations_inplace(r, pos) # getPairwiseSeparations(r, pos) 
    # rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))
    #rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((M, 1))
    rho = np.sum(m * kernel(dx, dy, dz, h), 1).reshape((r.shape[0], 1))
    return rho

def getPressure(rho, k, n):
    """
    Equation of State
    """
    n_np = np.asarray(n)
    P = k * rho**(1+1/n_np)
    return P
def calculate_a_numexpr(P, rho, dWx, dWy, dWz, N): 
    P_T = P.T
    rho_T = rho.T
    common_expr = "m * (P / rho**2 + P_T / rho_T**2)"
    common_val = ne.evaluate(common_expr)
    
    ax = -np.sum(common_val * dWx, axis=1).reshape((N, 1))
    ay = -np.sum(common_val * dWy, axis=1).reshape((N, 1))
    az = -np.sum(common_val * dWz, axis=1).reshape((N, 1))
def calculate_a_common_term(P, rho, m, dWx, dWy, dWz, N): 
    common_term = (P / rho**2 + P.T / rho.T**2)
    weighted_term = m * common_term

    ax = -np.sum(weighted_term * dWx, axis=1, keepdims=True)
    ay = -np.sum(weighted_term * dWy, axis=1, keepdims=True)
    az = -np.sum(weighted_term * dWz, axis=1, keepdims=True)
def calculate_a_einsum(P, rho, dWx, dWy, dWz, m, N): 
    common_expr = P / rho**2 + P.T / rho.T**2
    weighted_term = m * common_expr

    ax = -np.einsum("ij,ij->i", weighted_term, dWx).reshape((N, 1))
    ay = -np.einsum("ij,ij->i", weighted_term, dWy).reshape((N, 1))
    az = -np.einsum("ij,ij->i", weighted_term, dWz).reshape((N, 1)) 

def calculate_a_memoryview(m, P, rho, dWx, dWy, dWz, N): 
    m_mem = memoryview(m)
    P_mem = memoryview(P)
    rho_mem = memoryview(rho)
    dWx_mem = memoryview(dWx)
    dWy_mem = memoryview(dWy)
    dWz_mem = memoryview(dWz)

    common_term = np.array([m_mem[i] * (P_mem[i] / rho_mem[i]**2 + P_mem[:, i] / rho_mem[:, i]**2) for i in range(N)])

    ax = -np.sum(common_term * dWx_mem, axis=1).reshape((N, 1))
    ay = -np.sum(common_term * dWy_mem, axis=1).reshape((N, 1))
    az = -np.sum(common_term * dWz_mem, axis=1).reshape((N, 1))
    
def calculate_a_npdot(m, P, rho, dWx, dWy, dWz, N): 
    common_term = m * (P / rho**2 + P.T / rho.T**2)

    ax = -np.sum(np.dot(common_term, dWx), axis=1, keepdims=True)  # (400,1)
    ay = -np.sum(np.dot(common_term, dWy), axis=1, keepdims=True)  # (400,1)
    az = -np.sum(np.dot(common_term, dWz), axis=1, keepdims=True)  # (400,1)
    
#@profile 
def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    """
    Calculate the acceleration on each SPH particle
    """
    N = pos.shape[0]
    
    pos = np.ascontiguousarray(pos)
    m = np.ascontiguousarray(m)
    h = np.ascontiguousarray(h)
    k = np.ascontiguousarray(k)
    n = np.ascontiguousarray(n)

    # Create memoryviews for inputs
    pos_mv = memoryview(pos)
    m_mv = memoryview(m)
    h_mv = memoryview(h)
    k_mv = memoryview(k)
    n_mv = memoryview(n)
    rho = getDensity(pos_mv, pos_mv, m_mv, h_mv, optimizedW)
    #rho = optimizedGetDensity(pos, pos, m, h)
    P = getPressure(rho, k_mv, n_mv)
    dx, dy, dz = getPairwiseSeparations_inplace(pos_mv, pos_mv) #getPairwiseSeparations(pos_mv, pos_mv)#
    dWx, dWy, dWz = gradW_float32_ne(dx, dy, dz, h_mv, N) ##gradW(dx, dy, dz, h) #
    """ax = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWz, 1).reshape((N, 1))
    """
    # Precompute common terms
    P_over_rho_sq = np.ascontiguousarray(P / rho**2)
    P_over_rho_sq_T = np.ascontiguousarray(P.T / rho.T**2)

    # Calculate acceleration components
    ax = -np.sum(m_mv * (P_over_rho_sq + P_over_rho_sq_T) * dWx, axis=1).reshape((pos.shape[0], 1))
    ay = -np.sum(m_mv * (P_over_rho_sq + P_over_rho_sq_T) * dWy, axis=1).reshape((pos.shape[0], 1))
    az = -np.sum(m_mv * (P_over_rho_sq + P_over_rho_sq_T) * dWz, axis=1).reshape((pos.shape[0], 1))
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
            rho = getDensity( pos, pos, m, h )
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