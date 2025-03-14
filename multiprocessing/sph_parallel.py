import numpy as np
import sys

def W(x, y, z, h):
    """
    Gausssian Smoothing kernel (3D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        w     is the evaluated smoothing function
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    #print(r)
    #print(h, flush=True)
    w= np.array(POOL.starmap(calcw, [(i,h) for i in r]),dtype=np.double,like=x)
    return w

def calcw(r, h):
    return (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)


def gradW(x, y, z, h):
    """
    Gradient of the Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    n = np.array(POOL.starmap(calcn, [(i,h) for i in r]), like=x)
    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz
def calcn(r,h):
    return -2 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3 / 2)


def getPairwiseSeparations(ri, rj):
    # print("GetPairwiseSeparations")
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    # start = time.time_ns()
    # print("Start Pairwise")
    r = POOL.starmap(pairwiseTask, [(ri, M, 0), (ri, M, 1), (ri, M, 2), (rj, N, 0), (rj,N,1), (rj,N,2)])
    #print("r")
    # print(time.time_ns() - start)
    dx = r[0] - r[3].T
    dy = r[1] - r[4].T
    dz = r[2] - r[5].T
    #print("d")
    return dx, dy, dz

def pairwiseTask(r, M, i):
    #print("child says hello ", i, flush=True)
    a = r[:,i].reshape((M, 1))
    #print("child says goodbye ", i, flush=True)
    return a

#def pairwiseStore(ri, i):
#    return ri[i] - ri[i+3].T



def getDensity(r, pos, m, h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparations(r, pos)

    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho


def getPressure(rho, k, n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """

    P = k * rho ** (1 + 1 / n)

    return P


def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    N = pos.shape[0]

    # Calculate densities at the position of the particles
    rho = getDensity(pos, pos, m, h)

    # Get the pressures
    P = getPressure(rho, k, n)

    # Get pairwise distances and gradients
    #start = time.time()
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    #print(time.time() - start)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)

    # Add Pressure contribution to accelerations
    ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    # Add external potential force
    a -= lmbda * pos

    # Add viscosity
    a -= nu * vel

    return a



	


  
if __name__== "__main__":
    # import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import gamma
    import multiprocessing
    import time
    """
    Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
    Philip Mocz (2020) Princeton Univeristy, @PMocz

    Simulate the structure of a star with SPH
    """
    POOL = multiprocessing.Pool()
	# Simulation parameters
    N         = int(sys.argv[1])    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 12     # time at which simulation ends
    dt        = 0.04   # timestep
    M         = 2      # star mass
    R         = 0.75   # star radius
    h         = 0.1    # smoothing length
    k         = 0.1    # equation of state constant
    n         = 1      # polytropic index
    nu        = 1      # damping
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)
    print(POOL)
    # calculate initial gravitational accelerations
    acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2
        
        # drift
        pos += vel * dt
        
        # update accelerations
        acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
        
        # (1/2) kick
        vel += acc * dt/2
        
        # update time
        t += dt
        
        # get density for plotting
        rho = getDensity( pos, pos, m, h )

    # Save figure
    # plt.savefig('sph.png',dpi=240)
    # plt.show()
    POOL.close()
