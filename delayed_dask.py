import numpy as np
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute
from scipy.special import gamma

plt.ioff()

def W(x, y, z, h):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)

def gradW(x, y, z, h):
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-r**2 / h**2) / h**5 / (np.pi)**(3/2)
    return n * x, n * y, n * z

@delayed
def getPairwiseSeparations(ri, rj):
    M, N = ri.shape[0], rj.shape[0]
    rix, riy, riz = ri[:, 0].reshape((M, 1)), ri[:, 1].reshape((M, 1)), ri[:, 2].reshape((M, 1))
    rjx, rjy, rjz = rj[:, 0].reshape((N, 1)), rj[:, 1].reshape((N, 1)), rj[:, 2].reshape((N, 1))
    return rix - rjx.T, riy - rjy.T, riz - rjz.T

@delayed
def getDensity(r, pos, m, h):
    separations = getPairwiseSeparations(r, pos)  # Keep as a single delayed object
    dx, dy, dz = compute(separations)[0]  # Compute before unpacking
    return np.sum(m * W(dx, dy, dz, h), axis=1).reshape((r.shape[0], 1))
@delayed
def getPressure(rho, k, n):
    return k * rho**(1 + 1 / n)

@profile
def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    N = pos.shape[0]

    # Call delayed functions normally (no need to wrap in `delayed()` again)
    rho = getDensity(pos, pos, m, h)  
    P = getPressure(rho, k, n)
    separations = getPairwiseSeparations(pos, pos)

    # Compute all delayed dependencies at once
    dx, dy, dz = compute(separations)[0]
    rho, P = compute(rho, P)  

    dWx, dWy, dWz = gradW(dx, dy, dz, h)

    ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, axis=1).reshape((N, 1))
    ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, axis=1).reshape((N, 1))
    az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, axis=1).reshape((N, 1))

    a = np.hstack((ax, ay, az))
    a -= lmbda * pos
    a -= nu * vel
    return a

def main():
    N, t, tEnd, dt = 1000, 0, 12, 0.04
    M, R, h, k, n, nu = 2, 0.75, 0.1, 0.1, 1, 1
    plotRealTime = False

    np.random.seed(42)
    lmbda = 2 * k * (1 + n) * np.pi**(-3 / (2 * n)) * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n))**(1 / n) / R**2
    m = M / N
    pos = np.random.randn(N, 3)
    vel = np.zeros(pos.shape)

    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)

    Nt = int(np.ceil(tEnd / dt))

    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1, ax2 = plt.subplot(grid[0:2, 0]), plt.subplot(grid[2, 0])
    
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)

    for i in range(Nt):
        vel += acc * dt / 2
        pos += vel * dt

        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)  # Compute acceleration in parallel

        vel += acc * dt / 2
        t += dt

        rho = getDensity(pos, pos, m, h)

        if plotRealTime:
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho - 3) / 3, 1).flatten()
            plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])
            ax1.set_facecolor('black')

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            rho_radial = getDensity(rr, pos, m, h).compute()
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)

        if plotRealTime:
            plt.sca(ax2)
            plt.xlabel('radius')
            plt.ylabel('density')
            plt.savefig('sph.png', dpi=240)
            plt.show()

    return 0

if __name__ == "__main__":
    main()