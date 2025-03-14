import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
cimport numpy as np
from libc.math cimport sqrt, exp, pi, ceil
import cython
cimport cython
from cython.parallel import prange


"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""
@cython.boundscheck(False)	
def W(float[:,:] x, float[:,:] y, float[:,:] z, float h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	cdef int i,j,k
	cdef float[:,:] w = np.empty((x.shape[0], x.shape[1]), dtype=np.float32)
	cdef float r
	cdef float x0, y0, z0
	cdef int  hz = x.shape[0]
	cdef int vt = x.shape[1]
	cdef int stop = hz*vt
	for k in prange(stop, nogil=True):
		i = int(k / vt)
		j = int(k % vt)
		x0 = x[i][j]
		y0 = y[i][j]
		z0 = z[i][j]
		r = sqrt(x0*x0 + y0*y0 + z0*z0)
		w[i][j] = (1.0 / (h*sqrt(pi)))**3 * exp( -r*r / h*h)
	
	return w
	
@cython.boundscheck(False)		
def gradW(float[:,:] x, float[:,:] y, float[:,:] z, float h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	cdef float[:,:] wx, wy, wz
	wx = np.empty_like(x, dtype=np.float32)
	wy = np.empty_like(x, dtype=np.float32)
	wz = np.empty_like(x, dtype=np.float32)
	cdef float r, n, x0, y0, z0

	cdef int i,j 

	for i in prange(x.shape[0], nogil=True):
		for j in range(x.shape[1]):
			x0 = x[i][j]
			y0 = y[i][j]
			z0 = z[i][j]
			r = sqrt(x0*x0 + y0*y0 + z0*z0)
			n = -2 * exp( -r*r / h*h) / h**5 / (pi)**(3/2)
			wx[i][j] = n * x0
			wy[i][j] = n * y0
			wz[i][j] = n * z0
			
	return wx, wy, wz
	
@cython.boundscheck(False)	
def getPairwiseSeparations(float[:,:] ri, float[:,:] rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	cdef int M = ri.shape[0]
	cdef int N = rj.shape[0]
	cdef int i,j 
	cdef float[:,:] dx = np.empty((N,M), dtype=np.float32)
	cdef float[:,:] dy = np.empty((N,M), dtype=np.float32)
	cdef float[:,:] dz = np.empty((N,M), dtype=np.float32)

	for i in prange(M, nogil=True):
		for j in range(N):
			dx[i][j] = ri[i][0] - rj[j][0]
			dy[i][j] = ri[i][1] - rj[j][1]
			dz[i][j] = ri[i][2] - rj[j][2]
	
	return dx, dy, dz
	
@cython.boundscheck(False)	
def getDensity(float[:,:] r, float[:,:] pos, float m, float h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	cdef int M = r.shape[0]
	cdef int N = pos.shape[0]
	cdef float[:,:] dx, dy, dz 
	dx, dy, dz = getPairwiseSeparations( r, pos );
	cdef float[:] rho = np.zeros((M), dtype=np.float32)
	cdef int i,j,k 
	cdef float[:,:] w = W(dx, dy, dz, h)
	for i in prange(M, nogil=True):
		for k in range(w.shape[1]):
			rho[i] += w[k][1] * m 
			#rho = sum( m * (W(dx, dy, dz, h)), 1 ).reshape((M,1))
	
	return rho
	
@cython.boundscheck(False)		
def getPressure(float[:] rho, float k, int n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	cdef float[:] P = np.empty((rho.shape[0]), dtype=np.float32)
	cdef int i,j 
	cdef float r 
	for i in prange(rho.shape[0], nogil=True):
		r = rho[i]
		P[i] = k * r**(1+1/n)
	return P
	
@cython.boundscheck(False)	
def getAcc(float[:,:] pos,  float[:,:] vel, float m, float h, float k, int n, float lmbda, int nu ):
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
	
	cdef int N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	cdef float[:] rho = getDensity( pos, pos, m, h )
	cdef float[:] rho_T = rho.T
	
	# Get the pressures
	cdef float[:] P = getPressure(rho, k, n)
	cdef float[:] P_T = P.T
	
	# Get pairwise distances and gradients
	cdef float[:,:] dx, dy, dz, dWx, dWy, dWz
	dx, dy, dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	cdef int i, j
	cdef float[:,:] ax = np.zeros((N,1), dtype=np.float32)
	cdef float[:,:] ay = np.zeros((N,1), dtype=np.float32)
	cdef float[:,:] az = np.zeros((N,1), dtype=np.float32)
	cdef float ph, rh 
	for i in range (P.shape[0]):
		for j in range(P.shape[1]):
			ph = P[i]
			pht = P_T[i]
			rhi = rho[i]
			rhj = rho[j]
			ax[i][j] -= m * ( ph/rhi*rhi + pht/rhj*rhj  ) * dWx[i][j]
			ay[i][j] -= m * ( ph/rhi*rhi + pht/rhj*rhj  ) * dWy[i][j]
			az[i][j] -= m * ( ph/rhi*rhi + pht/rhj*rhj  ) * dWz[i][j]
			#ay = - sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
			#az = - sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	cdef float[:,:] a = np.hstack((ax,ay,az))
	
	# Add external potential force
	for i in prange(a.shape[0], nogil=True):
		for j in range(a.shape[1]):
			a[i][j] -= lmbda * pos[i][j]
			a[i][j] -= nu * vel[i][j]
	#a -= lmbda * pos
	
	# Add viscosity
	#a -= nu * vel
	
	return a
	

@cython.boundscheck(False)	
def main(int N):
	""" SPH simulation """
	
	# Simulation parameters
	# N       = 10000   # Number of particles
	cdef float t         = 0      # current time of the simulation
	cdef int tEnd      = 12     # time at which simulation ends
	cdef float dt        = 0.04   # timestep
	cdef float M         = 2      # star mass
	cdef float R         = 0.75   # star radius
	cdef float h         = 0.1    # smoothing length
	cdef float k         = 0.1    # equation of state constant
	cdef float n         = 1      # polytropic index
	cdef int nu        = 1      # damping
	rng = np.random.default_rng()
	# plotRealTime = False # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	cdef float lmbda = 2*k*(1+n)*pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R*R  # ~ 2.01
	cdef float m     = M/N                    # single particle mass
	cdef float[:,:] pos, vel, acc
	pos   = rng.standard_normal((N,3), dtype=np.float32)  # randomly selected positions and velocities
	vel   = np.zeros((N,3), dtype=np.float32)
	
	# calculate initial gravitational accelerations
	acc = getAcc(pos, vel, m, h, k, n, lmbda, nu )
	
	# number of timesteps
	cdef int Nt = int(ceil(tEnd/dt))

	cdef int i,j,l
	cdef float a,v
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		for j in prange(acc.shape[0], nogil=True):
			for l in range(acc.shape[1]):
				a = acc[j][l]
				vel[j][l] += a * dt/2
			
			# drift
				v = vel[j][l]
				pos[j][l] += v * dt
		
		# update accelerations
		acc = getAcc(pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		for j in prange(N, nogil=True):
			for l in range(3):
				a = acc[j][l]
				vel[j][l] += a * dt/2
		
		# update time
		t += dt
		
	return 0
