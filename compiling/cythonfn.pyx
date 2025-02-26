# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
import time
np.import_array()

import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W(double[:,:] x, double[:,:] y, double[:,:] z, double h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	cdef double two = 2
	cdef double[:,:] r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
	cdef double[:,:] w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -np.square(r) / h**2)
	
	return w
	
	
def gradW(double[:,:] x, double[:,:] y, double[:,:] z, double h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	cdef double[:,:] r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
	
	cdef double[:,:] n = -2 * np.exp( -np.square(r) / h**2) / h**5 / (np.pi)**(3/2)
	wx = np.multiply(n,x)
	wy = np.multiply(n,y)
	wz = np.multiply(n,z)
	
	return wx, wy, wz
	
	
def getPairwiseSeparations(ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	cdef int M = ri.shape[0]
	cdef int N = rj.shape[0]
	
	# positions ri = (x,y,z)
	cdef double[:,:] rix = ri[:,0].reshape((M,1))
	cdef double[:,:] riy = ri[:,1].reshape((M,1))
	cdef double[:,:] riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	cdef double[:,:] rjx = rj[:,0].reshape((N,1))
	cdef double[:,:] rjy = rj[:,1].reshape((N,1))
	cdef double[:,:] rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	cdef double[:,:] dx = np.subtract(rix,rjx.T)
	cdef double[:,:] dy = np.subtract(riy,rjy.T)
	cdef double[:,:] dz = np.subtract(riz,rjz.T)
	
  #print("dx:" , np.asarray(dx).shape, "\n dy: ", np.asarray(dy).shape, "\n dz: ",  np.asarray(dz).shape)
	return dx, dy, dz
	

def getDensity(r,pos, double m, double h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	cdef int M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos );

	#print("dx:" , np.asarray(dx).shape, "\n dy: ", np.asarray(dy).shape, "\n dz: ",  np.asarray(dz).shape)
	rho = (np.sum( np.multiply(m,W(dx, dy, dz, h)), 1 )).reshape((M))
	
	return rho
	
	
def getPressure(double[:] rho, double k, double n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	cdef double[:] P = np.multiply(k, np.power(rho,(1+1/n)))
	
	return P
	

def getAcc(pos, double[:,:] vel, double m, double h, double k, double n, double lmbda, double nu ):
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
	cdef double[:] rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	cdef double[:] P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	cdef double[:,:] dx, dy, dz
	dx, dy, dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	cdef double[:,:] ax = - np.sum( m * ( P/np.square(rho) + P.T/np.square(rho.T)  ) * dWx, 1).reshape((N,1))
	cdef double[:,:] ay = - np.sum( m * ( P/np.square(rho) + P.T/np.square(rho.T)  ) * dWy, 1).reshape((N,1))
	cdef double[:,:] az = - np.sum( m * ( P/np.square(rho) + P.T/np.square(rho.T)  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	cdef double[:,:] a = np.hstack((ax,ay,az))
	
	# Add external potential force
	sub = np.multiply(pos, lmbda)
	a -= sub
	
	# Add viscosity
	a -= np.multiply(nu, vel)
	
	return a
	


def main(numparticles):
	""" SPH simulation """
	
	# Simulation parameters
	cdef int N         = numparticles    # Number of particles
	cdef double t         = 0      # current time of the simulation
	cdef int tEnd      = 12     # time at which simulation ends
	cdef double dt        = 0.04   # timestep
	cdef int M         = 2      # star mass
	cdef double R         = 0.75   # star radius
	cdef double h         = 0.1    # smoothing length
	cdef double k         = 0.1    # equation of state constant
	cdef int n         = 1      # polytropic index
	cdef int nu        = 1      # damping
	cdef double[:] rho
	plotRealTime = False # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	cdef double lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
	cdef double m     = M/N                    # single particle mass
	pos   = np.random.randn(N,3)   # randomly selected positions and velocities
	cdef double[:,:] vel   = np.zeros((N,3), dtype=np.double)
	
	# calculate initial gravitational accelerations
	cdef double[:,:] acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
	
	# number of timesteps
	cdef int Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	rr = np.zeros((100,3))
	cdef double[:] rl = np.linspace(0,1,100)
	rlin = np.linspace(0,1,100)
	rr[:,0] =rlin
	rho_analytic = lmbda/(4*k) * (np.power(R,2) - np.power(rlin,2))
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += np.multiply(acc, dt/2)
		
		# drift
		pos += np.multiply(vel, dt)
		
		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += np.multiply(acc, dt/2)
		
		# update time
		t += dt
		
		# get density for plotting
		rho = getDensity( pos, pos, m, h )
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			cval = np.minimum(np.divide(np.subtract(rho,3),3),1).flatten()
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
	    
	
	
	# add labels/legend
	plt.sca(ax2)
	plt.xlabel('radius')
	plt.ylabel('density')
	
	# Save figure
	# plt.savefig('sph.png',dpi=240)
	# plt.show()
	    
	return 0
	