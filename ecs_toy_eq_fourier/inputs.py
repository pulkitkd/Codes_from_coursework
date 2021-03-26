import numpy as np
from numpy import sin, cos, exp, pi, log, cosh

import mpi4py
from mpi4py import MPI

import user_def_functions
from user_def_functions import *

# Input parameters-

# Peclet inverse (nu)
# time step (dt)
# total no. of time steps (nsteps)
# grid points (n) - must be an even number
# length of the domain (L)
# domain (x)
# scaling factor (sf) - maps (0 , 2pi) to (0 , L)
# initial conditions (init)

# Other variables defined in the code

# array of wavenumbers (k)
# initial condition (init)
# solution at nth time step (rho0)
# solution at n+1 th time step (rho1)
# solution at n+2 th time step (rho2)
# fft of u0 (fftu0)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
assert size < 9

# data handling
save_every = 50
# physical parameters
if rank == 0:
    nu = 0.01  # 1 / Pe
    mu = 0.1
    L = 4.0*pi
    
elif rank == 1:
    nu = 0.01  # 1 / Pe
    mu = 0.1
    L = 4.0*pi
    
elif rank == 2:
    nu = 0.1  # 1 / Pe
    mu = 0.01
    L = 4.0*pi
    
elif rank == 3:
    nu = 0.1  # 1 / Pe
    mu = 0.1
    L = 4.0*pi
    
elif rank == 4:
    nu = 0.01  # 1 / Pe
    mu = 0.01
    L = 10.0*pi
    
elif rank == 5:
    nu = 0.01  # 1 / Pe
    mu = 0.1
    L = 10.0*pi
    
elif rank == 6:
    nu = 0.1  # 1 / Pe
    mu = 0.01
    L = 10.0*pi
    
else:
    nu = 0.1  # 1 / Pe
    mu = 0.1
    L = 10.0*pi


# time discretization
dt = 0.0001
nsteps = 50000

# spatial discretization
n = 128
x = domain(n, L)
sf = (2.0*np.pi)/L

# initial condition

# init = 1.0 * np.ones(n-1,dtype=float)
# init = 0.5 + 0.1*sin(sf*x)
init = np.log(1 + np.cosh(20)**2/np.cosh(20*(x-L/2))**2) / (2*20)
k = wavenumbers(n)
