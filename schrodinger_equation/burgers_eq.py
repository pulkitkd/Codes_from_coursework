import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fft, ifft
from numpy import multiply
from post_processing import *

#===============================Introduction===================================#
'''
This code solves the viscous burgers equation using Fourier-Galerkin spectral
methods for periodic boundary conditions

u_t + u*u_x = nu * u_xx 

We want to evolve the Fourier coefficients of these terms in time using 
Adams-Bashforth 2 step method for the non-linear term and the Adams-Moulton 2
step method for the dissipation term.

To run the code:

Create a directory containing the following three elements-
* the file 'burgers_eq.py'
* the file 'post_processing.py'
* a subdirectory 'data'
Navigate to the above root directory via terminal and execute-
$ python3 ./burgers_eq.py 
'''
#=================================Functions====================================#
# clears all *.dat files from the subdirectory data/
# use before starting a new run of the program


def clear_datfiles():
    dirPath = "data"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath+"/"+fileName)

# write the real parts of the supplied array (x and u) to a data file
# with name solution"n".dat


def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()

# creates the wavenumber array of the form required for np.fft.ifft
# |0 | 1 | 2 | 3 | -3 | -2 | -1|


def wavenumbers(n):
    assert n % 2 == 0
    k1 = np.arange(0, n/2)
    k2 = np.arange(-n/2 + 1, 0)
    k = np.concatenate((k1, k2))
    return k

# grid points (excludes last point)
# enforces an even number of grid points


def domain(n, L):
    dx = L/(n-1)
    assert n % 2 == 0
    return np.linspace(0.0, L - dx, n - 1)

#=============================Main Program=====================================#

# Input parameters-

# diffusivity (nu)
# time step (dt)
# total no. of time steps (nsteps)
# grid points (n) - must be an even number
# length of the domain (L)
# domain (x)
# scaling factor (sf) - maps (0 , 2pi) to (0 , L)
# array of wavenumbers (k)
# initial conditions (init)

# Other variables defined in the code

# array of wavenumbers (k)
# initial condition (init)
# solution at nth time step (u0)
# solution at n+1 th time step (u1)
# solution at n+2 th time step (u2)
# fft of u0 (fftu0)


clear_datfiles()
# physical parameters
nu = 0.1j
# time
dt = 0.002
nsteps = 800
# space
n = 512
L = 50.0
x = domain(n, L)
sf = (2.0*np.pi)/L

k = wavenumbers(n)

# initial condition
# init = np.exp(-(x-L/2)**2) / np.sqrt(2*np.pi)
init = np.pi**(-0.25) * np.exp(-2*(x-0.5*L)**2 / 2.0) * np.exp(1.j*4*(x-0.5*L))
t1 = time.time()

# define u0 and write it to file
u0 = init
write_real(x, u0, 0)

# get u0x by differentiating in the Fourier space and then inverting it
fftu0 = fft(u0)

a = (np.ones(n-1) + sf**2 * 0.5 * 1j * dt * k**2)**(-1)
b = (np.ones(n-1) - 0.5 * sf**2 * 1j * dt * k**2)

for i in range(1, nsteps):
    fftu1 = (b * fft(u0)) * a
             
    u1 = ifft(fftu1)
    assert all(np.abs(u1) < 100)
    write_real(x, u1, i)

    u0 = u1
    # fftu0 = fft(u0)


t2 = time.time()
print("Total time for Fourier method = ", t2-t1)

'''
Post-processing : Following commands require the file 'post_processing.py'
'''
# The following commands create a plot showing time-evolution of the initial
# condition and creates a movie of the same. In case user needs to plot the data
# via another application e.g. gnuplot, these lines can be suppressed / removed
# without affecting the program.
# make_plot(8, name="Burgers_eq_(Fourier-Galerkin)")
make_movie([0, L], [-1, 1], name="Burgers_eq_(Fourier-Galerkin)")
