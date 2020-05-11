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

# takes the fft of function and returns the derivative of the function in
# physical space
def ifftik(fftfunc, k):
    ikfftfunc = 1j * k * fftfunc  # take derivative in frequency space
    return ifft(ikfftfunc)  # convert the derivative to physical space

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

clear_datfiles()
# physical parameters
nu = 0.03
# time
dt = 0.01
nsteps = 500
# space
n = 128 
L = 2.0*np.pi
x = domain(n, L)
sf = L/(2.0*np.pi)

k = wavenumbers(n)
init = np.sin(x)
t1 = time.time()
# define the initial condition and write it to file
u0 = init
write_real(x, u0, 0)
    
# determine the necessary quantities at t = n - 1
# take FFT of u
# get u_x in physical space
# get fft of u*u_x
fftu0 = fft(u0)
u0x = ifftik(fftu0, k)
fftu0u0x = fft(u0 * u0x)

# determine the necessary quantities at t = n
# -> fft(u*ux) @ t=n
# -> fft(u) @ t=n

# get u1 (in freq space) @ t=n from u @ t=n-1 using Euler's method
# convert u1 to physical space
# get u1_x in physical space
# get fft(u1 * u1_x)
fftu1 = fftu0 - dt * (fftu0u0x + nu*k**2*fftu0)
u1 = ifft(fftu1)
u1x = ifftik(fftu1, k)
fftu1u1x = fft(u1 * u1x)
write_real(x, u0, 1)


for i in range(2, nsteps):
    # Evaluate u2 in frequency space
    # convert u2 to physical space
    # update u0 to be new u1 and u1 to be new u2
    fftu2 = ((((1.0 - sf**2 *nu * k**2 * dt * 0.5) * fft(u1)) - 
             (sf*0.5 * dt * (3.0*fftu1u1x - fftu0u0x))) / 
             (1 + sf**2 *0.5 * nu * k**2 * dt))
    u2 = ifft(fftu2)
    assert all(np.abs(u2) < 100)
    write_real(x, u2, i)

    u0 = u1.real
    u0x = ifftik(fft(u0), k)
    fftu0u0x = fft(u0 * u0x)

    u1 = u2.real
    u1x = ifftik(fft(u1), k)
    fftu1u1x = fft(u1 * u1x)

t2 = time.time()
print("Total time for Fourier method = ", t2-t1)
make_plot(2,name="Burgers_eq_sine_wave_F")
make_movie([0,L],[-1,1],name="Burgers_eq_sine_wave_F")
