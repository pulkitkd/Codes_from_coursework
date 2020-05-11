import math
import numpy as np
import matplotlib.pyplot as plt
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
step method for the dissipation term. The final equation to be iterated over
time is

fftu2 = ((1.0 - nu * k**2 * dt * 0.5) * fft(u1) - 
          (0.5 * dt * (3.0*fftu1u1x - fftu0u0x))) / (1 + 0.5 * nu * k**2 * dt)

which gives us the Fourier Coefficients of u @ t = n +1 based on those at 
t = n  and t = n - 1. Here fftu indicates the array of Fourier coefficients of
u.
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
    k1 = np.arange(0, nx/2)
    k2 = np.arange(-nx/2 + 1, 0)
    k = np.concatenate((k1, k2))
    return k

# grid points (excludes last point)
# enforces an even number of grid points
def domain(n, L):
    assert n % 2 == 0
    return np.linspace(0.0, L - dx, nx - 1)

#=============================Main Program=====================================#

# define the parameters
# grid points / sampling points
# domain a to b
# step size
# time step
# diffusivity
# total no. of time steps
# save data every 'this' number of time steps
# the domain
# array of wavenumbers
# initial conditions
clear_datfiles()
nx = 128 
L = 1.0
dx = L/(nx-1)
sf = L/(2.0*np.pi)
dt = 0.01
nu = 0.05
Nt = 500

x = domain(nx, L)
k = wavenumbers(nx)
init = np.sin(2*np.pi*x)

# define the initial condition and write it to file
u0 = init
write_real(x, u0, 0)

for i in range(1, Nt):
    # determine the necessary quantities at t = n - 1
    # -> fft(u0*u0x) @ t = n - 1
    
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

    # Evaluate u2 in frequency space
    # convert u2 to physical space
    # update u0 to be new u1 and u1 to be new u2
    fftu2 = ((1.0 - sf**2*nu * k**2 * dt * 0.5) * fft(u1) - 
        (sf*0.5 * dt * (3.0*fftu1u1x - fftu0u0x))) / (1 + sf**2*0.5 * nu * k**2 * dt)
    u2 = ifft(fftu2)

    u0 = u1
    u1 = u2
    write_real(x, u2, i)
        
make_plot(name="Burgers_eq_sine_wave")
make_movie([0,1],[-1,1],name="Burgers_eq_sine_wave")

