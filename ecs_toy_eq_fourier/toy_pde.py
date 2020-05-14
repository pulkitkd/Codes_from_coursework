import math
import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
import time
from numpy.fft import fft, ifft
from numpy import multiply
from post_processing import *

#===============================Introduction===================================#
'''
This code solves a toy equation for ECS using Fourier-Galerkin spectral
methods for periodic boundary conditions

c_t = (1/Pe) c_xx + 2 c c_xxxx + c_xxxxxx + 4 c_x c_xxx + 1 - mu c

We want to evolve the Fourier coefficients of these terms in time using 
Adams-Bashforth 2 step method for the non-linear terms and the Adams-Moulton 2
step method for the linear terms.
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


# def update(rho0, rho1, rho2, ak0, bk0, ck0, ak1, bk1, ck1):
#     rho0 = rho1.real
#     ak0 = fft(rho0)

#     rho0x = ifft(1.0j * k * ak0)
#     rho0x3 = ifft(-1.0j * k**3 * ak0)
#     rho0x4 = ifft(k**4 * ak0)

#     bk0 = fft(rho0 * rho0x4)
#     ck0 = fft(rho0x * rho0x3)

#     rho1 = rho2.real
#     ak1 = fft(rho1)

#     rho1x = ifft(1.0j * k * ak1)
#     rho1x3 = ifft(-1.0j * k**3 * ak1)
#     rho1x4 = ifft(k**4 * ak1)

#     bk1 = fft(rho1 * rho1x4)
#     ck1 = fft(rho1x * rho1x3)


#=============================Main Program=====================================#

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
# solution at nth time step (u0)
# solution at n+1 th time step (u1)
# solution at n+2 th time step (u2)
# fft of u0 (fftu0)


# data handling
clear_datfiles()
save_every = 50
# physical parameters
nu = 0.01  # 1 / Pe
mu = 0.01
# time
dt = 0.001
nsteps = 5000
# space
n = 128
L = 2.0*pi
x = domain(n, L)
sf = (2.0*np.pi)/L

k = wavenumbers(n)

# initial condition
# init = np.ones(n-1)
init = 1.0 + 0.001*sin(sf*x)
t1 = time.time()

# define u0 and write it to file
rho0 = init
write_real(x, rho0, 0)
# make_plot_from("solution0.dat")

ak0 = fft(rho0)

rho0x = ifft(1.0j * k * ak0)
rho0x3 = ifft(-1.0j * k**3 * ak0)
rho0x4 = ifft(k**4 * ak0)

bk0 = fft(rho0 * rho0x4)
ck0 = fft(rho0x * rho0x3)

one = np.ones(n-1)
dk0 = fft(one)

# to begin the iterations, assume u1 = u0
rho1 = rho0
write_real(x, rho1, 1)

ak1 = ak0
bk1 = bk0
ck1 = ck0

A = (np.ones(n-1) +
     0.5 * nu * dt * sf**2 * k**2 +
     0.5 * dt * sf**6 * k**6 +
     0.5 * mu * dt)**(-1)

B = (np.ones(n-1) -
     0.5 * nu * dt * sf**2 * k**2 -
     0.5 * dt * sf**6 * k**6 -
     0.5 * mu * dt)

dtsf4 = dt * sf**4

# In this loop we use the AM2 scheme for the linear term and the AB2 scheme for
# the non linear term to determine u @ t = n+2 using u @ t = n+1 and n. The
# first line in the loop generates an array of fourier coefficients at t = n+2.
# This array is inverted using ifft to get u at t = n+2. Following this, rho0, rho1
# and their derivatives are updated and the loop continues. The generated data
# files are stored in a subdirectory /data. The assert statement aborts the code
# in case the solution blows-up to large values.
j = 2
for i in range(2, nsteps):
    ak2 = ((ak1 * B) +
           (bk1 * 3.0 * dtsf4) - (bk0 * dtsf4) +
           (ck1 * 6.0 * dtsf4) - (ck0 * 2.0 * dtsf4) +
           (dk0 * dt)) * A
    rho2 = ifft(ak2)
    # assert all(np.abs(rho2) < 1e5) , "Solution becomes unbounded at step "+str(i)
    if i % save_every == 0:
        write_real(x, rho2, j)
        j = j + 1

    rho0 = rho1.real
    ak0 = fft(rho0)
    rho0x = ifft(1.0j * k * ak0)
    rho0x3 = ifft(-1.0j * k**3 * ak0)
    rho0x4 = ifft(k**4 * ak0)

    bk0 = fft(rho0 * rho0x4)
    ck0 = fft(rho0x * rho0x3)

    rho1 = rho2.real
    ak1 = fft(rho1)
    rho1x = ifft(1.0j * k * ak1)
    rho1x3 = ifft(-1.0j * k**3 * ak1)
    rho1x4 = ifft(k**4 * ak1)

    bk1 = fft(rho1 * rho1x4)
    ck1 = fft(rho1x * rho1x3)

print("Finished the computation")
# '''
# Post-processing : Following commands require the file 'post_processing.py'
# '''
# The following commands create a plot showing time-evolution of the initial
# condition and creates a movie of the same. In case user needs to plot the data
# via another application e.g. gnuplot, these lines can be suppressed / removed
# without affecting the program.
# make_plot(2, name="Toy_PDE")
make_movie([0, L], [-3, 3], name="movie")
# make_plot_from("solution92.dat")
