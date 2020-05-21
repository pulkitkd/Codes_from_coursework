import math
import time
import numpy as np
import mpi4py
from mpi4py import MPI
from numpy.fft import fft, ifft
from numpy import multiply
import inputs
from inputs import *

#===============================Introduction===================================#
'''
This code solves a toy equation for ECS using Fourier-Galerkin spectral
methods for periodic boundary conditions

c_t = (1/Pe) c_xx + 2 c c_xxxx + c_xxxxxx + 4 c_x c_xxx + 1 - mu c

We want to evolve the Fourier coefficients of these terms in time using 
Adams-Bashforth 2 step method for the non-linear terms and the Adams-Moulton 2
step method for the linear terms.

To run the code:

* The root directory must contain
    -toy_pde.py
    -post_processing.py
    -postproc.py
    -data/

* Run the file toy_pde.py with desired parameter values. This will generate a set
of datafiles in the folder data/. 

* In order to plot the results, run the file postproc.py. 

User defined functions make_plot() and make_movie() generate the visualizations.
Details of the functions are given in file post_processing.py
'''
#=============================Main Program=====================================#
comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

new_dir_path = "data/mu_"+str(mu)+"_nu_"+str(nu)+"_L_"+str("{0:.2f}".format(L))

clear_dir(new_dir_path)

clean_print_matrix()
t1 = time.time()

# define u0 and write it to file
rho0 = init
write_to_dir(x, rho0, 0, new_dir_path)

ak0 = fft(rho0)

rho0x = ifft(1.0j * k * ak0).real
rho0x3 = ifft(-1.0j * k**3 * ak0).real
rho0x4 = ifft(k**4 * ak0).real

bk0 = fft(rho0 * rho0x4)
ck0 = fft(rho0x * rho0x3)

one = np.ones(n-1)
dk0 = fft(one)

# to begin the iterations, assume u1 = u0
rho1 = rho0.real
write_to_dir(x, rho1, 1, new_dir_path)

ak1 = ak0
bk1 = bk0
ck1 = ck0

A = ((np.ones(n-1) + 0.5 * nu * dt * sf**2 * k**2 +
      0.5 * dt * sf**6 * k**6 + 0.5 * mu * dt)**(-1))

B = (np.ones(n-1) - 0.5 * nu * dt * sf**2 * k**2 -
     0.5 * dt * sf**6 * k**6 - 0.5 * mu * dt)

dtsf4 = dt * sf**4

# In this loop we use the AM2 scheme for the linear term and the AB2 scheme for
# the non linear term to determine rho @ t = n+2 using rho @ t = n+1 and n. The
# first line in the loop generates an array of fourier coefficients at t = n+2.
# This array is inverted using ifft to get u at t = n+2. Following this, rho0, rho1
# and their derivatives are updated and the loop continues. The generated data
# files are stored in a subdirectory /data. The assert statement aborts the code
# in case the solution blows-up to large values.
t1 = time.time()
j = 2
T = dt * nsteps
print("Domain length       = ", L)
print("mu                  = ", mu)
print("nu                  = ", nu)
print("No of Fourier modes = ", n)
print("Total iterations    = ", nsteps)
print("Total time          = ", T)


for i in range(2, nsteps):
    ak2 = (((ak1 * B) + (bk1 * 3.0 * dtsf4) - (bk0 * dtsf4) +
            (ck1 * 6.0 * dtsf4) - (ck0 * 2.0 * dtsf4) + (dk0 * dt)) * A)
    rho2 = ifft(ak2)
    assert all(np.abs(rho2 < 1e3))
        
    if i % save_every == 0:
        write_to_dir(x, rho2, j, new_dir_path)
        j = j + 1

    rho0 = rho1.real
    ak0 = fft(rho0)
    rho0x = ifft(1.0j * k * ak0).real
    rho0x3 = ifft(-1.0j * k**3 * ak0).real
    rho0x4 = ifft(k**4 * ak0).real

    bk0 = fft(rho0 * rho0x4)
    ck0 = fft(rho0x * rho0x3)

    rho1 = rho2.real
    ak1 = fft(rho1)
    rho1x = ifft(1.0j * k * ak1).real
    rho1x3 = ifft(-1.0j * k**3 * ak1).real
    rho1x4 = ifft(k**4 * ak1).real

    bk1 = fft(rho1 * rho1x4)
    ck1 = fft(rho1x * rho1x3)

t2 = time.time()
print("Finished the computation. Time taken (s): ", t2 - t1)
