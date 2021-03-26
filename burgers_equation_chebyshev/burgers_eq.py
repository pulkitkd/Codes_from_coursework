'''
Code to solve the viscous Burgers equation using Chebyshev collocation method.

u_t = nu* u_xx - u* u_x

We have used Adams-Bashforth two-step scheme for the non-linear term and
Adams-Moulton two-step scheme for the linear (diffusion) term.

To run the code:

Create a directory containing the following four elements-
* the file 'burgers_eq.py'
* the file 'cheb_trefethen.py'
* the file 'post_processing.py'
* a subdirectory '/data'
Navigate to the above root directory via terminal and execute-
$ python3 ./burgers_eq.py 
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from numpy import pi, cos, sin, exp
from matplotlib import animation

from cheb_trefethen import *
from post_processing import *

# write data (x, u) to a file titled solution*.dat in subdirectory data/


def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()

# clears all *.dat files from the subdirectory data/
# use before starting a new run of the program


def clear_datfiles():
    dirPath = "data"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath+"/"+fileName)

# Apply dirichilet BC to the AX=B system of Chebyshev collocation matrices


def dirichilet_bc(A, B):
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    B[0] = 0.0
    B[-1] = 0.0

# a function to print all numbers as 0.123456 instead of 1.23*10^-1


def clean_print_matrix():
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)

#==============================================================================#

# Input parameters-

# diffusivity (nu)
# time step (dt)
# total no. of time steps (nsteps)
# grid points (n) - must be an odd number
# length of the domain (L)
# domain (x)
# physical domain (xp)
# scaling factor (sf) - maps (-1 , 1) to (0 , L)
# initial conditions (init)

clear_datfiles()
# physical parameters
nu = 0.01
# time
dt = 0.01
nsteps = 200
# space
# odd N yields much better results for small n
n = 127
L = 1.0
sf = 2.0/L # scaling factor
D, x = cheb(n)
xp = (x+1)/sf

# initial condition
u0 = sin(sf*np.pi*xp)
t1 = time.time()
uinit = u0
write_real(xp, u0, 0)

u1 = u0
write_real(xp, u1, 1)

# Using AM2 for linear and AB2 for non-linear term, we obtain u2 from u1 and u0.
# This generates a system of the form AX=B that is solved using Python's built
# in solver to obtain X (u2). At each time step, it is preferrable to check
# that the code doesn't blow up, hence we insert an assert statement to abort
# the code if output becomes too large. Finally, the output at each time step is
# written to a data file in subdirectory /data.

for i in range(2, nsteps):
    A = np.eye(n+1) - 0.5*nu*sf**2*dt*np.matmul(D, D)
    B = (u1 + 0.5*nu*sf**2*dt*np.dot(D, np.dot(D, u1)) - 
         0.25*3*sf*dt*np.dot(D, u1*u1) + 0.25*sf*dt*np.dot(D, u0*u0))
    dirichilet_bc(A, B)
    u2 = np.linalg.solve(A, B)
    assert all(np.abs(u2) < 100)
    u0 = u1
    u1 = u2
    write_real(xp, u2, i)

t2 = time.time()
print("Total time for Chebyshev collocation method = ", t2-t1)

'''
Post-processing : Following commands require the file 'post_processing.py'
'''
# The following commands create a plot showing time-evolution of the initial
# condition and creates a movie of the same. In case user needs to plot the data
# via another application e.g. gnuplot, these lines can be suppressed / removed
# without affecting the program.

make_plot(8, name="Burgers_eq_(Chebyshev_collocation)")
make_movie([0, 1], [-1, 1], name="Burgers_eq_(Chebyshev_collocation)")
