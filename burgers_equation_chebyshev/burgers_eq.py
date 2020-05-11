'''
Code to solve the viscous Burgers equation using Chebyshev collocation method.

u_t = nu* u_xx - u* u_x

We have used Adams-Bashforth two-step scheme for the non-linear term and
Adams-Moulton two-step scheme for the linear (diffusion) term.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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

clear_datfiles()
# physical parameters
nu = 0.03
# time
dt = 0.01
nsteps = 500
# space
N = 128
L = 1.0
D, x = cheb(N)
xp = (0.5*L)*(x+1)

#initial condition
u0 = sin(2*pi*xp)
t1 = time.time()
uinit = u0
rhs = u0
write_real(xp, u0, 0)

# Since AM2 is a two-step scheme, we need u0 and u1 to determine u2. So we take
# the first time step using Euler's method to determine u1 from u0

u1 = u0 + dt*(nu*np.dot(D, np.dot(D, u0))) - dt*0.5 * np.dot(D, u0*u0)
write_real(xp, u1, 1)

# Using AM2 for linear and AB2 for non-linear term, we obtain u2 from u1 and u0.
# This generates a system of the form AX=B that is solved using Python's built
# in solver to obtain X (= u2). At each time step, it is preferrable to check
# that the code doesn't blow up, hence we insert an assert statement to abort
# the code if output becomes too large. Finally, the output at each time step is
# written to a data file in subdirectory /data

for i in range(2, nsteps):
    A = 8*np.eye(N+1) - nu*L**2*dt*np.matmul(D, D)
    B = 8*u1 + nu*L**2*dt*np.dot(D, np.dot(D, u1)) - 3*dt*L*np.dot(D, u1*u1) + L*dt*np.dot(D, u0*u0)
    dirichilet_bc(A, B)
    u2 = np.linalg.solve(A, B)
    assert all(np.abs(u2) < 100)
    u0 = u1
    u1 = u2
    write_real(xp, u2, i)
    
t2 = time.time()
print("Total time for Chebyshev collocation method = ", t2-t1)
make_plot(2, name = "Burgers_eq_sine_wave_C", show=1)
# make_movie([0,1],[-1,1],name = "Burgers_eq_sine_wave_C", show=1)
