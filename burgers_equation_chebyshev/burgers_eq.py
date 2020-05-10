import numpy as np
import matplotlib.pyplot as plt
import os
import glob

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

np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)
clear_datfiles()

N = 127
dt = 0.01
nsteps = 500
nu = 0.02
D, x = cheb(N)
u0 = sin(pi*(x-1))  # Initial condition
uinit = u0
rhs = u0
write_real(x, u0, 0)

u1 = u0 + dt*(nu*np.dot(D, np.dot(D, u0))) - dt*0.5 * np.dot(D, u0*u0)
write_real(x, u1, 1)

for i in range(2, nsteps):
    A = 4*np.eye(N+1) - 2*nu*dt*np.matmul(D, D)
    B = 4*u1 + 2*nu*dt*np.dot(D, np.dot(D, u1)) - 3*dt*np.dot(D, u1*u1) + dt*np.dot(D, u0*u0)
    dirichilet_bc(A, B)
    u2 = np.linalg.solve(A, B)
    assert all(np.abs(u2) < 100)
    u0 = u1
    u1 = u2
    write_real(x, u2, i)

make_plot(7, show=1)
make_movie(show=1)