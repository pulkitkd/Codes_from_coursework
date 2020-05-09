import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import pi, cos, sin, exp
from matplotlib import animation

from cheb_trefethen import *
from post_processing import *

'''
Code to solve the Heat Equation using Chebyshev Collocation method. Time
stepping is done via the implicit Adams-Moulton two-step method. The boundary
conditions are homogeneous Dirichilet conditions.
'''


# writes x and u to file titled solution(n).dat in subdriectory data/
def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()

# clears all *.dat files from the subdirectory data/
def clear_datfiles():
    list = os.listdir("data/")
    file_count = len(list)
    for i in range(0, file_count):
        if ("data/solution"+str(i)+".dat" == 1):
            os.remove("data/solution"+str(i)+".dat")

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

N = 63
dt = 0.001
nsteps = 50
D, x = cheb(N)
u0 = sin(0.5*pi*x)*sin(5*pi*x)  # Initial condition
uinit = u0
rhs = u0
write_real(x, u0, 0)

for i in range(1, nsteps):
    A = 2*np.eye(N+1) - dt*np.matmul(D, D)
    B = 2*u0 + dt*np.dot(np.matmul(D, D), u0)
    dirichilet_bc(A, B)
    u1 = np.linalg.solve(A, B)
    u0 = u1
    write_real(x, u1, i)

make_multiplot(n=5,name="fig.png",show=1)
make_movie([-1, 1], [-1, 1], "movie.mp4", 1)