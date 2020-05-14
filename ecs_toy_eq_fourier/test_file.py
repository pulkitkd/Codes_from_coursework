import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fft, ifft
from numpy import multiply, sin, cos, exp, pi
from post_processing import *

#===============================Introduction===================================#
'''
This code solves the viscous burgers equation using Fourier-Galerkin spectral
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

def clean_print_matrix():
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)
#=============================Main Program=====================================#

    
clean_print_matrix()

n = 8
L = 2.0 * pi
x = domain(n, L)
u0 = np.ones(n-1)
# print(u0)
print(fft(u0)/len(u0))
