import numpy as np
from scipy.linalg import eig

import matplotlib.pyplot as plt
from numpy import pi, cos, sin

from cheb_trefethen import *

"""
Inputs
"""
Re = 5000.
Reinv = 1.0/Re
Ny = 128
kx = 1.
kz = 0.
omega = 1.

ksq = kx*kx + kz*kz
n = Ny+1
D, y = cheb(Ny)
D2 = D@D
D4 = D2@D2
Z = np.zeros((n, n))
Idn = np.eye(n)
U = 1. - y*y
Uy = D @ U
Uyy = D @ Uy

"""
M and L matrices
"""
M = np.block([[ksq - D2, Z], [Z, Idn]])

Los = 1j*kx*(U)*(ksq*Idn - D2) + 1j*kx*Uyy + Reinv*(ksq**2 + D4 - 2*ksq*D2)
Lsq = (1j*kx)*(U*Idn) + Reinv*(ksq*Idn - D2)
L = np.block([[Los, Z],[(1j*kz)*(Uy*Idn), Lsq]])

"""
BCs: v = dv/dy = eta = 0 at walls
"""
L[0, :] = 0.
L[n-1, :] = 0.
L[0, 0] = 1.
L[n-1, n-1] = 1.

L[n :] = 0.
L[2*n-1, :] = 0.
L[n, n] = 1.
L[2*n-1, 2*n-1] = 1.

L[1, :] = 0.
L[n-2, :] = 0.
L[1, 0:n] = D[0, :]
L[n-2, 0:n] = D[n-1, :]

M[0, :] = 0.
M[1, :] = 0.

M[n-1, :] = 0.
M[n-2, :] = 0.

M[n, :] = 0.
M[2*n-1, :] = 0.

"""
Solve the generalized eigenvalue problem M x = i omega L x
"""
w, v = eig(L, b=M, check_finite=True)

"""
Get the eigenvalues with largest imaginary parts
"""
w = -1j*w
w = np.sort_complex(w)
print("Eigenvalues = ", w)

"""
Plot the eigenvalues
"""
plt.scatter(np.real(w), np.imag(w))
plt.xlim((0,1))
plt.ylim((-1,0))
plt.xlabel("real")
plt.ylabel("imag")
plt.show()