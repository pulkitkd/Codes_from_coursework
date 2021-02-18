from cheb_trefethen import *

from numpy import pi, cos, sin

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

N = 16
D, x = cheb(N)
X = np.linspace(-1,1,num=100)
print ("D = \n ", D)
print ("x = \n ", x)
u = sin(pi*x); # Initial condition

ux_exact = -pi**2 * sin(pi*X)
# ux_cheb = np.dot(D, np.dot(D,u))
# ux_cheb = np.dot(D @ D, u)
ux_cheb = D @ D @ u

# plot the functions
plt.plot(X, ux_exact, label='exact')
plt.plot(x, ux_cheb,'.', label='cheb')
plt.legend()
plt.show()