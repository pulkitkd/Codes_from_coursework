from cheb_trefethen import *

from numpy import pi, cos, sin

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

N = 17
D, x = cheb(N)
print ("D = \n ", D)
print ("x = \n ", x)
u = sin(pi*x); # Initial condition

ux_exact = -pi**2 * sin(pi*x)
ux_cheb = np.dot(D, np.dot(D,u))

# plot the functions
plt.plot(x, ux_exact, label='exact')
plt.plot(x, ux_cheb,'o', label='cheb')
plt.legend()
plt.show()
plt.savefig("images/cheb_derivative.png")
