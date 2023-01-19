import numpy as np
import matplotlib.pyplot as plot
from numpy.fft import fft, ifft
import math

# takes the fft of function and returns the derivative of the function in
# physical space


def ifftik(fftfunc, k):
    fftfuncx = 1j * k * fftfunc  # take derivative in frequency space
    return ifft(fftfuncx)  # convert the derivative to physical space


# Define the parameters
nx = 128  # grid points / sampling points
a = 0  # domain a to b
b = 2.0*np.pi  # domain a to b
dx = (b-a)/(nx-1)  # step size
dt = 0.01  # time step
Nt = 50  # of time steps
nu = 0.1  # diffusivity
# grid points (excludes last point)
x = np.linspace(0.0, 2.0 * np.pi - dx, nx - 1)

# construct the array of wavenumbers
k1 = np.arange(0, nx/2)
k2 = np.arange(-nx/2 + 1, 0)
k = np.concatenate((k1, k2))

init = np.sin(3*x)*np.cos(8*x)  # define the initial condition
fftinit = fft(init)  # take its FFT
initx = ifftik(fftinit, k)

# fftsol = np.zeros(nx-1)

# for n in range(0, Nt):
#     fftsol = fftinit - nu * dt * np.multiply(k**2, fftinit)
#     sol = ifft(fftsol)
#     fftinit = fftsol

# plot the initial and final solutions
plot.plot(x, initx.real, 'r', x, ifft(fftinit).real, 'b')
plot.xlabel("domain (x)")
plot.ylabel("function (f(x))")
plot.title("Heat equation")
plot.grid(True)

plot.tight_layout()
plot.show()
