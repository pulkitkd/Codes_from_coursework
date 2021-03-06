import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# takes the fft of function and returns the derivative of the function in
# physical space


def ifftik(fftfunc, k):
    ikfftfunc = 1j * k * fftfunc  # take derivative in frequency space
    return ifft(ikfftfunc)  # convert the derivative to physical space

# write the real parts of the supplied array (x and u) to a data file
# with name solution"n".dat


def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()


# define the parameters
nx = 64  # grid points / sampling points
a = 0  # domain a to b
b = 2.0*np.pi  # domain a to b
dx = (b-a)/(nx-1)  # step size
dt = 0.05  # time step
nu = 0.1  # diffusivity
Nt = 25  # of time steps
save = 10  # save data every 'this' number of time steps
# grid points (excludes last point)
x = np.linspace(0.0, 2.0 * np.pi - dx, nx - 1)

# construct the array of wavenumbers
k1 = np.arange(0, nx/2)
k2 = np.arange(-nx/2 + 1, 0)
k = np.concatenate((k1, k2))

# define the initial condition and write it to file
init = np.sin(x)
u0 = init
write_real(x, u0, 0)

# print("k      = \n", k)
# print("fft(u0) = \n", fft(u0)/len(u0))
# Determine the necessary quantities at t = n-1
# -> fft(u*ux) at t = 0

# take FFT of u0
fftu0 = fft(u0)
# get u0_x in physical space
u0x = ifftik(fftu0, k)
# get fft of u0*u0_x
fftu0u0x = fft(u0 * u0x)

# determine the necessary quantities at t = n
# -> fft(u*ux) @ t=1
# -> fft(u) @ t=1
# get u @ t=n from u @ t=n-1 using Euler's method
# get u1 in frequency space
fftu1 = fftu0 - dt * (fftu0u0x + nu*k**2*fftu0)
# convert u1 to physical space
u1 = ifft(fftu1)
# get u1_x in physical space
u1x = ifftik(fftu1, k)
# get fft(u1 * u1_x)
fftu1u1x = fft(u1 * u1x)

# Evaluate u2 using AM2 for the linear term and AB2 for the non-linear term
fftu2 = ((1.0 - nu * k**2 * dt * 0.5) * fft(u1) - (0.5 * dt * (3.0*fftu1u1x - fftu0u0x))) / (1 + 0.5 * nu * k**2 * dt)
# convert u2 to physical space
u2 = ifft(fftu2)

u0 = u1
u1 = u2

for i in range(1, Nt):
    # take FFT of u
    fftu0 = fft(u0)
    # get u_x in physical space
    u0x = ifftik(fftu0, k)
    # get fft of u*u_x
    fftu0u0x = fft(u0 * u0x)

    # determine the necessary quantities at t = n
    # -> fft(u*ux) @ t=1
    # -> fft(u) @ t=1
    # get u @ t=n from u @ t=n-1 using Euler's method
    # get u1 in frequency space
    fftu1 = fftu0 - dt * (fftu0u0x + nu*k**2*fftu0)
    # convert u1 to physical space
    u1 = ifft(fftu1)
    # get u1_x in physical space
    u1x = ifftik(fftu1, k)
    # get fft(u1 * u1_x)
    fftu1u1x = fft(u1 * u1x)

    # Evaluate u2 using AM2 for the linear term and AB2 for the non-linear term
    fftu2 = ((1.0 - nu * k**2 * dt * 0.5) * fft(u1) - (0.5 * dt * (3.0*fftu1u1x - fftu0u0x))) / (1 + 0.5 * nu * k**2 * dt)
    # convert u2 to physical space
    u2 = ifft(fftu2)

    u0 = u1
    u1 = u2


# for i in range(1, Nt):
#     # take FFT of u
#     fftu = fft(u)
#     # get u_x in physical space
#     ux = ifftik(fftu, k)
#     # get fft of u*u_x
#     fftuux = fft(u * ux)
#     # get u at next time step in frequency space (u1)
#     fftu1 = fftu - dt * (fftuux + nu*k**2*fftu)
#     # convert u1 to physical space
#     u1 = ifft(fftu1)
#     # update u with u1
#     u = u1
#     # write x and u to a file
#     if i%save == 0:
#         write_real(x,u,i)

# load data from files to be plotted
# plotfiles = [10, 20, 50, 90]
# x0, u0 = np.loadtxt("data\solution0.dat", delimiter=",", unpack=True)
# x1, u1 = np.loadtxt(
#     "data\solution"+str(plotfiles[0])+".dat", delimiter=",", unpack=True)
# x2, u2 = np.loadtxt(
#     "data\solution"+str(plotfiles[1])+".dat", delimiter=",", unpack=True)
# x3, u3 = np.loadtxt(
#     "data\solution"+str(plotfiles[2])+".dat", delimiter=",", unpack=True)
# x4, u4 = np.loadtxt(
#     "data\solution"+str(plotfiles[3])+".dat", delimiter=",", unpack=True)

# plot the figure
# plt.plot(x0, u0, x1, u1, x2, u2, x3, u3, x4, u4)
plt.plot(x, init.real, x, u2.real)
plt.xlabel("domain (x)")
plt.ylabel("function u(x,t)")
plt.title("Viscous Burgers Equation")
plt.grid(True, linestyle='dotted')
plt.savefig("burgers_eqn_test.png", dpi=150)

plt.tight_layout()
plt.show()
