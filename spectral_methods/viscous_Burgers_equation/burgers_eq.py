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
dt = 0.02  # time step
Nt = 1000  # of time steps
nu = 0.1  # diffusivity
save = 50 # save data every 'this' number of time steps
# grid points (excludes last point)
x = np.linspace(0.0, 2.0 * np.pi - dx, nx - 1)

# construct the array of wavenumbers
k1 = np.arange(0, nx/2)
k2 = np.arange(-nx/2 + 1, 0)
k = np.concatenate((k1, k2))

# define the initial condition and write it to file
init = np.sin(x)
u = init
write_real(x,u,0)

for i in range(1, Nt):
    # take FFT of u
    fftu = fft(u)
    # get u_x in physical space
    ux = ifftik(fftu, k)
    # get fft of u*u_x
    fftuux = fft(u * ux)
    # get u at next time step in frequency space (u1)
    fftu1 = fftu - dt * (fftuux + nu*k**2*fftu)
    # convert u1 to physical space
    u1 = ifft(fftu1)
    # update u with u1
    u = u1
    # write x and u to a file
    if i%save == 0:
        write_real(x,u,i)

#load data from files to be plotted
plotfiles = [50,200,500,950]
x0, u0 = np.loadtxt("data\solution0.dat", delimiter="," , unpack=True)
x1, u1 = np.loadtxt("data\solution"+str(plotfiles[0])+".dat", delimiter="," , unpack=True)
x2, u2 = np.loadtxt("data\solution"+str(plotfiles[1])+".dat", delimiter="," , unpack=True)
x3, u3 = np.loadtxt("data\solution"+str(plotfiles[2])+".dat", delimiter="," , unpack=True)
x4, u4 = np.loadtxt("data\solution"+str(plotfiles[3])+".dat", delimiter="," , unpack=True)

#plot the figure
plt.plot(x0, u0, x1, u1, x2, u2, x3, u3, x4, u4)
plt.xlabel("domain (x)")
plt.ylabel("function u(x,t)")
plt.title("Viscous Burgers Equation")
plt.grid(True,linestyle='dotted')
plt.savefig("burgers_eqn_test.png",dpi=150)

plt.tight_layout()
plt.show()