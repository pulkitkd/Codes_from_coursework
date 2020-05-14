import numpy as np
import matplotlib.pyplot as plot
import post_processing
from post_processing import *
from numpy.fft import fft, ifft

def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open(".\data\solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()

n = 8  # grid points / sampling points
dx = 1/n  # step size
dt = 0.01 # time step
nsteps = 100 #no. of time steps
#t = 2500*dt #final time
# grid points where function is evaluated
x = np.linspace(0, 2 * np.pi - dx, n - 1)

# define the initial condition
init = np.exp(-(x-np.pi)**2)
# take its FFT
fftinit = fft(init)
# take the iFFT of FFT of the function
ifftinit = ifft(fftinit)

#print(fftfunction/len(function))
# construct the array of wavenumbers
k1 = np.arange(0, n/2)
k2 = np.arange(-n/2 + 1, 0)
#Two possible forms for the array
#k = np.concatenate((k1, np.zeros(1), k2))
k = np.concatenate((k1, k2))
print(k)


t = 0

for i in range(1, nsteps):
    sol = fftinit * np.exp(-k * k * t)
    ifftsol = ifft(sol)
    write_real(x, sol, i)
    t = t + dt

#plot the initial and final solutions
plot.plot(x, ifftsol.real, 'r', x, ifftinit.real, 'b')
plot.xlabel("domain (x)")
plot.ylabel("function (f(x))")
plot.title("Schrodinger equation")
plot.grid(True)

plot.tight_layout()
plot.show()
