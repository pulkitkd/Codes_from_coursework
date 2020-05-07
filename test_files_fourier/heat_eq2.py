import numpy as np
import matplotlib.pyplot as plot
from numpy.fft import fft, ifft

nx = 64  # grid points / sampling points
dx = 2.0*np.pi/nx  # step size
dt = 0.01  # time step
Nt = 5000; # # of time steps
# grid points where function is evaluated
x = np.linspace(0, 2 * np.pi - dx, nx - 1)
print("x = \n", x) 

nu = 0.1; # diffusivity

# define the initial condition
init = np.exp(-(x-np.pi)**2)
# take its FFT
fftinit = fft(init)
# take the iFFT of FFT of the function
# ifftinit = ifft(fftinit)

# construct the array of wavenumbers
k1 = np.arange(0, nx/2)
k2 = np.arange(-nx/2 + 1, 0)
k = np.concatenate((k1, k2))
# print("k = \n", k) 
fftsol = np.zeros(nx-1)

for n in range(0, Nt):
    fftsol = fftinit - nu * dt * np.multiply(k**2 , fftinit) 
    # fftsol = fftinit - nu*(dt * k**2 * fftinit );
    
    sol = ifft(fftsol)

    # if(n%5 == 0):
    #     plot.plot(x, sol.real, 'r', x, init.real, 'b')
        
    # update 
    fftinit = fftsol

        
    



# plot the initial and final solutions
plot.plot(x, sol.real, 'r', x, init.real, 'b')
plot.xlabel("domain (x)")
plot.ylabel("function (f(x))")
plot.title("Heat equation")
plot.grid(True)

plot.tight_layout()
plot.show()
