import numpy as np
import matplotlib.pyplot as plot
from numpy.fft import fft, ifft

n = 64  # grid points / sampling points
dx = 2.0*np.pi/(n-1)  # step size
# grid points where function is evaluated
x = np.linspace(0, 2.0 * np.pi - dx, n-1)

# define the function
function = np.sin(x)
# take its FFT
fftfunction = fft(function)
# take the iFFT of FFT of the function
ifftfunction = ifft(fftfunction)

print(fftfunction/len(function))
# construct the array of wavenumbers
k1 = np.arange(0, n/2)
# k[int(n/2)] = 0.0
k2 = np.arange(-n/2 + 1, 0)

# kc and k are two possible forms for the array
#k = np.concatenate((k1, np.zeros(1), k2))
k = np.concatenate((k1, k2))

# plot the frequency domain and physical domain representations
plot.subplot(2, 1, 1)
plot.plot(k, abs(fftfunction.imag)/len(function), '.')
plot.xlabel("wavenumbers (k)")
plot.ylabel("Fourier coefficient (C_k)")
plot.title("Frequency domain representation")
plot.grid(True)

plot.subplot(2, 1, 2)
plot.plot(x, ifftfunction.real, '.')
plot.xlabel("domain (x)")
plot.ylabel("function (f(x))")
plot.title("Physical domain representation")
plot.grid(True)

plot.tight_layout()
plot.show()
