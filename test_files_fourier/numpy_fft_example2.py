import numpy as np
import matplotlib.pyplot as plot
from numpy.fft import fft, ifft
def clean_print_matrix():
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)
    
clean_print_matrix()
n = 8
dx = 2.0*np.pi/n
x = np.linspace(0, 2 * np.pi, n)
print(x)
function = np.sin(x)
fftfunction = fft(function)
# fftfreq = np.fft.fftfreq(n)
# print(fftfunction)

k = np.zeros(n)
k1 = np.arange(0, n/2)
# k[int(n/2)] = 0.0
k2 = np.arange(-n/2, 0)
#k = np.concatenate((k1, np.zeros(1), k2))
k = np.concatenate((k1, k2))
print(k)
print(fftfunction/len(function))
plot.plot(k, abs(fftfunction.real)/len(function), '.')
plot.show()

# print(fftfreq)

# plot.plot(x,function)
