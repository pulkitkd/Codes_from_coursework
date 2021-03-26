
from scipy import *
from scipy.fftpack import fft, ifft, rfft
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
We adopt computational units
hbar = 1
omega = 1
m = 1
"""
# Initialising variables
N = 1000
L = 10.
time = 1000
dt = 1.

# Simple Harmonic Oscillator (SHO) potential
def V(x):
    return 0.5 * x**2

# Wavefunction at t = 0
def psi_0(x):
    psi = ( exp(-1j*((x-4.)/2)) * exp(-1j*((x-2.))/2) * exp(-1j*((x+1.))/2) * exp(-1j*((x+3.))/2) ) * (exp(-((x/sqrt(5))**2))/2 * sqrt(2*pi*5))
    return psi

# Split step operator FFT 
def split_step(time, dt, N, L):
    dx = L/N
    x = linspace(-L/2, L/2-dx, N)
    psi = psi_0(x)

    # Output array
    t_out = zeros(time + 1)
    psi_out = zeros((time + 1, N), dtype=complex)
    t_out[0], psi_out[0] = 0, copy(psi)

    # Potential operator
    def potential_step(psi):
        psi = exp(-1j * dt * V(x)) * psi
        return psi

    # For the kinetic step
    n = arange(N)  # oscillatory factor
    p = (-1)**n

    def kinetic_step(psi):
        # Kinetic operator
        expK = exp(-0.25j * dt * (-pi/dx + n * 2*pi/L)**2)

        psi = p * ifft(expK * fft(psi * p))
        return psi

    # split step run
    for i in range(time):
        psi = kinetic_step(psi)
        psi = potential_step(psi)
        psi = kinetic_step(psi)

        t_out[i+1] = i * dt
        psi_out[i+1] = copy(psi)

    return x, t_out, psi_out

# Evolving the wavefunction
x, t, psi = split_step(time, dt, N, L)

# Hanning window function
def win_fn(tstep, time_arr):
    return 1 - cos(2*pi*time_arr[tstep]/time_arr[-1])

# Correlation function
t = delete(t,0)
corr_arr = zeros(len(t), dtype = complex)
for s in range(len(t)):
    corr = conj(psi[0]) * psi[s]
    corr_arr[s] = trapz(corr,dx=L/N)
    corr_arr[s] *= win_fn(s, t)/t[-1]

# Fourier transform of correlation function
fcorr_arr = fft(corr_arr)

# Sieving peaks for curve fitting
peaks, _ = find_peaks(fcorr_arr, height =0.01)

# Line shape fitting
def Line(E, W, En):
    a = 1j * (E - En) * time
    b = (exp(a)-1)/a + 0.5*((exp(a+2*pi)-1)/(a+2*pi)) + 0.5*((exp(a-2*pi)-1)/(a-2*pi))
    return W*b

# Line fit for Real and Imag parts
def LineBoth(x, W, En):
    n = len(x)
    x_real = x[:n//2]
    x_imag = x[n//2:]
    y_real = real(Line(x_real, W, En))
    y_imag = imag(Line(x_imag, W, En))
    return hstack([y_real, y_imag])

# Data for the fitting
freq_arr = linspace(0,2*pi/dt, time)
yReal = fcorr_arr.real
yImag = fcorr_arr.imag
xdata = hstack([freq_arr, freq_arr])
ydata = hstack([yReal, yImag])

# Curve fitting according to no. of peaks
def BestFit(x, W1,W2,W3,W4,W5,W6, E1,E2,E3,E4,E5,E6):
    fn = LineBoth(x, W1, E1)+LineBoth(x, W2, E2)+LineBoth(x, W3, E3)+LineBoth(x, W4, E4)+LineBoth(x, W5, E5)+LineBoth(x, W6, E6)
    return fn

errfunc = lambda p, x, y: (BestFit(x, *p)-y)**2
guess = [.5,.5,.5,.5,.5,.5,.5,1.5,2.5,3.5,4.5,5.5]

optim, success = leastsq(errfunc, guess[:], args=(xdata,ydata))

err = sqrt(errfunc(optim, xdata, ydata)).sum()
print('Residual error in fitting: {}'.format(err))

# Plot visualisation
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(x, V(x), 'r', label='V(x)')
plt.plot(x, psi_0(x), 'b', label='$|\psi|^{2}$')
plt.title('Potential and initial wavefunction')
plt.legend(loc='best')
plt.xlabel('x')


plt.subplot(1,2,2)
plt.plot(freq_arr, BestFit(freq_arr, *optim))
plt.title('Energy spectrum')
plt.xticks(arange(0, freq_arr[-1], step=1))
plt.xlabel('Energy in $\hbar\omega$')
plt.ylabel('Spectrum')

plt.savefig('plot.png')


