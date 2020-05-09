from cheb_trefethen import *
from numpy import pi, cos, sin, exp
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt

def dirichilet_bc(A, B):
    A[0,:] = 0.0
    A[0,0] = 1.0
    A[-1,:] = 0.0
    A[-1,-1] = 1.0
    B[0] = 0.0
    B[-1] = 0.0

def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()
        


np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

N = 63
dt = 0.001
D, x = cheb(N)
u0 = exp(-25*x*x)  # Initial condition
uinit = u0
rhs = u0
write_real(x, u0, 0)    

for i in range(1,100):
    A = 2*np.eye(N+1) - dt*np.matmul(D,D)
    B = 2*u0 + dt*np.dot(np.matmul(D,D),u0)
    dirichilet_bc(A, B)
    u1 = np.linalg.solve(A, B)
    u0 = u1
    write_real(x, u1, i)    

#Plot
fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-0.1, 1))
line, = ax.plot([], [], lw=2)

def animate(j):
    x1, u1 = np.loadtxt(
    "data\solution"+str(j)+".dat", delimiter=",", unpack=True)
    line.set_data(x1, u1)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=i, interval=20, blit=True)
anim.save('diffusion.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
# x1, u1 = np.loadtxt(
#     "data\solution"+str(i)+".dat", delimiter=",", unpack=True)
# plt.plot(x1, u1)
# plt.show()
