from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os

'''
Code to make a movie from datafiles. In order for this code to work-
* Save datafiles in a folder called 'data'
* Name the files 'solution1.dat' , 'solution2.dat' etc.
* The files should contain only two columns  - x and y
* x and y must be separated by a comma
'''

def movie():
    list = os.listdir("data/")
    file_count = len(list)

    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
    line, = ax.plot([], [], lw=2)

    def animate(j):
        x1, u1 = np.loadtxt(
            "data\solution"+str(j)+".dat", delimiter=",", unpack=True)
        line.set_data(x1, u1)
        return line,

    anim = animation.FuncAnimation(
        fig, animate, frames=file_count, interval=20, blit=True)
    anim.save('diffusion.mp4', fps=120, extra_args=['-vcodec', 'libx264'])
    plt.show()