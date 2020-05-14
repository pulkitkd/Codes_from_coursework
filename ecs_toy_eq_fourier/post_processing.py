from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os

'''
Functions for post processing
In order for these codes to work-
* Save datafiles in a folder called 'data'
* Name the files 'solution1.dat' , 'solution2.dat' etc. starting with 'solution0.dat'
* The files should contain only two columns  - x and y
* x and y must be separated by a comma
'''

################################################################################
'''
make_movie()
inputs:
    xrange - array containing min and max x values
    yrange - array containing min and max y values
    name - a string containing the name and format, e.g. "movie.mp4"
    show - 0 on 1 depending on whether a popup window for the movie is desired
    
output:
    An mp4 / gif format video saved as <name>.mp4 in the root directory
    A popup window of the animation if show is 1 (default)
    
example call:
    # all default options
    make_movie()
    # all specified options
    make_movie([-1,1], [0, 5], name="mymovie.mp4", show=1)
'''
def make_movie(xr=[-1, 1],
               yr=[-1, 1],
               name="movie",
               show=1):
    framerate = 30
    intvl = 20
    list = os.listdir("data/")
    file_count = len(list)

    fig = plt.figure()
    ax = plt.axes(xlim=(xr[0], xr[1]), ylim=(yr[0], yr[1]))
    plt.xlabel("domain (x)")
    plt.ylabel("function u(x,t)")
    plt.title(name)
    line, = ax.plot([], [], lw=2)

    def animate(j):
        x1, u1 = np.loadtxt(
            "data/solution"+str(j)+".dat", delimiter=",", unpack=True)
        line.set_data(x1, u1)
        return line,

    anim = animation.FuncAnimation(
        fig, animate, frames=file_count, interval=intvl, blit=True)
    anim.save(str(name)+".mp4", fps=framerate, extra_args=['-vcodec', 'libx264'])
    
    if show==1:
        plt.show()
        
    plt.close()
################################################################################
'''
make_plot()
inputs:
    n - number of curves desired
    name - a string containing the name and format, e.g. "fig.png"
    show - 0 on 1 depending on whether a popup window for the movie is desired
output:
    A png image saved as <name.png> in the root directory
    A popup window of the figure if show is 1 (default)
    
example call:
    make_plot(name="fig.png", n=5, show=0)
    make_plot(5,"fig.png",1)
    
special cases:
    make_plot(1) - plots only the initial condition
    make_plot(2) - plots only the initial and final solution
'''
def make_plot(n=5, name="figure", show=1):
    plt.xlabel("domain (x)")
    plt.ylabel("function u(x,t)")
    plt.title(name)
    plt.grid(True, linestyle='dotted')
    list = os.listdir("data/")
    file_count = len(list)
    plotfiles = np.linspace(0,file_count-1,n)
    for i in range(0, len(plotfiles)):
        x, u = np.loadtxt("data/solution"+str(int(plotfiles[i]))+".dat", delimiter=",", unpack=True)
        plt.plot(x, u, label = "t ="+str("{0:.2f}".format(plotfiles[i]/file_count)))
        plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(str(name)+".png", dpi=150)
    if show == 1:
        plt.show()

    plt.close()

def make_plot_from(name="solution0.dat",save=0):
    plt.xlabel("domain (x)")
    plt.ylabel("function u(x,t)")
    plt.title(name)
    plt.grid(True, linestyle='dotted')
    x, u = np.loadtxt("data/"+name, delimiter=",", unpack=True)
    plt.plot(x, u, label = name)
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save == 1:
        plt.savefig(str(name)+".png", dpi=150)
    plt.show()

    plt.close()
    
# make_movie([0, 2.0*np.pi], [0, 3], name="movie")
