import post_processing
from post_processing import *
# from toy_pde import mu, nu, L, new_dir_path
import numpy as np


nu = 0.01  # 1 / Pe
mu = 0.01
L = 4.0*np.pi
dir_path = "data/mu_"+str(mu)+"_nu_"+str(nu)+"_L_"+str("{0:.2f}".format(L))

# make_plot(10,dir_path=dir_path,name="sin_wave")
make_movie([0,2*np.pi],[0,2],name="sin_wave", dir_path=dir_path,show=1)