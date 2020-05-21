import numpy as np
import os
import os.path
from os import path

# clears all *.dat files from the subdirectory data/
# use before starting a new run of the program


def clear_datfiles():
    dir_path = "data"
    filelist = os.listdir(dir_path)
    if len(filelist)!=0:
        print("removing all files from subdirectory",dir_path)
        for fileName in filelist:
            os.remove(dir_path+"/"+fileName)
    else:
        print("no files to remove")


def clear_dir(dir_path="data"):
    if path.exists(dir_path) == 0:
        os.mkdir(dir_path)
        print("Path does not exist. Creating new subdirectory",dir_path)
        
    else:
        filelist = os.listdir(dir_path)
        if len(filelist)!=0:
            print("removing all files from subdirectory",dir_path)
            for fileName in filelist:
                os.remove(dir_path+"/"+fileName)
        else:
            print("no files to remove")
      
# write the real parts of the supplied array (x and u) to a data file
# with name solution"n".dat


def write_real(x, u, n):
    data = np.array([x.real, u.real])
    data = data.T
    with open("data/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()


def write_to_dir(x, u, n, dir_path="data"):
    data = np.array([x.real, u.real])
    data = data.T
    with open(dir_path+"/solution"+str(n)+".dat", "w") as out_file:
        np.savetxt(out_file, data, delimiter=",")
        out_file.close()
        
         
# creates the wavenumber array of the form required for np.fft.ifft
# |0 | 1 | 2 | 3 | -3 | -2 | -1|


def wavenumbers(n):
    assert n % 2 == 0
    k1 = np.arange(0, n/2)
    k2 = np.arange(-n/2 + 1, 0)
    k = np.concatenate((k1, k2))
    return k

# grid points (excludes last point)
# enforces an even number of grid points


def domain(n, L):
    dx = L/(n-1)
    assert n % 2 == 0
    return np.linspace(0.0, L - dx, n - 1)

# print numbers in decimals rather than scientific notation


def clean_print_matrix():
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)