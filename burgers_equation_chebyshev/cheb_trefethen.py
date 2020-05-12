'''
This function is taken from an example given in the book 'Spectral Methods in
MATLAB' by  L N Trefethen. The Python translation of the codes is available at
http://blue.math.buffalo.edu/438/trefethen_spectral/all_py_files/

The function computes the Chebyshev differentiation matrix D and the grid
containing the Gauss-Lobatto grid points. 

Input:
	No. of grid points desired (N)
Output: 
	(N+1) X (N+1) differentiation matrix (D)
	A grid of N+1 Gauss Lobatto points between -1 and 1 (D)
Calling:
	D, x = cheb(N)
'''
from numpy import *

def cheb(N):
	n = arange(0,N+1)
	x = cos(pi*n/N).reshape(N+1,1) 
	c = (hstack(( [2.], ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
	X = tile(x,(1,N+1))
	dX = X - X.T
	D = dot(c,1./c.T)/(dX+eye(N+1))
	D -= diag(sum(D.T,axis=0))
	return D, x.reshape(N+1)
