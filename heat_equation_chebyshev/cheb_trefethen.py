# CHEB  compute D = differentiation matrix, x = Chebyshev grid
#       Translation 12/22/12
from numpy import *

def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = arange(0,N+1)
		x = cos(pi*n/N).reshape(N+1,1) 
		c = (hstack(( [2.], ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = tile(x,(1,N+1))
		dX = X - X.T
		D = dot(c,1./c.T)/(dX+eye(N+1))
		D -= diag(sum(D.T,axis=0))
	return D, x.reshape(N+1)





'''
set_printoptions(linewidth=200,precision=4)
D,x = cheb(4)
print D
print x
'''

