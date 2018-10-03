import scipy.linalg as ln
import math
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt  

def f (x):
	return np.sin(x / 5)*np.exp(x / 10) + 5*np.exp(-x / 2)
A = [[1,1],[1, 15]]
F = [1]*2
F[0]= f(1)
F[1] = f(15)
x = np.arange(1, 15, 0.1)
g = ln.solve(A, F) 
def ff(x):
    return g[0] + g[1]*x
plt.plot(x, f(x))
plt.plot(x, ff(x))
plt.show()