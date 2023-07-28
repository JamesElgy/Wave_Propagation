import numpy as np
import scipy.sparse as sr
import scipy.sparse.linalg as slg

N = int(1e5)
I = sr.eye(N,format='csr')
A = I + 1j*I
x = np.random.randn(N) + 1j*np.random.randn(N)
b = A.dot(x)

# scipy.gmres
x2 = slg.gmres(A,b)
print(np.linalg.norm(x2[0]-x))