import numpy as np
import scipy.sparse as sr
import scipy.sparse.linalg as slg

def scipy_random_solve():

    import scipy.sparse as sr
    import scipy.sparse.linalg as slg
    import numpy as np

    N = int(1e5)
    I = sr.eye(N,format='csr')
    A = I + 1j*I
    x = np.random.randn(N) + 1j*np.random.randn(N)
    b = A.dot(x)

    # scipy.gmres
    x2 = slg.gmres(A,b)

    err = np.linalg.norm(x2[0]-x)

    # print(f'Passed SciPy random solve : {err < 1e-10}')

    return err

if __name__ == '__main__':
    scipy_random_solve()