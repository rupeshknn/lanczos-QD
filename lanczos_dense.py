"""
lanczos diag using numpy dense array
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix

def lanczos_eig(A,v0,m):
    '''
    Finds the lowest m eigen values and eigen vectors of a symmetric array

    A : ndarray, shape (ndim, ndim)
    Array to diagnolize.
    v0 : ndarray, shape (ndim,)
    A vector to start the lanczos iteration
    m : scalar
    the dim of the krylov subspace

    returns 
    E : ndarray, shape (m,) Eigenvalues
    W : ndarray, shape (ndim, m) Eigenvectors
    '''

    n = v0.size
    Q = np.zeros((m,n))

    v = np.zeros_like(v0)
    r = np.zeros_like(v0) # v1

    b = np.zeros((m,))
    a = np.zeros((m,))

    v0 = v0 / np.linalg.norm(v0)
    Q[0,:] = v0

    r = A @ v0
    a[0] = v0 @ r
    r = r - a[0]*v0
    b[0] = np.linalg.norm(r)

    error = np.finfo(np.float64).eps

    for i in range(1,m,1):
        v = Q[i-1,:]

        Q[i,:] = r / b[i-1]

        r = A @ Q[i,:] # |n>

        r = r - b[i-1]*v

        a[i] = (Q[i,:] @ r) # real?
        r = r - a[i]*Q[i,:]

        b[i] = np.linalg.norm(r)

        if b[i] < error:
            m = i
            print('smaller space found',m)
            break

    E,V = eigh_tridiagonal(a[:m],b[:m-1])
    Q = Q[:m]
    W = (Q.T @ V)
    return E, W
