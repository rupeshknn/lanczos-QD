"""
Module contaning Lanczos diagonalization algorithm
"""
from typing import Tuple
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix
def lanczos_eig(array: csr_matrix, v_0: np.ndarray, k_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Finds the lowest k_dim eigen values and eigen vectors of alpha symmetric array

    array : ndarray, shape (ndim, ndim)
    Array to diagnolize.
    v_0 : ndarray, shape (ndim,)
    A vector to start the lanczos iteration
    k_dim : scalar
    the dim of the krylov subspace

    returns
    eigen_value : ndarray, shape (k_dim,) Eigenvalues
    eigen_vectors : ndarray, shape (ndim, k_dim) Eigenvectors
    '''

    array_dim = v_0.size
    q_basis = np.zeros((k_dim, array_dim))

    v_p = np.zeros_like(v_0)
    projection = np.zeros_like(v_0) # v1

    beta = np.zeros((k_dim,))
    alpha = np.zeros((k_dim,))

    v_0 = v_0 / np.linalg.norm(v_0)
    q_basis[0,:] = v_0

    projection = array @ v_0
    alpha[0] = v_0 @ projection
    projection = projection - alpha[0]*v_0
    beta[0] = np.linalg.norm(projection)

    error = np.finfo(np.float64).eps

    for i in range(1,k_dim,1):
        v_p = q_basis[i-1,:]

        q_basis[i,:] = projection / beta[i-1]

        projection = array @ q_basis[i,:] # |array_dim>

        projection = projection - beta[i-1]*v_p

        alpha[i] = (q_basis[i,:] @ projection) # real?
        projection = projection - alpha[i]*q_basis[i,:]

        beta[i] = np.linalg.norm(projection)

        if beta[i] < error:
            k_dim = i
            print('smaller space found', k_dim)
            break

    eigen_value, eigen_vectors_t = eigh_tridiagonal(alpha[:k_dim], beta[:k_dim-1])
    q_basis = q_basis[:k_dim]
    eigen_vectors_a = (q_basis.T @ eigen_vectors_t)
    return eigen_value, eigen_vectors_a
