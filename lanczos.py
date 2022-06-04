"""
Module contaning Lanczos diagonalization algorithm
"""
from typing import Tuple, Union
import numpy as np
# from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix

def lanczos_basis(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tridigonalises krylov subspace of dimension k_dim for a given sparse array

    Parameters
    ----------
    array : csr_matrix
        sparse array to diagonalise
    v_0 : np.ndarray
        inital vector
    k_dim : int
        dimension of the krylov subspace

    Returns
    -------
    Tridiagonal : np.ndarray
        tridigonalised matrix
    q_basis : np.ndarray
        basis of the krylov subspace
    """
    data_type = np.result_type(array.dtype, v_0.dtype)
    v_0 = np.array(v_0).reshape(-1,1) # ket
    array_dim = array.shape[0]
    q_basis = np.zeros((k_dim, array_dim), dtype=data_type)

    v_p = np.zeros_like(v_0)
    projection = np.zeros_like(v_0) # v1

    beta = np.zeros((k_dim,), dtype=data_type)
    alpha = np.zeros((k_dim,), dtype=data_type)

    v_0 = v_0 / np.sqrt(np.abs(v_0.conj().T @ v_0))
    q_basis[[0],:] = v_0.T

    projection = array @ v_0
    alpha[0] = v_0.conj().T @ projection
    projection = projection - alpha[0]*v_0
    beta[0] = np.sqrt(np.abs(projection.conj().T @ projection))

    error = np.finfo(np.float64).eps

    for i in range(1,k_dim,1):
        v_p = q_basis[i-1,:]

        q_basis[[i],:] = projection.T / beta[i-1]
        projection = array @ q_basis[i,:] # |array_dim>
        alpha[i] = q_basis[i,:].conj().T @ projection # real?
        projection = projection - alpha[i]*q_basis[i,:] - beta[i-1]*v_p
        beta[i] = np.sqrt(np.abs(projection.conj().T @ projection))
        
        # addtitional steps to increase accuracy
        delta = q_basis[i,:].conj().T @ projection
        projection -= delta*q_basis[i,:]
        alpha[i] += delta
        
        if beta[i] < error:
            k_dim = i
            # print('smaller space found', k_dim)
            break

    Tridiagonal = Tridiagonal = (
        np.diag(alpha[:k_dim], k=0)
        + np.diag(beta[: k_dim - 1], k=-1)
        + np.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis[:k_dim]
    q_basis = q_basis.T
    return Tridiagonal, q_basis

def lanczos_eig(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
    Finds the lowest k_dim eigenvalues and corresponding eigenvectors of a hermitian array

    Parameters
    ----------
    array : csr_matrix
        sparse array to diagonalise
    v_0 : np.ndarray
        inital vector
    k_dim : int
        dimension of the krylov subspace

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Eigenvalues and Eigenvectors
    """
    Tridiagonal, q_basis = lanczos_basis(array, v_0, k_dim)
    eigen_value, eigen_vectors_t = np.linalg.eigh(Tridiagonal)

    eigen_vectors_a = (q_basis @ eigen_vectors_t)

    return eigen_value, eigen_vectors_a, eigen_vectors_t, q_basis

def lanczos_exmp(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int, dt:float) -> Tuple[np.ndarray]:
    """Calculates action of matrix exponential on vector using lanczos algorithm

    Parameters
    ----------
    array : csr_matrix
        Array to exponentiate
    v_0 : np.ndarray
        Inital vector
    k_dim : int
        Dimension of the krylov subspace
    dt : float
        Maximum step size.

    Returns
    -------
    Tuple[np.ndarray]
        Action of matrix exponential on state
    """
    array = 1j*array
    eigen_value, _, eigen_vectors_t, q_basis = lanczos_eig(array, v_0, k_dim)
    return q_basis @ eigen_vectors_t @ (np.exp(-1j*dt*eigen_value)*eigen_vectors_t[0,:])
