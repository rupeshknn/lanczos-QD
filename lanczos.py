"""
Module containing Lanczos diagonalization and time evolution algorithm
"""

from typing import Union
import numpy as np
from scipy.sparse import csr_matrix


def lanczos_basis(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int):
    """Tridiagonalises a hermitian array in a krylov subspace of dimension k_dim
    using Lanczos algorithm. returns basis of krylov subspace and projection of array
    in krylov subspace

    Args:
        array : Array to tridiagonalise.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.

    Returns:
        q_basis : Basis of the krylov subspace.
        tridiagonal : Tridiagonal projection of ``array``.
    """

    error = np.finfo(np.float64).eps
    data_type = np.result_type(array.dtype, v_0.dtype)
    v_0 = np.array(v_0).reshape(-1, 1)
    v_0 = v_0 / np.linalg.norm(v_0)
    array_dim = array.shape[0]

    q_basis = np.zeros((k_dim, array_dim), dtype=data_type)
    v_p = np.zeros_like(v_0)
    projection = np.zeros_like(v_0)
    beta = np.zeros((k_dim,), dtype=data_type)
    alpha = np.zeros((k_dim,), dtype=data_type)

    q_basis[[0], :] = v_0.T
    projection = array @ v_0
    alpha[0] = v_0.conj().T @ projection
    projection = projection - alpha[0] * v_0
    beta[0] = np.linalg.norm(projection)

    for i in range(1, k_dim, 1):
        v_p = q_basis[i - 1, :]
        q_basis[[i], :] = projection.T / beta[i - 1]
        projection = array @ q_basis[i, :]
        alpha[i] = q_basis[i, :].conj().T @ projection
        projection = projection - alpha[i] * q_basis[i, :] - beta[i - 1] * v_p
        beta[i] = np.linalg.norm(projection)

        # additional steps to increase accuracy
        delta = q_basis[i, :].conj().T @ projection
        projection -= delta * q_basis[i, :]
        alpha[i] += delta

        # exit if beta is less than numerical precision
        if beta[i] < error:
            k_dim = i
            break

    tridiagonal = (
        np.diag(alpha[:k_dim], k=0)
        + np.diag(beta[: k_dim - 1], k=-1)
        + np.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis[:k_dim]
    return q_basis.T, tridiagonal


def lanczos_eig(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int):
    """
    Finds the lowest ``k_dim`` eigenvalues and corresponding eigenvectors of a hermitian array
    using Lanczos algorithm.

    Args:
        array : Array to diagonalize.
        v_0 : Vector to initialise Lanczos iteration.
        k_dim : Dimension of the krylov subspace.

    Returns:
        eigen_values : lowest ``k_dim`` Eigenvalues.
        eigen_vectors_a : Eigenvectors in hilbert-space.
    """

    q_basis, tridiagonal = lanczos_basis(array, v_0, k_dim)
    eigen_values, eigen_vectors_t = np.linalg.eigh(tridiagonal)

    eigen_vectors_a = q_basis @ eigen_vectors_t

    return eigen_values, eigen_vectors_a


def lanczos_expm(
    array: Union[csr_matrix, np.ndarray],
    v_0: np.ndarray,
    k_dim: int,
    max_dt: float,
):
    """Calculates action of matrix exponential on the state using Lanczos algorithm.

    Args:
        array : Array to exponentiate.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.
        max_dt : Maximum step size.

    Returns:
        v_dt : Action of matrix exponential on state.
    """

    q_basis, tridiagonal = lanczos_basis(array, v_0, k_dim)
    eigen_values, eigen_vectors_t = np.linalg.eigh(tridiagonal)

    v_dt = (
        q_basis
        @ eigen_vectors_t
        @ (np.exp(-1j * max_dt * eigen_values) * eigen_vectors_t[0, :])
    )

    return v_dt
