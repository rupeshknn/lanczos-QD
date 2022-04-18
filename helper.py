import numpy as np

def construct_tridiag(d,od):
    '''
    Constructs (ndim,ndim) symetric tridiagonal array given the diagonal and off-diagonal elements

    d : ndarray, shape (ndim,)
    The diagonal elements of the array.
    od : ndarray, shape (ndim-1,)
    The off-diagonal elements of the array.
    
    returns ndarray, shape (ndim,ndim)
    '''
    data_type = d.dtype
    n = d.size
    tri = np.zeros((n,n),dtype = data_type)
    for idx1 in range(n-1):
        tri[idx1,idx1] = d[idx1]
        tri[idx1+1,idx1] = od[idx1]
        tri[idx1,idx1+1] = od[idx1]
    tri[-1,-1] = d[-1]
    return tri

def eigh_tridiagonal(d,od):
    mat = construct_tridiag(d,od)
    return np.linalg.eigh(mat)