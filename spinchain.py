import numpy as np
from numba import njit


@njit(nogil=True)
def tensor(*args):
    ans = args[0]
    for mats in args[1:]:
        ans = np.kron(ans,mats)
    return ans

@njit(nogil=True)
def pauli_couple(particles, s1, s2, part_idx1, part_idx2):
#     particles, part_idx1, part_idx2 = particles, part_idx1-1, part_idx2-1
    sx = np.array([[0,1.0],[1.0,0]]) + 0.0*1j
    sy = np.array([[0,-1j],[1j,0]]) + 0.0*1j
    sz = np.array([[1.0,0],[0,-1.0]]) + 0.0*1j
    if s1 == 'x': sigma1 = sx
    if s1 == 'y': sigma1 = sy
    if s1 == 'z': sigma1 = sz
    if s1 == 'p': sigma1 = sx + 1j*sy
    if s1 == 'm': sigma1 = sx - 1j*sy
        
    if s2 == 'x': sigma2 = sx
    if s2 == 'y': sigma2 = sy
    if s2 == 'z': sigma2 = sz
    if s2 == 'p': sigma2 = sx + 1j*sy
    if s2 == 'm': sigma2 = sx - 1j*sy
    
#     print(part_idx1,part_idx2)
    if part_idx1 > particles-1: part_idx1 = part_idx1%(particles)
#     print(part_idx1,part_idx2)
    if part_idx2 > particles-1: part_idx2 = part_idx2%(particles)
#     print(part_idx1,part_idx2)
    if part_idx1 > part_idx2: part_idx1, part_idx2 = part_idx2, part_idx1
#     print(part_idx1,part_idx2)
    
    if part_idx1==part_idx2:
        return tensor(np.eye(2**(part_idx1)) + 0.0*1j, sigma1@sigma2, np.eye(2**(particles - part_idx1 - 1)) + 0.0*1j)
    
    pre = np.eye(2**(part_idx1)) + 0.0*1j
#     print(pre)
    mid = np.eye(2**(part_idx2 - part_idx1 - 1)) + 0.0*1j
    post = np.eye(2**(particles - part_idx2 - 1)) + 0.0*1j
    return tensor(pre,sigma1,mid,sigma2,post) + 0.0*1j

@njit(nogil=True)
def hamiltonian(particles,Exy,Ez):
    n = particles
    term = np.zeros((2**n, 2**n)) + 0.0*1j
    for idx in range(n):
        term = term + (Exy*pauli_couple(n,'x','x',idx,idx+1) + 
                       Exy*pauli_couple(n,'y','y',idx,idx+1) + 
                       Ez *pauli_couple(n,'z','z',idx,idx+1))
    return term