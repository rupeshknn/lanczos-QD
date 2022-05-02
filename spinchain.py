import numpy as np
from numba import njit

from quspin.operators import hamiltonian as qs_hamiltonian# Hamiltonians and operators
from quspin.basis import spin_basis_1d

# @njit(nogil=True)
# def tensor(*args:np.ndarray)->np.ndarray:
#     """tensor (np.kron) product of multiple arrays in order left to right

#     Returns:
#         np.ndarray: tensor product result
#     """
#     ans = args[0]
#     for mats in args[1:]:
#         ans = np.kron(ans,mats)
#     return ans

# @njit(nogil=True)
# def pauli_couple(particles:int, s1:str, s2:str, part_idx1:int, part_idx2:int)->np.ndarray:
#     """returns 2^particles-dim pauli matrices of different operator combinations

#     Args:
#         particles (int): number of particles
#         s1 (str): first pauli operator
#         s2 (str): second pauli  operator
#         part_idx1 (int): index for first operator
#         part_idx2 (int): index for second operator

#     Returns:
#         np.ndarray: 2^particles dimentional array
#     """
#     sx = np.array([[0,1.0],[1.0,0]], dtype = np.complex128)
#     sy = np.array([[0,-1j],[1j,0]], dtype = np.complex128)
#     sz = np.array([[1.0,0],[0,-1.0]], dtype = np.complex128)
#     if s1 == 'x': sigma1 = sx
#     if s1 == 'y': sigma1 = sy
#     if s1 == 'z': sigma1 = sz
#     if s1 == 'p': sigma1 = sx + 1j*sy
#     if s1 == 'm': sigma1 = sx - 1j*sy
        
#     if s2 == 'x': sigma2 = sx
#     if s2 == 'y': sigma2 = sy
#     if s2 == 'z': sigma2 = sz
#     if s2 == 'p': sigma2 = sx + 1j*sy
#     if s2 == 'm': sigma2 = sx - 1j*sy
    
#     if part_idx1 > particles-1: part_idx1 = part_idx1%(particles)
#     if part_idx2 > particles-1: part_idx2 = part_idx2%(particles)
#     if part_idx1 > part_idx2: part_idx1, part_idx2 = part_idx2, part_idx1
    
#     if part_idx1==part_idx2:
#         return tensor(np.eye(2**(part_idx1), dtype = np.complex128), sigma1@sigma2, np.eye(2**(particles - part_idx1 - 1), dtype = np.complex128))
    
#     pre = np.eye(2**(part_idx1), dtype = np.complex128)
#     mid = np.eye(2**(part_idx2 - part_idx1 - 1), dtype = np.complex128)
#     post = np.eye(2**(particles - part_idx2 - 1), dtype = np.complex128)
#     return tensor(pre,sigma1,mid,sigma2,post)

# @njit(nogil=True)
# def hamiltonian(particles:int, Exy:float, Ez:float)->np.ndarray:
#     """creates the Hisenberg XXZ spin chain hamiltonain
#     $$ H = J \sum_i^N \left( \sigma^x_i \sigma^x_{i+1} + \sigma^y_i \sigma^y_{i+1} \right) 
#     + \Delta \sum_i^N \left( \sigma^z_i \sigma^z_{i+1} \right) $$

#     Args:
#         particles (int): number of particles
#         Exy (float): coefficient of XY term
#         Ez (float): coeddicient of ZZ term

#     Returns:
#         np.ndarray: 2^particles x 2^particles array
#     """
#     n = particles
#     term = np.zeros((2**n, 2**n), dtype = np.complex128)
#     for idx in range(n):
#         term = term + (Exy*pauli_couple(n,'x','x',idx,idx+1) + 
#                        Exy*pauli_couple(n,'y','y',idx,idx+1) + 
#                        Ez *pauli_couple(n,'z','z',idx,idx+1))
#     return term

def hamiltonian_qu(particles,Exy,Ez):
    basis = spin_basis_1d(particles, pauli=False)
    L = particles
    kwargs = {'check_symm':False, 'check_herm':False, 'check_pcon':False}
    J_zz = [[Ez, i, (i+1)%L] for i in range(L)] # PBC
    J_xy = [[Exy, i, (i+1)%L] for i in range(L)] # PBC
    h_z =[[Exy/100,i] for i in range(L)]
    static = [["xx", J_xy], ["yy", J_xy], ["zz", J_zz], ["z",h_z]]
    
    H_XXZ = qs_hamiltonian(static, [], basis = basis,dtype=np.float64, **kwargs)
    # astring = ['LM', 'SM', 'LA', 'SA', 'BE']
    # w, v = H_XXZ.eigsh(k=1,which=astring[3])
    
    return H_XXZ #w, v#[:,0]