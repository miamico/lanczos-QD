"""
Module contaning Lanczos diagonalization algorithm

"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

from typing import Tuple


def lanczos_eig(A: np.ndarray, v0: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
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
    
    error = 1e-16

    for i in range(1,m,1):
        v = Q[i-1,:] 

        Q[i,:] = r / b[i-1]

        r = A @ Q[i,:] # |n> 

        a[i] = (r.conj() @ Q[i,:] ) # real? 

        r = r - a[i]*Q[i,:] - b[i-1]*v

        b[i] = np.linalg.norm(r)
        
        # addtitional steps to increase accuracy
        d = Q[i,:].conj() @ r
        r -= d*Q[i,:]
        a += d
        
        if b[i] < error:
            m = i
            print('smaller space found',m)
            break

    E,V = eigh_tridiagonal(a[:m],b[:m-1])
    Q = Q[:m]
    W = (Q.T @ V)
#     tri = constr_tridiag(a[:m],b[:m-1])
    return E, W
