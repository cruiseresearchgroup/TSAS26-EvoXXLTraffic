"""ASTGCN graph polynomials; matrix recurrence, including edgeless graphs.

The author's utils.py uses elementwise multiplication for order >=2.
We explicitly implement the Chebyshev matrix polynomial in the paper instead.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def scaled_Laplacian(a):
    a = np.asarray(a, dtype=np.float64)
    if not np.allclose(a, a.T):
        raise ValueError('ASTGCN requires the symmetric geographic graph')
    lap = np.diag(a.sum(1))-a
    if len(a) <= 2:
        largest = np.linalg.eigvalsh(lap).max()
    else:
        largest = eigsh(csr_matrix(lap), k=1, which='LA', return_eigenvectors=False,
                        v0=np.ones(len(a)), tol=1e-6)[0]
    return 2*lap/max(float(largest), 1e-12)-np.eye(len(a))


def cheb_polynomial(lap, order):
    terms = [np.eye(len(lap))]
    if order > 1:
        terms.append(lap.copy())
    for _ in range(2, order):
        terms.append(2*lap @ terms[-1]-terms[-2])
    return terms
