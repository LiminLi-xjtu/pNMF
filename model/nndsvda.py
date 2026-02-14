"""
This file contains components are inspired by the implementation of

Hozumi, Y., and Wei, G.-W.
Analyzing single cell RNA sequencing with topological nonnegative matrix factorization,
Journal of Computational and Applied Mathematics, 2024.

Original repository:
https://github.com/hozumiyu/TopologicalNMF-scRNAseq
"""


import numpy as np
from sklearn.utils.extmath import squared_norm
from math import sqrt
from typing import Tuple


def NNDSVDA(X: np.ndarray, dataname: str, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    NNDSVDA Initialization for Nonnegative Matrix Factorization (NMF).

    Nonnegative Double Singular Value Decomposition with Average Filling (NNDSVDA)
    initializes the factor matrices W and H for NMF.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Input data matrix where M is the number of features and N is the number of samples.
    dataname : str
        Dataset name, used to determine the number of components.
        Example: '3D_concentric_circles' -> n_components = 2.
    eps : float, default=1e-8
        Small value to replace near-zero entries in W and H to avoid numerical issues.

    Returns
    -------
    W : np.ndarray, shape (M, n_components)
        Initialized nonnegative basis matrix.
    H : np.ndarray, shape (n_components, N)
        Initialized nonnegative coefficient matrix.
    """

    # -----------------------------
    # 1. Determine the number of components (rank of factorization)
    # -----------------------------
    if dataname == '3D_concentric_circles':
        n_components = 2
    else:
        n_components = int(np.ceil(np.sqrt(X.shape[1])))
        # Ensure n_components does not exceed the number of features
        if n_components > X.shape[0]:
            n_components = int(np.ceil(np.sqrt(X.shape[0])))

    # -----------------------------
    # 2. Perform SVD on X
    # -----------------------------
    u, s, v = np.linalg.svd(X, full_matrices=False)
    U = u[:, :n_components]  # Left singular vectors
    V = v[:n_components, :]  # Right singular vectors
    S = s[:n_components]     # Singular values

    # -----------------------------
    # 3. Initialize W and H matrices
    # -----------------------------
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # Initialize first component using absolute values
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    # -----------------------------
    # 4. Initialize remaining components
    # -----------------------------
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # Split vectors into positive and negative parts
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # Compute norms of positive and negative parts
        x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
        x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))

        # Magnitudes for positive and negative parts
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # Choose the part with larger magnitude
        if m_p > m_n:
            u_vec, v_vec = x_p / x_p_nrm, y_p / y_p_nrm
            sigma = m_p
        else:
            u_vec, v_vec = x_n / x_n_nrm, y_n / y_n_nrm
            sigma = m_n

        # Update W and H for this component
        lbd = sqrt(S[j] * sigma)
        W[:, j] = lbd * u_vec
        H[j, :] = lbd * v_vec

    # -----------------------------
    # 5. Replace small entries with average of X
    # -----------------------------
    avg_val = np.mean(X)
    W[W < eps] = avg_val
    H[H < eps] = avg_val

    return W, H