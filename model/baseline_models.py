"""
This file contains components are inspired by the implementation of

Hozumi, Y., and Wei, G.-W.
Analyzing single cell RNA sequencing with topological nonnegative matrix factorization,
Journal of Computational and Applied Mathematics, 2024.

Original repository:
https://github.com/hozumiyu/TopologicalNMF-scRNAseq
"""


import numpy as np


from model.utils import Euclidean_distance


# ============================================================
# Basic Nonnegative Matrix Factorization (NMF)
# ============================================================

class NMF:
    """
    Standard NMF solved by multiplicative update rules.

    Objective:
        min ||X - WH||_F^2
        s.t. W >= 0, H >= 0
    """

    def __init__(self, n_components, max_iter=1000, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def compute_loss(self, W, H):
        """Frobenius reconstruction error."""
        return np.linalg.norm(self.X - W @ H, ord='fro') ** 2

    def updateW(self, W, H):
        """Multiplicative update of W."""
        return W * ((self.X @ H.T) / (W @ H @ H.T + 1e-9))

    def updateH(self, W, H):
        """Multiplicative update of H."""
        numerator = W.T @ self.X
        denominator = W.T @ W @ H + 1e-9
        return H * numerator / denominator

    def multiplicative_update(self, W, H):
        """One MU step with column normalization."""
        W = self.updateW(W, H)
        H = self.updateH(W, H)

        scale = np.linalg.norm(W, axis=0)
        scale[scale == 0] = 1
        W = W / scale[None, :]
        H = H * scale[:, None]
        return W, H

    def fit_transform(self, X, W, H):
        self.X = X

        iteration = 1
        loss_prev = 1e9
        curr_loss = self.compute_loss(W, H)

        while iteration < self.max_iter and abs(loss_prev - curr_loss) / loss_prev > self.tol:
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss = self.compute_loss(W, H)
            iteration += 1

        return W, H


# ============================================================
# Graph Regularized NMF (GNMF)
# ============================================================

class GNMF:
    """
    Graph-regularized NMF.

    Objective:
        min ||X - WH||_F^2 + λ Tr(H L H^T)
    """

    def __init__(self, n_components, l, max_iter=1000, tol=1e-3):
        self.n_components = n_components
        self.l = l
        self.max_iter = max_iter
        self.tol = tol

    def compute_loss(self, W, H):
        recon = np.linalg.norm(self.X - W @ H, ord='fro') ** 2
        geom = self.l * np.trace(H @ self.L @ H.T)
        return recon + geom

    def updateW(self, W, H):
        """Update W (same as standard NMF)."""
        return W * ((self.X @ H.T) / (W @ H @ H.T + 1e-9))

    def updateH(self, W, H):
        """
        Update H with graph regularization.

        Uses decomposition L = D - A.
        """
        numerator = W.T @ self.X + self.l * H @ self.A
        denominator = W.T @ W @ H + self.l * H @ self.D + 1e-9
        return H * numerator / denominator

    def multiplicative_update(self, W, H):
        W = self.updateW(W, H)
        H = self.updateH(W, H)

        scale = np.linalg.norm(W, axis=0)
        scale[scale == 0] = 1
        W = W / scale[None, :]
        H = H * scale[:, None]
        return W, H

    def fit_transform(self, X, W, H, A, D, L):
        self.X = X
        self.A = A
        self.D = D
        self.L = L

        iteration = 1
        loss_prev = 1e9
        curr_loss = self.compute_loss(W, H)

        while iteration < self.max_iter and abs(loss_prev - curr_loss) / loss_prev > self.tol:
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss = self.compute_loss(W, H)
            iteration += 1

        return W, H


# ============================================================
# Graph Construction Utilities
# ============================================================

def heatKernel(X, n_neighbors):
    """
    Construct a heat-kernel weighted adjacency matrix.

    Parameters
    ----------
    X : array (features × samples)
    n_neighbors : int

    Returns
    -------
    A : adjacency matrix
    """
    distance = Euclidean_distance(X.T)
    neighbor_indices = np.argsort(distance, axis=1)[:, 1:n_neighbors + 1]

    mask = np.zeros_like(distance, dtype=bool)
    rows = np.arange(distance.shape[0])[:, None]
    mask[rows, neighbor_indices] = True

    scale = np.max(distance)
    A = np.zeros_like(distance)
    A[mask] = np.exp(-distance[mask] ** 2 / scale ** 1.5)

    return A


def construct_graph(X, n_neighbors):
    """
    Build a symmetric kNN heat graph and its Laplacian.
    """
    distance = Euclidean_distance(X.T)
    neighbor_indices = np.argsort(distance, axis=1)[:, 1:n_neighbors + 1]

    mask = np.zeros_like(distance, dtype=bool)
    rows = np.arange(distance.shape[0])[:, None]
    mask[rows, neighbor_indices] = True

    scale = np.max(distance)
    A = np.zeros_like(distance)
    A[mask] = np.exp(-distance[mask] ** 2 / scale ** 1.5)

    A = np.maximum(A, A.T)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    return A, D, L


def construct_persistent_graph(X, n_neighbors, weights):
    """
    Persistent graph constructed via multi-threshold aggregation.
    """
    HEAT = heatKernel(X, n_neighbors)

    l_min = np.min(HEAT[HEAT > 0])
    l_max = np.max(HEAT)
    d = l_max - l_min

    weights = weights / np.sum(weights)
    T = len(weights)

    A = np.zeros_like(HEAT)
    for idx in range(1, T + 1):
        A[HEAT >= (idx / T) * d + l_min] += weights[T - idx]

    A = A + A.T - A * A.T
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    return A, D, L


def construct_knn_persistent_graph(X, weights):
    """
    Persistent graph by progressively enlarging k in kNN.
    """
    n_neighbors = len(weights)
    weights = weights / np.sum(weights)

    distance = Euclidean_distance(X.T)
    neighbor_indices = np.argsort(distance, axis=1)[:, 1:n_neighbors + 1]

    A = np.zeros_like(distance)
    for k in range(n_neighbors):
        rows = np.arange(distance.shape[0])
        cols = neighbor_indices[:, :k + 1].reshape(-1)
        A[rows.repeat(k + 1), cols] += weights[k]

    A = A + A.T - A * A.T
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    return A, D, L
