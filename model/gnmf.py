"""
This file contains components are inspired by the implementation of

Hozumi, Y., and Wei, G.-W.
Analyzing single cell RNA sequencing with topological nonnegative matrix factorization,
Journal of Computational and Applied Mathematics, 2024.

Original repository:
https://github.com/hozumiyu/TopologicalNMF-scRNAseq
"""


import numpy as np
from typing import Tuple


class GNMF:
    """
    Graph-regularized Non-negative Matrix Factorization (GNMF) model.
    
    Performs NMF with a geometric regularization term using a graph Laplacian.
    """

    def __init__(self, n_components: int, l: float, max_iter: int = 1000, tol: float = 1e-3):
        """
        Initialize the GNMF model.

        Parameters
        ----------
        n_components : int
            Number of latent components.
        l : float
            Regularization parameter for the graph Laplacian term.
        max_iter : int
            Maximum number of iterations (default 1000).
        tol : float
            Relative tolerance for convergence (default 1e-3).
        """
        self.n_components = n_components
        self.l = l
        self.max_iter = max_iter
        self.tol = tol

        # Internal attributes to store data and graph matrices
        self.X: np.ndarray = None
        self.A: np.ndarray = None
        self.D: np.ndarray = None
        self.L: np.ndarray = None

    def compute_loss(self, W: np.ndarray, H: np.ndarray) -> float:
        """
        Compute the GNMF loss: reconstruction error + geometric regularization.

        Parameters
        ----------
        W : np.ndarray, shape (M, n_components)
            Basis matrix.
        H : np.ndarray, shape (n_components, N)
            Coefficient matrix.

        Returns
        -------
        float
            Total loss value.
        """
        recon_loss = np.sum((self.X - W @ H) ** 2)  # Reconstruction error
        geom_loss = self.l * np.sum((H @ self.L) * H)  # Graph Laplacian regularization
        return recon_loss + geom_loss

    def updateW(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Update basis matrix W using multiplicative update rule.

        Parameters
        ----------
        W : np.ndarray, shape (M, n_components)
        H : np.ndarray, shape (n_components, N)

        Returns
        -------
        np.ndarray
            Updated W.
        """
        return W * ((self.X @ H.T) / (W @ H @ H.T + 1e-9))

    def updateH(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Update coefficient matrix H using multiplicative update rule.

        Parameters
        ----------
        W : np.ndarray, shape (M, n_components)
        H : np.ndarray, shape (n_components, N)

        Returns
        -------
        np.ndarray
            Updated H.
        """
        numerator = W.T @ self.X + self.l * H @ self.A
        denominator = W.T @ W @ H + self.l * H @ self.D + 1e-9
        return H * numerator / denominator

    def multiplicative_update(self, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one multiplicative update for both W and H, then normalize W.

        Parameters
        ----------
        W : np.ndarray, shape (M, n_components)
        H : np.ndarray, shape (n_components, N)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated W and H.
        """
        W = self.updateW(W, H)
        H = self.updateH(W, H)

        # Normalize columns of W to unit norm and scale H accordingly
        scale = np.linalg.norm(W, axis=0)
        W /= scale[None, :]
        H *= scale[:, None]

        return W, H

    def fit_transform(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        L: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit GNMF to the data using multiplicative updates.

        Parameters
        ----------
        X : np.ndarray, shape (M, N)
            Input data matrix.
        W : np.ndarray, shape (M, n_components)
            Initial basis matrix.
        H : np.ndarray, shape (n_components, N)
            Initial coefficient matrix.
        A : np.ndarray
            Adjacency matrix.
        D : np.ndarray
            Degree matrix.
        L : np.ndarray
            Laplacian matrix.

        Returns
        -------
        W : np.ndarray
            Optimized basis matrix.
        H : np.ndarray
            Optimized coefficient matrix.
        curr_loss : float
            Final loss value.
        """
        self.X = X
        self.A = A
        self.D = D
        self.L = L

        iteration = 0
        loss_prev = 1e9
        curr_loss = self.compute_loss(W, H)

        while iteration < self.max_iter and np.abs(loss_prev - curr_loss) / loss_prev > self.tol:
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss = self.compute_loss(W, H)
            iteration += 1

        if iteration == self.max_iter:
            print("Warning: GNMF reached the maximum number of iterations.")

        return W, H, curr_loss