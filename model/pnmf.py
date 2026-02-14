from typing import Tuple
import numpy as np


class pNMF:
    def __init__(self, l1: float, l2: float, alpha: float, beta: float, l3: float,
                 max_iter: int = 1000, tol: float = 1e-3):
        """
        Initialize the pNMF model with regularization parameters.

        Parameters
        ----------
        l1 : float
            Geometric regularization hyperparameter.
        l2 : float
            Scale smoothness regularization hyperparameter.
        alpha : float
            Weight for previous scale smoothness.
        beta : float
            Weight for next scale smoothness.
        l3 : float
            Scale anchoring regularization hyperparameter.
        max_iter : int
            Maximum number of iterations for optimization (default 1000).
        tol : float
            Tolerance for convergence (default 1e-3).
        """
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.l3 = l3
        self.max_iter = max_iter
        self.tol = tol

    def compute_loss(self, W: np.ndarray, H: np.ndarray) -> Tuple[float, float]:
        """
        Compute the total loss and the reconstruction + geometric loss.

        Parameters
        ----------
        W : np.ndarray
            Basis matrix.
        H : np.ndarray
            Coefficient matrix.

        Returns
        -------
        loss : float
            Total loss including scale regularization and anchoring.
        rg_loss : float
            Loss without the scale smoothness/anchoring terms.
        """
        recon_loss = np.sum((self.X - W @ H) ** 2)
        geom_loss = self.l1 * np.sum((H @ self.L) * H)
        scale_smooth_loss = self.alpha * np.sum((H - self.H_before) ** 2) + \
                            self.beta * np.sum((H - self.H_after) ** 2)
        scale_anchor_loss = self.l3 * np.sum(H ** 2)

        loss = recon_loss + geom_loss + scale_smooth_loss + scale_anchor_loss
        rg_loss = recon_loss + geom_loss
        return loss, rg_loss

    def updateW(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Update the basis matrix W using multiplicative gradient-style updates.

        Parameters
        ----------
        W : np.ndarray
            Basis matrix.
        H : np.ndarray
            Coefficient matrix.

        Returns
        -------
        W : np.ndarray
            Updated basis matrix.
        """
        grad_W = -self.X @ H.T + W @ H @ H.T
        # Prevent negative updates that might produce negative W
        W_safe = np.where(grad_W >= 0, W, np.maximum(W, 1e-9))
        W = W - (W_safe / (W_safe @ H @ H.T + 1e-9)) * grad_W
        return W

    def updateH(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Update the coefficient matrix H using multiplicative gradient-style updates.

        Parameters
        ----------
        W : np.ndarray
            Basis matrix.
        H : np.ndarray
            Coefficient matrix.

        Returns
        -------
        H : np.ndarray
            Updated coefficient matrix.
        """
        grad_H = - W.T @ self.X + W.T @ W @ H + self.l1 * H @ self.L + \
                 self.alpha * (H - self.H_before) + self.beta * (H - self.H_after) + \
                 self.l3 * H
        # Prevent negative updates that might produce negative H
        H_safe = np.where(grad_H >= 0, H, np.maximum(H, 1e-9))
        denominator = W.T @ W @ H_safe + self.l1 * H_safe @ self.D + \
                      (self.alpha + self.beta + self.l3) * H_safe + 1e-9
        H = H - (H_safe / denominator) * grad_H
        return H

    def multiplicative_update(self, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform multiplicative updates for both W and H, including column normalization of W.

        Parameters
        ----------
        W : np.ndarray
            Basis matrix.
        H : np.ndarray
            Coefficient matrix.

        Returns
        -------
        W : np.ndarray
            Updated basis matrix.
        H : np.ndarray
            Updated coefficient matrix.
        """
        W = self.updateW(W, H)
        H = self.updateH(W, H)

        # Normalize columns of W to unit norm and scale H accordingly
        scale = np.linalg.norm(W, axis=0)
        W = W / scale[None, :]
        H = H * scale[:, None]
        return W, H

    def fit_transform(self, X: np.ndarray, W: np.ndarray, H: np.ndarray,
                      H_before: np.ndarray, H_after: np.ndarray,
                      A: np.ndarray, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit the pNMF model to the data at a single scale and return updated W and H.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        W : np.ndarray
            Initial basis matrix.
        H : np.ndarray
            Initial coefficient matrix.
        H_before : np.ndarray
            Coefficient matrix from previous scale.
        H_after : np.ndarray
            Coefficient matrix from next scale.
        A : np.ndarray
            Adjacency matrix for this scale.
        D : np.ndarray
            Degree matrix for this scale.
        L : np.ndarray
            Laplacian matrix for this scale.

        Returns
        -------
        W : np.ndarray
            Updated basis matrix.
        H : np.ndarray
            Updated coefficient matrix.
        rg_loss : float
            Reconstruction + geometric loss (without scale regularization).
        """
        self.X = X
        self.A = A
        self.D = D
        self.L = L
        self.H_before = H_before
        self.H_after = H_after

        iteration = 0
        loss_prev = 1e9
        curr_loss, rg_loss = self.compute_loss(W, H)

        while iteration < self.max_iter and np.abs(loss_prev - curr_loss) / loss_prev > self.tol:
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss, rg_loss = self.compute_loss(W, H)
            iteration += 1

        if iteration == self.max_iter:
            print("Maximum iteration reached")

        return W, H, rg_loss