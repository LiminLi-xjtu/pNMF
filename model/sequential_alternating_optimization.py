from typing import List, Tuple
import numpy as np


from model.pnmf import pNMF


def sequential_alternating_optimization(
    X: np.ndarray,
    W_matrices: List[np.ndarray],
    H_matrices: List[np.ndarray],
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    A: List[np.ndarray],
    D: List[np.ndarray],
    L: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Perform sequential alternating optimization across multiple scales
    for the pNMF model.

    At each scale, this function updates the basis matrix W and coefficient matrix H
    using the pNMF update rules, considering the geometric regularization
    and scale smoothness/anchoring terms.

    Parameters
    ----------
    X : np.ndarray of shape (M, N)
        Input data matrix.
    W_matrices : list of np.ndarray
        List of basis matrices W for each scale.
    H_matrices : list of np.ndarray
        List of coefficient matrices H for each scale.
    lambda_1 : float
        Geometric regularization hyperparameter.
    lambda_2 : float
        Scale smoothness regularization hyperparameter.
    lambda_3 : float
        Scale anchoring regularization hyperparameter.
    A : list of np.ndarray
        Adjacency matrices for each scale.
    D : list of np.ndarray
        Degree matrices for each scale.
    L : list of np.ndarray
        Laplacian matrices for each scale.

    Returns
    -------
    W_matrices : list of np.ndarray
        Updated basis matrices W.
    H_matrices : list of np.ndarray
        Updated coefficient matrices H.
    RG_loss : list of float
        List of reconstruction + geometric regularization loss for each scale.
    """
    RG_loss: List[float] = []
    T = len(H_matrices)  # Total number of scales

    for t in range(T):
        # Determine the correct pNMF hyperparameters depending on the scale
        if t == 0:
            # First scale: only H_after matters (next scale's H)
            pnmf_model = pNMF(l1=lambda_1, l2=lambda_2, alpha=0, beta=lambda_2, l3=lambda_3)
            W_matrices[t], H_matrices[t], rg_loss = pnmf_model.fit_transform(
                X,
                W_matrices[t],
                H_matrices[t],
                H_before=H_matrices[t],      # No previous scale
                H_after=H_matrices[t + 1],   # Next scale
                A=A[t],
                D=D[t],
                L=L[t]
            )
        elif t == T - 1:
            # Last scale: only H_before matters (previous scale's H)
            pnmf_model = pNMF(l1=lambda_1, l2=lambda_2, alpha=lambda_2, beta=0, l3=lambda_3)
            W_matrices[t], H_matrices[t], rg_loss = pnmf_model.fit_transform(
                X,
                W_matrices[t],
                H_matrices[t],
                H_before=H_matrices[t - 1],  # Previous scale
                H_after=H_matrices[t],        # No next scale
                A=A[t],
                D=D[t],
                L=L[t]
            )
        else:
            # Middle scales: consider both H_before and H_after
            pnmf_model = pNMF(l1=lambda_1, l2=lambda_2, alpha=lambda_2, beta=lambda_2, l3=lambda_3)
            W_matrices[t], H_matrices[t], rg_loss = pnmf_model.fit_transform(
                X,
                W_matrices[t],
                H_matrices[t],
                H_before=H_matrices[t - 1],
                H_after=H_matrices[t + 1],
                A=A[t],
                D=D[t],
                L=L[t]
            )

        RG_loss.append(rg_loss)

    return W_matrices, H_matrices, RG_loss