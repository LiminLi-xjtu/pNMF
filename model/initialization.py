from typing import List, Tuple
import numpy as np

from model.nndsvda import NNDSVDA
from model.multi_scale_topology_graphs import construct_multi_scale_topology_graphs
from model.gnmf import GNMF


def initialize(
    X: np.ndarray,
    lambda_1: float,
    dataname: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """
    Initialize matrices W and H using NNDSVDA, then refine them with GNMF
    across multiple scales of topology graphs.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Input data matrix where M is the number of features and N is the number of samples.
    lambda_1 : float
        Regularization parameter for the GNMF model.
    dataname : str
        Dataset name, used to determine the number of components in NNDSVDA.

    Returns
    -------
    W_matrices : list of np.ndarray
        List of refined basis matrices W across different scales.
    H_matrices : list of np.ndarray
        List of refined coefficient matrices H across different scales.
    A : list of np.ndarray
        List of adjacency matrices for each scale.
    D : list of np.ndarray
        List of degree matrices for each scale.
    L : list of np.ndarray
        List of Laplacian matrices for each scale.
    RG_loss : list of float
        List of reconstruction + geometric regularization losses for each scale.
    """

    # -----------------------------
    # 1. Initialize W and H using NNDSVDA
    # -----------------------------
    W, H = NNDSVDA(X, dataname)

    # -----------------------------
    # 2. Construct multi-scale topology graphs
    # -----------------------------
    # Returns adjacency matrices (A), degree matrices (D), Laplacians (L), and thresholds
    A, D, L = construct_multi_scale_topology_graphs(X)

    # -----------------------------
    # 3. Initialize lists to store W, H, and losses for each scale
    # -----------------------------
    W_matrices: List[np.ndarray] = []
    H_matrices: List[np.ndarray] = []
    RG_loss: List[float] = []

    n_scales = len(L)

    # -----------------------------
    # 4. Iterate over each scale and refine W and H using GNMF
    # -----------------------------
    for t in range(n_scales):
        # Initialize GNMF model for current scale
        gnmf_model = GNMF(n_components=W.shape[1], l=lambda_1)

        # Fit GNMF and update W, H
        W, H, rg_loss = gnmf_model.fit_transform(X, W, H, A[t], D[t], L[t])

        # Store results for this scale
        W_matrices.append(W)
        H_matrices.append(H)
        RG_loss.append(rg_loss)

    # Reverse lists to have the finest scale first
    return W_matrices[::-1], H_matrices[::-1], A[::-1], D[::-1], L[::-1], RG_loss[::-1]