from typing import List
import numpy as np


def Euclidean_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance matrix for a set of samples.

    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Input data matrix where N is the number of samples and M is the number of features.

    Returns
    -------
    dist_mat : np.ndarray of shape (N, N)
        Symmetric distance matrix where dist_mat[i, j] is the Euclidean distance
        between samples i and j.
    """
    # Compute squared norms of each row (column vector)
    squared_norms = np.sum(X ** 2, axis=1, keepdims=True)  # Shape (N, 1)

    # Compute pairwise squared Euclidean distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    dist_mat = squared_norms + squared_norms.T - 2 * (X @ X.T)

    # Fix potential negative values due to floating-point errors
    dist_mat = np.maximum(dist_mat, 0.0)

    # Take the square root to get actual distances
    dist_mat = np.sqrt(dist_mat)

    # Ensure symmetry
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat


def compute_scale_regularization(H: List[np.ndarray], alpha: float, beta: float) -> float:
    """
    Compute the scale regularization for a list of coefficient matrices.

    The regularization consists of:
        alpha * sum_{t=2}^T ||H_t - H_{t-1}||_F^2
      + beta * sum_{t=1}^T ||H_t||_F^2

    Parameters
    ----------
    H : list of np.ndarray
        List of coefficient matrices H_t, each of shape (d, n)
    alpha : float
        Weight for the scale smoothness term
    beta : float
        Weight for the scale anchoring term

    Returns
    -------
    float
        Total regularization value
    """
    if len(H) == 0:
        return 0.0

    # Stack H into a 3D array of shape (T, d, n)
    H_arr = np.stack(H, axis=0)

    # Scale anchoring: sum of Frobenius norms squared
    anchoring = np.sum(H_arr ** 2)

    # Scale smoothness: sum of squared differences between consecutive scales
    smoothness = np.sum((H_arr[1:] - H_arr[:-1]) ** 2) if H_arr.shape[0] > 1 else 0.0

    return alpha * smoothness + beta * anchoring