import numpy as np
from typing import List, Tuple

from model.utils import Euclidean_distance
from model.persistent_homology import Persistent_Homology_Calculation


def induced_adjacency_matrix(distance: np.ndarray, threshold: float) -> np.ndarray:
    """
    Construct an adjacency matrix based on a distance matrix and a threshold.

    Parameters
    ----------
    distance : np.ndarray, shape (N, N)
        Pairwise symmetric distance matrix between N samples.
    threshold : float
        Threshold for connectivity. An edge (i, j) is created if distance[i, j] < threshold.

    Returns
    -------
    adjacency_matrix : np.ndarray, shape (N, N)
        Weighted adjacency matrix. Entries (i, j) = exp(-distance^2 / threshold^1.5)
        if distance[i, j] < threshold; 0 otherwise. Diagonal entries are always 0.
    """
    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros_like(distance)

    # Mask for entries below threshold (exclude self-loops)
    mask = (distance < threshold) & ~np.eye(distance.shape[0], dtype=bool)

    # Apply weighting
    adjacency_matrix[mask] = np.exp(-distance[mask] ** 2 / threshold ** 1.5)

    return adjacency_matrix


def construct_multi_scale_topology_graphs(X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Construct multi-scale topology graphs using 0-dimensional persistent homology.

    For each threshold derived from persistent homology, construct adjacency,
    degree, and Laplacian matrices.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Input data matrix (M features × N samples).

    Returns
    -------
    adjacency_matrices : List[np.ndarray]
        List of adjacency matrices, one per threshold.
    degree_matrices : List[np.ndarray]
        List of degree matrices, one per threshold.
    laplacian_matrices : List[np.ndarray]
        List of Laplacian matrices, one per threshold.
    thresholds : List[float]
        List of thresholds used to generate adjacency matrices, sorted descending.
    """

    # -----------------------------
    # 1. Initialize lists
    # -----------------------------
    adjacency_matrices, degree_matrices, laplacian_matrices = [], [], []

    # -----------------------------
    # 2. Compute pairwise Euclidean distance
    # -----------------------------
    distance = Euclidean_distance(X.T)  # Shape: (N, N)

    # -----------------------------
    # 3. Persistent homology calculation
    # -----------------------------
    ph_calculator = Persistent_Homology_Calculation()
    pairs_0 = ph_calculator(distance)  # Returns 0-dimensional pairs (birth, death)

    # -----------------------------
    # 4. Extract thresholds from distances at connected pairs
    # -----------------------------
    thresholds = distance[(pairs_0[:, 0], pairs_0[:, 1])]
    thresholds = np.append(thresholds, np.max(distance) + 1e-9)  # Add max distance as last threshold
    thresholds = sorted(thresholds, reverse=True)  # Descending order

    # -----------------------------
    # 5. Construct adjacency, degree, Laplacian matrices per threshold
    # -----------------------------
    for threshold in thresholds:
        # Adjacency
        A = induced_adjacency_matrix(distance, threshold)
        A = np.maximum(A, A.T)  # Ensure symmetry

        # Degree matrix
        D = np.diag(np.sum(A, axis=0))

        # Laplacian matrix
        L = D - A

        # Store matrices
        adjacency_matrices.append(A)
        degree_matrices.append(D)
        laplacian_matrices.append(L)

    return adjacency_matrices, degree_matrices, laplacian_matrices