"""
This file contains components are inspired by the implementation of

Hozumi, Y., and Wei, G.-W.
Analyzing single cell RNA sequencing with topological nonnegative matrix factorization,
Journal of Computational and Applied Mathematics, 2024.

Original repository:
https://github.com/hozumiyu/TopologicalNMF-scRNAseq
"""


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import warnings


def compute_kmeans(X, y, max_state=10):
    """
    Perform KMeans clustering multiple times with different random states.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        True labels.
    max_state : int
        Number of different random states to run KMeans.

    Returns
    -------
    LABELS : ndarray of shape (max_state, n_samples)
        Cluster assignments for each random state.
    """
    n_clusters = len(np.unique(y))  # Determine number of clusters
    LABELS = np.zeros((max_state, X.shape[0]), dtype=int)  # Initialize cluster labels array

    for state in range(max_state):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=state).fit(X)
        LABELS[state, :] = kmeans.labels_

    return LABELS


def compute_ari(LABELS, y):
    """
    Compute the mean Adjusted Rand Index (ARI) across multiple random states.

    Parameters
    ----------
    LABELS : ndarray of shape (max_state, n_samples)
        Cluster labels.
    y : ndarray of shape (n_samples,)
        True labels.

    Returns
    -------
    float
        Mean ARI score.
    """
    ARI = np.array([adjusted_rand_score(y, LABELS[state, :]) for state in range(LABELS.shape[0])])
    return np.mean(ARI)


def compute_nmi(LABELS, y):
    """
    Compute the mean Normalized Mutual Information (NMI) across multiple random states.

    Parameters
    ----------
    LABELS : ndarray of shape (max_state, n_samples)
        Cluster labels.
    y : ndarray of shape (n_samples,)
        True labels.

    Returns
    -------
    float
        Mean NMI score.
    """
    NMI = np.array([normalized_mutual_info_score(y, LABELS[state, :]) for state in range(LABELS.shape[0])])
    return np.mean(NMI)


def compute_purity(LABELS, y):
    """
    Compute the mean Purity score across multiple random states.

    Parameters
    ----------
    LABELS : ndarray of shape (max_state, n_samples)
        Cluster labels.
    y : ndarray of shape (n_samples,)
        True labels.

    Returns
    -------
    float
        Mean Purity score.
    """
    PURITY = np.array([
        np.sum(np.max(contingency_matrix(y, LABELS[state, :]), axis=0)) / len(y)
        for state in range(LABELS.shape[0])
    ])
    return np.mean(PURITY)


def compute_acc(LABELS, y):
    """
    Compute the mean Clustering Accuracy (ACC) across multiple random states.
    Uses the Hungarian algorithm to match cluster labels to true labels.

    Parameters
    ----------
    LABELS : ndarray of shape (max_state, n_samples)
        Cluster labels.
    y : ndarray of shape (n_samples,)
        True labels.

    Returns
    -------
    float
        Mean ACC score.
    """
    ACC = []
    for state in range(LABELS.shape[0]):
        cm = contingency_matrix(y, LABELS[state, :])  # Compute contingency matrix
        row_ind, col_ind = linear_sum_assignment(cm, maximize=True)  # Hungarian matching
        acc_score = cm[row_ind, col_ind].sum() / len(y)
        ACC.append(acc_score)
    return np.mean(ACC)


def compute_clustering_score(X, y, max_state=10):
    """
    Compute clustering evaluation metrics (ARI, NMI, Purity, ACC) by performing
    KMeans clustering multiple times.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        True labels.
    max_state : int
        Number of different random states for KMeans.

    Returns
    -------
    tuple
        Mean (ARI, NMI, Purity, ACC) scores across all random states.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress KMeans warnings
        LABELS = compute_kmeans(X, y, max_state)

    ari = compute_ari(LABELS, y)
    nmi = compute_nmi(LABELS, y)
    purity = compute_purity(LABELS, y)
    acc = compute_acc(LABELS, y)

    return ari, nmi, purity, acc