"""K-means clustering and evaluation for a fixed pNMF embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


METRIC_NAMES = ("ARI", "NMI", "Purity", "Accuracy")


@dataclass(frozen=True)
class ClusteringResult:
    """Repeated K-means assignments and one metric vector per repetition."""

    cluster_labels: np.ndarray
    seeds: np.ndarray
    metrics: Dict[str, np.ndarray]
    n_clusters: int
    n_init: int

    def metric_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "mean": float(np.mean(self.metrics[name])),
                "std": float(np.std(self.metrics[name], ddof=0)),
            }
            for name in METRIC_NAMES
        }


def purity_score(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    table = contingency_matrix(true_labels, cluster_labels)
    return float(np.sum(np.max(table, axis=0)) / true_labels.size)


def clustering_accuracy(
    true_labels: np.ndarray, cluster_labels: np.ndarray
) -> float:
    table = contingency_matrix(true_labels, cluster_labels)
    row_indices, column_indices = linear_sum_assignment(table, maximize=True)
    return float(table[row_indices, column_indices].sum() / true_labels.size)


def evaluate_assignment(
    true_labels: np.ndarray, cluster_labels: np.ndarray
) -> Dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(true_labels, cluster_labels)),
        "NMI": float(
            normalized_mutual_info_score(true_labels, cluster_labels)
        ),
        "Purity": purity_score(true_labels, cluster_labels),
        "Accuracy": clustering_accuracy(true_labels, cluster_labels),
    }


def normalize_factor_pair_for_clustering(
    W: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2-normalize W columns and reciprocally rescale H rows.

    This is a post-optimization transformation.  It preserves ``W @ H`` but
    intentionally does not modify the raw factors returned by the solver.
    """

    W = np.asarray(W)
    H = np.asarray(H)
    if W.ndim != 2 or H.ndim != 2:
        raise ValueError("W and H must both be two-dimensional.")
    if W.shape[1] != H.shape[0]:
        raise ValueError("W columns and H rows must have the same component count.")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(H)):
        raise ValueError("W and H must contain only finite values.")

    column_norms = np.linalg.norm(W, axis=0)
    dtype = np.result_type(W.dtype, H.dtype, np.float32)
    epsilon = np.finfo(dtype).eps
    safe_norms = np.where(column_norms > 0.0, column_norms, 1.0)
    W_normalized = W / safe_norms[None, :]
    H_rescaled = H * column_norms[:, None]
    return W_normalized, H_rescaled, column_norms


def cluster_embedding(
    embedding: np.ndarray,
    true_labels: np.ndarray,
    *,
    n_clusters: int | None = None,
    repeats: int = 1,
    random_seed: int = 1,
    n_init: int = 10,
) -> ClusteringResult:
    """Cluster a samples-by-components embedding with repeated K-means."""

    embedding = np.asarray(embedding)
    true_labels = np.asarray(true_labels).reshape(-1)
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape (n_samples, n_components).")
    if embedding.shape[0] != true_labels.size:
        raise ValueError("The embedding and true labels have different sample counts.")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("The embedding contains NaN or infinite values.")
    if repeats <= 0 or n_init <= 0:
        raise ValueError("repeats and n_init must be positive.")

    inferred_clusters = int(np.unique(true_labels).size)
    n_clusters = inferred_clusters if n_clusters is None else int(n_clusters)
    if n_clusters <= 1 or n_clusters > embedding.shape[0]:
        raise ValueError("n_clusters must lie in [2, n_samples].")
    if inferred_clusters != n_clusters:
        raise ValueError(
            f"Configured n_clusters={n_clusters}, but the labels contain "
            f"{inferred_clusters} classes."
        )

    seeds = np.arange(random_seed, random_seed + repeats, dtype=np.int64)
    assignments = np.empty((repeats, embedding.shape[0]), dtype=np.int64)
    metric_values = {
        name: np.empty(repeats, dtype=np.float64) for name in METRIC_NAMES
    }

    for index, seed in enumerate(seeds):
        assignments[index] = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=int(seed),
        ).fit_predict(embedding)
        scores = evaluate_assignment(true_labels, assignments[index])
        for name in METRIC_NAMES:
            metric_values[name][index] = scores[name]

    return ClusteringResult(
        cluster_labels=assignments,
        seeds=seeds,
        metrics=metric_values,
        n_clusters=n_clusters,
        n_init=n_init,
    )


__all__ = [
    "ClusteringResult",
    "METRIC_NAMES",
    "cluster_embedding",
    "clustering_accuracy",
    "evaluate_assignment",
    "normalize_factor_pair_for_clustering",
    "purity_score",
]
