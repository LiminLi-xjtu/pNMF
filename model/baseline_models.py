"""Baseline NMF models used by the real-data comparison experiments.

The multiplicative-update denominators include a stabilizer of 1e-9.
After each update the models normalize the basis columns and rescale the
embedding rows. GNMF, TNMF, and k-TNMF share an update and differ in the
construction of their graph.
"""

from __future__ import annotations

import numpy as np


_DENOMINATOR_EPSILON = 1.0e-9


def _euclidean_distance(samples: np.ndarray) -> np.ndarray:
    """Return the symmetric Euclidean distance matrix."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional array.")
    squared_norms = np.sum(samples**2, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (samples @ samples.T)
    distances = np.sqrt(np.maximum(distances, 0.0))
    return np.maximum(distances, distances.T)


def _validate_factor_inputs(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or W.ndim != 2 or H.ndim != 2:
        raise ValueError("X, W, and H must all be two-dimensional.")
    expected_w = (X.shape[0], int(n_components))
    expected_h = (int(n_components), X.shape[1])
    if W.shape != expected_w or H.shape != expected_h:
        raise ValueError(
            f"Expected W{expected_w} and H{expected_h}; got W{W.shape} and H{H.shape}."
        )
    if not all(np.all(np.isfinite(value)) for value in (X, W, H)):
        raise ValueError("X, W, and H must contain only finite values.")
    if any(float(np.min(value)) < 0.0 for value in (X, W, H)):
        raise ValueError("X, W, and H must be nonnegative.")
    return X, W, H


def _validate_graph(A: np.ndarray, D: np.ndarray, L: np.ndarray, n: int):
    values = tuple(np.asarray(value, dtype=np.float64) for value in (A, D, L))
    if any(value.shape != (n, n) for value in values):
        raise ValueError(f"A, D, and L must all have shape ({n}, {n}).")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("A, D, and L must contain only finite values.")
    return values


class NMF:
    """Standard NMF solved by the original multiplicative-update rules."""

    def __init__(self, n_components: int, max_iter: int = 1000, tol: float = 1e-3):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be positive.")
        if float(tol) < 0.0:
            raise ValueError("tol must be nonnegative.")
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_iter_ = 0
        self.objective_ = float("nan")
        self.converged_ = False

    def compute_loss(self, W: np.ndarray, H: np.ndarray) -> float:
        return float(np.linalg.norm(self.X - W @ H, ord="fro") ** 2)

    def updateW(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        return W * (
            (self.X @ H.T) / (W @ H @ H.T + _DENOMINATOR_EPSILON)
        )

    def updateH(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        numerator = W.T @ self.X
        denominator = W.T @ W @ H + _DENOMINATOR_EPSILON
        return H * numerator / denominator

    def multiplicative_update(
        self, W: np.ndarray, H: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        W = self.updateW(W, H)
        H = self.updateH(W, H)
        scale = np.linalg.norm(W, axis=0)
        scale[scale == 0.0] = 1.0
        W = W / scale[None, :]
        H = H * scale[:, None]
        return W, H

    def fit_transform(
        self, X: np.ndarray, W: np.ndarray, H: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.X, W, H = _validate_factor_inputs(X, W, H, self.n_components)

        # Keep the iteration and tolerance semantics of the original script.
        iteration = 1
        loss_prev = 1.0e9
        curr_loss = self.compute_loss(W, H)
        while (
            iteration < self.max_iter
            and abs(loss_prev - curr_loss) / loss_prev > self.tol
        ):
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss = self.compute_loss(W, H)
            iteration += 1

        self.n_iter_ = iteration - 1
        self.objective_ = float(curr_loss)
        self.converged_ = bool(
            abs(loss_prev - curr_loss) / loss_prev <= self.tol
        )
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(H)):
            raise FloatingPointError("NMF produced a NaN or infinite factor.")
        return W, H


class GNMF:
    """Graph-regularized NMF with the original update rules."""

    def __init__(
        self,
        n_components: int,
        l: float,
        max_iter: int = 1000,
        tol: float = 1e-3,
    ):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if float(l) < 0.0:
            raise ValueError("l must be nonnegative.")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be positive.")
        if float(tol) < 0.0:
            raise ValueError("tol must be nonnegative.")
        self.n_components = int(n_components)
        self.l = float(l)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_iter_ = 0
        self.objective_ = float("nan")
        self.converged_ = False

    def compute_loss(self, W: np.ndarray, H: np.ndarray) -> float:
        reconstruction = np.linalg.norm(self.X - W @ H, ord="fro") ** 2
        geometry = self.l * np.trace(H @ self.L @ H.T)
        return float(reconstruction + geometry)

    def updateW(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        return W * (
            (self.X @ H.T) / (W @ H @ H.T + _DENOMINATOR_EPSILON)
        )

    def updateH(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        numerator = W.T @ self.X + self.l * H @ self.A
        denominator = W.T @ W @ H + self.l * H @ self.D + _DENOMINATOR_EPSILON
        return H * numerator / denominator

    def multiplicative_update(
        self, W: np.ndarray, H: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        W = self.updateW(W, H)
        H = self.updateH(W, H)
        scale = np.linalg.norm(W, axis=0)
        scale[scale == 0.0] = 1.0
        W = W / scale[None, :]
        H = H * scale[:, None]
        return W, H

    def fit_transform(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        L: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.X, W, H = _validate_factor_inputs(X, W, H, self.n_components)
        self.A, self.D, self.L = _validate_graph(A, D, L, self.X.shape[1])

        iteration = 1
        loss_prev = 1.0e9
        curr_loss = self.compute_loss(W, H)
        while (
            iteration < self.max_iter
            and abs(loss_prev - curr_loss) / loss_prev > self.tol
        ):
            loss_prev = curr_loss
            W, H = self.multiplicative_update(W, H)
            curr_loss = self.compute_loss(W, H)
            iteration += 1

        self.n_iter_ = iteration - 1
        self.objective_ = float(curr_loss)
        self.converged_ = bool(
            abs(loss_prev - curr_loss) / loss_prev <= self.tol
        )
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(H)):
            raise FloatingPointError("GNMF produced a NaN or infinite factor.")
        return W, H


def _validate_neighbors(n_neighbors: int, n_samples: int) -> int:
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0 or n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors must lie in [1, {n_samples - 1}]; got {n_neighbors}."
        )
    return n_neighbors


def _validate_weights(weights) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("weights cannot be empty.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("at least one filtration weight must be positive.")
    return values / total


def heatKernel(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Construct the directed heat-kernel graph used by TNMF."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (features, samples).")
    n_neighbors = _validate_neighbors(n_neighbors, X.shape[1])
    distance = _euclidean_distance(X.T)
    scale = float(np.max(distance))
    if scale <= 0.0:
        raise ValueError("A heat-kernel graph is undefined when all samples coincide.")
    neighbor_indices = np.argsort(distance, axis=1)[:, 1 : n_neighbors + 1]
    mask = np.zeros_like(distance, dtype=bool)
    rows = np.arange(distance.shape[0])[:, None]
    mask[rows, neighbor_indices] = True
    A = np.zeros_like(distance)
    A[mask] = np.exp(-(distance[mask] ** 2) / scale**1.5)
    return A


def construct_graph(
    X: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the symmetric kNN heat graph and graph Laplacian for GNMF."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (features, samples).")
    n_neighbors = _validate_neighbors(n_neighbors, X.shape[1])
    distance = _euclidean_distance(X.T)
    scale = float(np.max(distance))
    if scale <= 0.0:
        raise ValueError("A heat-kernel graph is undefined when all samples coincide.")
    neighbor_indices = np.argsort(distance, axis=1)[:, 1 : n_neighbors + 1]
    mask = np.zeros_like(distance, dtype=bool)
    rows = np.arange(distance.shape[0])[:, None]
    mask[rows, neighbor_indices] = True
    A = np.zeros_like(distance)
    A[mask] = np.exp(-(distance[mask] ** 2) / scale**1.5)
    A = np.maximum(A, A.T)
    D = np.diag(np.sum(A, axis=0))
    return A, D, D - A


def construct_persistent_graph(
    X: np.ndarray, n_neighbors: int, weights
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the original heat-threshold persistent graph for TNMF."""

    HEAT = heatKernel(X, n_neighbors)
    positive = HEAT[HEAT > 0.0]
    if positive.size == 0:
        raise ValueError("The heat graph has no positive edge weights.")
    l_min = float(np.min(positive))
    l_max = float(np.max(HEAT))
    interval = l_max - l_min
    weights = _validate_weights(weights)
    threshold_count = weights.size
    A = np.zeros_like(HEAT)
    for index in range(1, threshold_count + 1):
        A[HEAT >= (index / threshold_count) * interval + l_min] += weights[
            threshold_count - index
        ]
    A = A + A.T - A * A.T
    D = np.diag(np.sum(A, axis=0))
    return A, D, D - A


def construct_knn_persistent_graph(
    X: np.ndarray, weights
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the original nested-kNN persistent graph for k-TNMF."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (features, samples).")
    weights = _validate_weights(weights)
    n_neighbors = _validate_neighbors(weights.size, X.shape[1])
    distance = _euclidean_distance(X.T)
    neighbor_indices = np.argsort(distance, axis=1)[:, 1 : n_neighbors + 1]
    A = np.zeros_like(distance)
    for k in range(n_neighbors):
        rows = np.arange(distance.shape[0])
        columns = neighbor_indices[:, : k + 1].reshape(-1)
        A[rows.repeat(k + 1), columns] += weights[k]
    A = A + A.T - A * A.T
    D = np.diag(np.sum(A, axis=0))
    return A, D, D - A


__all__ = [
    "GNMF",
    "NMF",
    "construct_graph",
    "construct_knn_persistent_graph",
    "construct_persistent_graph",
    "heatKernel",
]
