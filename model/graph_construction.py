"""Compact, accelerator-friendly storage for the multi-scale Laplacians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

from model.array_backend import array_module, scalar_value, to_numpy
from model.persistent_homology import Persistent_Homology_Calculation


def pairwise_euclidean_columns(X):
    """Pairwise Euclidean distances between the columns of ``X``."""
    xp = array_module(X)
    squared_norms = xp.sum(X * X, axis=0)
    distance_squared = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X.T @ X)
    )
    xp.maximum(distance_squared, 0.0, out=distance_squared)
    # Averaging removes the tiny asymmetry introduced by GEMM roundoff.
    distance_squared = 0.5 * (distance_squared + distance_squared.T)
    xp.fill_diagonal(distance_squared, 0.0)
    distance = xp.sqrt(distance_squared)
    return distance, distance_squared


def persistence_thresholds(distance) -> np.ndarray:
    """Return the distinct positive finite H0 death values in increasing order.

    These are the positive model scales for the non-strict rule d(i,j) <= eps.
    The theoretical sufficient set also contains zero; zero is not a model
    scale. No auxiliary complete-graph scale is appended.
    """
    distance_cpu = np.asarray(to_numpy(distance), dtype=np.float64)
    if distance_cpu.ndim != 2 or distance_cpu.shape[0] != distance_cpu.shape[1]:
        raise ValueError("distance must have shape (n, n).")
    if not np.all(np.isfinite(distance_cpu)) or np.any(distance_cpu < 0.0):
        raise ValueError("Pairwise distances must be finite and nonnegative.")
    pairs = Persistent_Homology_Calculation()(distance_cpu)
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    deaths = distance_cpu[pairs[:, 0], pairs[:, 1]]
    # Equal death values represent simultaneous mergers and contribute only
    # one scale. Graph construction includes every edge at that distance.
    return np.unique(deaths[deaths > 0.0])


@dataclass
class DenseLaplacianStack:
    """One contiguous tensor for all scale-specific graph Laplacians.

    Parameters
    ----------
    values
        Tensor with shape ``(T, n, n)``.
    degree
        Tensor with shape ``(T, n)`` containing the Laplacian diagonals.
    thresholds
        CPU array containing the persistence-selected scales.
    matmul_chunk_size
        Optional number of scales per batched right multiplication.  ``0``
        lets the backend process every scale in a single batched GEMM.
    """

    values: object
    degree: object
    thresholds: np.ndarray
    matmul_chunk_size: int = 0

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[1] != self.values.shape[2]:
            raise ValueError("Laplacians must have shape (T, n, n).")
        if self.degree.shape != (self.values.shape[0], self.values.shape[1]):
            raise ValueError("degree must have shape (T, n).")

    @property
    def shape(self):
        return self.values.shape

    @property
    def n_scales(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[1])

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def xp(self):
        return array_module(self.values)

    def right_multiply(self, matrices):
        """Compute ``matrices[t] @ L[t]`` without a global Hessian."""
        if matrices.shape[0] != self.n_scales or matrices.shape[-1] != self.n_samples:
            raise ValueError("matrices must have shape (T, ..., n).")
        xp = self.xp
        chunk = int(self.matmul_chunk_size)
        if chunk <= 0 or chunk >= self.n_scales:
            return xp.matmul(matrices, self.values)

        result = xp.empty_like(matrices)
        for start in range(0, self.n_scales, chunk):
            stop = min(start + chunk, self.n_scales)
            result[start:stop] = xp.matmul(
                matrices[start:stop], self.values[start:stop]
            )
        return result

    def scale_right_multiply(self, matrix, scale: int):
        """Compute ``matrix @ L[scale]`` on the active backend."""
        scale = int(scale)
        if scale < 0 or scale >= self.n_scales:
            raise IndexError("scale index is out of range.")
        if array_module(matrix) is not self.xp:
            raise ValueError("matrix and graph must use the same backend.")
        if matrix.shape[-1] != self.n_samples:
            raise ValueError("matrix must have n_samples columns.")
        return matrix @ self.values[scale]

    def quadratic(self, H):
        HL = self.right_multiply(H)
        return self.xp.sum(H * HL)

    @classmethod
    def from_sequence(
        cls,
        laplacians,
        *,
        dtype=None,
        thresholds: Optional[np.ndarray] = None,
        matmul_chunk_size: int = 0,
    ) -> "DenseLaplacianStack":
        if len(laplacians) == 0:
            raise ValueError("At least one Laplacian is required.")
        xp = array_module(laplacians[0])
        values = xp.stack(laplacians, axis=0)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        degree = xp.diagonal(values, axis1=1, axis2=2).copy()
        if thresholds is None:
            thresholds = np.arange(values.shape[0], dtype=np.float64)
        return cls(values, degree, np.asarray(thresholds), matmul_chunk_size)


@dataclass
class SparseLaplacianStack:
    """CSR storage for the complete persistence-scale Laplacian sequence.

    One CSR matrix per scale is retained for scale-wise initialization.
    Optional block-diagonal CSR chunks batch sparse--dense multiplications
    across scales on CPU or CUDA.
    """

    values: list[object]
    degree: object
    thresholds: np.ndarray
    matmul_chunk_size: int = 0
    block_matrices: Optional[list[tuple[int, int, object]]] = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("At least one sparse Laplacian is required.")
        n_samples = int(self.values[0].shape[0])
        if any(matrix.shape != (n_samples, n_samples) for matrix in self.values):
            raise ValueError("Every sparse Laplacian must have shape (n, n).")
        if self.degree.shape != (len(self.values), n_samples):
            raise ValueError("degree must have shape (T, n).")
        if len(self.thresholds) != len(self.values):
            raise ValueError("thresholds must contain one value per scale.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_scales, self.n_samples, self.n_samples)

    @property
    def n_scales(self) -> int:
        return len(self.values)

    @property
    def n_samples(self) -> int:
        return int(self.values[0].shape[0])

    @property
    def dtype(self):
        return self.degree.dtype

    @property
    def xp(self):
        return array_module(self.degree)

    @property
    def storage_label(self) -> str:
        if self.xp is np:
            if self.block_matrices:
                return "CPU per-scale CSR + block-diagonal CSR"
            return "CPU per-scale CSR"
        return "CUDA per-scale CSR + block-diagonal CSR"

    @staticmethod
    def _csr_bytes(matrix) -> int:
        return int(
            matrix.data.nbytes
            + matrix.indices.nbytes
            + matrix.indptr.nbytes
        )

    @property
    def csr_storage_bytes(self) -> int:
        """Per-scale CSR buffers only, the manuscript storage measure."""
        return sum(self._csr_bytes(matrix) for matrix in self.values)

    @property
    def storage_bytes(self) -> int:
        """Bytes retained by CSR arrays, degrees, and scale thresholds."""
        matrix_bytes = sum(self._csr_bytes(matrix) for matrix in self.values)
        block_bytes = sum(
            self._csr_bytes(matrix)
            for _, _, matrix in (self.block_matrices or [])
        )
        return int(
            matrix_bytes
            + block_bytes
            + self.degree.nbytes
            + self.thresholds.nbytes
        )

    def scale_right_multiply(self, matrix, scale: int):
        """Compute ``matrix @ L[scale]`` on the active backend."""
        scale = int(scale)
        if scale < 0 or scale >= self.n_scales:
            raise IndexError("scale index is out of range.")
        if array_module(matrix) is not self.xp:
            raise ValueError("matrix and sparse graph must use the same backend.")
        if matrix.shape[-1] != self.n_samples:
            raise ValueError("matrix must have n_samples columns.")
        original_shape = matrix.shape
        flattened = matrix.reshape(-1, self.n_samples)
        # Each L_t is symmetric.  This order is supported efficiently by both
        # SciPy CSR and CuPy CSR, unlike dense @ sparse on every backend.
        multiplied = self.values[scale].dot(flattened.T).T
        return multiplied.reshape(original_shape)

    def right_multiply(self, matrices):
        """Compute ``matrices[t] @ L[t]`` without dense graph storage."""
        if (
            matrices.shape[0] != self.n_scales
            or matrices.shape[-1] != self.n_samples
        ):
            raise ValueError("matrices must have shape (T, ..., n).")
        xp = self.xp
        if array_module(matrices) is not xp:
            raise ValueError("matrices and sparse graph must use the same backend.")

        result = xp.empty_like(matrices)
        if xp is np and not self.block_matrices:
            for t in range(self.n_scales):
                result[t] = self.scale_right_multiply(matrices[t], t)
            return result

        if not self.block_matrices:
            raise RuntimeError("CUDA sparse graph is missing block CSR storage.")
        middle_size = int(np.prod(matrices.shape[1:-1], dtype=np.int64))
        for start, stop, block_matrix in self.block_matrices:
            scale_count = stop - start
            if xp is np:
                packed = matrices[start:stop].reshape(scale_count, middle_size, self.n_samples)
                rhs = packed.transpose(0, 2, 1).reshape(scale_count * self.n_samples, middle_size)
                product = block_matrix.dot(rhs).reshape(scale_count, self.n_samples, middle_size)
                result[start:stop] = product.transpose(0, 2, 1).reshape(matrices[start:stop].shape)
                continue
            packed_shape = matrices.shape[1:-1] + (scale_count, self.n_samples)
            # CuPy CSRMM requires Fortran-contiguous input. Packing directly
            # into that layout avoids both a C-order reshape buffer and its
            # implicit full-size asfortranarray copy inside sparse.dot.
            rhs = xp.empty(
                (scale_count * self.n_samples, middle_size),
                dtype=matrices.dtype,
                order="F",
            )
            rhs.T.reshape(packed_shape)[...] = xp.moveaxis(
                matrices[start:stop], 0, -2
            )
            multiplied = block_matrix.dot(rhs)
            # CSRMM also returns Fortran order: transpose first so unpacking
            # can use views instead of a full-size C-order reshape copy.
            result[start:stop] = xp.moveaxis(
                multiplied.T.reshape(packed_shape), -2, 0
            )
            # Do not retain a previous chunk while allocating the next one.
            del rhs, multiplied
        return result

    def quadratic(self, H):
        return self.xp.sum(
            H * self.right_multiply(H),
            dtype=self.xp.float64,
        )


def build_dense_laplacian_stack(
    X,
    *,
    matmul_chunk_size: int = 0,
    alpha: float = 1.5,
) -> DenseLaplacianStack:
    """Construct every persistence-scale Laplacian in one preallocated tensor.

    The graph rule is

    ``A_t[i,j] = exp(-d(i,j)^2 / eps_t^1.5)`` when
    ``d(i,j) <= eps_t`` and ``i != j``.

    Only the Laplacian and its diagonal are retained.  This removes the three
    simultaneous ``A/D/L`` copies used by the original implementation.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if X.ndim != 2:
        raise ValueError("X must have shape (p, n).")
    xp = array_module(X)
    if not np.issubdtype(np.dtype(X.dtype), np.floating):
        X = xp.asarray(X, dtype=xp.float64)
    if scalar_value(X.min()) < 0:
        raise ValueError("pNMF requires a nonnegative input matrix.")

    distance, distance_squared = pairwise_euclidean_columns(X)
    thresholds = persistence_thresholds(distance)
    n_scales = int(thresholds.size)
    n_samples = int(X.shape[1])
    if n_scales == 0:
        raise ValueError(
            "No positive H0 death scales exist. The positive-scale pNMF model "
            "requires at least two samples with positive pairwise distance."
        )

    values = xp.empty((n_scales, n_samples, n_samples), dtype=X.dtype)
    degree = xp.empty((n_scales, n_samples), dtype=X.dtype)
    mask = xp.empty((n_samples, n_samples), dtype=bool)
    diagonal = xp.arange(n_samples)
    tiny = float(np.finfo(np.dtype(X.dtype)).tiny)

    for t, threshold in enumerate(thresholds):
        Lt = values[t]
        denominator = max(float(threshold) ** alpha, tiny)
        xp.multiply(distance_squared, -1.0 / denominator, out=Lt)
        xp.exp(Lt, out=Lt)
        xp.less_equal(distance, threshold, out=mask)
        xp.multiply(Lt, mask, out=Lt)
        Lt[diagonal, diagonal] = 0.0
        degree[t] = xp.sum(Lt, axis=1)
        Lt *= -1.0
        Lt[diagonal, diagonal] = degree[t]

    return DenseLaplacianStack(
        values=values,
        degree=degree,
        thresholds=thresholds,
        matmul_chunk_size=matmul_chunk_size,
    )


def _cpu_sparse_laplacians(
    distance: np.ndarray,
    distance_squared: np.ndarray,
    *,
    dtype,
    alpha: float = 1.5,
) -> tuple[list[sparse.csr_matrix], np.ndarray, np.ndarray]:
    """Build the positive persistence-scale graphs with d(i,j) <= eps as CSR."""
    thresholds = persistence_thresholds(distance)
    n_scales = int(thresholds.size)
    n_samples = int(distance.shape[0])
    if distance.shape != (n_samples, n_samples):
        raise ValueError("distance must have shape (n, n).")
    if distance_squared.shape != distance.shape:
        raise ValueError("distance and distance_squared must have equal shapes.")
    if n_scales == 0:
        raise ValueError(
            "No positive H0 death scales exist. The positive-scale pNMF model "
            "requires at least two samples with positive pairwise distance."
        )

    upper_row, upper_col = np.triu_indices(n_samples, k=1)
    upper_distance = distance[upper_row, upper_col]
    upper_distance_squared = distance_squared[upper_row, upper_col]
    order = np.argsort(upper_distance, kind="stable")
    sorted_distance = upper_distance[order]
    sorted_distance_squared = upper_distance_squared[order]
    sorted_row = upper_row[order].astype(np.int32, copy=False)
    sorted_col = upper_col[order].astype(np.int32, copy=False)
    degree = np.empty((n_scales, n_samples), dtype=dtype)
    laplacians: list[sparse.csr_matrix] = []
    tiny = float(np.finfo(np.dtype(dtype)).tiny)

    for t, threshold in enumerate(thresholds):
        # ``side='right'`` includes all edges with d(i,j) <= epsilon_t,
        # matching ``build_dense_laplacian_stack``.
        edge_count = int(
            np.searchsorted(sorted_distance, threshold, side="right")
        )
        if edge_count == 0:
            degree[t] = 0.0
            laplacians.append(
                sparse.csr_matrix(
                    (n_samples, n_samples),
                    dtype=dtype,
                )
            )
            continue

        rows = sorted_row[:edge_count]
        cols = sorted_col[:edge_count]
        selected_distance_squared = sorted_distance_squared[:edge_count]
        denominator = max(float(threshold) ** alpha, tiny)
        weights = np.exp(
            -selected_distance_squared / denominator
        ).astype(dtype, copy=False)
        degree_t = (
            np.bincount(rows, weights=weights, minlength=n_samples)
            + np.bincount(cols, weights=weights, minlength=n_samples)
        ).astype(dtype, copy=False)
        degree[t] = degree_t

        upper_laplacian = sparse.csr_matrix(
            (-weights, (rows, cols)),
            shape=(n_samples, n_samples),
            dtype=dtype,
        )
        laplacian = (upper_laplacian + upper_laplacian.T).tocsr()
        laplacian.setdiag(degree_t)
        laplacian.eliminate_zeros()
        laplacian.sort_indices()
        laplacians.append(laplacian)

    return laplacians, degree, np.asarray(thresholds, dtype=np.float64)


def build_sparse_laplacian_stack(
    X,
    *,
    matmul_chunk_size: int = 0,
    alpha: float = 1.5,
) -> SparseLaplacianStack:
    """Construct the exact persistence-scale Laplacians in sparse CSR form.

    Pairwise distances are evaluated on the selected NumPy/CuPy backend, as in
    the dense implementation.  CSR assembly and persistence pairing are then
    performed on CPU before the sparse arrays are transferred to CUDA when
    requested.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if X.ndim != 2:
        raise ValueError("X must have shape (p, n).")
    xp = array_module(X)
    if not np.issubdtype(np.dtype(X.dtype), np.floating):
        X = xp.asarray(X, dtype=xp.float64)
    if scalar_value(X.min()) < 0:
        raise ValueError("pNMF requires a nonnegative input matrix.")
    matmul_chunk_size = int(matmul_chunk_size)
    if matmul_chunk_size < 0:
        raise ValueError("matmul_chunk_size must be nonnegative.")

    distance, distance_squared = pairwise_euclidean_columns(X)
    distance_cpu = np.ascontiguousarray(to_numpy(distance))
    distance_squared_cpu = np.ascontiguousarray(to_numpy(distance_squared))
    laplacians_cpu, degree_cpu, thresholds = _cpu_sparse_laplacians(
        distance_cpu,
        distance_squared_cpu,
        dtype=np.dtype(X.dtype),
        alpha=alpha,
    )
    cpu_graphs = SparseLaplacianStack(
        values=laplacians_cpu,
        degree=degree_cpu,
        thresholds=thresholds,
        matmul_chunk_size=matmul_chunk_size,
    )
    if xp is np:
        return cpu_graphs
    return sparse_graphs_to_backend(cpu_graphs, xp)


def sparse_graphs_to_backend(graphs: SparseLaplacianStack, xp) -> SparseLaplacianStack:
    """Pack a CPU graph stack for batched products on NumPy or CuPy.

    The CSR buffers come from the supplied graphs, allowing a CUDA fit to
    use thresholds and edge weights computed once in CPU precision.
    """
    if graphs.xp is not np:
        raise ValueError("Expected a sparse graph stack built on CPU.")
    if xp is np:
        values = graphs.values
        convert = sparse.csr_matrix
    else:
        from cupyx.scipy import sparse as cupy_sparse

        def convert(matrix):
            return cupy_sparse.csr_matrix(
                (xp.asarray(matrix.data), xp.asarray(matrix.indices), xp.asarray(matrix.indptr)),
                shape=matrix.shape,
            )

        values = [convert(matrix) for matrix in graphs.values]

    n_scales = graphs.n_scales
    effective_chunk = graphs.matmul_chunk_size or n_scales
    block_matrices = []
    for start in range(0, n_scales, effective_chunk):
        stop = min(start + effective_chunk, n_scales)
        block_cpu = sparse.block_diag(
            graphs.values[start:stop],
            format="csr",
        )
        block_matrices.append((start, stop, convert(block_cpu)))

    return SparseLaplacianStack(
        values=values,
        degree=xp.asarray(graphs.degree),
        thresholds=graphs.thresholds,
        matmul_chunk_size=graphs.matmul_chunk_size,
        block_matrices=block_matrices,
    )


def estimate_dense_graph_bytes(n_samples: int, dtype, n_scales: int) -> int:
    """Memory used by the Laplacian stack and its degree vectors."""
    itemsize = np.dtype(dtype).itemsize
    return itemsize * n_scales * n_samples**2
