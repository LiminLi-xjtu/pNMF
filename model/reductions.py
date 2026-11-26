"""Bounded-workspace reductions for large NumPy/CuPy factor tensors."""

from __future__ import annotations

import numpy as np

from model.array_backend import array_module, scalar_value


DEFAULT_WORKSPACE_BYTES = 32 * 1024**2


def _element_budget(x, max_workspace_bytes: int) -> int:
    if np.dtype(x.dtype).kind != "f":
        raise TypeError("Factor reductions require floating-point arrays.")
    if max_workspace_bytes < x.dtype.itemsize:
        raise ValueError("Workspace must hold at least one array element.")
    return int(max_workspace_bytes) // x.dtype.itemsize


def _bounded_views(x, max_elements: int):
    """Yield views without flattening/copying a noncontiguous complete input."""
    if x.size <= max_elements:
        yield x
        return
    axis = max(range(x.ndim), key=lambda i: x.shape[i])
    elements_per_index = x.size // x.shape[axis]
    width = max(1, max_elements // elements_per_index)
    for start in range(0, x.shape[axis], width):
        index = [slice(None)] * x.ndim
        index[axis] = slice(start, min(start + width, x.shape[axis]))
        yield from _bounded_views(x[tuple(index)], max_elements)


def sum_squares(x, *, max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES) -> float:
    """Sum squares in float64 using bounded input-dtype square temporaries.

    Multiplication retains the input dtype, as in the original reduction.
    Only the reduction order changes; no full-sized square/copy is created.
    """
    max_elements = _element_budget(x, max_workspace_bytes)
    xp = array_module(x)
    total = xp.zeros((), dtype=xp.float64)
    for block in _bounded_views(x, max_elements):
        total += xp.sum(block * block, dtype=xp.float64)
    return scalar_value(total)


def column_l2_norms(W, *, max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES):
    """Return final W column norms, shape (m,d), without a full W*W array.

    W has shape (m,p,d). Accumulation uses float64; returned norms retain W's
    dtype. The budget bounds each square temporary, not all allocator/library
    memory or the small output/partial-sum buffers.
    """
    if W.ndim != 3:
        raise ValueError("Expected W with shape (m,p,d).")
    max_elements = _element_budget(W, max_workspace_bytes)
    xp = array_module(W)
    m, p, d = W.shape
    squared = xp.zeros((m, d), dtype=xp.float64)
    scale_chunk = max(1, min(m, max_elements // max(1, p * d)))
    for scale_start in range(0, m, scale_chunk):
        scale_stop = min(scale_start + scale_chunk, m)
        count = scale_stop - scale_start
        column_chunk = max(1, min(d, max_elements // count))
        for column_start in range(0, d, column_chunk):
            column_stop = min(column_start + column_chunk, d)
            columns = column_stop - column_start
            feature_chunk = max(1, max_elements // (count * columns))
            for feature_start in range(0, p, feature_chunk):
                feature_stop = min(feature_start + feature_chunk, p)
                block = W[scale_start:scale_stop, feature_start:feature_stop,
                          column_start:column_stop]
                squared[scale_start:scale_stop, column_start:column_stop] += (
                    xp.sum(block * block, axis=1, dtype=xp.float64)
                )
    return xp.sqrt(squared).astype(W.dtype, copy=False)
