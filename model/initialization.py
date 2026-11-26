"""Memory-bounded NNDSVDA and sequential-GNMF initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.utils.extmath import randomized_svd

from model.array_backend import array_module, scalar_value, to_numpy
from model.graph_construction import DenseLaplacianStack, SparseLaplacianStack


@dataclass
class InitializationInfo:
    method: str
    rank: int
    gnmf_total_iterations: int
    gnmf_iterations_per_scale: list


def nndsvda(
    X,
    rank: int,
    *,
    svd_solver: str = "exact",
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """NNDSVDA on CPU, returning arrays in the input dtype."""
    X_cpu = np.asarray(to_numpy(X))
    if rank <= 0 or rank > min(X_cpu.shape):
        raise ValueError("rank must lie between 1 and min(X.shape).")
    if svd_solver == "randomized":
        U, singular_values, Vt = randomized_svd(
            X_cpu,
            n_components=rank,
            n_iter=4,
            random_state=random_state,
        )
    elif svd_solver == "exact":
        U_full, singular_values_full, Vt_full = np.linalg.svd(
            X_cpu, full_matrices=False
        )
        U = U_full[:, :rank]
        singular_values = singular_values_full[:rank]
        Vt = Vt_full[:rank]
    else:
        raise ValueError("svd_solver must be 'randomized' or 'exact'.")

    W = np.zeros((X_cpu.shape[0], rank), dtype=X_cpu.dtype)
    H = np.zeros((rank, X_cpu.shape[1]), dtype=X_cpu.dtype)
    W[:, 0] = np.sqrt(singular_values[0]) * np.abs(U[:, 0])
    H[0] = np.sqrt(singular_values[0]) * np.abs(Vt[0])
    tiny = np.finfo(X_cpu.dtype).eps

    for component in range(1, rank):
        u = U[:, component]
        v = Vt[component]
        u_pos, v_pos = np.maximum(u, 0.0), np.maximum(v, 0.0)
        u_neg, v_neg = np.maximum(-u, 0.0), np.maximum(-v, 0.0)
        pos_norms = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
        neg_norms = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)
        if pos_norms >= neg_norms:
            u_part, v_part, product = u_pos, v_pos, pos_norms
        else:
            u_part, v_part, product = u_neg, v_neg, neg_norms
        if product <= tiny:
            continue
        u_part = u_part / max(np.linalg.norm(u_part), tiny)
        v_part = v_part / max(np.linalg.norm(v_part), tiny)
        coefficient = np.sqrt(singular_values[component] * product)
        W[:, component] = coefficient * u_part
        H[component] = coefficient * v_part

    fill = float(np.mean(X_cpu))
    W[W <= tiny] = fill
    H[H <= tiny] = fill
    return W, H


def _sequential_gnmf_scale(
    X,
    W,
    H,
    laplacians,
    scale_index: int,
    lambda_1: float,
    *,
    max_iterations: int,
    tolerance: float,
    check_every: int,
) -> Tuple[object, object, int]:
    """Optimized multiplicative GNMF used only to warm-start pNMF."""
    xp = array_module(X)
    epsilon = 16.0 * np.finfo(np.dtype(X.dtype)).eps
    X_norm_squared = scalar_value(xp.sum(X * X, dtype=xp.float64))
    previous_loss = float("inf")
    check_every = max(1, int(check_every))
    degree = laplacians.degree[scale_index]

    for iteration in range(1, max_iterations + 1):
        HHt = H @ H.T
        XHt = X @ H.T
        W *= XHt / xp.maximum(W @ HHt, epsilon)

        WtW = W.T @ W
        WtX = W.T @ X
        HL = laplacians.scale_right_multiply(H, scale_index)
        HA = H * degree[None, :] - HL
        numerator = WtX + lambda_1 * HA
        denominator = WtW @ H + lambda_1 * H * degree[None, :]
        H *= xp.maximum(numerator, 0.0) / xp.maximum(denominator, epsilon)

        # GNMF is only an initializer; reciprocal rescaling here does not form
        # part of the pNMF iteration analyzed in the paper.
        scale = xp.linalg.norm(W, axis=0)
        scale = xp.maximum(scale, epsilon)
        W /= scale[None, :]
        H *= scale[:, None]

        if iteration % check_every == 0 or iteration == max_iterations:
            WtW = W.T @ W
            WtX = W.T @ X
            HL = laplacians.scale_right_multiply(H, scale_index)
            loss = scalar_value(
                X_norm_squared
                - 2.0 * xp.sum(H * WtX, dtype=xp.float64)
                + xp.sum(H * (WtW @ H), dtype=xp.float64)
                + lambda_1 * xp.sum(H * HL, dtype=xp.float64)
            )
            if np.isfinite(previous_loss) and (
                abs(previous_loss - loss) / max(1.0, abs(previous_loss))
                <= tolerance
            ):
                return W, H, iteration
            previous_loss = loss
    return W, H, max_iterations


def initialize_two_block_factors(
    X,
    laplacians: DenseLaplacianStack | SparseLaplacianStack,
    rank: int,
    lambda_1: float,
    *,
    method: str = "sequential-gnmf",
    svd_solver: str = "exact",
    random_state: int = 0,
    gnmf_max_iterations: int = 100,
    gnmf_tolerance: float = 1.0e-3,
    gnmf_check_every: int = 5,
    verbose: bool = True,
):
    """Create ``W(T,p,d)`` and ``H(T,d,n)`` without Python lists."""
    xp = array_module(X)
    W_cpu, H_cpu = nndsvda(
        X, rank, svd_solver=svd_solver, random_state=random_state
    )
    W_current = xp.asarray(W_cpu, dtype=X.dtype)
    H_current = xp.asarray(H_cpu, dtype=X.dtype)
    T = laplacians.n_scales
    W_all = xp.empty((T, X.shape[0], rank), dtype=X.dtype)
    H_all = xp.empty((T, rank, X.shape[1]), dtype=X.dtype)
    iterations = []

    normalized_method = method.strip().lower()
    if normalized_method not in {"nndsvda", "sequential-gnmf"}:
        raise ValueError("method must be 'nndsvda' or 'sequential-gnmf'.")

    if normalized_method == "nndsvda":
        W_all[:] = W_current[None, :, :]
        H_all[:] = H_current[None, :, :]
        iterations = [0] * T
    else:
        # Match the original warm-start path: solve from the coarsest graph to
        # the finest one, while storing tensors in increasing-scale order.
        iterations = [0] * T
        for progress, t in enumerate(range(T - 1, -1, -1), start=1):
            W_current, H_current, used = _sequential_gnmf_scale(
                X,
                W_current,
                H_current,
                laplacians,
                t,
                lambda_1,
                max_iterations=gnmf_max_iterations,
                tolerance=gnmf_tolerance,
                check_every=gnmf_check_every,
            )
            W_all[t] = W_current
            H_all[t] = H_current
            iterations[t] = used
            if verbose and (progress == 1 or progress % 50 == 0 or progress == T):
                print(
                    f"Initialization scale {progress}/{T}: {used} GNMF iterations"
                )

    return W_all, H_all, InitializationInfo(
        method=normalized_method,
        rank=rank,
        gnmf_total_iterations=int(sum(iterations)),
        gnmf_iterations_per_scale=iterations,
    )
