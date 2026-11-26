"""Active-set projected Newton solver for strongly convex nonnegative QPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from model.array_backend import array_module, scalar_value


Array = object
LinearOperator = Callable[[Array], Array]


@dataclass
class ProjectedNewtonInfo:
    converged: bool
    status: str
    newton_iterations: int
    pcg_iterations: int
    line_search_steps: int
    initial_kkt: float
    final_kkt: float
    final_relative_kkt: float
    objective_change: float


def _norm(x) -> float:
    xp = array_module(x)
    return scalar_value(xp.sqrt(xp.sum(x * x, dtype=xp.float64)))


def projected_kkt_residual(z, half_gradient):
    """Residual ``z - Pi_+(z - grad q)`` for ``grad q = 2*half_gradient``."""
    xp = array_module(z, half_gradient)
    return z - xp.maximum(z - 2.0 * half_gradient, 0.0)


def _relative_kkt(z, half_gradient) -> Tuple[float, float]:
    residual = projected_kkt_residual(z, half_gradient)
    raw = _norm(residual)
    return raw, raw / max(1.0, _norm(z))


def masked_pcg(
    operator: LinearOperator,
    rhs,
    free_mask,
    diagonal,
    *,
    preconditioner: Optional[LinearOperator] = None,
    tolerance: float,
    max_iterations: int,
    check_every: int = 1,
) -> Tuple[Array, int, float]:
    """Solve a free-set SPD system with preconditioned CG.

    ``operator`` is never materialized. Multiplication by ``free_mask`` on
    both sides gives the principal free-set system used by projected Newton.
    When no callable preconditioner is supplied, the provided diagonal defines
    a Jacobi fallback.
    """
    xp = array_module(rhs)
    free = free_mask.astype(rhs.dtype, copy=False)
    b = rhs * free
    solution = xp.zeros_like(rhs)
    residual = b.copy()
    b_norm = _norm(b)
    if b_norm == 0.0:
        return solution, 0, 0.0

    if preconditioner is None:
        safe_diagonal = xp.maximum(
            diagonal, np.finfo(np.dtype(rhs.dtype)).tiny
        )
        preconditioned = residual * free / safe_diagonal
    else:
        safe_diagonal = None
        preconditioned = preconditioner(residual * free) * free
    direction = preconditioned.copy()
    rz = xp.sum(residual * preconditioned)
    residual_norm = b_norm
    check_every = max(1, int(check_every))
    scalar_tiny = np.finfo(np.dtype(rhs.dtype)).tiny

    for iteration in range(1, max_iterations + 1):
        Adirection = operator(direction * free) * free
        denominator = xp.sum(direction * Adirection)
        # The free-set operator is SPD. Keeping these scalars on-device avoids
        # two CUDA synchronizations per PCG iteration; tiny clipping only
        # handles the already-converged zero-direction case between checks.
        alpha = rz / xp.maximum(denominator, scalar_tiny)
        solution += alpha * direction
        residual -= alpha * Adirection

        if iteration % check_every == 0 or iteration == max_iterations:
            residual_norm = _norm(residual)
            if not np.isfinite(residual_norm):
                return solution, iteration, float("inf")
            if residual_norm <= tolerance * b_norm:
                return solution, iteration, residual_norm / b_norm

        if preconditioner is None:
            preconditioned = residual * free / safe_diagonal
        else:
            preconditioned = preconditioner(residual * free) * free
        rz_new = xp.sum(residual * preconditioned)
        beta = rz_new / xp.maximum(rz, scalar_tiny)
        direction = preconditioned + beta * direction
        rz = rz_new

    return solution, max_iterations, residual_norm / b_norm


def _projected_armijo(
    z,
    half_gradient,
    direction,
    operator: LinearOperator,
    *,
    sigma: float,
    beta: float,
    max_steps: int,
):
    """Exact quadratic Armijo test without recomputing the full objective."""
    xp = array_module(z)
    alpha = 1.0
    for line_step in range(max_steps + 1):
        candidate = xp.maximum(z + alpha * direction, 0.0)
        delta = candidate - z
        directional = xp.sum(half_gradient * delta)
        Adelta = operator(delta)
        change = 2.0 * directional + xp.sum(delta * Adelta)
        rhs = 2.0 * sigma * directional
        change_value = scalar_value(change)
        rhs_value = scalar_value(rhs)
        roundoff = 64.0 * np.finfo(np.dtype(z.dtype)).eps * max(
            1.0, abs(rhs_value)
        )
        if change_value <= rhs_value + roundoff:
            return candidate, delta, Adelta, change_value, line_step, True
        alpha *= beta
    return z, xp.zeros_like(z), xp.zeros_like(z), 0.0, max_steps, False


def solve_nonnegative_qp(
    initial,
    operator: LinearOperator,
    rhs,
    diagonal,
    *,
    tolerance: float,
    max_newton_iterations: int,
    pcg_tolerance: float,
    pcg_max_iterations: int,
    preconditioner: Optional[LinearOperator] = None,
    active_tolerance: Optional[float] = None,
    armijo_sigma: float = 1.0e-4,
    line_search_beta: float = 0.5,
    max_line_search_steps: int = 20,
    pcg_check_every: int = 1,
) -> Tuple[Array, Array, ProjectedNewtonInfo]:
    """Minimize ``z.T A z - 2 rhs.T z`` over the nonnegative orthant.

    The returned gradient is the *half gradient* ``A z - rhs``.  Using this
    convention removes an irrelevant factor two from every Newton system.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 0.0 < armijo_sigma < 1.0:
        raise ValueError("armijo_sigma must lie in (0, 1)")
    if not 0.0 < line_search_beta < 1.0:
        raise ValueError("line_search_beta must lie in (0, 1)")

    xp = array_module(initial, rhs)
    z = xp.maximum(initial, 0.0).copy()
    half_gradient = operator(z) - rhs
    initial_kkt, relative_kkt = _relative_kkt(z, half_gradient)
    final_kkt = initial_kkt
    total_pcg = 0
    total_lines = 0
    total_change = 0.0
    status = "maximum Newton iterations reached"
    iterations_used = 0

    if active_tolerance is None:
        active_tolerance = float(
            max(100.0 * np.finfo(np.dtype(z.dtype)).eps, 1.0e-12)
        )

    for newton_iteration in range(max_newton_iterations + 1):
        final_kkt, relative_kkt = _relative_kkt(z, half_gradient)
        if relative_kkt <= tolerance:
            status = "projected KKT tolerance reached"
            return z, half_gradient, ProjectedNewtonInfo(
                True,
                status,
                newton_iteration,
                total_pcg,
                total_lines,
                initial_kkt,
                final_kkt,
                relative_kkt,
                total_change,
            )
        if newton_iteration == max_newton_iterations:
            break

        iterations_used = newton_iteration + 1
        threshold = active_tolerance * max(1.0, scalar_value(xp.max(z)))
        binding = (z <= threshold) & (half_gradient > 0.0)
        free = ~binding
        direction, pcg_iterations, _ = masked_pcg(
            operator,
            -half_gradient,
            free,
            diagonal,
            preconditioner=preconditioner,
            tolerance=pcg_tolerance,
            max_iterations=pcg_max_iterations,
            check_every=pcg_check_every,
        )
        total_pcg += pcg_iterations

        # An inaccurate PCG direction can occasionally lose descent.  The
        # diagonally scaled projected-gradient direction is a safe fallback.
        descent = scalar_value(xp.sum(half_gradient * direction))
        if not np.isfinite(descent) or descent >= 0.0:
            free_values = free.astype(z.dtype, copy=False)
            if preconditioner is None:
                safe_diagonal = xp.maximum(
                    diagonal, np.finfo(np.dtype(z.dtype)).tiny
                )
                direction = -free_values * half_gradient / safe_diagonal
            else:
                direction = -free_values * preconditioner(
                    free_values * half_gradient
                )

        trial = _projected_armijo(
            z,
            half_gradient,
            direction,
            operator,
            sigma=armijo_sigma,
            beta=line_search_beta,
            max_steps=max_line_search_steps,
        )
        candidate, delta, Adelta, change, line_steps, accepted = trial
        total_lines += line_steps

        if not accepted:
            free_values = free.astype(z.dtype, copy=False)
            if preconditioner is None:
                safe_diagonal = xp.maximum(
                    diagonal, np.finfo(np.dtype(z.dtype)).tiny
                )
                fallback = -free_values * half_gradient / safe_diagonal
            else:
                fallback = -free_values * preconditioner(
                    free_values * half_gradient
                )
            trial = _projected_armijo(
                z,
                half_gradient,
                fallback,
                operator,
                sigma=armijo_sigma,
                beta=line_search_beta,
                max_steps=max_line_search_steps,
            )
            candidate, delta, Adelta, change, line_steps, accepted = trial
            total_lines += line_steps

        if not accepted or _norm(delta) == 0.0:
            status = "line search stalled"
            break

        z = candidate
        half_gradient = half_gradient + Adelta
        total_change += change

    final_kkt, relative_kkt = _relative_kkt(z, half_gradient)
    return z, half_gradient, ProjectedNewtonInfo(
        relative_kkt <= tolerance,
        status,
        iterations_used,
        total_pcg,
        total_lines,
        initial_kkt,
        final_kkt,
        relative_kkt,
        total_change,
    )
