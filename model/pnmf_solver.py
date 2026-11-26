"""Structure-exploiting two-block optimizer for persistent NMF."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.array_backend import array_module, scalar_value, synchronize
from model.reductions import sum_squares
from model.graph_construction import DenseLaplacianStack, SparseLaplacianStack
from model.projected_newton import (
    ProjectedNewtonInfo,
    projected_kkt_residual,
    solve_nonnegative_qp,
)


@dataclass
class SolverConfig:
    max_outer_iterations: int = 25
    stop_on_convergence: bool = True
    stopping_rule: str = "objective_and_kkt"
    inner_tolerance_initial: float = 1.0e-2
    inner_tolerance_decay: float = 0.5
    inner_tolerance_minimum: float = 1.0e-5
    outer_objective_tolerance: float = 1.0e-5
    outer_kkt_tolerance: float = 1.0e-4
    w_scale_chunk_size: int = 8
    w_max_newton_iterations: int = 20
    h_max_newton_iterations: int = 20
    w_pcg_max_iterations: int = 40
    h_pcg_max_iterations: int = 80
    pcg_tolerance: float = 1.0e-3
    pcg_check_every: int = 2
    active_tolerance: Optional[float] = None
    armijo_sigma: float = 1.0e-4
    line_search_beta: float = 0.5
    max_line_search_steps: int = 20
    kkt_check_every: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.max_outer_iterations <= 0:
            raise ValueError("max_outer_iterations must be positive.")
        self.stopping_rule = self.stopping_rule.strip().lower()
        if self.stopping_rule not in {"objective", "objective_and_kkt"}:
            raise ValueError(
                "stopping_rule must be 'objective' or 'objective_and_kkt'."
            )
        if self.inner_tolerance_initial <= 0.0 or self.inner_tolerance_minimum <= 0.0:
            raise ValueError("Inner tolerances must be positive.")
        if not 0.0 < self.inner_tolerance_decay <= 1.0:
            raise ValueError("inner_tolerance_decay must lie in (0, 1].")
        if self.outer_objective_tolerance <= 0.0 or self.outer_kkt_tolerance <= 0.0:
            raise ValueError("Outer tolerances must be positive.")
        integer_fields = (
            self.w_scale_chunk_size,
            self.w_max_newton_iterations,
            self.h_max_newton_iterations,
            self.w_pcg_max_iterations,
            self.h_pcg_max_iterations,
            self.pcg_check_every,
            self.max_line_search_steps,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("Iteration limits, chunk sizes, and check intervals must be positive.")
        if self.kkt_check_every < 0:
            raise ValueError("kkt_check_every must be nonnegative; use 0 to disable it.")
        if (
            self.stop_on_convergence
            and self.stopping_rule == "objective_and_kkt"
            and self.kkt_check_every == 0
        ):
            raise ValueError(
                "objective_and_kkt stopping requires kkt_check_every to be positive."
            )


@dataclass
class PNMFResult:
    W: object
    H: object
    objective: float
    converged: bool
    stopping_reason: str
    history: List[Dict[str, object]] = field(default_factory=list)

    def history_as_dicts(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self.history]


def _sum_squares(x) -> float:
    # Never materialize the square of the complete multi-scale W tensor.
    return sum_squares(x)


class TwoBlockPNMF:
    """Parallel ``W`` updates plus a joint block-tridiagonal ``H`` update."""

    def __init__(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
        *,
        config: Optional[SolverConfig] = None,
    ) -> None:
        if lambda_1 < 0.0 or lambda_2 < 0.0:
            raise ValueError("lambda_1 and lambda_2 must be nonnegative.")
        if lambda_3 <= 0.0:
            raise ValueError(
                "lambda_3 must be positive for strong convexity and the stated convergence results."
            )
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.lambda_3 = float(lambda_3)
        self.config = config or SolverConfig()

    def _validate(
        self,
        X,
        W,
        H,
        laplacians: DenseLaplacianStack | SparseLaplacianStack,
    ) -> None:
        if X.ndim != 2 or W.ndim != 3 or H.ndim != 3:
            raise ValueError("Expected X(p,n), W(T,p,d), and H(T,d,n).")
        T, p, d = W.shape
        if X.shape != (p, H.shape[2]):
            raise ValueError("X, W, and H dimensions are inconsistent.")
        if H.shape[:2] != (T, d):
            raise ValueError("W and H dimensions are inconsistent.")
        if laplacians.shape != (T, X.shape[1], X.shape[1]):
            raise ValueError("The Laplacian stack has incompatible dimensions.")
        xp = array_module(X)
        if (
            array_module(W) is not xp
            or array_module(H) is not xp
            or laplacians.xp is not xp
        ):
            raise ValueError("X, factors, and Laplacians must use the same backend.")
        if scalar_value(xp.min(X)) < 0.0:
            raise ValueError("pNMF requires X >= 0.")
        if scalar_value(xp.min(W)) < 0.0 or scalar_value(xp.min(H)) < 0.0:
            raise ValueError("Initial factors must be nonnegative.")

    @staticmethod
    def _scale_degrees(T: int, xp, dtype):
        nu = xp.full((T,), 2.0, dtype=dtype)
        if T == 1:
            nu[0] = 0.0
        else:
            nu[0] = 1.0
            nu[-1] = 1.0
        return nu

    def _hessian_H(
        self,
        P,
        WtW,
        laplacians: DenseLaplacianStack | SparseLaplacianStack,
        nu,
    ):
        xp = array_module(P)
        result = xp.matmul(WtW, P)
        if self.lambda_1 != 0.0:
            graph_action = laplacians.right_multiply(P)
            graph_action *= self.lambda_1
            result += graph_action
        result += (self.lambda_3 + self.lambda_2 * nu[:, None, None]) * P
        if P.shape[0] > 1 and self.lambda_2 != 0.0:
            result[1:] -= self.lambda_2 * P[:-1]
            result[:-1] -= self.lambda_2 * P[1:]
        return result

    def _factor_products(self, X, W):
        """Compute the two small products needed by the joint H subproblem."""
        xp = array_module(X)
        T, _, d = W.shape
        n = X.shape[1]
        WtW = xp.empty((T, d, d), dtype=W.dtype)
        WtX = xp.empty((T, d, n), dtype=W.dtype)
        chunk = max(1, int(self.config.w_scale_chunk_size))
        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            W_chunk = W[start:stop]
            WtW[start:stop] = xp.matmul(
                W_chunk.swapaxes(1, 2), W_chunk
            )
            WtX[start:stop] = xp.matmul(
                W_chunk.swapaxes(1, 2), X
            )
        return WtW, WtX, _sum_squares(W)

    def _update_W(self, X, W, H, tolerance: float):
        """Solve independent scale problems in GPU-friendly scale batches."""
        xp = array_module(X)
        T, _, d = W.shape
        n = X.shape[1]
        WtW = xp.empty((T, d, d), dtype=W.dtype)
        WtX = xp.empty((T, d, n), dtype=W.dtype)
        identity = xp.eye(d, dtype=W.dtype)
        infos: List[ProjectedNewtonInfo] = []
        chunk = max(1, int(self.config.w_scale_chunk_size))

        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            H_chunk = H[start:stop]
            W_chunk = W[start:stop]
            gram = xp.matmul(H_chunk, H_chunk.swapaxes(1, 2))
            gram += self.lambda_3 * identity[None, :, :]
            rhs = xp.matmul(X, H_chunk.swapaxes(1, 2))
            diagonal = xp.diagonal(gram, axis1=1, axis2=2)[:, None, :]
            gram_inverse = xp.linalg.inv(gram)

            def operator(P, gram=gram):
                return xp.matmul(P, gram)

            def preconditioner(P, gram_inverse=gram_inverse):
                return xp.matmul(P, gram_inverse)

            solved, _, info = solve_nonnegative_qp(
                W_chunk,
                operator,
                rhs,
                diagonal,
                tolerance=tolerance,
                max_newton_iterations=self.config.w_max_newton_iterations,
                pcg_tolerance=self.config.pcg_tolerance,
                pcg_max_iterations=self.config.w_pcg_max_iterations,
                preconditioner=preconditioner,
                active_tolerance=self.config.active_tolerance,
                armijo_sigma=self.config.armijo_sigma,
                line_search_beta=self.config.line_search_beta,
                max_line_search_steps=self.config.max_line_search_steps,
                pcg_check_every=self.config.pcg_check_every,
            )
            W[start:stop] = solved
            infos.append(info)
            WtW[start:stop] = xp.matmul(solved.swapaxes(1, 2), solved)
            WtX[start:stop] = xp.matmul(solved.swapaxes(1, 2), X)

        return W, WtW, WtX, _sum_squares(W), infos

    def _update_H(self, H, WtW, WtX, laplacians, nu, tolerance: float):
        xp = array_module(H)

        def operator(P):
            return self._hessian_H(P, WtW, laplacians, nu)

        shift = (
            self.lambda_1 * laplacians.degree[:, None, :]
            + (self.lambda_3 + self.lambda_2 * nu[:, None, None])
        )
        # Spectral block preconditioner.  It retains the full d-by-d Gram
        # coupling and the Laplacian diagonal, but drops graph off-diagonals and
        # neighboring-scale blocks.  One tiny eigendecomposition per scale
        # makes its inverse cheap for every PCG right-hand side.
        eigenvalues, eigenvectors = xp.linalg.eigh(WtW)
        xp.maximum(eigenvalues, 0.0, out=eigenvalues)
        preconditioner_denominator = eigenvalues[:, :, None] + shift

        def preconditioner(R):
            transformed = xp.matmul(eigenvectors.swapaxes(1, 2), R)
            transformed /= preconditioner_denominator
            return xp.matmul(eigenvectors, transformed)

        return solve_nonnegative_qp(
            H,
            operator,
            WtX,
            preconditioner_denominator,
            tolerance=tolerance,
            max_newton_iterations=self.config.h_max_newton_iterations,
            pcg_tolerance=self.config.pcg_tolerance,
            pcg_max_iterations=self.config.h_pcg_max_iterations,
            preconditioner=preconditioner,
            active_tolerance=self.config.active_tolerance,
            armijo_sigma=self.config.armijo_sigma,
            line_search_beta=self.config.line_search_beta,
            max_line_search_steps=self.config.max_line_search_steps,
            pcg_check_every=self.config.pcg_check_every,
        )

    def _objective(
        self,
        X_norm_squared: float,
        W_norm_squared: float,
        H,
        WtW,
        WtX,
        laplacians,
        nu,
        *,
        AH=None,
    ) -> float:
        xp = array_module(H)
        if AH is None:
            AH = self._hessian_H(H, WtW, laplacians, nu)
        q_H = xp.sum(H * AH, dtype=xp.float64) - 2.0 * xp.sum(
            H * WtX, dtype=xp.float64
        )
        value = (
            H.shape[0] * X_norm_squared
            + self.lambda_3 * W_norm_squared
            + scalar_value(q_H)
        )
        return float(value)

    def _current_kkt(
        self,
        X,
        W,
        H,
        WtW,
        WtX,
        laplacians,
        nu,
        *,
        H_half_gradient=None,
    ) -> Tuple[float, float, float]:
        xp = array_module(X)
        if H_half_gradient is None:
            H_half_gradient = self._hessian_H(
                H, WtW, laplacians, nu
            ) - WtX
        H_residual = projected_kkt_residual(H, H_half_gradient)
        H_raw_squared = scalar_value(
            xp.sum(H_residual * H_residual, dtype=xp.float64)
        )

        W_raw_squared = 0.0
        W_norm_squared = 0.0
        d = W.shape[2]
        identity = xp.eye(d, dtype=W.dtype)
        chunk = max(1, int(self.config.w_scale_chunk_size))
        for start in range(0, W.shape[0], chunk):
            stop = min(start + chunk, W.shape[0])
            W_chunk = W[start:stop]
            H_chunk = H[start:stop]
            gram = xp.matmul(H_chunk, H_chunk.swapaxes(1, 2))
            gram += self.lambda_3 * identity[None, :, :]
            rhs = xp.matmul(X, H_chunk.swapaxes(1, 2))
            half_gradient = xp.matmul(W_chunk, gram) - rhs
            residual = projected_kkt_residual(W_chunk, half_gradient)
            W_raw_squared += scalar_value(
                xp.sum(residual * residual, dtype=xp.float64)
            )
            W_norm_squared += scalar_value(
                xp.sum(W_chunk * W_chunk, dtype=xp.float64)
            )

        H_norm_squared = _sum_squares(H)
        W_raw = float(np.sqrt(max(W_raw_squared, 0.0)))
        H_raw = float(np.sqrt(max(H_raw_squared, 0.0)))
        joint = float(np.sqrt(W_raw_squared + H_raw_squared)) / max(
            1.0, float(np.sqrt(W_norm_squared + H_norm_squared))
        )
        return W_raw, H_raw, joint

    def fit(
        self,
        X,
        W_initial,
        H_initial,
        laplacians: DenseLaplacianStack | SparseLaplacianStack,
        *,
        take_ownership: bool = False,
    ) -> PNMFResult:
        """Run the finite-tolerance two-block algorithm.

        By default, the initial factors are protected for compatibility with
        paired experiments that reuse a common initialization.  Setting
        ``take_ownership=True`` permits in-place updates and avoids eager
        copies of the complete multi-scale factor sequences.
        """
        self._validate(X, W_initial, H_initial, laplacians)
        if take_ownership:
            W = W_initial
            H = H_initial
        else:
            W = W_initial.copy()
            H = H_initial.copy()
        xp = array_module(X)
        T = int(H.shape[0])
        nu = self._scale_degrees(T, xp, H.dtype)
        X_norm_squared = _sum_squares(X)
        WtW, WtX, W_norm_squared = self._factor_products(X, W)
        AH = self._hessian_H(H, WtW, laplacians, nu)
        objective = self._objective(
            X_norm_squared,
            W_norm_squared,
            H,
            WtW,
            WtX,
            laplacians,
            nu,
            AH=AH,
        )
        history: List[Dict[str, object]] = []
        converged = False
        stopping_reason = "maximum outer iterations reached"

        if self.config.verbose:
            print(f"Initial objective: {objective:.8e}")

        for outer in range(self.config.max_outer_iterations):
            start_time = time.perf_counter()
            tolerance = max(
                self.config.inner_tolerance_minimum,
                self.config.inner_tolerance_initial
                * (self.config.inner_tolerance_decay**outer),
            )
            previous_objective = objective

            W, WtW, WtX, W_norm_squared, W_infos = self._update_W(
                X, W, H, tolerance
            )
            H, H_half_gradient, H_info = self._update_H(
                H, WtW, WtX, laplacians, nu, tolerance
            )
            AH = H_half_gradient + WtX
            objective = self._objective(
                X_norm_squared,
                W_norm_squared,
                H,
                WtW,
                WtX,
                laplacians,
                nu,
                AH=AH,
            )
            relative_change = abs(previous_objective - objective) / max(
                1.0, abs(previous_objective)
            )

            kkt_needed_for_stopping = (
                self.config.stop_on_convergence
                and self.config.stopping_rule == "objective_and_kkt"
            )
            periodic_kkt_check = (
                self.config.kkt_check_every > 0
                and (
                    (outer + 1) % self.config.kkt_check_every == 0
                    or outer + 1 == self.config.max_outer_iterations
                )
            )
            should_check_kkt = (
                periodic_kkt_check
                or (
                    kkt_needed_for_stopping
                    and relative_change <= self.config.outer_objective_tolerance
                )
            )
            if should_check_kkt:
                W_kkt, H_kkt, joint_kkt = self._current_kkt(
                    X,
                    W,
                    H,
                    WtW,
                    WtX,
                    laplacians,
                    nu,
                    H_half_gradient=H_half_gradient,
                )
            else:
                W_kkt = H_kkt = joint_kkt = float("nan")

            synchronize(xp)
            elapsed = time.perf_counter() - start_time
            W_newton = sum(info.newton_iterations for info in W_infos)
            W_pcg = sum(info.pcg_iterations for info in W_infos)
            W_newton_values = [info.newton_iterations for info in W_infos]
            W_relative_kkt_values = [info.final_relative_kkt for info in W_infos]
            row = {
                "iteration": outer + 1,
                "inner_tolerance": tolerance,
                "objective": objective,
                "relative_objective_change": relative_change,
                "w_kkt_raw": W_kkt,
                "h_kkt_raw": H_kkt,
                "joint_relative_kkt": joint_kkt,
                "w_newton_iterations": W_newton,
                "w_newton_iterations_max": max(W_newton_values, default=0),
                "w_newton_iterations_mean": float(
                    np.mean(W_newton_values) if W_newton_values else 0.0
                ),
                "w_pcg_iterations": W_pcg,
                "w_relative_kkt_max": max(W_relative_kkt_values, default=0.0),
                "w_relative_kkt_mean": float(
                    np.mean(W_relative_kkt_values) if W_relative_kkt_values else 0.0
                ),
                "w_unconverged_batches": sum(
                    not info.converged for info in W_infos
                ),
                "w_block_objective_change": sum(
                    info.objective_change for info in W_infos
                ),
                "h_newton_iterations": H_info.newton_iterations,
                "h_pcg_iterations": H_info.pcg_iterations,
                "h_converged": H_info.converged,
                "h_relative_kkt": H_info.final_relative_kkt,
                "h_block_objective_change": H_info.objective_change,
                "elapsed_seconds": elapsed,
            }
            history.append(row)

            # Every accepted inner step is descending.  A material increase in
            # the recomputed objective therefore signals loss of numerical
            # accuracy rather than an algorithmic step and should be visible.
            monotonic_slack = 256.0 * np.finfo(np.dtype(H.dtype)).eps * max(
                1.0, abs(previous_objective)
            )
            if objective > previous_objective + monotonic_slack:
                stopping_reason = "stopped after a numerical monotonicity violation"
                if self.config.verbose:
                    print(
                        "Warning: objective increased beyond floating-point slack; "
                        "consider float64 or tighter PCG tolerances."
                    )
                break

            if self.config.verbose:
                kkt_text = (
                    f"{joint_kkt:.3e}" if np.isfinite(joint_kkt) else "not checked"
                )
                print(
                    f"Outer {outer + 1:03d} | objective {objective:.8e} | "
                    f"rel.change {relative_change:.3e} | KKT {kkt_text} | "
                    f"{elapsed:.2f}s"
                )

            if self.config.stop_on_convergence:
                objective_reached = (
                    relative_change <= self.config.outer_objective_tolerance
                )
                if (
                    self.config.stopping_rule == "objective"
                    and objective_reached
                ):
                    converged = True
                    stopping_reason = "relative objective tolerance reached"
                    break
                if (
                    self.config.stopping_rule == "objective_and_kkt"
                    and objective_reached
                    and np.isfinite(joint_kkt)
                    and joint_kkt <= self.config.outer_kkt_tolerance
                ):
                    converged = True
                    stopping_reason = (
                        "objective and projected KKT tolerances reached"
                    )
                    break

        return PNMFResult(
            W=W,
            H=H,
            objective=objective,
            converged=converged,
            stopping_reason=stopping_reason,
            history=history,
        )


def solver_config_dict(config: SolverConfig) -> Dict[str, object]:
    return asdict(config)
