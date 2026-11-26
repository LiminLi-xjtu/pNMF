"""Fit the pNMF model using the settings supplied with a dataset."""

from dataclasses import fields

import numpy as np

from model.array_backend import array_module, free_memory_pool
from model.initialization import initialize_two_block_factors
from model.pnmf_solver import PNMFResult, SolverConfig, TwoBlockPNMF


def fit_pnmf(X, graphs, config, *, verbose=True) -> PNMFResult:
    """Fit all selected scales for a feature-by-sample matrix on the graph backend.

    The returned factors have shapes (scales, features, rank) and
    (scales, rank, samples). The fit uses all scales before any embedding
    is selected or rescaled for evaluation.
    """
    rank = config.get("rank")
    if rank is None:
        rank = int(np.ceil(np.sqrt(X.shape[1])))
    lambda1 = float(config["lambda1"])
    W, H, _ = initialize_two_block_factors(
        X, graphs, int(rank), lambda1,
        method=config.get("initialization", "nndsvda"),
        svd_solver=config.get("svd_solver", "exact"),
        random_state=int(config.get("random_seed", 0)),
        gnmf_max_iterations=int(config.get("gnmf_max_iterations", 1000)),
        gnmf_tolerance=float(config.get("gnmf_tolerance", 1e-3)),
        gnmf_check_every=int(config.get("gnmf_check_every", 1)),
        verbose=verbose,
    )
    free_memory_pool(array_module(X))
    options = {f.name: config[f.name] for f in fields(SolverConfig) if f.name in config}
    options["verbose"] = verbose
    model = TwoBlockPNMF(
        lambda1, float(config["lambda2"]), float(config["lambda3"]),
        config=SolverConfig(**options),
    )
    return model.fit(X, W, H, graphs, take_ownership=True)
