"""Recompute clustering scores from saved embeddings or refit the five models."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model.array_backend import activate_device, to_numpy
from model.baseline_models import (
    GNMF,
    NMF,
    construct_graph,
    construct_knn_persistent_graph,
    construct_persistent_graph,
)
from model.clustering import METRIC_NAMES, cluster_embedding
from model.data_loader import load_dataset, load_dataset_config
from model.fit import fit_pnmf
from model.graph_construction import build_sparse_laplacian_stack
from model.initialization import nndsvda
from model.reductions import column_l2_norms


DATA = Path(__file__).resolve().parents[1] / "data" / "single_cell"
DATASETS = ("GSE75748time", "Dendritic_batch1")
METHODS = ("NMF", "GNMF", "TNMF", "kTNMF", "pNMF")


def fit_baseline(method, X, initial, parameters):
    W, H = initial
    options = dict(
        n_components=H.shape[0], max_iter=parameters["max_iter"], tol=parameters["tol"]
    )
    if method == "NMF":
        _, H = NMF(**options).fit_transform(X, W, H)
    else:
        if method == "GNMF":
            graph = construct_graph(X, parameters["n_neighbors"])
        elif method == "TNMF":
            graph = construct_persistent_graph(
                X, parameters["n_neighbors"], parameters["weights"]
            )
        else:
            graph = construct_knn_persistent_graph(X, parameters["weights"])
        _, H = GNMF(l=parameters["l"], **options).fit_transform(X, W, H, *graph)
    # Baseline multiplicative updates already absorb the basis-column norms into H.
    return H


def fit_first_scale(X, config, device):
    xp = activate_device(device)
    X = xp.asarray(X, dtype=config["dtype"])
    graphs = build_sparse_laplacian_stack(
        X, alpha=config["alpha"], matmul_chunk_size=config["graph_matmul_chunk"]
    )
    print(f"Fitting all {graphs.n_scales} positive scales on {xp.__name__}.", flush=True)
    result = fit_pnmf(X, graphs, config)
    norms = column_l2_norms(result.W[:1])[0]
    return to_numpy(norms[:, None] * result.H[0])


def evaluate(dataset, method, H, labels):
    with threadpool_limits(limits=1):
        result = cluster_embedding(H.T, labels, repeats=1, random_seed=1, n_init=10)
    scores = {name: float(result.metrics[name][0]) for name in METRIC_NAMES}
    values = " ".join(f"{scores[k]:9.4f}" for k in METRIC_NAMES)
    print(f"{dataset:17} {method:6} {values}", flush=True)
    return {"dataset": dataset, "method": method, **scores}


def dataset_rows(mode, dataset, methods, device):
    if mode == "score":
        config = load_dataset_config(dataset, DATA)
        labels = pd.read_csv(DATA / dataset / config["label_file"], index_col=0)[
            config["label_column"]
        ].to_numpy()
        reference = DATA / dataset / "reference_embeddings.npz"
        with np.load(reference, allow_pickle=False) as embeddings:
            return [evaluate(dataset, method, embeddings[method], labels) for method in methods]

    bundle = load_dataset(dataset, DATA)
    config = bundle.config
    initial = None
    if any(method != "pNMF" for method in methods):
        rank = int(np.ceil(np.sqrt(bundle.X.shape[1])))
        initial = nndsvda(bundle.X, rank, svd_solver="exact")
    rows = []
    for method in methods:
        if method == "pNMF":
            H = fit_first_scale(bundle.X, config["pnmf"], device)
        else:
            H = fit_baseline(method, bundle.X, initial, config["baselines"][method])
        rows.append(evaluate(dataset, method, H, bundle.labels))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("score", "fit"),
        help="score clusters the supplied evaluation embeddings. fit trains from expression.",
    )
    parser.add_argument("--dataset", choices=DATASETS, help="Default: both datasets.")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=METHODS, help="Default: all five models."
    )
    parser.add_argument(
        "--device", default="auto", help="pNMF fitting device: auto, cpu, cuda or cuda:<id>."
    )
    parser.add_argument(
        "--output", type=Path, help="Optionally save the displayed scores to a CSV file."
    )
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    rows = []
    print(f"{'Dataset':17} {'Method':6} " + " ".join(f"{k:>9}" for k in METRIC_NAMES))
    for dataset in datasets:
        rows.extend(dataset_rows(args.mode, dataset, args.methods, args.device))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dataset", "method", *METRIC_NAMES))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
