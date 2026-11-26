"""Fit pNMF to a fixed synthetic dataset and animate the raw embedding sequence."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GROUP_COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "^", "s")
DATASETS = {
    "circles": [
        (f"Circle {i}", color, "o")
        for i, color in enumerate((*GROUP_COLORS, "#CC79A7"), start=1)
    ],
    "disks": [
        (f"Group {g}, disk {d}", color, marker)
        for g, color in enumerate(GROUP_COLORS, start=1)
        for d, marker in enumerate(MARKERS, start=1)
    ],
    "branching": [
        ("Trunk", "#666666", "o"),
        ("Left main", "#2878B5", "o"),
        ("Right main", "#C85C0E", "o"),
        ("Left outer", "#215A86", "^"),
        ("Left inner", "#4BA3CB", "s"),
        ("Right inner", "#9A4810", "^"),
        ("Right outer", "#EA8A32", "s"),
    ],
}


def render(dataset, labels, H, thresholds, output):
    """Draw the fitted embeddings from the largest threshold to the smallest."""
    styles = DATASETS[dataset]
    groups = np.unique(labels)
    thresholds = np.asarray(thresholds).reshape(-1)
    if len(groups) != len(styles) or H.shape != (len(thresholds), 2, labels.size):
        raise ValueError("Embedding dimensions or sample groups disagree with the dataset.")
    if not np.isfinite(H).all() or not np.isfinite(thresholds).all():
        raise ValueError("Embeddings and thresholds must be finite.")
    if len(thresholds) == 0 or thresholds[0] <= 0 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("The animation requires positive, strictly increasing thresholds.")

    minimum = H.min(axis=(0, 2))
    maximum = H.max(axis=(0, 2))
    center = (minimum + maximum) / 2
    half_span = max(float((maximum - minimum).max()) * 0.55, 1e-8)
    fig = Figure(figsize=(4.8, 4.75), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.16, 0.27, 0.72, 0.70], aspect="equal")
    ax.set(xlim=(center[0] - half_span, center[0] + half_span),
           ylim=(center[1] - half_span, center[1] + half_span),
           xlabel="Embedding coordinate 1", ylabel="Embedding coordinate 2")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(color="#e1e5e8", linewidth=0.6)
    ax.set_axisbelow(True)
    points = []
    for group, (name, color, marker) in zip(groups, styles):
        mask = labels == group
        scatter = ax.scatter([], [], label=name, color=color, marker=marker,
                             s=18, linewidths=0, alpha=0.84)
        points.append((scatter, mask))
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.018), frameon=False,
               ncol=2 if dataset == "circles" else 3, fontsize=8.5,
               columnspacing=1.3, handletextpad=0.3)

    frames = []
    for embedding in reversed(H):
        for scatter, mask in points:
            scatter.set_offsets(embedding.T[mask])
        canvas.draw()
        rgb = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        frames.append(rgb.quantize(colors=128))
    duration = [max(40, min(120, round(14000 / len(H))))] * len(H)
    duration[0] = duration[-1] = 800
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)


def fit(X_cpu, device, max_iterations):
    from threadpoolctl import threadpool_limits
    from model.array_backend import activate_device, to_numpy
    from model.fit import fit_pnmf
    from model.graph_construction import build_sparse_laplacian_stack, sparse_graphs_to_backend

    config = json.loads(Path(__file__).with_name("synthetic_settings.json").read_text())
    if max_iterations is not None:
        config["max_outer_iterations"] = max_iterations
    xp = activate_device(device)
    with threadpool_limits(limits=1):
        cpu_graphs = build_sparse_laplacian_stack(X_cpu, alpha=config["alpha"])
        graphs = sparse_graphs_to_backend(cpu_graphs, xp)
        result = fit_pnmf(xp.asarray(X_cpu), graphs, config)
    return {
        "H": to_numpy(result.H),
        "thresholds": graphs.thresholds,
        "outer_iterations": len(result.history),
        "stopping_reason": result.stopping_reason,
        "joint_relative_kkt": result.history[-1]["joint_relative_kkt"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="circles")
    parser.add_argument("--device", default="cpu", help="Fit on cpu or cuda:0.")
    parser.add_argument("--max-outer-iterations", type=int, help="Override the 1500-iteration budget.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "synthetic")
    parser.add_argument("--from-fit", type=Path,
                        help="Render a saved NPZ containing H and thresholds, skipping the fit.")
    args = parser.parse_args()
    if args.max_outer_iterations is not None and args.max_outer_iterations < 1:
        parser.error("--max-outer-iterations must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "data" / "synthetic" / f"{args.dataset}.npz") as data:
        labels = data["labels"]
        if args.from_fit is None:
            result = fit(data["X"], args.device, args.max_outer_iterations)
            fit_path = args.output_dir / f"{args.dataset}.npz"
            np.savez_compressed(fit_path, **result)
            print(f"Saved {fit_path} ({result['outer_iterations']} iterations, "
                  f"{result['stopping_reason']}, relative KKT {result['joint_relative_kkt']:.3g})")
            H, thresholds = result["H"], result["thresholds"]
        else:
            with np.load(args.from_fit) as saved:
                H, thresholds = saved["H"], saved["thresholds"]
    gif = args.output_dir / f"{args.dataset}.gif"
    render(args.dataset, labels, H, thresholds, gif)
    print(f"Saved {gif} ({len(thresholds)} scales)")


if __name__ == "__main__":
    main()
