import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

from model.load_datasets import load_datasets
from model.clustering_metrics import compute_clustering_score


# -----------------------------
# Configuration
# -----------------------------
dataset = 'Dendritic_batch1' 

# Define colors for different methods
method_colors = {
    "pNMF": "#1f77b4",   # deep blue
    "NMF": "#d62728",    # deep red
    "GNMF": "#2ca02c",   # deep green
    "TNMF": "#9467bd",   # deep purple
    "kTNMF": "#4d4d4d"   # deep gray
}

baseline_methods = ["NMF", "GNMF", "TNMF", "kTNMF"]
metrics = ['ARI', 'NMI', 'Purity', 'Accuracy']

# =====================================================
# Load dataset
# =====================================================
X, y = load_datasets(dataset)

# =====================================================
# Load pNMF multi-scale embeddings
# =====================================================
H_result = scipy.io.loadmat(f'results/{dataset}/pNMF.mat')
H_matrices = H_result['pNMF']

timesteps = np.arange(1, len(H_matrices) + 1)

# =====================================================
# Compute pNMF multi-scale scores
# =====================================================
pnmf_scores = {m: [] for m in metrics}

for H in H_matrices:
    ari, nmi, purity, acc = compute_clustering_score(H.T, y, max_state=1)
    pnmf_scores["ARI"].append(ari)
    pnmf_scores["NMI"].append(nmi)
    pnmf_scores["Purity"].append(purity)
    pnmf_scores["Accuracy"].append(acc)

for m in metrics:
    pnmf_scores[m] = np.array(pnmf_scores[m])

# =====================================================
# Select best scale (average of four metrics)
# =====================================================
mean_score = np.mean(
    np.vstack([pnmf_scores[m] for m in metrics]),
    axis=0
)

best_idx = mean_score.argmax()
best_t = timesteps[best_idx]

# =====================================================
# Compute baseline scores
# =====================================================
baseline_scores = {}

for method in baseline_methods:
    H_result = scipy.io.loadmat(f"results/{dataset}/{method}.mat")
    H_matrix = H_result[method]

    ari, nmi, purity, acc = compute_clustering_score(
        H_matrix.T, y, max_state=1
    )

    baseline_scores[method] = {
        "ARI": ari,
        "NMI": nmi,
        "Purity": purity,
        "Accuracy": acc
    }

# =====================================================
# Print report
# =====================================================
print(f"\nScores for {dataset}\n")
print("Method\t\t\tARI\t\tNMI\t\tPurity\tAccuracy\tAverage")

# pNMF at first scale
avg_first = np.mean([
    pnmf_scores["ARI"][0],
    pnmf_scores["NMI"][0],
    pnmf_scores["Purity"][0],
    pnmf_scores["Accuracy"][0]
])

print(f"pNMF(t=1)\t\t"
      f"{pnmf_scores['ARI'][0]:.3f}\t"
      f"{pnmf_scores['NMI'][0]:.3f}\t"
      f"{pnmf_scores['Purity'][0]:.3f}\t"
      f"{pnmf_scores['Accuracy'][0]:.3f}\t\t"
      f"{avg_first:.3f}")

# pNMF best scale
avg_best = np.mean([
    pnmf_scores["ARI"][best_idx],
    pnmf_scores["NMI"][best_idx],
    pnmf_scores["Purity"][best_idx],
    pnmf_scores["Accuracy"][best_idx]
])

print(f"pNMF-best(t={best_t})\t"
      f"{pnmf_scores['ARI'][best_idx]:.3f}\t"
      f"{pnmf_scores['NMI'][best_idx]:.3f}\t"
      f"{pnmf_scores['Purity'][best_idx]:.3f}\t"
      f"{pnmf_scores['Accuracy'][best_idx]:.3f}\t\t"
      f"{avg_best:.3f}")


# =====================================================
# Plot multi-scale clustering curves
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    # pNMF curve
    ax.plot(
        timesteps,
        pnmf_scores[metric],
        color=method_colors["pNMF"],
        label="pNMF"
    )

    # baseline horizontal lines
    for method in baseline_methods:
        ax.axhline(
            baseline_scores[method][metric],
            linestyle='--',
            color=method_colors[method],
        )

    ax.set_title(metric, fontsize=20)
    ax.tick_params(labelsize=16)

# Global labels
fig.text(0.5, 0.08, 'Index $t$ of $H_t$', ha='center', fontsize=20)
fig.text(0.01, 0.5, 'Score', va='center', rotation='vertical', fontsize=20)

# Unified legend
handles = []
labels = []

handles.append(plt.Line2D([0], [0], color=method_colors["pNMF"]))
labels.append("pNMF")

for method in baseline_methods:
    handles.append(plt.Line2D([0], [0], linestyle='--',
                              color=method_colors[method], lw=2))
    labels.append(method)

fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=5,
    fontsize=20,
    frameon=False,
    bbox_to_anchor=(0.5, 0)
)

plt.tight_layout(rect=[0.02, 0.1, 0.99, 1])

# Save figure
os.makedirs(f'results/{dataset}', exist_ok=True)
plt.savefig(f'results/{dataset}/{dataset}_multi_scale_clustering.pdf', dpi=300)
plt.show()