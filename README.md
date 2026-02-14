# Persistent Non-negative Matrix Factorization (pNMF)

This repository implements **Persistent Non-negative Matrix Factorization (pNMF)** for multi-scale data embedding and clustering. The project includes experiments on both simulation and single-cell RNA-seq datasets.

---

## Folder Structure

```
pnmf-master/
│
├─ datasets/                  # Input datasets
│   ├─ Simulation/             # Synthetic datasets
│   ├─ scRNA-seq/              # Single-cell datasets (e.g., GSE75748time, Dendritic_batch1)
│
├─ results/                    # Results and visualizations
│   ├─ {dataset}/              # Dataset-specific outputs
│
├─ multi_scale_visualization.py    # Visualize embeddings across scales (simulation)
├─ multi_scale_clustering.py       # Compute multi-scale clustering (scRNA-seq)
├─ clustering_metrics.py           # ARI, NMI, Purity, ACC calculation
├─ sequential_alternating_optimization.py  # Optimization routine
├─ pnmf.py                        # pNMF model
├─ utils.py                        # Helper functions
├─ load_datasets.py                # Load datasets
├─ initialization.py               # Initialize W, H, and graph matrices
├─ run_main.py                     # Main script to run pNMF
└─ README.md
```

---

## Installation

1. Install Python 3.11+ and required libraries:

```bash
pip install numpy pandas scipy scikit-learn matplotlib umap-learn
```

2. Clone or download this repository.

---

## Usage

### 1. Run pNMF on a dataset

```bash
python run_main.py --data 3D_concentric_circles --l1 100 --l2 100 --l3 1
```

- `--data` : Dataset name (`3D_concentric_circles`, `GSE75748time`, `Dendritic_batch1`, etc.)
- `--l1`, `--l2`, `--l3` : Regularization parameters

**Notes**:

- For **simulation datasets** (e.g., `3D_concentric_circles`), default l1/l2/l3 can be tuned.
- For **single-cell datasets** (e.g., `GSE75748time`, `Dendritic_batch1`), use:

```bash
--l1 1 --l2 1 --l3 1
```

---

### 2. Multi-scale Embedding Visualization (Simulation)

```bash
python multi_scale_visualization.py
```

- Loads embeddings saved by `run_main.py`.
- Generates PDF and GIF visualizations for each scale.
- Colors are consistent across scales.

---

### 3. Multi-scale Clustering (Single-cell data)

```bash
python multi_scale_clustering.py
```

- Computes **ARI, NMI, Purity, and Accuracy** at each scale.
- Applied to **single-cell datasets**:
  - `GSE75748time`
  - `Dendritic_batch1`
- Uses **l1 = l2 = l3 = 1** by default for all single-cell experiments.
- Generates plots showing score progression across scales and identifies the best scale.

---

## Citation

If you use this code in your work, please cite our paper:  
[Your paper reference here]

