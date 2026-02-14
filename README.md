# Persistent Nonnegative Matrix Factorization via Multi-Scale Graph Regularization

This repository implements **Persistent Non-negative Matrix Factorization (pNMF)** for multi-scale data embedding and clustering. The project includes experiments on both simulation and single-cell RNA-seq datasets.

---

## Installation and Requirements

To run this project, ensure you have Python 3.11.7 and install the following packages:

- **numpy**: 1.26.3
- **pandas**: 2.1.4
- **scikit-learn**: 1.2.2
- **scipy**: 1.13.1
- **matplotlib**: 3.10.1

You can install the required packages using `pip` from the `requirements.txt` file. Create a `requirements.txt` file with the following content:

```plaintext
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.2.2
scipy==1.13.1
matplotlib==3.10.1
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

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

### 2. Run baselines on a single-cell dataset (e.g., `GSE75748time`, `Dendritic_batch1`)

```bash
python run_baselines.py --data GSE75748time
```

---

### 3. Multi-scale Embedding Visualization (Simulation)

```bash
python multi_scale_visualization.py
```

---

### 4. Multi-scale Clustering (Single-cell data)

```bash
python multi_scale_clustering.py
```

---
