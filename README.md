# Persistent Nonnegative Matrix Factorization via Multi-Scale Graph Regularization

This repository implements **Persistent Nonnegative Matrix Factorization
(pNMF)** for learning multi-scale low-rank embeddings.\
The project includes experiments on both simulation datasets and
single-cell RNA-seq datasets.

`<img src="results/3D_concentric_circles/animation_3D_concentric_circles_pNMF.gif">`{=html}

------------------------------------------------------------------------

## Installation and Requirements

To run this project, ensure you have **Python 3.11.7** and install the
following packages:

-   **numpy**: 1.26.3\
-   **pandas**: 2.1.4\
-   **scikit-learn**: 1.2.2\
-   **scipy**: 1.13.1\
-   **matplotlib**: 3.10.1

You can install the required packages using `pip` from the
`requirements.txt` file.

Create a `requirements.txt` file with the following content:

``` plaintext
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.2.2
scipy==1.13.1
matplotlib==3.10.1
```

Then install the dependencies via:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### 1. Run pNMF on a dataset

``` bash
python run_main.py --data 3D_concentric_circles --l1 100 --l2 100 --l3 1
```

-   `--data`: Dataset name (`3D_concentric_circles`, `GSE75748time`,
    `Dendritic_batch1`, etc.)\
-   `--l1`, `--l2`, `--l3`: Regularization parameters

**Notes**:

-   For **simulation datasets** (e.g., `3D_concentric_circles`), we
    recommend:

``` bash
--l1 100 --l2 100 --l3 1
```

-   For **single-cell datasets** (e.g., `GSE75748time`,
    `Dendritic_batch1`), we recommend:

``` bash
--l1 1 --l2 1 --l3 1
```

------------------------------------------------------------------------

### 2. Run baselines on a single-cell dataset (e.g., `GSE75748time`, `Dendritic_batch1`)

``` bash
python run_baselines.py --data Dendritic_batch1
```

------------------------------------------------------------------------

### 3. Multi-scale Embeddings Visualization (Simulation datasets)

``` bash
python multi_scale_visualization.py
```

------------------------------------------------------------------------

### 4. Multi-scale Clustering (Single-cell datasets)

``` bash
python multi_scale_clustering.py
```

------------------------------------------------------------------------
