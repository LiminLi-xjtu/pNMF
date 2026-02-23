import numpy as np
import argparse
import scipy.io
import os

from model.load_datasets import load_datasets
from model.nndsvda import NNDSVDA
from model.baseline_models import (
    NMF, GNMF,
    construct_graph,
    construct_persistent_graph,
    construct_knn_persistent_graph
)
from model.clustering_metrics import compute_clustering_score


# ============================================================
# Arguments (global)
# ============================================================

parser = argparse.ArgumentParser(description='Run baseline methods')
parser.add_argument('--data', type=str, default='Dendritic_batch1')

args = parser.parse_args()


# ============================================================
# Load dataset
# ============================================================

X, y = load_datasets(args.data)
W_init, H_init = NNDSVDA(X, args.data)

save_dir = f'results/{args.data}'
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# Dataset-Method-specific hyperparameters
# ============================================================

dataset_configs = {

    'GSE75748time': {
        'NMF': {},

        'GNMF': {
            'n_neighbors': 5,
            'l': 0.1,
        },

        'TNMF': {
            'n_neighbors': 20,
            'l': 0.1,
            'weights': np.array([1, 0, 1, 0, 1, 0, 1, 1], dtype=float)
        },

        'kTNMF': {
            'l': 1,
            'weights': np.array([1, 0, 1, 0, 1, 0, 1, 1], dtype=float)
        }
    },

    'Dendritic_batch1': {
        'NMF': {},

        'GNMF': {
            'n_neighbors': 15,
            'l': 0.1,
        },

        'TNMF': {
            'n_neighbors': 10,
            'l': 0.1,
            'weights': np.array([1, 0, 1, 0, 1, 0, 1, 1], dtype=float)
        },

        'kTNMF': {
            'l': 1,
            'weights': np.array([1, 0, 1, 0, 1, 0, 1, 1], dtype=float)
        }
    }
}

method_configs = dataset_configs[args.data]


# ============================================================
# Run
# ============================================================

results = {}

for method, config in method_configs.items():
    print(f'\nRunning {method} ...')

    W, H = W_init.copy(), H_init.copy()

    # -----------------------
    # NMF
    # -----------------------
    if method == 'NMF':
        model = NMF(n_components=W.shape[1])
        W, H = model.fit_transform(X, W, H)

    # -----------------------
    # GNMF
    # -----------------------
    elif method == 'GNMF':
        A, D, L = construct_graph(X, config['n_neighbors'])
        model = GNMF(n_components=W.shape[1], l=config['l'])
        W, H = model.fit_transform(X, W, H, A, D, L)

    # -----------------------
    # TNMF
    # -----------------------
    elif method == 'TNMF':
        A, D, L = construct_persistent_graph(
            X,
            config['n_neighbors'],
            config['weights']
        )
        model = GNMF(n_components=W.shape[1], l=config['l'])
        W, H = model.fit_transform(X, W, H, A, D, L)

    # -----------------------
    # kTNMF
    # -----------------------
    elif method == 'kTNMF':
        A, D, L = construct_knn_persistent_graph(
            X,
            config['weights']
        )
        model = GNMF(n_components=W.shape[1], l=config['l'])
        W, H = model.fit_transform(X, W, H, A, D, L)

    # -----------------------
    # Evaluation
    # -----------------------
    ari, nmi, purity, acc = compute_clustering_score(H.T, y, max_state=1)
    avg = np.mean([ari, nmi, purity, acc])

    results[method] = (ari, nmi, purity, acc, avg)

    # save embedding
    scipy.io.savemat(f'{save_dir}/{method}.mat', {method: H})


# ============================================================
# Summary
# ============================================================

print('\n' + '=' * 30)
print(f'Results on {args.data}')
print('=' * 30)
print('\nMethod\tARI\tNMI\tPurity\tAccuracy\tAverage\n')

for m, (ari, nmi, purity, acc, avg) in results.items():
    print(f'{m}\t{ari:.3f}\t{nmi:.3f}\t{purity:.3f}\t{acc:.3f}\t\t{avg:.3f}\n')
