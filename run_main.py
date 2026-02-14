import argparse
import numpy as np
import os
import scipy.io as sio

from model.load_datasets import load_datasets
from model.initialization import initialize
from model.sequential_alternating_optimization import sequential_alternating_optimization
from model.utils import compute_scale_regularization


def main():
    """
    Main function for running persistent Nonnegative Matrix Factorization (pNMF).

    Workflow:
    1. Parse command-line arguments.
    2. Load dataset and labels.
    3. Initialize factor matrices W, H and graph-related matrices (A, D, L).
    4. Perform iterative sequential alternating optimization over multiple scales.
    5. Save the resulting H matrices into a .mat file.
    """

    # -----------------------------
    # 1. Command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description='Persistent Nonnegative Matrix Factorization (pNMF)')
    parser.add_argument('--data', type=str, default='3D_concentric_circles',
                        help='Dataset name. Options: 3D_concentric_circles, GSE75748time, GSE94820'
                        'GSE75140, Dendritic_batch1, Dendritic_batch2')
    parser.add_argument('--l1', type=float, default=100,
                        help='Hyperparameter for the Geometric Regularization term.')
    parser.add_argument('--l2', type=float, default=100,
                        help='Hyperparameter for the Scale Smoothness Regularization term.')
    parser.add_argument('--l3', type=float, default=1,
                        help='Hyperparameter for the Scale Anchoring Regularization term.')
    args = parser.parse_args()

    # -----------------------------
    # 2. Load dataset
    # -----------------------------
    X, y = load_datasets(args.data)
    print(f"\nLoaded dataset '{args.data}' with shape X: {X.shape}")

    # -----------------------------
    # 3. Initialize factor matrices and graph structures
    # -----------------------------
    W_matrices, H_matrices, A, D, L, RG_loss = initialize(X, args.l1, args.data)
    print('\nDimension of W_matrices:', (len(W_matrices), *np.array(W_matrices[-1]).shape))
    print('\nDimension of H_matrices:', (len(H_matrices), *np.array(H_matrices[-1]).shape))

    # -----------------------------
    # 4. Iterative optimization
    # -----------------------------
    iteration = 0
    curr_loss = np.sum(RG_loss) + compute_scale_regularization(H_matrices, args.l2, args.l3)

    max_iterations = 10
    print("\nStarting sequential alternating optimization...\n")
    while iteration < max_iterations:
        loss_prev = curr_loss

        # Update W and H matrices across scales
        W_matrices, H_matrices, RG_loss = sequential_alternating_optimization(
            X, W_matrices, H_matrices, args.l1, args.l2, args.l3, A, D, L)
        
        # Compute total loss
        curr_loss = np.sum(RG_loss) + compute_scale_regularization(H_matrices, args.l2, args.l3)
        
        # Print iteration info
        print(f"Iteration {iteration}: Current loss = {curr_loss:.4f}, "
              f"Relative change = {np.abs(loss_prev - curr_loss) / loss_prev:.6f}")

        iteration += 1

    # -----------------------------
    # 5. Save results
    # -----------------------------
    result_dir = os.path.join('results', args.data)
    os.makedirs(result_dir, exist_ok=True)  # Ensure folder exists
    
    save_path = os.path.join(result_dir, 'pNMF.mat')
    sio.savemat(save_path, {'pNMF': H_matrices})
    print(f"\nOptimization finished.\n \nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
