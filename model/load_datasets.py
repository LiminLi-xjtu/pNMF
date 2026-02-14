import os
import numpy as np
import scipy.io as sio
import pandas as pd


def load_datasets(dataname: str):
    """
    Load dataset by name.

    Supports:
    - Simulation dataset: '3D_concentric_circles'
    - Dendritic datasets: 'Dendritic_batch1', 'Dendritic_batch2'
    - Other datasets: 'GSE75748time', 'GSE94820', 'GSE75140'

    Parameters
    ----------
    dataname : str
        Name of the dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix (features x samples)
    y : np.ndarray
        Labels vector
    """

    # -----------------------------
    # 1. Simulation dataset
    # -----------------------------
    if dataname == '3D_concentric_circles':
        mat_path = os.path.join('datasets', 'Simulation', f'{dataname}.mat')
        mat = sio.loadmat(mat_path)
        X = mat['X'].T  # Transpose so columns are samples
        y = mat['Y'].reshape(-1)

    # -----------------------------
    # 2. Dendritic batch datasets
    # -----------------------------
    elif dataname in ['Dendritic_batch1', 'Dendritic_batch2']:
        data_path = os.path.join('datasets', dataname, f'{dataname}.csv')
        X = pd.read_csv(data_path, index_col=0)
        X = X.values.T  # Transpose so columns are samples
        X = np.log10(1 + X)  # Log-transform
        X = X / np.linalg.norm(X, axis=0)[None, :]  # Column normalization

        label_path = os.path.join('datasets', dataname, f'{dataname}label.txt')
        y = pd.read_csv(label_path).squeeze().x  # Extract series
        y = pd.factorize(y)[0]  # Convert to integer labels

    # -----------------------------
    # 3. Other datasets
    # -----------------------------
    else:
        data_path = os.path.join('datasets', dataname, f'{dataname}_data.csv')
        X = pd.read_csv(data_path)
        X = X.values[:, 1:].astype(float)  # Skip first column if it's index
        X = np.log10(1 + X)
        X = X / np.linalg.norm(X, axis=0)[None, :]  # Column normalization

        label_path = os.path.join('datasets', dataname, f'{dataname}_labels.csv')
        y = pd.read_csv(label_path)
        y = np.array(list(y['Label'])).astype(int)

    return X, y