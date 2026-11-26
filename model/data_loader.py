"""Configuration-driven dataset loading for pNMF.

Each dataset lives in a named directory under its data root and supplies a
``config.json``.  Adding a dataset therefore does not require editing Python
source or adding a dataset-name branch to this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.io as sio


@dataclass(frozen=True)
class DatasetBundle:
    """A loaded dataset in the column-sample convention used by pNMF."""

    name: str
    X: np.ndarray
    labels: Optional[np.ndarray]
    config: Dict[str, Any]
    directory: Path


def _safe_dataset_directory(datasets_root: Path, dataset_name: str) -> Path:
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("dataset_name must be one directory name without separators.")
    root = datasets_root.expanduser().resolve()
    directory = (root / dataset_name).resolve()
    if directory.parent != root:
        raise ValueError("Dataset path must remain inside the datasets directory.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    return directory


def _safe_relative_file(directory: Path, relative_name: str) -> Path:
    if not relative_name:
        raise ValueError("Dataset file name cannot be empty.")
    path = (directory / relative_name).resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise ValueError("Dataset files must remain inside their dataset directory.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return path


def _csv_header(value):
    if value == "infer" or value is None or isinstance(value, int):
        return value
    raise ValueError("CSV header must be 'infer', null, or an integer row index.")


def _read_array(
    path: Path,
    *,
    key: Optional[str],
    role: str,
    config: Dict[str, Any],
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = Path(path.stem).suffix.lower()
        if suffix not in {".csv", ".tsv", ".txt"}:
            raise ValueError("Gzip input must contain a CSV, TSV, or TXT file.")
    if suffix == ".mat":
        if not key:
            raise ValueError(f"{role}_key is required for a MAT file.")
        payload = sio.loadmat(path)
        if key not in payload:
            visible = sorted(name for name in payload if not name.startswith("__"))
            raise KeyError(f"{key!r} is absent from {path.name}; available keys: {visible}")
        return np.asarray(payload[key])
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        if not key:
            raise ValueError(f"{role}_key is required for an NPZ file.")
        with np.load(path, allow_pickle=False) as payload:
            if key not in payload:
                raise KeyError(f"{key!r} is absent from {path.name}.")
            return np.asarray(payload[key])
    if suffix in {".csv", ".tsv", ".txt"}:
        default_separator = "\t" if suffix == ".tsv" else ","
        separator = config.get(f"{role}_delimiter", default_separator)
        frame = pd.read_csv(
            path,
            sep=separator,
            header=_csv_header(config.get(f"{role}_header", "infer")),
            index_col=config.get(f"{role}_index_col"),
        )
        if config.get(f"{role}_drop_first_column", False):
            frame = frame.iloc[:, 1:]
        selected_column = config.get(f"{role}_column")
        if selected_column is not None:
            if isinstance(selected_column, int):
                try:
                    frame = frame.iloc[:, [selected_column]]
                except IndexError as exc:
                    raise ValueError(
                        f"{role}_column index {selected_column} is outside "
                        f"{path.name}, which has {frame.shape[1]} columns."
                    ) from exc
            else:
                if selected_column not in frame.columns:
                    available = [str(column) for column in frame.columns]
                    raise KeyError(
                        f"{selected_column!r} is absent from {path.name}; "
                        f"available columns: {available}"
                    )
                frame = frame.loc[:, [selected_column]]
        return frame.to_numpy()
    raise ValueError(
        f"Unsupported {role} file type {suffix!r}; use MAT, NPY, NPZ, CSV, TSV, or TXT."
    )


def load_dataset_config(dataset_name: str, datasets_root: Path | str) -> Dict[str, Any]:
    """Load and minimally validate the dataset's config.json."""

    directory = _safe_dataset_directory(Path(datasets_root), dataset_name)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Dataset config.json must contain one JSON object.")
    if "data_file" not in config:
        raise ValueError("Dataset config.json must define 'data_file'.")
    return config


def _preprocess(X: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    transform = str(config.get("log_transform", "none")).lower()
    if transform == "none":
        pass
    elif transform == "log1p":
        X = np.log1p(X)
    elif transform in {"log10p1", "log10(1+x)"}:
        X = np.log10(1.0 + X)
    else:
        raise ValueError("log_transform must be one of: none, log1p, log10p1.")

    if config.get("column_l2_normalize", False):
        norms = np.linalg.norm(X, axis=0)
        if np.any(norms == 0.0):
            raise ValueError("Column L2 normalization is undefined for a zero column.")
        X = X / norms[None, :]
    return X


def load_dataset(dataset_name: str, datasets_root: Path | str) -> DatasetBundle:
    """Load a configured dataset and return ``X`` with shape ``(p, n)``."""

    datasets_root = Path(datasets_root)
    directory = _safe_dataset_directory(datasets_root, dataset_name)
    config = load_dataset_config(dataset_name, datasets_root)
    data_path = _safe_relative_file(directory, str(config["data_file"]))
    X = _read_array(
        data_path,
        key=config.get("data_key"),
        role="data",
        config=config,
    )
    if X.ndim != 2:
        raise ValueError(f"Loaded data must be two-dimensional; got shape {X.shape}.")

    layout = str(config.get("layout", "features_by_samples")).lower()
    if layout == "samples_by_features":
        X = X.T
    elif layout != "features_by_samples":
        raise ValueError("layout must be 'features_by_samples' or 'samples_by_features'.")
    X = _preprocess(X, config)

    if not np.all(np.isfinite(X)):
        raise ValueError("The data matrix contains NaN or infinite values.")
    negative_tolerance = float(config.get("negative_tolerance", 0.0))
    if negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be nonnegative.")
    if float(np.min(X)) < -negative_tolerance:
        raise ValueError("pNMF requires a nonnegative data matrix.")
    if negative_tolerance > 0.0:
        X = np.maximum(X, 0.0)
    if min(X.shape) == 0:
        raise ValueError("The data matrix cannot have an empty dimension.")

    labels = None
    label_key = config.get("label_key")
    label_file = config.get("label_file")
    if label_key is not None or label_file is not None:
        label_path = (
            data_path
            if label_file is None
            else _safe_relative_file(directory, str(label_file))
        )
        labels = _read_array(
            label_path,
            key=label_key,
            role="label",
            config=config,
        ).reshape(-1)
        if labels.size != X.shape[1]:
            raise ValueError(
                f"Expected {X.shape[1]} labels (one per sample); got {labels.size}."
            )
        if labels.dtype.kind not in "biufc" or config.get("factorize_labels", False):
            labels = pd.factorize(labels, sort=True)[0]
        labels = np.asarray(labels)

    return DatasetBundle(
        name=dataset_name,
        X=np.ascontiguousarray(X),
        labels=labels,
        config=config,
        directory=directory,
    )


__all__ = ["DatasetBundle", "load_dataset", "load_dataset_config"]
