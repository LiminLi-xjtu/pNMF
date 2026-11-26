"""NumPy/CuPy compatibility helpers for the pNMF solver.

CuPy is imported lazily.  A CPU-only installation therefore does not need a
CUDA runtime, while the numerical kernels can use the same source on either
backend.
"""

from typing import Any, Tuple

import numpy as np


def parse_device(device: str) -> Tuple[str, int]:
    """Parse ``auto``, ``cpu``, ``cuda`` or ``cuda:<id>`` device strings."""
    normalized = device.strip().lower()
    if normalized == "auto":
        try:
            cp = require_cupy()
            if cp.cuda.runtime.getDeviceCount() > 0:
                return "cuda", 0
        except Exception:
            pass
        return "cpu", 0
    if normalized == "cpu":
        return "cpu", 0
    if normalized == "cuda":
        return "cuda", 0
    if normalized.startswith("cuda:"):
        try:
            device_id = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid device {device!r}; expected cpu, cuda, or cuda:<id>."
            ) from exc
        if device_id < 0:
            raise ValueError("CUDA device id must be non-negative.")
        return "cuda", device_id
    raise ValueError(
        f"Invalid device {device!r}; expected auto, cpu, cuda, or cuda:<id>."
    )


def require_cupy():
    """Import CuPy with an actionable error when GPU support is unavailable."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU execution requires CuPy. For CUDA 12, install "
            "`python -m pip install cupy-cuda12x==13.6.0`."
        ) from exc
    return cp


def activate_device(device: str):
    """Select a device for subsequent operations and return its array module."""
    kind, device_id = parse_device(device)
    if kind == "cpu":
        return np
    cp = require_cupy()
    visible_devices = cp.cuda.runtime.getDeviceCount()
    if device_id >= visible_devices:
        raise RuntimeError(
            f"CUDA device {device_id} is not visible; "
            f"CuPy sees {visible_devices} device(s)."
        )
    cp.cuda.Device(device_id).use()
    return cp


def resolve_dtype(dtype: str, xp):
    """Resolve ``auto`` to float32 on CUDA and float64 on CPU."""
    normalized = dtype.strip().lower()
    if normalized == "auto":
        return xp.float64 if xp is np else xp.float32
    if normalized == "float32":
        return xp.float32
    if normalized == "float64":
        return xp.float64
    raise ValueError("dtype must be one of: auto, float32, float64")


def array_module(*arrays: Any):
    """Return CuPy when any input is a CuPy array, otherwise return NumPy."""
    for array in arrays:
        module_name = type(array).__module__.split(".", 1)[0]
        if module_name == "cupy":
            return require_cupy()
    return np


def to_numpy(array: Any) -> np.ndarray:
    """Return a NumPy view/copy suitable for SciPy and ``savemat``."""
    if type(array).__module__.split(".", 1)[0] == "cupy":
        return require_cupy().asnumpy(array)
    return np.asarray(array)


def scalar_value(value: Any) -> float:
    """Convert a NumPy/CuPy scalar or zero-dimensional array to ``float``."""
    if type(value).__module__.split(".", 1)[0] == "cupy":
        return float(value.item())
    return float(value)


def synchronize(xp) -> None:
    """Synchronize the active CUDA stream; no-op for NumPy."""
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def free_memory_pool(xp) -> None:
    """Release cached CuPy blocks after a large phase transition."""
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def backend_name(xp) -> str:
    return "numpy" if xp is np else "cupy"
