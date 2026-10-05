from __future__ import annotations
import os
import numpy as np

# Set DMRG_BACKEND (or BACKEND) to "numpy" | "cupy" | "jax" | "auto"
BACKEND = os.environ.get("DMRG_BACKEND", os.environ.get("BACKEND", "auto"))

_has_cupy = False
_has_jax = False
cp = None
jnp = None

if BACKEND in ("auto", "cupy"):
    try:
        import cupy as cp
        _has_cupy = True
    except Exception:
        pass

if BACKEND in ("auto", "jax"):
    try:
        import jax
        import jax.numpy as jnp
        _has_jax = True
    except Exception:
        pass


def _select():
    if BACKEND == "numpy":
        return "numpy"
    if BACKEND == "cupy":
        return "cupy" if _has_cupy else "numpy"
    if BACKEND == "jax":
        return "jax" if _has_jax else "numpy"
    # auto
    if _has_cupy:
        return "cupy"
    if _has_jax:
        return "jax"
    return "numpy"


BACK = _select()

if BACK == "cupy":
    xp = cp
elif BACK == "jax":
    xp = jnp
else:
    xp = np


def get_array_module(arr):
    """Return the array module (numpy/cupy) matching the given array."""
    if _has_cupy:
        return cp.get_array_module(arr)
    return np


def to_device(arr):
    """Move a numpy array to the active compute device."""
    if BACK == "cupy":
        return cp.asarray(arr)
    if BACK == "jax":
        import jax
        return jax.device_put(np.asarray(arr))
    return arr


def to_numpy(arr):
    """Move an array from the compute device back to numpy."""
    if BACK == "cupy":
        return cp.asnumpy(arr)
    if BACK == "jax":
        return np.asarray(arr)
    return np.asarray(arr)
