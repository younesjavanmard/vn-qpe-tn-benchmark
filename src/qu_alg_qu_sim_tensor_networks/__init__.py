# SPDX-License-Identifier: MIT
"""Top-level package for qu_alg_qu_sim_tensor_networks."""

from __future__ import annotations

try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # backport

try:
    __version__ = version("qu_alg_qu_sim_tensor_networks")
except PackageNotFoundError:  # e.g., running from source without install
    __version__ = "0.0.0"

from qu_alg_qu_sim_tensor_networks import lattice  # noqa: F401

__all__ = ["__version__", "lattice"]
