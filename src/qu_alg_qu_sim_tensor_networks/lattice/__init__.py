"""
Lattice module — Qiskit-Nature-inspired lattice geometries and Hamiltonian models.

Geometries
----------
SquareLattice, TriangularLattice, HoneycombLattice, KagomeLattice,
HeavyHexLattice (IBM heavy-hex), CustomLattice

Models  (return a ``Hamiltonian`` ready for ``get_mpo``)
---------------------------------------------------------
tfi, heisenberg, xxz, hubbard, fkm

Ordering
--------
All lattices support 'hilbert' (default), 'row_major', and 'snake' MPS
site orderings.  Hilbert-curve ordering minimises long-range bond lengths
for 2-D systems and works for non-power-of-2 grids.

Quick start
-----------
>>> from qu_alg_qu_sim_tensor_networks.lattice import SquareLattice, heisenberg
>>> lat = SquareLattice(4, 4)
>>> H = heisenberg(lat, Jx=1.0, Jy=1.0, Jz=1.0)
>>> print(lat.max_bond_length())
>>> lat.draw()
"""

from .geometry import (
    Lattice,
    SquareLattice,
    TriangularLattice,
    HoneycombLattice,
    KagomeLattice,
    HeavyHexLattice,
    CustomLattice,
)
from .models import tfi, heisenberg, xxz, hubbard, fkm
from .ordering import hilbert_order, row_major_order, snake_order

__all__ = [
    # Geometries
    "Lattice",
    "SquareLattice",
    "TriangularLattice",
    "HoneycombLattice",
    "KagomeLattice",
    "HeavyHexLattice",
    "CustomLattice",
    # Models
    "tfi",
    "heisenberg",
    "xxz",
    "hubbard",
    "fkm",
    # Ordering utilities
    "hilbert_order",
    "row_major_order",
    "snake_order",
]
