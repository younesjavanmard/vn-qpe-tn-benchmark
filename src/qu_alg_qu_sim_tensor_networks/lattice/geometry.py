"""
Lattice geometries for MPS/MPO construction.

Design is inspired by Qiskit Nature's lattice module: each lattice is a thin
wrapper around a NetworkX graph whose nodes carry (x, y) position attributes.
The `mps_ordering` method maps the 2-D site labels to a 1-D MPS index sequence
using a Hilbert-curve ordering by default.

Supported geometries
--------------------
- SquareLattice
- TriangularLattice
- HoneycombLattice
- KagomeLattice
- HeavyHexLattice   (matches IBM heavy-hex connectivity pattern)
- CustomLattice

Usage
-----
>>> lat = SquareLattice(3, 4)
>>> lat.num_sites          # 12
>>> lat.bonds              # list of (i, j) in MPS index space
>>> lat.draw()             # requires matplotlib
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

from .ordering import xy_to_hilbert, _next_power_of_two


def _hilbert_sort(
    node_list: List[int], float_coords: List[Tuple[float, float]]
) -> List[int]:
    """
    Sort node_list by Hilbert-curve distance using float (x, y) positions.

    Scales float positions to the smallest integer grid that keeps all sites
    distinct — i.e. scale = 1 / min_gap, doubled until uniqueness holds.
    This preserves Hilbert locality (no spurious large-grid artefacts).
    """
    if not node_list:
        return []
    xs = [p[0] for p in float_coords]
    ys = [p[1] for p in float_coords]
    min_x, min_y = min(xs), min(ys)

    # Find the minimum non-zero gap between distinct coordinate values
    unique_xs = sorted(set(xs))
    unique_ys = sorted(set(ys))
    gaps_x = [b - a for a, b in zip(unique_xs, unique_xs[1:])] if len(unique_xs) > 1 else [1.0]
    gaps_y = [b - a for a, b in zip(unique_ys, unique_ys[1:])] if len(unique_ys) > 1 else [1.0]
    min_gap = min(min(gaps_x), min(gaps_y))

    # Iteratively double scale until all integer coords are unique
    scale = 1.0 / min_gap
    for _ in range(10):
        int_coords = [
            (int(round((x - min_x) * scale)), int(round((y - min_y) * scale)))
            for x, y in float_coords
        ]
        if len(set(int_coords)) == len(float_coords):
            break
        scale *= 2.0

    max_c = max(max(c[0] for c in int_coords), max(c[1] for c in int_coords), 1)
    N = _next_power_of_two(max_c + 1)
    return [
        node
        for _, node in sorted(
            zip(int_coords, node_list),
            key=lambda t: xy_to_hilbert(N, t[0][0], t[0][1]),
        )
    ]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Lattice(ABC):
    """
    Abstract base class for lattice geometries.

    Internally stores a NetworkX graph where each node is an integer label and
    carries a ``pos`` attribute with its (x, y) floating-point coordinates.

    After construction call ``set_ordering(method)`` (or pass ``ordering=`` to
    the subclass constructor) to fix the MPS site indices.

    Attributes
    ----------
    num_sites : int
    bonds : List[Tuple[int,int]]   -- (mps_i, mps_j) nearest-neighbour pairs
    site_coords : Dict[int, Tuple[float,float]]  -- mps_index → (x, y)
    coord_to_index : Dict[Tuple,int]             -- (x, y) → mps_index
    """

    def __init__(self, ordering: str = "hilbert"):
        if not _HAS_NX:
            raise ImportError(
                "networkx is required for the lattice module. "
                "Install it with:  pip install networkx"
            )
        self._graph: nx.Graph = nx.Graph()
        self._ordering_method = ordering
        self._build()
        self._apply_ordering(ordering)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build(self) -> None:
        """Populate self._graph with nodes (each with 'pos' attr) and edges."""

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def set_ordering(self, method: str) -> None:
        """Re-apply a different site ordering. method ∈ {'hilbert','row_major','snake'}."""
        _valid = {"hilbert", "row_major", "snake"}
        if method not in _valid:
            raise ValueError(f"Unknown ordering '{method}'. Choose from {_valid}")
        self._apply_ordering(method)
        self._ordering_method = method

    def _apply_ordering(self, method: str) -> None:
        node_list: List[int] = list(self._graph.nodes)
        float_coords: List[Tuple[float, float]] = [
            self._graph.nodes[n]["pos"] for n in node_list
        ]

        if method == "hilbert":
            ordered_nodes = _hilbert_sort(node_list, float_coords)
        elif method == "row_major":
            ordered_nodes = [
                n for _, n in sorted(
                    zip(float_coords, node_list), key=lambda t: (t[0][1], t[0][0])
                )
            ]
        elif method == "snake":
            ordered_nodes = [
                n for _, n in sorted(
                    zip(float_coords, node_list),
                    key=lambda t: (t[0][1], t[0][0] if round(t[0][1]) % 2 == 0 else -t[0][0]),
                )
            ]
        else:
            raise ValueError(f"Unknown ordering '{method}'")

        self._mps_to_node: List[int] = ordered_nodes
        self._node_to_mps: Dict[int, int] = {
            node: mps_idx for mps_idx, node in enumerate(ordered_nodes)
        }

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_sites(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def graph(self) -> "nx.Graph":
        return self._graph

    @property
    def ordering(self) -> str:
        return self._ordering_method

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        """Nearest-neighbour bonds as (mps_i, mps_j) pairs, i < j."""
        result = []
        for u, v in self._graph.edges():
            a = self._node_to_mps[u]
            b = self._node_to_mps[v]
            result.append((min(a, b), max(a, b)))
        return sorted(result)

    @property
    def site_coords(self) -> Dict[int, Tuple[float, float]]:
        """mps_index → (x, y) position."""
        return {
            mps_idx: self._graph.nodes[node]["pos"]
            for mps_idx, node in enumerate(self._mps_to_node)
        }

    @property
    def coord_to_index(self) -> Dict[Tuple[float, float], int]:
        """(x, y) → mps_index."""
        return {pos: mps_idx for mps_idx, pos in self.site_coords.items()}

    def node_to_mps(self, node: int) -> int:
        """Convert internal graph node label to MPS site index."""
        return self._node_to_mps[node]

    def mps_to_node(self, mps_idx: int) -> int:
        """Convert MPS site index to internal graph node label."""
        return self._mps_to_node[mps_idx]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def max_bond_length(self) -> int:
        """Maximum |i - j| over all bonds in MPS ordering."""
        return max(abs(b - a) for a, b in self.bonds) if self.bonds else 0

    def bond_length_histogram(self) -> Dict[int, int]:
        """Count how many bonds have each |i - j| distance."""
        hist: Dict[int, int] = {}
        for a, b in self.bonds:
            d = abs(b - a)
            hist[d] = hist.get(d, 0) + 1
        return dict(sorted(hist.items()))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(
        self,
        ax=None,
        node_size: int = 300,
        with_mps_labels: bool = True,
        **kwargs,
    ) -> None:
        """
        Draw the lattice using matplotlib + networkx.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        node_size : int
        with_mps_labels : bool
            If True labels show MPS index; otherwise the internal node label.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for drawing. pip install matplotlib")

        pos = nx.get_node_attributes(self._graph, "pos")
        labels = (
            {node: self._node_to_mps[node] for node in self._graph.nodes}
            if with_mps_labels
            else None
        )
        if ax is None:
            _, ax = plt.subplots()
        nx.draw(
            self._graph,
            pos=pos,
            ax=ax,
            labels=labels,
            node_size=node_size,
            node_color="steelblue",
            font_color="white",
            font_size=8,
            **kwargs,
        )
        ax.set_title(f"{type(self).__name__} — ordering: {self._ordering_method}")

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} sites={self.num_sites} "
            f"bonds={len(self.bonds)} ordering={self._ordering_method}>"
        )


# ---------------------------------------------------------------------------
# Square lattice
# ---------------------------------------------------------------------------

class SquareLattice(Lattice):
    """
    2-D square lattice with nearest-neighbour bonds.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (rows = y-size, cols = x-size).
    boundary : str
        'open' or 'periodic'.
    ordering : str
        Site ordering for MPS ('hilbert', 'row_major', 'snake').
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        boundary: str = "open",
        ordering: str = "hilbert",
    ):
        self.rows = rows
        self.cols = cols
        self.boundary = boundary
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                node = r * self.cols + c
                self._graph.add_node(node, pos=(float(c), float(r)))

        for r in range(self.rows):
            for c in range(self.cols):
                n = r * self.cols + c
                if c + 1 < self.cols:
                    self._graph.add_edge(n, n + 1)
                elif self.boundary == "periodic":
                    self._graph.add_edge(n, r * self.cols)
                if r + 1 < self.rows:
                    self._graph.add_edge(n, n + self.cols)
                elif self.boundary == "periodic":
                    self._graph.add_edge(n, c)


# ---------------------------------------------------------------------------
# Triangular lattice
# ---------------------------------------------------------------------------

class TriangularLattice(Lattice):
    """
    2-D triangular lattice (nearest-neighbour bonds include the diagonal).

    Each site (r, c) connects to: right, up, and upper-right.

    Parameters
    ----------
    rows, cols : int
    boundary : str  'open' | 'periodic'
    ordering : str
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        boundary: str = "open",
        ordering: str = "hilbert",
    ):
        self.rows = rows
        self.cols = cols
        self.boundary = boundary
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                node = r * self.cols + c
                # Offset every other row by 0.5 for visual clarity
                x = float(c) + (0.5 if r % 2 else 0.0)
                y = float(r) * (3 ** 0.5 / 2)
                self._graph.add_node(node, pos=(x, y))

        for r in range(self.rows):
            for c in range(self.cols):
                n = r * self.cols + c
                # Horizontal bond
                right = r * self.cols + (c + 1) % self.cols
                if c + 1 < self.cols or self.boundary == "periodic":
                    self._graph.add_edge(n, right)
                # Vertical / diagonal bonds to row above
                if r + 1 < self.rows:
                    up = (r + 1) * self.cols + c
                    self._graph.add_edge(n, up)
                    up_right = (r + 1) * self.cols + (c + 1) % self.cols
                    if c + 1 < self.cols or self.boundary == "periodic":
                        self._graph.add_edge(n, up_right)
                elif self.boundary == "periodic":
                    up = c
                    self._graph.add_edge(n, up)


# ---------------------------------------------------------------------------
# Honeycomb lattice
# ---------------------------------------------------------------------------

class HoneycombLattice(Lattice):
    """
    2-D honeycomb (hexagonal) lattice.

    Unit cell has 2 sites (A and B sublattice).  Dimensions are in unit cells:
    ``rows`` unit-cell rows, ``cols`` unit-cell columns → 2 * rows * cols sites.

    Parameters
    ----------
    rows, cols : int   (unit-cell counts)
    boundary : str     'open' | 'periodic'
    ordering : str
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        boundary: str = "open",
        ordering: str = "hilbert",
    ):
        self.rows = rows
        self.cols = cols
        self.boundary = boundary
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        # Site A: index 2*(r*cols+c), Site B: index 2*(r*cols+c)+1
        sqrt3 = 3 ** 0.5
        for r in range(self.rows):
            for c in range(self.cols):
                a_idx = 2 * (r * self.cols + c)
                b_idx = a_idx + 1
                x_base = 3.0 * c
                y_base = sqrt3 * r
                self._graph.add_node(a_idx, pos=(x_base, y_base))
                self._graph.add_node(b_idx, pos=(x_base + 1.0, y_base))
                # A-B bond within unit cell
                self._graph.add_edge(a_idx, b_idx)

        for r in range(self.rows):
            for c in range(self.cols):
                b_idx = 2 * (r * self.cols + c) + 1
                # B → A right neighbour
                cr = (c + 1) % self.cols
                a_right = 2 * (r * self.cols + cr)
                if c + 1 < self.cols or self.boundary == "periodic":
                    self._graph.add_edge(b_idx, a_right)
                # B → A upper neighbour
                rr = (r + 1) % self.rows
                a_up = 2 * (rr * self.cols + c)
                if r + 1 < self.rows or self.boundary == "periodic":
                    self._graph.add_edge(b_idx, a_up)


# ---------------------------------------------------------------------------
# Kagome lattice
# ---------------------------------------------------------------------------

class KagomeLattice(Lattice):
    """
    2-D Kagome lattice.

    Unit cell has 3 sites.  Dimensions are in unit cells:
    ``rows`` × ``cols`` unit cells → 3 * rows * cols sites.

    Parameters
    ----------
    rows, cols : int
    boundary : str  'open' | 'periodic'
    ordering : str
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        boundary: str = "open",
        ordering: str = "hilbert",
    ):
        self.rows = rows
        self.cols = cols
        self.boundary = boundary
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        sqrt3 = 3 ** 0.5
        # Unit cell: A at (0,0), B at (1,0), C at (0.5, sqrt3/2)
        offsets = [(0.0, 0.0), (1.0, 0.0), (0.5, sqrt3 / 2.0)]

        def _idx(r, c, s):
            return 3 * (r * self.cols + c) + s

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = 2.0 * c
                y0 = sqrt3 * r
                for s, (dx, dy) in enumerate(offsets):
                    self._graph.add_node(_idx(r, c, s), pos=(x0 + dx, y0 + dy))
                # Intra-cell bonds: A-B, A-C, B-C
                for sa, sb in [(0, 1), (0, 2), (1, 2)]:
                    self._graph.add_edge(_idx(r, c, sa), _idx(r, c, sb))

        for r in range(self.rows):
            for c in range(self.cols):
                cr = (c + 1) % self.cols
                rr = (r + 1) % self.rows
                # B of (r,c) → A of (r, c+1)
                if c + 1 < self.cols or self.boundary == "periodic":
                    self._graph.add_edge(_idx(r, c, 1), _idx(r, cr, 0))
                # C of (r,c) → A of (r+1, c)
                if r + 1 < self.rows or self.boundary == "periodic":
                    self._graph.add_edge(_idx(r, c, 2), _idx(rr, c, 0))
                # C of (r,c) → B of (r+1, c-1)
                cl = (c - 1) % self.cols
                if r + 1 < self.rows and (c - 1 >= 0 or self.boundary == "periodic"):
                    self._graph.add_edge(_idx(r, c, 2), _idx(rr, cl, 1))


# ---------------------------------------------------------------------------
# Heavy-Hex lattice  (IBM architecture)
# ---------------------------------------------------------------------------

class HeavyHexLattice(Lattice):
    """
    IBM heavy-hexagonal lattice.

    Matches the connectivity pattern used in IBM Falcon, Eagle, and Heron
    processors.  All qubits have degree ≤ 3 (bipartite, no triangles).

    Layout
    ------
    ``rows`` backbone chains, each with ``cols`` qubits.
    Between adjacent chains, vertical bridge bonds are added at alternating
    column positions (period-3 pattern, offset alternates by row pair):

        Even row-pairs: bridges at cols  2, 5, 8, ...
        Odd  row-pairs: bridges at cols  0, 3, 6, ...  (shifted by -2)

    This reproduces the IBM heavy-hex unit cell with a spacing of 3.

    Parameters
    ----------
    rows : int   Number of backbone qubit chains (≥ 1).
    cols : int   Number of qubits per chain (≥ 1).
    ordering : str
    """

    def __init__(self, rows: int, cols: int, ordering: str = "hilbert"):
        self.rows = rows
        self.cols = cols
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                node = r * self.cols + c
                self._graph.add_node(node, pos=(float(c), float(r)))

        # Intra-row (horizontal) bonds
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self._graph.add_edge(r * self.cols + c, r * self.cols + c + 1)

        # Inter-row (vertical bridge) bonds — IBM heavy-hex pattern
        for r in range(self.rows - 1):
            # Bridges alternate: even row-pair offset 2, odd row-pair offset 0
            offset = 2 if r % 2 == 0 else 0
            period = 3
            c = offset
            while c < self.cols:
                self._graph.add_edge(r * self.cols + c, (r + 1) * self.cols + c)
                c += period

    @property
    def num_qubits(self) -> int:
        """Alias for num_sites (IBM terminology)."""
        return self.num_sites


# ---------------------------------------------------------------------------
# Custom lattice
# ---------------------------------------------------------------------------

class CustomLattice(Lattice):
    """
    Lattice defined by an explicit list of sites and bonds.

    Parameters
    ----------
    sites : sequence of (x, y)
        Coordinates of each site (0-based labels in list order).
    bonds : sequence of (i, j)
        Edges between site indices (matching ``sites`` list order).
    ordering : str
        'hilbert' | 'row_major' | 'snake'
    """

    def __init__(
        self,
        sites: Sequence[Tuple[float, float]],
        bonds: Sequence[Tuple[int, int]],
        ordering: str = "hilbert",
    ):
        self._raw_sites = list(sites)
        self._raw_bonds = list(bonds)
        super().__init__(ordering=ordering)

    def _build(self) -> None:
        for i, (x, y) in enumerate(self._raw_sites):
            self._graph.add_node(i, pos=(float(x), float(y)))
        for u, v in self._raw_bonds:
            self._graph.add_edge(u, v)
