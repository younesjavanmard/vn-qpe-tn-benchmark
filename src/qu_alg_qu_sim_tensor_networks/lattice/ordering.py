"""
Hilbert-curve and alternative site orderings for MPS/MPO.

For non-power-of-2 grids the algorithm embeds the lattice into the smallest
enclosing 2^k × 2^k grid, sorts all sites by their Hilbert distance, and
keeps relative order — this minimises the length of long-range bonds compared
to row-major or column-major orderings.
"""

from __future__ import annotations
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core Hilbert-curve primitives (no external dependency)
# ---------------------------------------------------------------------------

def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def xy_to_hilbert(n: int, x: int, y: int) -> int:
    """Hilbert distance for point (x, y) in an n×n grid (n must be a power of 2)."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_to_xy(n: int, d: int) -> Tuple[int, int]:
    """Inverse: Hilbert distance → (x, y) for an n×n grid (n power of 2)."""
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 if (t & 2) else 0
        ry = 1 if (t & 1) ^ rx else 0
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _next_power_of_two(k: int) -> int:
    n = 1
    while n < k:
        n <<= 1
    return n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hilbert_order(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sort a list of (x, y) lattice coordinates by Hilbert-curve distance.

    Works for arbitrary rectangular regions — not just power-of-2 grids.
    The enclosing grid size is chosen as 2^k ≥ max(max_x+1, max_y+1).

    Parameters
    ----------
    coords : list of (int, int)
        Site coordinates to order.

    Returns
    -------
    List[Tuple[int, int]]
        Same coordinates, sorted by Hilbert distance (ascending).
    """
    if not coords:
        return []
    max_x = max(p[0] for p in coords)
    max_y = max(p[1] for p in coords)
    N = _next_power_of_two(max(max_x + 1, max_y + 1, 2))
    return sorted(coords, key=lambda p: xy_to_hilbert(N, p[0], p[1]))


def row_major_order(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort coordinates in row-major (y first, then x) order."""
    return sorted(coords, key=lambda p: (p[1], p[0]))


def snake_order(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Boustrophedon (snake) ordering: left-to-right for even rows,
    right-to-left for odd rows.  Minimises nearest-neighbour bond length for
    regular grids, but Hilbert is usually better for 2D.
    """
    return sorted(coords, key=lambda p: (p[1], p[0] if p[1] % 2 == 0 else -p[0]))
