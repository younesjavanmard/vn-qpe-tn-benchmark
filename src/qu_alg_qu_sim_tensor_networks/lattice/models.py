"""
Lattice Hamiltonian models.

Each function takes a ``Lattice`` object and returns a ``Hamiltonian``
(the Pauli-string dict used by ``get_mpo``).

Spin models (qubit = spin-1/2)
-------------------------------
- tfi          : Transverse-Field Ising
- heisenberg   : Heisenberg (general Jx, Jy, Jz)
- xxz          : XXZ  (J, Delta)

Fermionic models (Jordan-Wigner mapping)
-----------------------------------------
- hubbard      : Fermi-Hubbard  (spinful, 2 qubits/site)
- fkm          : Falicov-Kimball (1 itinerant + 1 localised fermion / site)

For fermionic models the qubit register is *interleaved*:
    qubit 2*i   = spin-up (or itinerant) fermion at lattice site i
    qubit 2*i+1 = spin-down (or localised) fermion at lattice site i

where i is the MPS site index assigned by the lattice ordering.
"""

from __future__ import annotations

from typing import List, Tuple

from qu_alg_qu_sim_tensor_networks.hamiltonians import Hamiltonian
from .geometry import Lattice


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jw_hop(H: Hamiltonian, qubit_i: int, qubit_j: int, t: float) -> None:
    """
    Add the Jordan-Wigner hopping  -t (c†_i c_j + h.c.)  to H.

    For i < j:
        c†_i c_j + h.c. = (1/2)[X_i (∏_{k=i+1}^{j-1} Z_k) X_j
                                + Y_i (∏_{k=i+1}^{j-1} Z_k) Y_j]
    Coefficient absorbed: -t * (1/2).
    """
    if qubit_i == qubit_j:
        return
    if qubit_i > qubit_j:
        qubit_i, qubit_j = qubit_j, qubit_i

    jw = [(k, "Z") for k in range(qubit_i + 1, qubit_j)]
    coeff = -t / 2.0
    H.add_term([(qubit_i, "X")] + jw + [(qubit_j, "X")], coeff)
    H.add_term([(qubit_i, "Y")] + jw + [(qubit_j, "Y")], coeff)


def _density_density(H: Hamiltonian, qi: int, qj: int, U: float) -> None:
    """
    Add  U * n_i * n_j  where n_k = (I - Z_k)/2.

    Expands to U/4 * (I - Z_i - Z_j + Z_i Z_j).
    The identity term shifts the energy by U/4 and is stored as 'I…I'.
    """
    n = H.n
    identity_str = "I" * n
    H.terms[identity_str] = H.terms.get(identity_str, 0.0) + U / 4.0
    H.add_term([(qi, "Z")], -U / 4.0)
    H.add_term([(qj, "Z")], -U / 4.0)
    H.add_term([(qi, "Z"), (qj, "Z")], U / 4.0)


# ---------------------------------------------------------------------------
# Spin models
# ---------------------------------------------------------------------------

def tfi(
    lattice: Lattice,
    J: float = 1.0,
    h: float = 1.0,
    boundary: str = "open",
) -> Hamiltonian:
    """
    Transverse-Field Ising model.

    H = -J Σ_{<i,j>} Z_i Z_j  -  h Σ_i X_i

    Parameters
    ----------
    lattice : Lattice
    J : float   ZZ coupling (positive = ferromagnetic).
    h : float   Transverse field.
    boundary : str  'open' or 'periodic' (for bond wrapping on custom lattices,
        this parameter is currently unused — boundary is encoded in the lattice
        graph itself).
    """
    n = lattice.num_sites
    H = Hamiltonian(n)
    for i, j in lattice.bonds:
        H.add_term([(i, "Z"), (j, "Z")], -J)
    for i in range(n):
        H.add_term([(i, "X")], -h)
    return H


def heisenberg(
    lattice: Lattice,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
) -> Hamiltonian:
    """
    Heisenberg model.

    H = Σ_{<i,j>} (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j)
    """
    n = lattice.num_sites
    H = Hamiltonian(n)
    for i, j in lattice.bonds:
        H.add_term([(i, "X"), (j, "X")], Jx)
        H.add_term([(i, "Y"), (j, "Y")], Jy)
        H.add_term([(i, "Z"), (j, "Z")], Jz)
    return H


def xxz(
    lattice: Lattice,
    J: float = 1.0,
    Delta: float = 1.0,
) -> Hamiltonian:
    """
    XXZ model.

    H = J Σ_{<i,j>} (X_i X_j + Y_i Y_j + Δ Z_i Z_j)

    Δ=1 → isotropic Heisenberg, Δ=0 → XX model, Δ→∞ → Ising.
    """
    return heisenberg(lattice, Jx=J, Jy=J, Jz=J * Delta)


# ---------------------------------------------------------------------------
# Fermionic models (Jordan-Wigner)
# ---------------------------------------------------------------------------

def hubbard(
    lattice: Lattice,
    t: float = 1.0,
    U: float = 4.0,
) -> Hamiltonian:
    """
    Fermi-Hubbard model (spinful).

    H = -t Σ_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.)  +  U Σ_i n_{i↑} n_{i↓}

    Qubit layout (interleaved):
        qubit 2*i   → site i, spin-up
        qubit 2*i+1 → site i, spin-down

    Total qubits: 2 * lattice.num_sites.

    Parameters
    ----------
    lattice : Lattice
    t : float   Hopping amplitude (positive).
    U : float   On-site Coulomb repulsion.
    """
    n_sites = lattice.num_sites
    n_qubits = 2 * n_sites
    H = Hamiltonian(n_qubits)

    for i, j in lattice.bonds:
        # Spin-up hopping: qubits 2i → 2j
        _jw_hop(H, 2 * i, 2 * j, t)
        # Spin-down hopping: qubits 2i+1 → 2j+1
        _jw_hop(H, 2 * i + 1, 2 * j + 1, t)

    for i in range(n_sites):
        # On-site interaction U n_{i↑} n_{i↓}
        _density_density(H, 2 * i, 2 * i + 1, U)

    return H


def fkm(
    lattice: Lattice,
    t: float = 1.0,
    U: float = 4.0,
) -> Hamiltonian:
    """
    Falicov-Kimball model.

    H = -t Σ_{<i,j>} (c†_i c_j + h.c.)  +  U Σ_i n^c_i n^f_i

    c_i: itinerant (mobile) fermions.
    f_i: localised fermions (no hopping term).

    Qubit layout (interleaved):
        qubit 2*i   → site i, itinerant (c)
        qubit 2*i+1 → site i, localised  (f)

    Total qubits: 2 * lattice.num_sites.

    Parameters
    ----------
    lattice : Lattice
    t : float   Hopping amplitude for itinerant fermions.
    U : float   Coupling between itinerant and localised densities.
    """
    n_sites = lattice.num_sites
    n_qubits = 2 * n_sites
    H = Hamiltonian(n_qubits)

    for i, j in lattice.bonds:
        # Only itinerant fermions hop: qubits 2i → 2j
        _jw_hop(H, 2 * i, 2 * j, t)

    for i in range(n_sites):
        # Interaction U n^c_i n^f_i
        _density_density(H, 2 * i, 2 * i + 1, U)

    return H
