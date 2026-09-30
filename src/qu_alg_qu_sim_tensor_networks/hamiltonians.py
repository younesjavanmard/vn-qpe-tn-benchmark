"""
pauli_hamiltonians.py
---------------------

Core utilities for constructing Pauli-string Hamiltonians, performing small-scale
exact diagonalization (ED), evaluating Pauli expectations, and building MPOs via
your `qu_alg_qu_sim_tensor_networks` back-end.

This module intentionally preserves the original public interface while adding:
- Rich docstrings and inline documentation
- Light input validation and clearer error messages
- Tidier layout and consistent naming/style

Public API (unchanged):
- Class: Hamiltonian
- Functions:
    ed_ground_state, pauli_expectation, pauli_digits_to_string, random_pauli_string,
    operator_from_pauli_string, operator_from_pauli_ops, build_pauli_mpo_from_string,
    build_pauli_mpo_from_digits, get_mpo, from_openfermion,
    heisenberg_chain, transverse_field_ising, cluster_hamiltonian,
    build_H_odd, build_string_op
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterator, Any, Literal, Sequence
from itertools import product  # (kept; used by some users externally)
import numpy as np


from qu_alg_qu_sim_tensor_networks import graph_builder_new_version_just_to_keep_it as gb
from qu_alg_qu_sim_tensor_networks import mpo_builder as qmpo


# ============================ Pauli base matrices ============================

PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PI = np.eye(2, dtype=complex)

# Map single-letter Pauli to 2x2 matrices. Kept as a dict for simplicity.
P_MAP: Dict[str, np.ndarray] = {"I": PI, "X": PX, "Y": PY, "Z": PZ}

# Base-4 digits -> Pauli lookup
_DIGIT2P = np.array(["I", "X", "Y", "Z"], dtype="<U1")


# ============================== Hamiltonian class =============================

class Hamiltonian:
    """
    A Pauli-string Hamiltonian on `n` qubits stored as a dict[str, float].

    Each term is represented by a length-`n` string over {'I','X','Y','Z'}.
    Coefficients are real or complex numbers (floats used by default).

    Notes
    -----
    - Addition and in-place addition merge coefficients on matching strings.
    - Scalar multiplication scales all coefficients.
    - `to_dense()` builds the (2^n)x(2^n) dense matrix; use only for small `n`.
    - Iteration yields (term, coeff) pairs.

    Parameters
    ----------
    n_qubits
        Number of qubits / sites.
    """
    __array_priority__ = 1000  # ensure Python ops take precedence over NumPy ufuncs

    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError(f"`n_qubits` must be a positive int, got {n_qubits!r}")
        self.n = n_qubits
        self.terms: Dict[str, float] = {}

    # ----------------------------- Construction -----------------------------

    def add_term(self, pauli_ops: List[Tuple[int, str]], coeff: float) -> None:
        """
        Add a Pauli-product term.

        Parameters
        ----------
        pauli_ops
            List of (site_index, 'X'|'Y'|'Z'). 'I' sites are omitted implicitly.
        coeff
            Scalar coefficient for that Pauli product.

        Notes
        -----
        - If the term already exists, coefficients accumulate.
        - Any site not mentioned is assumed to be an identity 'I'.
        """
        s = ["I"] * self.n
        for idx, P in pauli_ops:
            if not (0 <= idx < self.n):
                raise IndexError(f"Qubit index {idx} out of range [0,{self.n-1}]")
            if P not in P_MAP or P == "I":
                raise ValueError(f"Invalid Pauli '{P}'. Use only 'X','Y','Z' here.")
            s[idx] = P
        key = "".join(s)
        self.terms[key] = self.terms.get(key, 0.0) + coeff

    # ------------------------------ Properties ------------------------------

    @property
    def ham_terms(self) -> List[str]:
        """List of Pauli-string keys in insertion order."""
        return list(self.terms.keys())

    @property
    def coeff_list(self) -> List[float]:
        """List of coefficients aligned with `ham_terms` order."""
        return list(self.terms.values())

    @property
    def as_dict(self) -> Dict[str, float]:
        """A shallow copy of the term→coefficient mapping."""
        return dict(self.terms)

    @property
    def term_indices(self) -> Dict[str, int]:
        """Map term→sequential index (0..len-1) in current ordering."""
        return {term: idx for idx, term in enumerate(self.ham_terms)}

    # -------------------------- Container Protocols -------------------------

    def __repr__(self) -> str:
        return f"<Hamiltonian n_qubits={self.n} n_terms={len(self)}>"

    def __str__(self) -> str:
        if not self.terms:
            return "Hamiltonian: <empty>"
        lines = [f"{term}: {coef}" for term, coef in self.terms.items()]
        return "Hamiltonian:\n  " + "\n  ".join(lines)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        return iter(self.terms.items())

    def __contains__(self, term: str) -> bool:
        return term in self.terms

    def __getitem__(self, term: str) -> float:
        return self.terms.get(term, 0.0)

    def __setitem__(self, term: str, coeff: float) -> None:
        if not isinstance(term, str) or len(term) != self.n:
            raise ValueError(
                f"Pauli string length mismatch: expected {self.n}, got {len(term) if isinstance(term,str) else term!r}"
            )
        # Validate characters
        bad = [ch for ch in term if ch not in P_MAP]
        if bad:
            raise ValueError(f"Invalid character(s) in term: {bad} (allowed: I,X,Y,Z)")
        self.terms[term] = coeff

    # ------------------------------ Arithmetic ------------------------------

    def __add__(self, other: Any) -> Any:
        if not isinstance(other, Hamiltonian) or self.n != other.n:
            return NotImplemented
        H = Hamiltonian(self.n)
        for t, c in self.terms.items():
            H.terms[t] = c
        for t, c in other.terms.items():
            H.terms[t] = H.terms.get(t, 0.0) + c
        return H

    def __iadd__(self, other: Any) -> Any:
        if not isinstance(other, Hamiltonian) or self.n != other.n:
            return NotImplemented
        for t, c in other.terms.items():
            self.terms[t] = self.terms.get(t, 0.0) + c
        return self

    def __neg__(self) -> "Hamiltonian":
        H = Hamiltonian(self.n)
        for t, c in self.terms.items():
            H.terms[t] = -c
        return H

    def __sub__(self, other: Any) -> Any:
        return self.__add__(-other)

    def __mul__(self, scalar: Any) -> Any:
        # allow NumPy/scalar-like values
        try:
            _ = scalar + 0
        except Exception:
            return NotImplemented
        H = Hamiltonian(self.n)
        for t, c in self.terms.items():
            H.terms[t] = c * scalar
        return H

    def __rmul__(self, scalar: Any) -> Any:
        return self.__mul__(scalar)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Hamiltonian) and self.n == other.n and self.terms == other.terms

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        # Only support scalar multiplication via NumPy ufuncs
        if method != "__call__":
            return NotImplemented
        if ufunc not in (np.multiply,):
            return NotImplemented
        a, b = inputs
        if isinstance(a, Hamiltonian) and not isinstance(b, Hamiltonian):
            return a * b
        if isinstance(b, Hamiltonian) and not isinstance(a, Hamiltonian):
            return b * a
        return NotImplemented

    # --------------------------- Dense Representation ---------------------------

    def to_dense(self) -> np.ndarray:
        """
        Build the dense (2^n x 2^n) matrix of the Hamiltonian.

        Returns
        -------
        H : np.ndarray (complex)
            The dense matrix representation.

        Notes
        -----
        Suitable for small n (e.g., n <= 12). For larger systems, prefer
        sparse/MPO workflows.
        """
        d = 2 ** self.n
        H = np.zeros((d, d), dtype=complex)
        for term, c in self.terms.items():
            M = np.array([[1]], dtype=complex)
            for ch in term:
                M = np.kron(M, P_MAP[ch])
            H += c * M
        return H


# =============================== ED & Observables ===============================

def ed_ground_state(H: Hamiltonian) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Exact diagonalization for small systems using `numpy.linalg.eigh`.

    Parameters
    ----------
    H
        Hamiltonian object.

    Returns
    -------
    E0 : float
        Ground-state energy (real).
    psi0 : np.ndarray
        Normalized ground-state eigenvector.
    H_dense : np.ndarray
        Dense matrix used in the diagonalization.
    """
    Hmat = H.to_dense()
    evals, evecs = np.linalg.eigh(Hmat)  # Hermitian eigensolver
    idx = int(np.argmin(evals.real))
    E0 = float(evals[idx].real)
    psi0 = evecs[:, idx]
    psi0 = psi0 / np.linalg.norm(psi0)  # normalize defensively
    return E0, psi0, Hmat


def pauli_expectation(psi: np.ndarray, term: str) -> float:
    """
    Compute ⟨psi| (⊗_i P_i) |psi⟩ for a single Pauli string.

    Parameters
    ----------
    psi
        Statevector (normalized or not; normalization cancels in expectation).
    term
        Pauli string over {'I','X','Y','Z'}.

    Returns
    -------
    float
        Real value of the expectation (Pauli strings are Hermitian).
    """
    # Validate quickly
    bad = [ch for ch in term if ch not in P_MAP]
    if bad:
        raise ValueError(f"Invalid character(s) in term: {bad} (allowed: I,X,Y,Z)")
    M = np.array([[1]], dtype=complex)
    for ch in term:
        M = np.kron(M, P_MAP[ch])
    val = np.vdot(psi, M @ psi)  # vdot conjugates the first argument
    return float(np.real_if_close(val))


# =============================== Helper Builders ===============================

def pauli_digits_to_string(digs: Sequence[int]) -> str:
    """
    Convert base-4 digits to a Pauli string.

    Example
    -------
    [3, 0, 1, 2] -> 'ZIXY'
    """
    arr = np.asarray(digs, dtype=int)
    if np.any((arr < 0) | (arr > 3)):
        raise ValueError("Digits must be in {0,1,2,3} for {I,X,Y,Z}.")
    return "".join(_DIGIT2P[arr])


def random_pauli_string(n_qubits: int, *, p_I: float = 0.25, rng: np.random.Generator | None = None) -> str:
    """
    Sample a random length-`n_qubits` Pauli string with tunable identity bias.

    Parameters
    ----------
    n_qubits
        Length of the string to sample.
    p_I
        Probability for 'I'. Remaining mass is split equally among 'X','Y','Z'.
    rng
        Optional NumPy Generator.

    Returns
    -------
    str
        Random Pauli string.

    Notes
    -----
    Useful for generating sparse-ish strings when `p_I` is large.
    """
    if n_qubits <= 0:
        raise ValueError("`n_qubits` must be positive.")
    if not (0.0 <= p_I <= 1.0):
        raise ValueError("`p_I` must lie in [0,1].")
    rng = np.random.default_rng() if rng is None else rng
    probs = np.array([p_I, (1 - p_I) / 3, (1 - p_I) / 3, (1 - p_I) / 3], dtype=float)  # I, X, Y, Z
    digs = rng.choice(4, size=n_qubits, p=probs)
    return pauli_digits_to_string(digs)


def operator_from_pauli_string(term: str, coeff: float = 1.0) -> Hamiltonian:
    """
    Create a single-term Hamiltonian: H = coeff · (⊗_i term[i]).

    Parameters
    ----------
    term
        Pauli string over {'I','X','Y','Z'}.
    coeff
        Coefficient.

    Returns
    -------
    Hamiltonian
        A Hamiltonian containing exactly one term (possibly all-identity).
    """
    n = len(term)
    H = Hamiltonian(n)
    bad = [ch for ch in term if ch not in P_MAP]
    if bad:
        raise ValueError(f"Invalid character(s) in term: {bad} (allowed: I,X,Y,Z)")
    ops: List[Tuple[int, str]] = [(i, P) for i, P in enumerate(term) if P != "I"]
    if not ops:
        # All-identity is allowed; store it explicitly.
        H[term] = coeff
        return H
    H.add_term(ops, coeff)
    return H


def operator_from_pauli_ops(n_qubits: int, ops: List[Tuple[int, str]], coeff: float = 1.0) -> Hamiltonian:
    """
    Build a Hamiltonian from explicit non-identity Pauli ops.

    Parameters
    ----------
    n_qubits
        Total number of sites.
    ops
        List of (site, 'X'|'Y'|'Z'). 'I' entries are not allowed here.
    coeff
        Coefficient.

    Returns
    -------
    Hamiltonian
    """
    H = Hamiltonian(n_qubits)
    H.add_term(ops, coeff)
    return H


# ================================ MPO Builders ================================

def build_pauli_mpo_from_string(term: str, coeff: float = 1.0, epsilon: float = 1e-8, chi_max: int = 256):
    """
    Convenience wrapper: Pauli string → Hamiltonian → `get_mpo(...)`.

    Parameters
    ----------
    term, coeff, epsilon, chi_max
        Forwarded to `operator_from_pauli_string` and `get_mpo`.

    Returns
    -------
    (wmpo, norm, discarded)
        As returned by `get_mpo`.
    """
    H = operator_from_pauli_string(term, coeff=coeff)
    wmpo, norm, discarded = get_mpo(len(term), H, epsilon=epsilon, chi_max=chi_max)
    return wmpo, norm, discarded


def build_pauli_mpo_from_digits(digs: Sequence[int], coeff: float = 1.0, **kwargs):
    """
    Same as `build_pauli_mpo_from_string`, starting from base-4 digits (0=I,1=X,2=Y,3=Z).
    """
    term = pauli_digits_to_string(digs)
    return build_pauli_mpo_from_string(term, coeff=coeff, **kwargs)


def get_mpo(N_qubit: int, sys_Ham: Hamiltonian, epsilon: float = 1e-8, chi_max: int = 256, show_info: bool = False):
    """
    Build and compress an MPO from a Hamiltonian using your back-end.

    Parameters
    ----------
    N_qubit
        Number of physical qubits/sites.
    sys_Ham
        Hamiltonian (this object encapsulates the Pauli terms and coefficients).
    epsilon
        Truncation threshold passed to the MPO compressor.
    chi_max
        Maximum bond dimension for MPO compression.
    show_info
        If True, prints basic diagnostic info.

    Returns
    -------
    wmpo : MPO
        Compressed MPO object.
    norm_mpo : float
        Normalization absorbed into the first tensor.
    discarded : float
        Discarded weight during compression.

    Notes
    -----
    Public signature and behavior preserved. Graph construction is delegated
    to `gb` and the compression to `qmpo`.
    """
    if not isinstance(sys_Ham, Hamiltonian):
        raise TypeError("`sys_Ham` must be a Hamiltonian.")
    if sys_Ham.n != N_qubit:
        raise ValueError(f"Mismatched sizes: N_qubit={N_qubit}, Hamiltonian.n={sys_Ham.n}")

    # Derive structures from the Hamiltonian
    coefs_dict = sys_Ham.as_dict           # term -> coefficient
    coefsym_dict = sys_Ham.term_indices    # term -> integer index
    Ham_terms = sys_Ham.ham_terms          # list of Pauli strings

    # Optional graph building hooks (kept for compatibility)
    g_l = gb.get_left_G(Ham_terms, N_qubit)
    g_r = gb.get_right_G(Ham_terms, N_qubit)
    #print(g_l)

    # Build raw MPO tensors from terms
    ws = gb.build_mpo_tensors(
        Ham_terms,
        N_qubit,
        coefs_dict,
        coefsym_dict
    )

    wmpo_raw = qmpo.MPO(ws, Ss=None, bonds=None)
    if show_info:
        print(f"number of terms: {len(sys_Ham)}")
        print("mpo raw bonds dimensions", wmpo_raw.bonds)

    # Compress and absorb normalization
    wmpo, norm_mpo, discarded = qmpo.compress_mpo(wmpo_raw, epsilon, chi_max)
    wmpo.Ms[0] *= norm_mpo
    return wmpo, norm_mpo, discarded


# ============================== Interoperability ==============================

def from_openfermion(of_hamiltonian: Any, n_qubits: int) -> Hamiltonian:
    """
    Convert an OpenFermion-style QubitOperator into this module's Hamiltonian.

    Parameters
    ----------
    of_hamiltonian
        An object with `.terms` like OpenFermion's QubitOperator, i.e.,
        a dict mapping tuples of ((qubit_index, 'X'|'Y'|'Z'), ...) to coeff.
    n_qubits
        Number of qubits (needed to place identities on unspecified sites).

    Returns
    -------
    Hamiltonian
    """
    H = Hamiltonian(n_qubits)
    # Expected: of_hamiltonian.terms: Dict[Tuple[(int,str),...], coeff]
    for ops, coeff in of_hamiltonian.terms.items():
        # OpenFermion uses a tuple of (int, 'X'|'Y'|'Z'), identities omitted.
        H.add_term(list(ops), coeff)
    return H



########################################## Renormalizer #############################
def rn_terms_from_hamiltonian(H, *, drop_zeros: bool = True, atol: float = 0.0):
    """
    Convert your Hamiltonian (Pauli-string dict) -> Renormalizer Op terms.

    Returns
    -------
    rn_terms : list[Op]
        Each entry is a product of Op(...) objects, with the coefficient attached
        to the first factor (same convention you used).
    constant_shift : complex
        Coefficient of the all-identity term 'I...I' (energy shift).
        Renormalizer usually doesn't need it as an Op; you can add it to energies.
    """
    from renormalizer import Op  # import locally to keep your module import-light

    n = H.n
    rn_terms = []
    constant_shift = 0.0 + 0j

    for term, coeff in H.terms.items():
        if drop_zeros and (abs(coeff) <= atol):
            continue

        if len(term) != n:
            raise ValueError(f"Bad term length: {term} (expected length {n})")

        # collect non-identity factors
        active = [(i, ch) for i, ch in enumerate(term) if ch != "I"]

        if not active:
            # all identity -> constant energy shift
            constant_shift += complex(coeff)
            continue

        # build Op product, attach coefficient to first factor
        i0, ch0 = active[0]
        op_term = Op(ch0, i0, complex(coeff))
        for i, ch in active[1:]:
            op_term = op_term * Op(ch, i)

        rn_terms.append(op_term)

    return rn_terms, constant_shift


def renormalizer_model_from_hamiltonian(H, *, build_mpo: bool = True, **kwargs):
    """
    Build a Renormalizer Model (and optionally an MPO) from your Hamiltonian.

    Returns
    -------
    model, mpo_or_None, constant_shift
    """
    from renormalizer import Model, BasisHalfSpin, Mpo

    rn_terms, constant_shift = rn_terms_from_hamiltonian(H, **kwargs)

    basis = [BasisHalfSpin(i) for i in range(H.n)]

    if rn_terms:
        ham_terms = rn_terms[0]
        for t in rn_terms[1:]:
            ham_terms = ham_terms + t
    else:
        # no non-identity terms; make a harmless zero term if needed
        # (some versions of renormalizer dislike an empty ham_terms)
        ham_terms = 0.0

    model = Model(basis, ham_terms)
    mpo = Mpo(model) if build_mpo else None
    return model, mpo, constant_shift





import numpy as np
from scipy.linalg import hadamard
from pyscf import gto, scf, ao2mo


def build_fq_hamiltonian_mpo(
    atom_str: str,
    basis: str = 'sto-3g',
    charge: int = 0,
    spin: int = 0,
    epsilon: float = 1e-8,
    chi_max: int = 256
) -> tuple[Hamiltonian, Any, float, float]:
    """
    Constructs the First Quantized Hamiltonian for a molecule and converts it to an MPO.
    
    Parameters
    ----------
    atom_str : str
        PySCF atom string (e.g., "H 0 0 0; H 0 0 0.74").
    basis : str
        Basis set name (e.g., 'sto-3g').
    charge : int
        Molecular charge.
    spin : int
        Molecular spin multiplicity difference (n_alpha - n_beta).
    epsilon : float
        MPO truncation threshold.
    chi_max : int
        Maximum bond dimension.

    Returns
    -------
    H_obj : pauli_hamiltonians.Hamiltonian
        The constructed Hamiltonian object containing all Pauli terms.
    wmpo : MPO
        The compressed MPO object.
    norm : float
        The normalization factor of the MPO.
    discarded : float
        The discarded weight during MPO compression.
    """
    
    # --- 1. System Setup (PySCF) ---
    print(f"Building FQ Hamiltonian for: {atom_str}")
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, verbose=0)
    mol.build()
    
    # Run Hartree-Fock
    if spin == 0:
        mf = scf.RHF(mol).run()
    else:
        mf = scf.ROHF(mol).run()
        
    n_elec = mol.nelectron
    n_orb = mf.mo_coeff.shape[1]
    
    # Calculate qubits needed per electron register
    bits_per_reg = int(np.ceil(np.log2(n_orb)))
    if bits_per_reg == 0: bits_per_reg = 1
    
    n_qubits = n_elec * bits_per_reg
    print(f"  Electrons: {n_elec}, Orbitals: {n_orb}")
    print(f"  Register Size: {bits_per_reg} bits")
    print(f"  Total Qubits: {n_qubits}")

    # --- 2. Compute Integrals & LCU Coefficients ---
    h1 = mf.get_hcore()
    
    # Handle effective potential for Open Shell if needed
    if spin != 0:
        dm = mf.make_rdm1()
        v_eff = mf.get_veff(mol, dm)
        if v_eff.ndim == 3: v_eff = v_eff[0] # Take alpha potential
        h1 += v_eff # Add V_eff to H_core for active space consistency if needed
        # Note: For full Hamiltonian, standard h1/h2 is usually sufficient, 
        # but freezing/active space logic requires care. We use raw integrals here.

    # Transform integrals to MO basis
    h1_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, h1, mf.mo_coeff)
    h2_mo_flat = ao2mo.kernel(mol, mf.mo_coeff)
    h2_mo = ao2mo.restore(1, h2_mo_flat, n_orb)
    
    # LCU Transform (Walsh-Hadamard)
    D_pad = 2**bits_per_reg
    H_mat = hadamard(D_pad)
    
    # Compute Omega1 (1-Body LCU)
    omega1 = np.zeros((D_pad, D_pad))
    for p in range(D_pad):
        for q in range(D_pad):
            val = 0.0
            for a in range(D_pad):
                if (p ^ a) < n_orb and a < n_orb:
                    val += h1_mo[p ^ a, a] * H_mat[a, q]
            omega1[p, q] = val / D_pad

    # Compute Omega2 (2-Body LCU)
    omega2 = np.zeros((D_pad, D_pad, D_pad, D_pad))
    for p in range(D_pad):
        for q in range(D_pad):
            for r in range(D_pad):
                for s in range(D_pad):
                    val = 0.0
                    for a in range(D_pad):
                        for b in range(D_pad):
                            i, j, k, l = p^a, a, r^b, b
                            if i<n_orb and j<n_orb and k<n_orb and l<n_orb:
                                val += h2_mo[i, j, k, l] * H_mat[a, q] * H_mat[b, s]
                    omega2[p, q, r, s] = val / (D_pad**2)

    # --- 3. Construct Pauli Strings ---
    H_obj = Hamiltonian(n_qubits)
    
    # Helper: Convert LCU index (p, q) to list of (qubit_idx, PauliStr)
    # Mapping: p is X-bitmask, q is Z-bitmask.
    # If p_k=1, q_k=0 -> X
    # If p_k=0, q_k=1 -> Z
    # If p_k=1, q_k=1 -> Y (ignoring phase for this structural demo, strictly iY)
    def get_reg_ops(p, q, reg_idx):
        ops = []
        offset = reg_idx * bits_per_reg
        for k in range(bits_per_reg):
            is_x = (p >> k) & 1
            is_z = (q >> k) & 1
            
            p_char = None
            if is_x and not is_z: p_char = 'X'
            elif not is_x and is_z: p_char = 'Z'
            elif is_x and is_z: p_char = 'Y'
            
            if p_char:
                ops.append((offset + k, p_char))
        return ops

    # A. One-Body Terms
    # H = sum_{k} Omega1[p,q] * Op_k(p,q)
    for p in range(D_pad):
        for q in range(D_pad):
            c = omega1[p, q]
            if abs(c) < 1e-9: continue
            
            # Sum over all electron registers
            for k in range(n_elec):
                ops = get_reg_ops(p, q, k)
                if not ops:
                    # Identity term (adds to scalar)
                    # We handle scalar via a dummy identity string or tracking constant
                    # But Hamiltonian class supports "I"*N implicit add
                    H_obj.add_term([], c) 
                else:
                    H_obj.add_term(ops, c)

    # B. Two-Body Terms
    # H = 0.5 * sum_{k != l} Omega2[p,q,r,s] * Op_k(p,q) * Op_l(r,s)
    # We iterate unique pairs (k < l) and add both k,l and l,k (symmetric)
    # 0.5 * 2 = 1.0 multiplier for the pair sum
    pairs = [(k, l) for k in range(n_elec) for l in range(k + 1, n_elec)]
    
    for p in range(D_pad):
        for q in range(D_pad):
            for r in range(D_pad):
                for s in range(D_pad):
                    c = omega2[p, q, r, s]
                    if abs(c) < 1e-9: continue
                    
                    for k, l in pairs:
                        ops_k = get_reg_ops(p, q, k)
                        ops_l = get_reg_ops(r, s, l)
                        
                        # Combine lists
                        full_ops = ops_k + ops_l
                        H_obj.add_term(full_ops, c) # coeff c, not 0.5*c, because we sum pairs

    # C. Nuclear Repulsion (Scalar)
    e_nuc = mol.energy_nuc()
    if abs(e_nuc) > 1e-9:
        H_obj.add_term([], e_nuc) # Adds to Identity

    print(f"  Terms generated: {len(H_obj)}")

    # --- 4. Build MPO ---
    print("Building MPO...")
    wmpo, norm, discarded = get_mpo(n_qubits, H_obj, epsilon=epsilon, chi_max=chi_max, show_info=True)
    
    return H_obj, wmpo, norm, discarded

# --- Usage Example ---
if __name__ == "__main__":
    # Example: H2 Molecule
    atom_str = "Li 0 0 0; H 0 0 0.74"
    H_fq, mpo, norm, disc = build_fq_hamiltonian_mpo(atom_str)
    
    print("\nCheck MPO:")
    
    print(f"Norm: {norm}")
    print(f"Discarded Weight: {disc}")
    
    # Optional: Verify Ground State if small enough
    if H_fq.n <= 12:
        print("\nVerifying with Exact Diagonalization...")
        e0, _, _ = ed_ground_state(H_fq)
        print(f"Ground State Energy: {e0:.6f} Ha")




# =============================== Model Generators ==============================

def heisenberg_chain(n: int,
                     Jx: float,
                     Jy: float,
                     Jz: float,
                     boundary: Literal["open", "periodic"] = "open") -> Hamiltonian:
    """
    Nearest-neighbor Heisenberg model (XX+YY+ZZ) on a chain.

    Parameters
    ----------
    n
        Number of sites.
    Jx, Jy, Jz
        Coupling constants.
    boundary
        'open' or 'periodic'.
    """
    H = Hamiltonian(n)
    rng = range(n - 1) if boundary == "open" else range(n)
    for i in rng:
        j = (i + 1) % n
        H.add_term([(i, "X"), (j, "X")], Jx)
        H.add_term([(i, "Y"), (j, "Y")], Jy)
        H.add_term([(i, "Z"), (j, "Z")], Jz)
    return H


def transverse_field_ising(n: int,
                           Jz: float,
                           hx: float,
                           hz: float,
                           boundary: Literal["open", "periodic"] = "open") -> Hamiltonian:
    """
    Transverse-field Ising model with on-site X and Z fields.

    H = Jz ∑ Z_i Z_{i+1} + hx ∑ X_i + hz ∑ Z_i
    """
    H = Hamiltonian(n)
    rng = range(n - 1) if boundary == "open" else range(n)
    for i in rng:
        j = (i + 1) % n
        H.add_term([(i, "Z"), (j, "Z")], Jz)
    for i in range(n):
        H.add_term([(i, "X")], hx)
        H.add_term([(i, "Z")], hz)
    return H


def cluster_hamiltonian(n: int, K: float, boundary: Literal["open", "periodic"] = "open") -> Hamiltonian:
    """
    1D cluster Hamiltonian: H = K ∑ Z_i X_{i+1} Z_{i+2}.
    """
    H = Hamiltonian(n)
    rng = range(n - 2) if boundary == "open" else range(n)
    for i in rng:
        i0 = i
        i1 = (i + 1) % n
        i2 = (i + 2) % n
        H.add_term([(i0, "Z"), (i1, "X"), (i2, "Z")], K)
    return H


def build_H_odd(L: int,
                J1: float = +1.0,
                J2: float = -1.0,
                J3: float = +1.0,
                boundary: Literal["open", "periodic"] = "periodic") -> Hamiltonian:
    """
    Build the model:

        H_odd = J1 ∑ Z_{2n-1} X_{2n}   Z_{2n+1}
              + J2 ∑ Y_{2n}   X_{2n+1} Y_{2n+2}
              + J3 ∑ Z_{2n-1} Z_{2n} X_{2n+1} Z_{2n+2} Z_{2n+3}

    Parameters
    ----------
    L
        Chain length (must be even).
    J1, J2, J3
        Couplings as described above.
    boundary
        'open' or 'periodic'.
    """
    if L % 2 != 0:
        raise ValueError("L must be even")
    H = Hamiltonian(L)

    for n in range(1, L // 2 + 1):
        # 1-based indexing in description; convert to 0-based
        i0 = 2 * n - 2
        i1 = 2 * n - 1
        i2 = 2 * n
        i3 = 2 * n + 1
        i4 = 2 * n + 2

        if boundary == "periodic":
            i0 %= L; i1 %= L; i2 %= L; i3 %= L; i4 %= L
        else:
            if i4 >= L:
                continue

        H.add_term([(i0, "Z"), (i1, "X"), (i2, "Z")], J1)
        H.add_term([(i1, "Y"), (i2, "X"), (i3, "Y")], J2)
        H.add_term([(i0, "Z"), (i1, "Z"), (i2, "X"), (i3, "Z"), (i4, "Z")], J3)

    return H


def build_string_op(N: int, L: int, n: int, parity: str = "even") -> Hamiltonian:
    """
    Build the length-(2L+2) string operator

        R^L_{parity,n} = Z  (XI)^L  Z

    placed within an N-site chain. There are L X's on every other site between
    the two Z's.

    Parameters
    ----------
    N
        Total number of qubits.
    L
        Number of X's in the alternating 'XI' pattern.
    n
        1-based "unit-cell" index that sets the starting position.
    parity
        'even' → first Z on site 2n−2 (0-based),
        'odd'  → first Z on site 2n−1.

    Returns
    -------
    Hamiltonian
        A one-term Hamiltonian with coefficient 1.0.
    """
    H = Hamiltonian(N)
    if parity == "even":
        start = 2 * n - 2
    elif parity == "odd":
        start = 2 * n - 1
    else:
        raise ValueError("parity must be 'even' or 'odd'")

    end = start + 2 * L + 1
    if not (0 <= start < N and 0 <= end < N):
        raise IndexError(f"String runs off the chain: start={start}, end={end}, N={N}")

    ops = [(start, "Z")]
    for i in range(L):
        pos = start + 1 + 2 * i
        ops.append((pos, "X"))
    ops.append((end, "Z"))

    H.add_term(ops, 1.0)
    return H


# =================================== DEMO ===================================

if __name__ == "__main__":
    # --- Example A: Heisenberg chain construction ---
    H = heisenberg_chain(n=6, Jx=1.0, Jy=1.0, Jz=1.0, boundary="open")
    print("Heisenberg terms:", H.ham_terms[:6], " ...")
    print("First few coeffs:", H.coeff_list[:6])
    print("Term→index (head):", {k: H.term_indices[k] for k in list(H.term_indices)[:6]})
    print("=" * 60)

    # --- Example B: Basic arithmetic, indexing, equality ---
    H1 = Hamiltonian(n_qubits=4)
    H1.add_term([(0, "X"), (1, "X")], 1.0)
    H1.add_term([(2, "Z")], 2.0)

    H2 = Hamiltonian(n_qubits=4)
    H2.add_term([(0, "X"), (1, "X")], 0.5)
    H2.add_term([(2, "Z"), (3, "Z")], 3.0)

    print(repr(H1))
    print(H1)  # human-readable

    print("len(H1) =", len(H1))
    print("XXII in H1?", "XXII" in H1)
    print("Coeff[XXII] =", H1["XXII"])
    print("Coeff[ZZZZ] (missing) =", H1["ZZZZ"])

    H1["ZZZZ"] = 5.0
    Hsum = H1 + H2
    Hscaled = 2.0 * H2
    Hneg = -H2
    print("Hsum.as_dict:", Hsum.as_dict)
    print("Hscaled.as_dict:", Hscaled.as_dict)
    print("Hneg.as_dict:", Hneg.as_dict)

    H3 = Hamiltonian(4)
    H3.add_term([(0, "X"), (1, "X")], 0.5)
    H3.add_term([(2, "Z"), (3, "Z")], 3.0)
    print("H2 == H3 ?", H2 == H3)
    print("=" * 60)

    # --- Example C: OpenFermion conversion ---
    try:
        import openfermion as of
        of_ham = of.QubitOperator("X0 X1", 1.0) + of.QubitOperator("Z2", 2.0)
        H_of = from_openfermion(of_ham, n_qubits=4)
        print("From OpenFermion:", H_of.as_dict)
    except Exception as e:
        print("OpenFermion not available or failed to import; skipping:", e)
    print("=" * 60)

    # --- Example D: Random Pauli → MPO ---
    n = 8
    term = random_pauli_string(n, p_I=0.2)  # e.g. 'ZIXYIYXI'
    print("Random term:", term)
    H_rand = operator_from_pauli_string(term, coeff=1.0)
    wmpo, norm, discarded = get_mpo(n, H_rand, epsilon=1e-9, chi_max=512)
    print("compressed MPO bonds:", getattr(wmpo, "bonds", None), "norm:", norm, "discarded:", discarded)

    # --- Example E: Explicit ops → MPO ---
    n = 6
    ops = [(0, "X"), (3, "Y"), (5, "Z")]  # X ⊗ I ⊗ I ⊗ Y ⊗ I ⊗ Z
    H_ops = operator_from_pauli_ops(n, ops, coeff=0.75)
    print(H_ops)
    wmpo2, norm2, discarded2 = get_mpo(n, H_ops)
    print("compressed MPO (ops) bonds:", getattr(wmpo2, "bonds", None), "norm:", norm2, "discarded:", discarded2)
