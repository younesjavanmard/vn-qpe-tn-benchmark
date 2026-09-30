import numpy as np
from scipy.linalg import svd as _scipy_svd
from typing import List
import warnings

from qu_alg_qu_sim_tensor_networks.backend import BACK, to_numpy, get_array_module

def svd(M, full_matrices=False, compute_uv=True):
    """Backend-aware SVD. On GPU uses xp.linalg.svd; on CPU uses scipy with fallbacks."""
    xp = get_array_module(M)
    if BACK in ("cupy", "jax") or xp is not np:
        return xp.linalg.svd(M, full_matrices=full_matrices)
    # CPU: scipy with fallbacks
    try:
        return _scipy_svd(M, full_matrices=full_matrices, compute_uv=True, check_finite=False)
    except Exception:
        try:
            return np.linalg.svd(M, full_matrices=full_matrices, compute_uv=True)
        except Exception:
            # Last-resort: EVD of M^† M
            H = M.conj().T @ M
            w, V = np.linalg.eigh(H)
            order = np.argsort(w)[::-1]
            s = np.sqrt(np.clip(w[order], 0.0, None))
            V = V[:, order]
            tol = 1e-14 * (float(np.max(s)) if s.size > 0 and np.max(s) > 0 else 1.0)
            inv_s = np.where(s > tol, 1.0 / s, 0.0)
            U = M @ (V * inv_s[None, :])
            return U, s, V.conj().T
class MPS:
    """Simple class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We *assume* that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss, bc:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.Array[ndim=3]
        The 'matrices', in right-canonical form, one for each physical site
        (within the unit-cell for an infinite MPS).
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``.
    Ss : list of np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    bc : 'infinite', 'finite'
        Boundary conditions.
    L : int
        Number of sites (in the unit-cell for an infinite MPS).
    nbonds : int
        Number of (non-trivial) bonds: L-1 for 'finite' boundary conditions, L for 'infinite'.
    """
    def __init__(self, Bs, Ss, bc='finite'):
        assert bc in ['finite', 'infinite']
        self.Bs = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'finite' else self.L

    def copy(self):
        """Return a deep copy of this MPS (tensors and Schmidt values are copied)."""
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss], self.bc)

    def one_site_center(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        xp = get_array_module(self.Bs[i])
        return xp.tensordot(xp.diag(self.Ss[i]), self.Bs[i], (1, 0))  # vL [vL'], [vL] i vR

    def two_site_center(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        xp = get_array_module(self.Bs[i])
        return xp.tensordot(self.one_site_center(i), self.Bs[j], (2, 0))  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.nbonds)]

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            xp = get_array_module(self.Bs[i])
            theta = self.one_site_center(i)  # vL i vR
            op_dev = xp.asarray(op)
            op_theta = xp.tensordot(op_dev, theta, axes=(1, 1))
            result.append(to_numpy(xp.tensordot(theta.conj(), op_theta, ([0, 1, 2], [1, 0, 2]))))
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            xp = get_array_module(self.Bs[i])
            theta = self.two_site_center(i)  # vL i j vR
            op_dev = xp.asarray(op[i])
            op_theta = xp.tensordot(op_dev, theta, axes=([2, 3], [1, 2]))
            result.append(to_numpy(xp.tensordot(theta.conj(), op_theta, ([0, 1, 2, 3], [2, 0, 1, 3]))))
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            xp = get_array_module(self.Ss[i])
            S = self.Ss[i].copy()
            S = xp.where(S < 1.e-20, xp.zeros_like(S), S)
            S2 = S * S
            assert abs(float(xp.linalg.norm(S)) - 1.) < 1.e-13
            result.append(float(-xp.sum(S2 * xp.log(xp.where(S2 > 0, S2, xp.ones_like(S2))))))
        return np.array(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        import scipy.sparse.linalg as arp
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        # Convert to numpy: scipy ARPACK is CPU-only
        Bs_cpu = [to_numpy(B) for B in self.Bs]
        B = Bs_cpu[0]  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))
        T = np.transpose(T, [0, 2, 1, 3])
        for i in range(1, self.L):
            B = Bs_cpu[i]
            T = np.tensordot(T, B, axes=(2, 0))
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
        T = np.reshape(T, (chi**2, chi**2))
        eta = arp.eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi = -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j."""
        assert i < j
        xp = get_array_module(self.Bs[i])
        op_i_dev, op_j_dev = xp.asarray(op_i), xp.asarray(op_j)
        theta = self.one_site_center(i)
        C = xp.tensordot(op_i_dev, theta, axes=(1, 1))
        C = xp.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]
            C = xp.tensordot(C, B, axes=(1, 0))
            C = xp.tensordot(B.conj(), C, axes=([0, 1], [0, 1]))
        j = j % self.L
        B = self.Bs[j]
        C = xp.tensordot(C, B, axes=(1, 0))
        C = xp.tensordot(op_j_dev, C, axes=(1, 1))
        C = xp.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2]))
        return to_numpy(C)
    
    
    @classmethod
    def product_state_from_bits(cls, bits: List[int], d=2, dtype=complex):
        """
        Build a product-state MPS from computational-basis bits.
        bits[q] in {0,1} sets |bits[0] bits[1] ...>.
        Returns an MPS with all bond dimensions = 1 and B tensors (1,d,1).
        """
        Bs = []
        Ss = []
        for b in bits:
            B = np.zeros((1, d, 1), dtype=dtype)
            B[0, int(b), 0] = 1.0
            Bs.append(B)
            Ss.append(np.array([1.0], dtype=float))
        Ss.append(np.array([1.0], dtype=float))
        bonds = [1] * (len(bits) + 1)
        return cls(Bs=Bs, Ss=Ss, bc="finite")


def init_FM_MPS(L, d, bc='finite'):
    """Return a ferromagnetic MPS: product state with all spins in state |0⟩.

    Parameters
    ----------
    L : int
        Number of sites.
    d : int
        Physical dimension per site.
    bc : str, optional
        Boundary condition, 'finite' or 'infinite' (default 'finite').

    Returns
    -------
    MPS
        Right-canonical MPS with all sites in state |0⟩ and bond dimension 1.
    """
    B = np.zeros([1, d, 1], dtype=float)
    B[0, 0, 0] = 1.
    S = np.ones([1], dtype=float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss, bc=bc)


def init_Neel_MPS(L, d, bc='finite'):
    """Return a Néel-ordered MPS: alternating |0⟩ and |d-1⟩ product state.

    Parameters
    ----------
    L : int
        Number of sites.
    d : int
        Physical dimension per site.
    bc : str, optional
        Boundary condition, 'finite' or 'infinite' (default 'finite').

    Returns
    -------
    MPS
        Right-canonical MPS alternating between |0⟩ (even sites) and |d-1⟩ (odd sites).
    """
    S = np.ones([1], dtype=float)
    Bs = []
    for i in range(L):
        B = np.zeros([1, d, 1], dtype=float)
        if i % 2 == 0:
            B[0, 0, 0] = 1.
        else:
            B[0, -1, 0] = 1.
        Bs.append(B)
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss, bc=bc)


def init_random_bond_chi_MPS(L, d, chi=2, bc='finite', seed=None, normalize=True):
    """
    Return a random MPS with bond dimension `chi`.
    Shapes: B[0]  -> (1,  d, chi)
            B[i]  -> (chi, d, chi) for 0<i<L-1
            B[-1] -> (chi, d, 1)
    S: list of length-L singular-value stubs (kept as ones for compatibility).
    """
    rng = np.random.default_rng(seed)
    S = np.ones([1], dtype=float)
    Bs = []

    for i in range(L):
        if i == 0:
            B = rng.normal(0.0, 1.0, size=(1, d, chi))
        elif i == L - 1:
            B = rng.normal(0.0, 1.0, size=(chi, d, 1))
        else:
            B = rng.normal(0.0, 1.0, size=(chi, d, chi))

        if normalize:
            nrm = np.linalg.norm(B)
            if nrm > 0:
                B = B / nrm

        Bs.append(B)

    Ss = [S.copy() for _ in range(L)]
    return MPS(Bs, Ss, bc=bc)


def init_xf_state(L, d, bc="finite"):
    """Return an X-polarized product MPS: alternating |+⟩ amplitudes on even/odd sites.

    Parameters
    ----------
    L : int
        Number of sites.
    d : int
        Physical dimension per site (must be 2 for this state to be meaningful).
    bc : str, optional
        Boundary condition, 'finite' or 'infinite' (default 'finite').

    Returns
    -------
    MPS
        Right-canonical MPS with amplitude 1/√2 on the physical index matching site parity.
    """
    S = np.ones([1], dtype=float)
    Bs = []
    for i in range(L):
        B = np.zeros([1, d, 1], dtype=float)
        if i % 2 == 0:
            B[0, 0, 0] = 1.0 / np.sqrt(2.0)
        else:
            B[0, 1, 0] = 1.0 / np.sqrt(2.0)
        Bs.append(B)
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss, bc=bc)    


def svd_split_truncate(theta, chi_max, eps, return_norm=False):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    If ``return_norm`` is True, additionally return the 2-norm of the kept
    singular values *before* they are renormalized to unit norm. For
    non-Hermitian (norm non-preserving) evolution this factor is the amount by
    which the local update rescaled the state; retaining it across the sweep
    lets a caller reconstruct the unnormalized (scale-carrying) trajectory.

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    xp = get_array_module(theta)
    chivL, dL, dR, chivR = theta.shape
    theta = xp.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False)

    # truncate
    chivC = min(chi_max, int(xp.sum(Y > eps)))
    assert chivC >= 1
    piv = xp.argsort(Y)[::-1][:chivC]
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    kept_norm = xp.linalg.norm(Y)
    S = Y / kept_norm
    # split legs of X and Z
    A = xp.reshape(X, [chivL, dL, chivC])
    B = xp.reshape(Z, [chivC, dR, chivR])
    if return_norm:
        return A, S, B, kept_norm
    return A, S, B

rng = np.random.default_rng(0)
def rand_mps(L=6, d=2, chi=3):
    """Return a random complex MPS with given bond dimension (unit-norm tensors, no canonical form).

    Parameters
    ----------
    L : int, optional
        Number of sites (default 6).
    d : int, optional
        Physical dimension (default 2).
    chi : int, optional
        Bond dimension for interior tensors (default 3).

    Returns
    -------
    MPS
        Finite MPS with complex tensors drawn from a complex normal distribution.
        Each tensor is independently normalized; Schmidt values are all set to 1.
    """
    Bs = []
    for i in range(L):
        if i == 0:
            B = rng.normal(size=(1, d, chi)) + 1j*rng.normal(size=(1, d, chi))
        elif i == L-1:
            B = rng.normal(size=(chi, d, 1)) + 1j*rng.normal(size=(chi, d, 1))
        else:
            B = rng.normal(size=(chi, d, chi)) + 1j*rng.normal(size=(chi, d, chi))
        Bs.append(B / np.linalg.norm(B))
    return MPS(Bs, Ss=[np.ones(1) for _ in range(L)], bc='finite')

# ====================== isometry diagnostics ======================

def right_isometry_error(B: np.ndarray) -> float:
    """Measure how far a tensor is from right-isometric: ‖∑_σ B^σ (B^σ)† − I‖_F.

    A right-canonical tensor satisfies ∑_σ B^σ (B^σ)† = I_{χL}.

    Parameters
    ----------
    B : np.ndarray, shape (χL, d, χR)
        Site tensor to check.

    Returns
    -------
    float
        Frobenius norm of the deviation from right-isometry (0 = perfectly right-canonical).
    """
    chiL, d, chiR = B.shape
    E = np.zeros((chiL, chiL), dtype=B.dtype)
    for s in range(d):
        E += B[:, s, :] @ B[:, s, :].conj().T
    return np.linalg.norm(E - np.eye(chiL, dtype=B.dtype))

def left_isometry_error(B: np.ndarray) -> float:
    """Measure how far a tensor is from left-isometric: ‖∑_σ (B^σ)† B^σ − I‖_F.

    A left-canonical tensor satisfies ∑_σ (B^σ)† B^σ = I_{χR}.

    Parameters
    ----------
    B : np.ndarray, shape (χL, d, χR)
        Site tensor to check.

    Returns
    -------
    float
        Frobenius norm of the deviation from left-isometry (0 = perfectly left-canonical).
    """
    chiL, d, chiR = B.shape
    E = np.zeros((chiR, chiR), dtype=B.dtype)
    for s in range(d):
        E += B[:, s, :].conj().T @ B[:, s, :]
    return np.linalg.norm(E - np.eye(chiR, dtype=B.dtype))

# ==================== Right-canonical (R → L) =====================

def right_canonicalize(
    psi: MPS,
    method: str = "svd",      # {"svd","qr"}
    chi_max: int | None = None,
    eps: float | None = None,
) -> MPS:
    """
    TEXTBOOK right-canonicalization (sweep RIGHT → LEFT).

    SVD version (Orús/Schollwöck, matches your screenshot):
      For site i = L-1 ... 0:
        - M = reshape(B_i, (chi_L, d*chi_R))
        - SVD: M = U S V†
        - Choose r (≤ rank), optionally by chi_max and eps.
        - Set B_i' from *rows of V†*:  V†[:r,:].reshape(r, d, chi_R) → (chi_L'=r, d, chi_R)
        - Push G_left = U[:, :r] @ diag(S[:r]) into left neighbor (its right leg).

    QR variant:
      - Use M^T = Q R  ⇒  M = R^T Q^T
      - Set B_i' from rows of Q^T, push R^T left.
    """
    assert method in ("svd", "qr")
    Bs = [B.copy() for B in psi.Bs]
    L = len(Bs)

    for i in range(L - 1, -1, -1):
        chiL, d, chiR = Bs[i].shape
        M = Bs[i].reshape(chiL, d * chiR)  # (chi_L, d*chi_R)

        if method == "svd":
            U, S, Vh = np.linalg.svd(M, full_matrices=False)  # U:(chiL,r), S:(r,), Vh:(r,d*chiR)
            r = len(S)
            if eps is not None:
                r = int(np.count_nonzero(S > eps)) or 1
            if chi_max is not None:
                r = min(r, chi_max)

            # Build B_i' from rows of Vh (V†) – textbook right-canonical step
            Vh_r = Vh[:r, :]                                   # (r, d*chiR)
            Bi_new = Vh_r.reshape(r, d, chiR)                  # (chiL_new=r, d, chiR)
            Bs[i] = Bi_new

            # Push U S into left neighbor (shape (chiL, r))
            G_left = U[:, :r] * S[:r]                          # broadcasting (chiL,r)
            if i > 0:
                # Bs[i-1]: (chiL_prev, d_prev, chiL)
                Bs[i-1] = np.tensordot(Bs[i-1], G_left, axes=(2, 0))  # -> (chiL_prev, d_prev, r)

        else:  # method == "qr"
            # Do QR on M^T to mimic the same pattern: M^T = Q R  ⇒  M = R^T Q^T
            Q, R = np.linalg.qr(M.T, mode="reduced")           # Q:(d*chiR,r), R:(r,chiL)
            r = Q.shape[1]
            if eps is not None:
                diag_abs = np.abs(np.diag(R))
                r = int(np.count_nonzero(diag_abs > eps)) or 1
            if chi_max is not None:
                r = min(r, chi_max)

            Q = Q[:, :r]                                       # (d*chiR,r)
            R = R[:r, :]                                       # (r,chiL)

            Bi_new = Q.T.reshape(r, d, chiR)                   # rows of Q^T ⇒ (r,d,chiR)
            Bs[i] = Bi_new

            if i > 0:
                G_left = R.T                                   # (chiL, r)
                Bs[i-1] = np.tensordot(Bs[i-1], G_left, axes=(2, 0))

    return MPS(Bs, [S.copy() for S in psi.Ss], psi.bc)




import numpy as np



def schmidt_vals(Blist, ChiVec, L, d=2):
    """Compute Schmidt values for each bond by sweeping left-to-right with SVD.

    Updates ``Blist`` in-place so that site tensors and Schmidt lists are mutually
    consistent after the sweep.

    Parameters
    ----------
    Blist : list of np.ndarray
        Site tensors, each of shape (χL, d, χR), modified in-place.
    ChiVec : list of int
        Target bond dimensions at each bond (length L+1).
    L : int
        Number of sites.
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    Blist : list of np.ndarray
        Updated site tensors (in-place).
    llist : list of np.ndarray
        Schmidt values at each bond, including boundary trivial bonds.
    """
    llist = []
    llist.append(np.array([1.0]))
    for i in range(L - 1):
        # Blist[i]   : (chi_l_i, d_i, chi_r_i)
        # Blist[i+1] : (chi_l_{i+1}=chi_r_i, d_{i+1}, chi_r_{i+1})

        # two-site theta: contract right of i with left of i+1
        theta = np.tensordot(Blist[i], Blist[i+1], axes=(2, 0))  # (chi_l_i, d_i, d_{i+1}, chi_r_{i+1})
        theta_bar = theta

        # composite dims for SVD
        chi_l_i, d_i, d_next, chi_r_next = theta.shape
        a = d * Blist[i].shape[0]            # = chi_l_i * d
        b = d * Blist[i+1].shape[2]          # = d_{i+1} * chi_r_{i+1}

        # insert current Schmidt on the left bond
        theta = np.tensordot(np.diag(llist[i]), theta, axes=(1, 0))  # (chi_l_i, d_i, d_{i+1}, chi_r_{i+1})
        theta = np.reshape(theta, (a, b))

        try:
            X, Y, Z = np.linalg.svd(theta, compute_uv=True, full_matrices=True)  # X=U, Y=S, Z=Vh
        except np.linalg.LinAlgError:
            print('SVD did not converge, diagonalizing theta_dagger*theta')
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        # truncate to target rank r = ChiVec[i+1]
        r = ChiVec[i + 1]
        # update B_{i+1} from Vh (Z): shape (r, d_{i+1}, chi_r_{i+1})
        Blist[i + 1] = np.reshape(Z[:r, :], (r, d, Z.shape[1] // d))

        # normalize Schmidt values kept
        tmp = np.linalg.norm(Y[:r])
        llist.append(Y[:r] / tmp if tmp != 0.0 else Y[:r])

        # update B_i by contracting theta_bar with conj(B_{i+1}) over (d_{i+1}, chi_r_{i+1})
        # result shape: (chi_l_i, d_i, r)  → matches (chi_l, d, chi_r)
        Blist[i] = np.tensordot(theta_bar, np.conjugate(Blist[i + 1]),
                                 axes=([2, 3], [1, 2])) / (tmp if tmp != 0.0 else 1.0)

    llist.append(np.array([1.0]))
    return Blist, llist



# ===================== Left-canonical (L → R) =====================

def left_canonicalize(
    psi: MPS,
    method: str = "svd",      # {"svd","qr"}
    chi_max: int | None = None,
    eps: float | None = None,
) -> MPS:
    """
    TEXTBOOK left-canonicalization (sweep LEFT → RIGHT).

    SVD:
      - M = reshape(B_i, (chi_L*d, chi_R)) = U S V†
      - Keep r, set A_i from U[:, :r] reshaped to (chi_L, d, r)
      - Push right: G_right = diag(S[:r]) @ V†[:r,:] into next site's left leg.

    QR:
      - M = Q R; set A_i from Q, push R right.
    """
    assert method in ("svd", "qr")
    Bs = [B.copy() for B in psi.Bs]
    L = len(Bs)

    for i in range(L):
        chiL, d, chiR = Bs[i].shape
        M = Bs[i].reshape(chiL * d, chiR)  # (chi_L*d, chi_R)

        if method == "svd":
            U, S, Vh = np.linalg.svd(M, full_matrices=False)   # U:(chiL*d,r), S:(r,), Vh:(r,chiR)
            r = len(S)
            if eps is not None:
                r = int(np.count_nonzero(S > eps)) or 1
            if chi_max is not None:
                r = min(r, chi_max)

            A_i = U[:, :r].reshape(chiL, d, r)                 # left-isometric
            Bs[i] = A_i

            if i < L - 1:
                G_right = (S[:r, None] * Vh[:r, :])            # (r, chiR)
                # Push into next site's left leg (size chiR)
                Bs[i+1] = np.tensordot(G_right, Bs[i+1], axes=(1, 0))  # -> (r, d_next, chi_next)

        else:  # method == "qr"
            Q, R = np.linalg.qr(M, mode="reduced")             # Q:(chiL*d,r), R:(r,chiR)
            r = Q.shape[1]
            if eps is not None:
                diag_abs = np.abs(np.diag(R))
                r = int(np.count_nonzero(diag_abs > eps)) or 1
            if chi_max is not None:
                r = min(r, chi_max)

            Q = Q[:, :r]
            R = R[:r, :]
            Bs[i] = Q.reshape(chiL, d, r)
            if i < L - 1:
                Bs[i+1] = np.tensordot(R, Bs[i+1], axes=(1, 0))

    return MPS(Bs, [S.copy() for S in psi.Ss], psi.bc)

# ============================== quick test ==============================

if __name__ == "__main__":
    # random finite MPS
    rng = np.random.default_rng(0)
    psi0 = rand_mps()

    # Right-canonical (R→L), textbook SVD (V† makes the new site)
    psiR = right_canonicalize(psi0.copy(), method="svd")
    print("Right-isometry errors (R→L SVD):",
          [f"{right_isometry_error(B):.2e}" for B in psiR.Bs])

    for B in psiR.Bs:
        print(np.tensordot(B, B.conj(), axes=([1,2],[1,2])))  # should be identity
    
    print("="*10)
    # Left-canonical (L→R), textbook SVD (U makes the new site)
    psiL = left_canonicalize(psi0.copy(), method="svd")
    print("Left-isometry  errors (L→R SVD):",
          [f"{left_isometry_error(B):.2e}" for B in psiL.Bs])
    for B in psiL.Bs:
        print(np.tensordot(B, B.conj(), axes=([0,1],[0,1])))  # should be identity