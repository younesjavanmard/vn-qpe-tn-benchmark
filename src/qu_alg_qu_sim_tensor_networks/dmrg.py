from __future__ import annotations
import numpy as np
import scipy.sparse
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator as NP_LinearOperator
from scipy.sparse.linalg import eigsh as np_eigsh, lobpcg as np_lobpcg

from qu_alg_qu_sim_tensor_networks.backend import (
    BACK, xp, cp, jnp, _has_cupy, _has_jax, to_device, to_numpy,
)

# CuPy sparse eigensolver (optional)
cp_eigsh = cp_lobpcg = None
if _has_cupy:
    try:
        from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
        from cupyx.scipy.sparse.linalg import lobpcg as cp_lobpcg
    except Exception:
        pass


# ========= Robust eigensolver helpers =========

class _HermitizedLO(NP_LinearOperator):
    """
    Hermitize a (possibly slightly non-Hermitian) LinearOperator A by applying (A + A^H)/2.
    This often stabilizes Krylov solvers like ARPACK/LOBPCG.

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        The original operator.

    Notes
    -----
    Works in real/complex dtype. Uses *NumPy/SciPy* host-side LinearOperator.
    """
    def __init__(self, A: NP_LinearOperator):
        self.A = A
        super().__init__(dtype=A.dtype, shape=A.shape)

    def _matvec(self, x):
        y = self.A.matvec(x)
        return 0.5 * (y + np.conjugate(y))


def safe_smallest_eigpair(
    A: NP_LinearOperator,
    v0=None, which='SA',
    maxiter=4000, tol=1e-6, ncv=None,
    retries=3, random_state=1234,
    hermitize=True,
    dense_limit_cpu=4096,
    dense_limit_accel=8192,
):
    """
    Compute the smallest eigenpair of an (approximately) Hermitian operator robustly.

    Strategy (by backend)
    ---------------------
    NumPy/CPU:
        1) ARPACK (`eigsh`) with restarts
        2) LOBPCG fallback
        3) Dense `eigh` if dim <= dense_limit_cpu
    CuPy/GPU:
        1) Try GPU ARPACK/LOBPCG via `cupyx.scipy.sparse.linalg` if available
        2) Dense `cupy.linalg.eigh` if dim <= dense_limit_accel
        3) Fallback: move to CPU path
    JAX/GPU/TPU:
        1) Dense `jax.numpy.linalg.eigh` if dim <= dense_limit_accel
        2) Fallback: move to CPU path

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        Two-site effective Hamiltonian as a LinearOperator (NumPy/SciPy API).
    v0 : ndarray, optional
        Initial guess vector.
    which : str
        ARPACK selector (default 'SA' = smallest algebraic).
    maxiter : int
        Maximum iterations for iterative solvers.
    tol : float
        Tolerance for eigensolve.
    ncv : int or None
        Krylov subspace size; sensible defaults chosen when None.
    retries : int
        Number of ARPACK random restarts on failure.
    random_state : int
        RNG seed for restarts.
    hermitize : bool
        If True, wrap `A` with Hermitization.
    dense_limit_cpu : int
        Dimension cutoff for dense CPU `eigh`.
    dense_limit_accel : int
        Dimension cutoff for dense GPU/TPU `eigh`.

    Returns
    -------
    eigval : float
        The smallest eigenvalue.
    eigvec : ndarray
        Corresponding eigenvector (1D array).

    Notes
    -----
    - For JAX/TPU we *materialize* a dense matrix for sizes up to `dense_limit_accel`.
      For larger problems we fall back to the CPU path.
    - For CuPy, we first try `cupyx.scipy.sparse.linalg.eigsh/lobpcg`. If they are
      unavailable for your CuPy version, we use dense `cupy.linalg.eigh` within
      `dense_limit_accel`, otherwise fall back to CPU.
    """
    rng = np.random.default_rng(random_state)
    n = A.shape[0]
    A_use = _HermitizedLO(A) if (hermitize and isinstance(A, NP_LinearOperator)) else A

    def _assemble_dense_numpy(op: NP_LinearOperator):
        I = np.eye(n, dtype=op.dtype)
        M = np.column_stack([op.matvec(I[:, i]) for i in range(n)])
        M = 0.5 * (M + M.conj().T)
        return M

    # ---------- GPU JAX path (dense only) ----------
    if BACK == "jax":
        if n <= dense_limit_accel:
            M = _assemble_dense_numpy(A_use)
            Mx = jnp.asarray(M)
            w, V = jnp.linalg.eigh(Mx)
            # move back to host
            w = np.array(w) ; V = np.array(V)
            return float(w[0]), V[:, 0]
        # too big for dense on accel: fall back to CPU path below

    # ---------- GPU CuPy path ----------
    if BACK == "cupy":
        # Try GPU ARPACK/LOBPCG
        try:
            # Wrap a CuPy LinearOperator that calls A_use.matvec on host→device
            def _cp_matvec(xd):
                xh = cp.asnumpy(xd)
                yh = A_use.matvec(xh)
                return cp.asarray(yh)

            CP_LO = CP_LinearOperator(
                shape=A_use.shape,
                matvec=_cp_matvec,
                dtype=A_use.dtype
            )

            # Try ARPACK first
            try:
                w, V = cp_eigsh(CP_LO, k=1, which=which, maxiter=maxiter, tol=tol)
                w = cp.asnumpy(w).ravel()
                v = cp.asnumpy(V)[:, 0]
                return float(w[0]), v
            except Exception:
                # Try LOBPCG
                X0 = cp.random.standard_normal((n, 1), dtype=CP_LO.dtype)
                w, V = cp_lobpcg(CP_LO, X0, tol=max(tol, 1e-6), maxiter=2000, largest=False)
                w = cp.asnumpy(w).ravel()
                v = cp.asnumpy(V)[:, 0]
                return float(w[0]), v

        except Exception:
            # Dense on GPU?
            if n <= dense_limit_accel:
                M = _assemble_dense_numpy(A_use)
                Mg = cp.asarray(M)
                w, V = cp.linalg.eigh(Mg)
                w = cp.asnumpy(w) ; V = cp.asnumpy(V)
                return float(w[0]), V[:, 0]
            # else fall through to CPU path

    # ---------- CPU NumPy/SciPy path ----------
    if ncv is None:
        ncv = min(n, max(20, 14))

    v = None
    if v0 is not None:
        v = np.array(v0, dtype=A.dtype, copy=True).ravel()
        nv = norm(v)
        if np.isfinite(nv) and nv > 0:
            v /= nv
        else:
            v = None

    this_tol = tol
    this_ncv = ncv

    for _ in range(retries):
        try:
            w, V = np_eigsh(A_use, k=1, which=which, v0=v,
                            maxiter=maxiter, tol=this_tol, ncv=this_ncv,
                            return_eigenvectors=True)
            return float(w[0]), V[:, 0]
        except Exception:
            v = rng.standard_normal(n, dtype=A.dtype)
            v /= max(norm(v), 1e-16)
            this_ncv = min(n, max(this_ncv + 10, int(1.25 * this_ncv)))
            this_tol = max(this_tol, 5e-6)

    try:
        X = rng.standard_normal((n, 1), dtype=A.dtype)
        X /= max(norm(X), 1e-16)
        w, V = np_lobpcg(A_use, X, tol=max(tol, 1e-6), maxiter=2000, largest=False)
        w = np.array(w).ravel()
        return float(w[0]), np.array(V)[:, 0]
    except Exception:
        pass

    if n <= dense_limit_cpu:
        from scipy.linalg import eigh
        M = _assemble_dense_numpy(A_use)
        w, V = eigh(M, overwrite_a=True, check_finite=False)
        return float(w[0]), V[:, 0]

    raise RuntimeError("safe_smallest_eigpair: ARPACK/LOBPCG failed; "
                       "problem too large for dense fallback.")


# ========= DMRG engine =========

class DMRGEngine:
    """
    Two-site DMRG with right-canonical MPS storage and a robust eigensolver.

    Parameters
    ----------
    psi : MPS-like
        State object with attributes:
          - L (int), bc (str)
          - Bs (list of rank-3 arrays): right-canonical site tensors (vL, i, vR)
          - Ss (list of 1D arrays): Schmidt singular values per bond (vC)
          - nbonds (int)
          - two_site_center(i) -> rank-4 array: current two-site center (vL, i, j, vR)
    model : object
        Must expose `H_mpo`, a list of rank-4 MPO tensors per site: (wL, wR, i, i*)
    chi_max : int
        Truncation bond dimension cap after SVD.
    eps : float
        Truncation tolerance for discarded weight.
    eig_* : various
        Parameters forwarded to `safe_smallest_eigpair`.

    Notes
    -----
    - Backend CPU/GPU/TPU is selected globally via BACK variable at import time.
    - The two-site effective Hamiltonian is represented as a SciPy LinearOperator,
      so the CPU path is always available. GPU/TPU paths are used opportunistically.
    """
    def __init__(self, psi, model, chi_max, eps,
                 eig_maxiter=4000, eig_tol=1e-6, eig_retries=3,
                 eig_ncv=None, eig_dense_limit=4096, eig_random_state=1234,
                 eig_hermitize=True):
        from qu_alg_qu_sim_tensor_networks.mps import svd_split_truncate

        assert psi.L == model.L and psi.bc == model.bc
        self.H_mpo = model.H_mpo
        self.psi = psi
        self.svd_split_truncate = svd_split_truncate

        self.left_envs = [None] * psi.L
        self.right_envs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps

        # eigensolver knobs
        self._eig_maxiter = eig_maxiter
        self._eig_tol = eig_tol
        self._eig_retries = eig_retries
        self._eig_ncv = eig_ncv
        self._eig_dense_limit = eig_dense_limit
        self._eig_random_state = eig_random_state
        self._eig_hermitize = eig_hermitize

        # move MPS tensors and MPO to compute device
        psi.Bs = [to_device(B) for B in psi.Bs]
        psi.Ss = [to_device(S) for S in psi.Ss]
        self.H_mpo = [to_device(W) for W in self.H_mpo]

        # initialize left/right environments on device
        D = self.H_mpo[0].shape[0]
        chi = psi.Bs[0].shape[0]
        left_env = xp.zeros([chi, D, chi], dtype=float)  # vL wL* vL*
        right_env = xp.zeros([chi, D, chi], dtype=float)  # vR* wR* vR
        left_env[:, 0, :] = xp.eye(chi)
        right_env[:, D - 1, :] = xp.eye(chi)
        self.left_envs[0] = left_env
        self.right_envs[-1] = right_env

        # prepare right environments
        for i in range(psi.L - 1, 1, -1):
            self.absorb_right_site(i, psi.Bs[i])

    def sweep(self) -> float:
        """
        Perform one full left-to-right and right-to-left two-site DMRG sweep.

        Returns
        -------
        E0 : float
            Ground-state energy estimate from the last updated bond.
        """
        # left -> right
        for i in range(self.psi.nbonds - 1):
            self.update_bond(i)
        # right -> left
        for i in range(self.psi.nbonds - 1, 0, -1):
            E0 = self.update_bond(i)
        return E0

    def update_bond(self, i: int) -> float:
        """
        Optimize the two-site center at bond i,(i+1).

        Steps
        -----
        1) Build two-site effective Hamiltonian Heff as LinearOperator
        2) Diagonalize for the smallest eigenpair
        3) Split & truncate the optimized theta → (Ai, Sj, Bj)
        4) Update canonical MPS tensors and environments

        Parameters
        ----------
        i : int
            Left site index of the optimized bond.

        Returns
        -------
        E0 : float
            Local ground-state energy for that bond.
        """
        j = (i + 1) % self.psi.L
        Heff = Heff2(self.left_envs[i], self.right_envs[j], self.H_mpo[i], self.H_mpo[j])
        theta_guess = self.psi.two_site_center(i)
        E0, theta = self.diag(Heff, theta_guess)

        # diag() returns theta on the host (the eigensolver hands back a NumPy
        # vector); move it back to the compute device so svd_split_truncate
        # and the following contractions stay on-device (no-op on CPU).
        theta = to_device(theta)
        Ai, Sj, Bj = self.svd_split_truncate(theta, self.chi_max, self.eps)

        # write back into right-canonical storage
        Gi = xp.tensordot(xp.diag(xp.asarray(self.psi.Ss[i]) ** (-1)), Ai, axes=(1, 0))
        self.psi.Bs[i] = xp.tensordot(Gi, xp.diag(Sj), axes=(2, 0))
        self.psi.Ss[j] = Sj                                                  # vC
        self.psi.Bs[j] = Bj                                                  # vC j vR

        # environments
        self.absorb_left_site(i, Ai)
        self.absorb_right_site(j, Bj)
        return E0

    def diag(self, Heff: 'Heff2', guess: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Diagonalize the two-site effective Hamiltonian for the smallest eigenpair.

        Parameters
        ----------
        Heff : Heff2
            Two-site effective Hamiltonian as a LinearOperator.
        guess : ndarray
            Initial theta tensor (vL, i, j, vR).

        Returns
        -------
        E0 : float
            Lowest eigenvalue.
        theta_opt : ndarray
            Optimized theta (vL, i, j, vR).
        """
        # eigensolver runs host-side; ensure the initial guess is on the host
        # (guess may arrive as a device array on GPU backends).
        g = to_numpy(guess).reshape([Heff.shape[1]])

        # Early NaN/Inf probe
        z = np.random.standard_normal(Heff.shape[1]).astype(Heff.dtype, copy=False)
        probe = Heff._matvec(z)
        if not np.all(np.isfinite(probe)):
            raise ValueError("Heff matvec produced NaN/Inf; check MPO/env contractions.")

        E0, v0 = safe_smallest_eigpair(
            Heff, v0=g, which='SA',
            maxiter=self._eig_maxiter, tol=self._eig_tol, ncv=self._eig_ncv,
            retries=self._eig_retries, random_state=self._eig_random_state,
            hermitize=self._eig_hermitize
        )
        return E0, np.reshape(v0, Heff.theta_shape)

    def absorb_right_site(self, i: int, B: np.ndarray) -> None:
        """
        Update right environment tensor immediately to the left of site i.

        Parameters
        ----------
        i : int
            Site index whose right environment is updated.
        B : ndarray
            Right-canonical site tensor at i with shape (vL, i, vR).
        """
        j = (i - 1) % self.psi.L
        right_env = self.right_envs[i]     # vR* wR* vR
        Bc = B.conj()        # vL* i* vR*
        W = self.H_mpo[i]    # wL wR i i*

        right_env = xp.tensordot(B, right_env, axes=(2, 0))
        right_env = xp.tensordot(right_env, W, axes=([1, 2], [3, 1]))
        right_env = xp.tensordot(right_env, Bc, axes=([1, 3], [2, 1]))
        self.right_envs[j] = right_env

    def absorb_left_site(self, i: int, A: np.ndarray) -> None:
        """
        Update left environment tensor immediately to the right of site i.

        Parameters
        ----------
        i : int
            Site index whose left environment is updated.
        A : ndarray
            Left-canonical site tensor at i with shape (vL, i, vR).
        """
        j = (i + 1) % self.psi.L
        left_env = self.left_envs[i]     # vL wL vL*
        Ac = A.conj()        # vL* i* vR*
        W = self.H_mpo[i]    # wL wR i i*

        left_env = xp.tensordot(left_env, A, axes=(2, 0))
        left_env = xp.tensordot(W, left_env, axes=([0, 3], [1, 2]))
        left_env = xp.tensordot(Ac, left_env, axes=([0, 1], [2, 1]))
        self.left_envs[j] = left_env


class Heff2(NP_LinearOperator):
    """
    Two-site effective Hamiltonian as a SciPy LinearOperator.

    Contracts left_env -- W1 -- W2 -- right_env with a two-site vector theta (flattened)
    without ever materializing the full Heff matrix.

    Parameters
    ----------
    left_env : ndarray
        Left environment (vL, wL*, vL*).
    right_env : ndarray
        Right environment (vR*, wR*, vR).
    W1 : ndarray
        Left-site MPO tensor (wL, wC, i, i*).
    W2 : ndarray
        Right-site MPO tensor (wC, wR, j, j*).

    Attributes
    ----------
    theta_shape : tuple
        Shape (vL, i, j, vR) used for reshaping vectors inside matvec.
    """
    def __init__(self, left_env, right_env, W1, W2):
        self.left_env = left_env  # vL wL* vL*
        self.right_env = right_env  # vR* wR* vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi1, chi2 = left_env.shape[0], right_env.shape[2]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chi1, d1, d2, chi2)   # vL i j vR
        n = chi1 * d1 * d2 * chi2
        super().__init__(dtype=W1.dtype, shape=(n, n))

    def _matvec(self, theta):
        """Compute y = Heff * theta for flattened theta (n,)."""
        x = xp.reshape(xp.asarray(theta), self.theta_shape)
        x = xp.tensordot(self.left_env, x, axes=(2, 0))
        x = xp.tensordot(x, self.W1, axes=([1, 2], [0, 3]))
        x = xp.tensordot(x, self.W2, axes=([3, 1], [0, 3]))
        x = xp.tensordot(x, self.right_env, axes=([1, 3], [0, 1]))
        return to_numpy(xp.reshape(x, self.shape[0]))


