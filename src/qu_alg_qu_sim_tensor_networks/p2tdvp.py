"""
Parallel two-site TDVP (p2TDVP) in the inverse canonical gauge.

Reference: P. Secular, N. Gourianov, M. Lubasch, S. Dolgov, S. R. Clark, D. Jaksch,
"Parallel time-dependent variational principle algorithm for matrix product states",
Phys. Rev. B 101, 235123 (2020), arXiv:1912.06127 (Sec. III and Appendix B).

State:   |psi> = Psi_0 V_0 Psi_1 V_1 ... V_{N-2} Psi_{N-1},   V_j = Lambda_j^{-1}
         Psi_j : (vL, d, vR),  V_j : 1D vector on the bond between sites j and j+1.
         Left environments are built from A_j = Psi_j V_j, right environments from
         B_j = V_{j-1} Psi_j.
Update:  Theta = Psi_L V Psi_R  ->  exp(-i Heff dt/2) Theta  ->  SVD = A' L' B'
         Psi_L = A' L',  V = 1/L',  Psi_R = L' B';  one-site backward: exp(+i Heff dt/2).
Timestep (p partitions, p = 1 or even):
         1. "start" boundaries (neighbours sweep away from each other): forward dt/2
         2. every partition sweeps (in parallel) towards its "end" side
         3. "end" boundaries (neighbours sweep towards each other): forward dt
         4. every partition sweeps back
         5. "start" boundaries: forward dt/2
         The two central partitions sweep away from the centre in the first half.
         For p = 1 this is exactly serial two-site TDVP.

MPO convention: W[i] with legs (wL, wR, i, i*), start state 0, final state D-1
(same as qu_alg_qu_sim_tensor_networks.tdvp / dmrg).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import expm, eigh_tridiagonal

__all__ = ["ParallelTDVP"]


# ----------------------------------------------------------------- local kernels
def _heff2(LE, RE, W1, W2, x, shape):
    x = x.reshape(shape)
    x = np.tensordot(LE, x, axes=(2, 0))
    x = np.tensordot(x, W1, axes=([1, 2], [0, 3]))
    x = np.tensordot(x, W2, axes=([3, 1], [0, 3]))
    x = np.tensordot(x, RE, axes=([1, 3], [0, 1]))
    return x.reshape(-1)


def _heff1(LE, RE, W, x, shape):
    x = x.reshape(shape)
    x = np.tensordot(LE, x, axes=(2, 0))
    x = np.tensordot(x, W, axes=([1, 2], [0, 3]))
    x = np.tensordot(x, RE, axes=([1, 2], [0, 1]))
    return x.reshape(-1)


def _expm_krylov(matvec, v, tau, k=30, tol=1e-12):
    """exp(-i H tau) v with a Lanczos/Krylov approximation (H Hermitian)."""
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return v
    V = [v / nrm]
    alpha, beta = [], []
    w = matvec(V[0])
    a = np.vdot(V[0], w).real
    alpha.append(a)
    w = w - a * V[0]
    for j in range(1, k):
        b = np.linalg.norm(w)
        if b < tol:
            break
        beta.append(b)
        V.append(w / b)
        w = matvec(V[-1]) - b * V[-2]
        a = np.vdot(V[-1], w).real
        alpha.append(a)
        w = w - a * V[-1]
        # full re-orthogonalisation (small k, keeps it stable)
        for q in V:
            w = w - np.vdot(q, w) * q
    m = len(alpha)
    T = np.diag(alpha) + np.diag(beta[: m - 1], 1) + np.diag(beta[: m - 1], -1)
    c = expm(-1j * tau * T)[:, 0]
    return nrm * (np.array(V[:m]).T @ c)


def _update_LE(LE, A, W):
    x = np.tensordot(LE, A, axes=(2, 0))
    x = np.tensordot(W, x, axes=([0, 3], [1, 2]))
    return np.tensordot(A.conj(), x, axes=([0, 1], [2, 1]))


def _update_RE(RE, B, W):
    x = np.tensordot(B, RE, axes=(2, 0))
    x = np.tensordot(x, W, axes=([1, 2], [3, 1]))
    return np.tensordot(x, B.conj(), axes=([1, 3], [2, 1]))


# ----------------------------------------------------------------- engine
class ParallelTDVP:
    """p2TDVP engine. `psi` is a right-canonical MPS (attributes tensors, singular_values) as produced by the package."""

    def __init__(self, psi, H_mpo, chi_max, n_parts=1, eps_rel=1e-12, eps_abs=1e-14,
                 krylov_dim=30, n_threads=None):
        L = len(psi.tensors)
        if n_parts != 1 and (n_parts % 2 or n_parts > L // 2):
            raise ValueError("n_parts must be 1 or an even number <= L//2")
        self.L, self.p, self.chi_max = L, n_parts, chi_max
        self.eps_rel, self.eps_abs, self.k = eps_rel, eps_abs, krylov_dim
        self.W = [np.asarray(W, dtype=complex) for W in H_mpo]
        tensors = [np.asarray(B, dtype=complex) for B in psi.tensors]
        singular_values = [np.asarray(S, dtype=float) for S in psi.singular_values]      # singular_values[i]: bond left of site i
        # inverse canonical gauge: Psi_i = Lambda_{i-1} B_i ; V_i = 1/Lambda_i (bond i|i+1 = singular_values[i+1])
        self.Psi = [singular_values[i][:, None, None] * tensors[i] for i in range(L)]
        self.V = [1.0 / singular_values[i + 1] for i in range(L - 1)]
        D0, DL = self.W[0].shape[0], self.W[-1].shape[1]
        self.LE, self.RE = [None] * L, [None] * L
        self.LE[0] = np.zeros((1, D0, 1), complex); self.LE[0][0, 0, 0] = 1
        self.RE[-1] = np.zeros((1, DL, 1), complex); self.RE[-1][0, DL - 1, 0] = 1
        for i in range(L - 1):
            self.LE[i + 1] = _update_LE(self.LE[i], self.Psi[i] * self.V[i][None, None, :], self.W[i])
        for i in range(L - 1, 0, -1):
            self.RE[i - 1] = _update_RE(self.RE[i], self.V[i - 1][:, None, None] * self.Psi[i], self.W[i])
        self._setup_partitions()
        self.discarded = 0.0                                     # accumulated per timestep
        self.pool = ThreadPoolExecutor(max_workers=n_threads or max(1, self.p))

    # ---- partition layout and sweep directions (Fig. 6 / Appendix B)
    def _setup_partitions(self):
        L, p = self.L, self.p
        base, rem = divmod(L, p)
        self.parts, s = [], 0
        for k in range(p):
            e = s + base + (1 if k < rem else 0)
            self.parts.append((s, e - 1))
            s = e
        if p == 1:
            self.right_first = [True]
        else:
            h = p // 2
            self.right_first = [((h - 1 - k) % 2 == 1) if k < h else ((k - h) % 2 == 0) for k in range(p)]
        # boundaries between partition k and k+1: site pair (e_k, s_{k+1})
        self.start_b, self.end_b = [], []
        for k in range(p - 1):
            pair = (self.parts[k][1], self.parts[k + 1][0])
            if (not self.right_first[k]) and self.right_first[k + 1]:
                self.start_b.append(pair)                        # sweeps start here (away from each other)
            else:
                self.end_b.append(pair)                          # sweeps meet here

    # ---- local operations
    def _split(self, theta):
        cl, d1, d2, cr = theta.shape
        U, S, Vh = np.linalg.svd(theta.reshape(cl * d1, d2 * cr), full_matrices=False)
        tot = float(np.sum(S ** 2))
        keep = int(np.sum(S > max(self.eps_abs, self.eps_rel * S[0])))
        keep = max(1, min(keep, self.chi_max))
        disc = float(np.sum(S[keep:] ** 2)) / tot
        S = S[:keep] / np.linalg.norm(S[:keep])
        return U[:, :keep].reshape(cl, d1, keep), S, Vh[:keep].reshape(keep, d2, cr), disc

    def _two_site(self, i, tau):
        """Forward-evolve sites (i, i+1) by tau; returns discarded weight."""
        theta = np.tensordot(self.Psi[i] * self.V[i][None, None, :], self.Psi[i + 1], axes=(2, 0))
        shape = theta.shape
        LE, RE, W1, W2 = self.LE[i], self.RE[i + 1], self.W[i], self.W[i + 1]
        theta = _expm_krylov(lambda x: _heff2(LE, RE, W1, W2, x, shape), theta.reshape(-1), tau, self.k).reshape(shape)
        A, S, B, disc = self._split(theta)
        self.Psi[i] = A * S[None, None, :]
        self.Psi[i + 1] = S[:, None, None] * B
        self.V[i] = 1.0 / S
        return disc, A, B

    def _one_site_back(self, i, tau):
        shape = self.Psi[i].shape
        LE, RE, W = self.LE[i], self.RE[i], self.W[i]
        self.Psi[i] = _expm_krylov(lambda x: _heff1(LE, RE, W, x, shape), self.Psi[i].reshape(-1), -tau, self.k).reshape(shape)

    def _boundary(self, pair, tau):
        i, j = pair
        disc, A, B = self._two_site(i, tau)
        self.LE[j] = _update_LE(self.LE[i], A, self.W[i])
        self.RE[i] = _update_RE(self.RE[j], B, self.W[j])
        return disc

    def _sweep(self, k, right, half_dt, first_half):
        """Partial sweep of partition k. Starts with a backward step if it starts at a boundary."""
        a, b = self.parts[k]
        L = self.L
        start_is_boundary = (a > 0) if right else (b < L - 1)
        end_is_chain = (b == L - 1) if right else (a == 0)
        disc = 0.0
        if start_is_boundary:
            self._one_site_back(a if right else b, half_dt)
        rng = range(a, b) if right else range(b - 1, a - 1, -1)
        for i in rng:
            d, A, B = self._two_site(i, half_dt)
            disc += d
            if right:
                self.LE[i + 1] = _update_LE(self.LE[i], A, self.W[i])
                if not (i + 1 == b and end_is_chain):
                    self._one_site_back(i + 1, half_dt)
            else:
                self.RE[i] = _update_RE(self.RE[i + 1], B, self.W[i + 1])
                if not (i == a and end_is_chain):
                    self._one_site_back(i, half_dt)
        return disc

    # ---- public API
    def step(self, dt):
        h = 0.5 * dt
        run = lambda fn, items: list(self.pool.map(fn, items)) if self.p > 1 else [fn(x) for x in items]
        disc = sum(run(lambda pr: self._boundary(pr, h), self.start_b))
        disc += sum(run(lambda k: self._sweep(k, self.right_first[k], h, True), range(self.p)))
        disc += sum(run(lambda pr: self._boundary(pr, dt), self.end_b))
        disc += sum(run(lambda k: self._sweep(k, not self.right_first[k], h, False), range(self.p)))
        disc += sum(run(lambda pr: self._boundary(pr, h), self.start_b))
        self.discarded = disc
        return self

    def mps_tensors(self):
        """Plain MPS tensors (Psi_i V_i ..., Psi_{N-1}) of the current state."""
        return [self.Psi[i] * self.V[i][None, None, :] for i in range(self.L - 1)] + [self.Psi[-1]]

    def max_chi(self):
        return max(len(v) for v in self.V)
