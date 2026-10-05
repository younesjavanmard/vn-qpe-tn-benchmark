"""Time evolution of finite matrix product states with the time-dependent variational principle.

The integrator evolves an MPS under a Hamiltonian given as an MPO. One time step consists of a
left-to-right and a right-to-left sweep. Site tensors (one-site variant) or pairs of site tensors
(two-site variant) are evolved forward by half a time step with the local effective Hamiltonian;
the tensor between two consecutive forward updates is evolved backward by half a time step, which
keeps the evolution on the manifold of MPS. The local exponentials are computed with a Krylov
(Lanczos) approximation.

Tensor conventions
------------------
MPS tensors have legs (left bond, physical, right bond).
MPO tensors have legs (left bond, right bond, physical out, physical in); the first MPO channel
is the start state and the last one the final state.
Environments have legs (bra bond, MPO bond, ket bond) on the left and (ket bond, MPO bond, bra
bond) on the right.
"""
import numpy as np
import scipy.sparse.linalg
from qu_alg_qu_sim_tensor_networks.lanczos import lanczos_expm_multiply

from qu_alg_qu_sim_tensor_networks.dmrg import EffectivePairOperator
from qu_alg_qu_sim_tensor_networks import mps
from qu_alg_qu_sim_tensor_networks.backend import xp, to_device, to_numpy


class TDVPIntegrator:
    """Time-dependent variational principle (TDVP) integrator for a finite MPS.

    One time step is a symmetric left-to-right and right-to-left sweep: site (or pair)
    tensors are evolved forward by half a step and the intermediate bond (or site)
    tensors backward by half a step, so that the evolution stays on the MPS manifold.

    * ``step_one_site(dt)``  -- one-site TDVP; the bond dimensions are kept fixed.
    * ``step_two_site(dt)``  -- two-site TDVP; bond dimensions may grow and are truncated
      to ``chi_max`` and the singular-value cutoff ``eps``.

    During a sweep the tensors left of the orthogonality center are stored in
    left-canonical form; at the end of every step all tensors are right-canonical again.

    Parameters
    ----------
    psi : MPS
        Initial state; finite boundary conditions, right-canonical tensors.
    model :
        Object providing the generator of the evolution as ``model.H_mpo`` and the number
        of sites as ``model.L``.
    chi_max : int
        Maximal bond dimension in two-site steps.
    eps : float
        Singular values below ``eps`` are discarded in two-site steps.

    Attributes
    ----------
    psi : MPS
        The evolved state (updated in place).
    H_mpo : list of ndarray
        MPO tensors of the generator.
    left_envs, right_envs : list of ndarray
        Cached environments: ``left_envs[i]`` contracts <psi|H|psi> over all sites left of
        site ``i`` and ``right_envs[i]`` over all sites right of site ``i``.
    """

    def __init__(self, psi, model, chi_max, eps):
        if psi.L != model.L:
            raise ValueError(f"State has {psi.L} sites but the MPO has {model.L}.")
        if psi.bc != 'finite':
            raise ValueError("TDVPIntegrator requires an MPS with finite boundary conditions.")
        # move the state and the MPO to the compute device
        psi.tensors = [to_device(B) for B in psi.tensors]
        psi.singular_values = [to_device(S) for S in psi.singular_values]
        self.H_mpo = [to_device(W) for W in model.H_mpo]
        self.psi = psi
        self.left_envs = [None] * psi.L
        self.right_envs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        # boundary environments: the MPO starts in channel 0 and ends in channel D-1
        D = self.H_mpo[0].shape[0]
        chi = psi.tensors[0].shape[0]
        left_env = xp.zeros([chi, D, chi], dtype=float)
        right_env = xp.zeros([chi, D, chi], dtype=float)
        left_env[:, 0, :] = xp.eye(chi)
        right_env[:, D - 1, :] = xp.eye(chi)
        self.left_envs[0] = left_env
        self.right_envs[-1] = right_env
        # all right environments are needed before the first left-to-right sweep
        for i in range(psi.L - 1, 0, -1):
            self.grow_right_environment(i, psi.tensors[i])

    # ------------------------------------------------------------------ time steps
    def step_one_site(self, dt):
        """Advance the state by one time step ``dt`` with one-site TDVP.

        The bond dimensions are not changed. Every site tensor is evolved forward by
        ``dt/2`` in each of the two sweeps (the last site once by ``dt``), and the bond
        matrix between consecutive sites is evolved backward by ``dt/2``.

        Parameters
        ----------
        dt : float or complex
            Time step; ``dt = -1j * tau`` gives imaginary-time evolution.

        Returns
        -------
        MPS
            The evolved state, all tensors right-canonical.
        """
        psi = self.psi
        L = self.psi.L
        # left-to-right sweep
        theta = self.psi.site_center_tensor(0)
        for i in range(L - 1):
            theta = self.evolve_site(i, 0.5*dt, theta)             # forward
            Ai, theta = self.split_site_tensor(i, theta, move_right=True)
            # theta is now the bond matrix between sites i and i+1
            psi.tensors[i] = Ai                                       # left-canonical until the return sweep
            self.grow_left_environment(i, Ai)
            theta = self.evolve_bond(i, -0.5*dt, theta)             # backward
            j = i + 1
            Bj = self.psi.tensors[j]
            theta = xp.tensordot(theta, Bj, axes=(1, 0))              # bond matrix times next site tensor
        # last site: forward by a full step
        i = L - 1
        theta = self.evolve_site(i, dt, theta)
        theta, Bi = self.split_site_tensor(i, theta, move_right=False)
        self.psi.tensors[i] = Bi
        self.grow_right_environment(i, Bi)
        # right-to-left sweep
        for i in reversed(range(L - 1)):
            theta = self.evolve_bond(i, -0.5*dt, theta)             # backward
            Ai = self.psi.tensors[i]                                  # left-canonical from the first sweep
            theta = xp.tensordot(Ai, theta, axes=(2, 0))
            theta = self.evolve_site(i, 0.5*dt, theta)             # forward
            theta, Bi = self.split_site_tensor(i, theta, move_right=False)
            self.psi.tensors[i] = Bi
            if i > 0:  # for i = 0 it would overwrite the right boundary right_envs[L-1]
                self.grow_right_environment(i, Bi)
        # What remains is the 1x1 matrix of the trivial left boundary bond: a pure phase for a
        # normalized state. It is multiplied back into the first tensor.
        if theta.shape != (1, 1) or abs(abs(theta[0]) - 1.) >= 1.e-10:
            raise RuntimeError("One-site TDVP step did not preserve the norm of the state.")
        self.psi.tensors[0] *= theta[0, 0]
        return psi

    def step_two_site(self, dt):
        """Advance the state by one time step ``dt`` with two-site TDVP.

        Pairs of neighbouring site tensors are evolved forward by ``dt/2`` in each sweep
        (the last pair once by ``dt``) and split by a truncated SVD, so the bond dimensions
        can grow up to ``chi_max``; the site between two consecutive pairs is evolved
        backward by ``dt/2``.

        Parameters
        ----------
        dt : float or complex
            Time step; ``dt = -1j * tau`` gives imaginary-time evolution.

        Returns
        -------
        MPS
            The evolved state, all tensors right-canonical.
        """
        psi = self.psi
        L = self.psi.L
        # left-to-right sweep
        theta = self.psi.pair_center_tensor(0)
        for i in range(L - 2):
            j = i + 1
            k = i + 2
            Ai, S, Bj = self.evolve_pair(i, 0.5*dt, theta)          # forward
            psi.tensors[i] = Ai                                       # left-canonical until the return sweep
            self.grow_left_environment(i, Ai)
            theta = xp.tensordot(xp.diag(S), Bj, axes=(1, 0))         # center tensor of site j
            theta = self.evolve_site(j, -0.5*dt, theta)             # backward
            Bk = self.psi.tensors[k]
            theta = xp.tensordot(theta, Bk, axes=(2, 0))              # pair (j, k)
        # last pair: forward by a full step
        i = L - 2
        j = L - 1
        Ai, S, Bj = self.evolve_pair(i, dt, theta)
        theta = xp.tensordot(Ai, xp.diag(S), axes=(2, 0))             # center tensor of site i
        self.psi.tensors[j] = Bj
        self.grow_right_environment(j, Bj)
        # right-to-left sweep
        for i in reversed(range(L - 2)):
            j = i + 1
            theta = self.evolve_site(j, -0.5*dt, theta)             # backward
            Ai = self.psi.tensors[i]                                  # left-canonical from the first sweep
            theta = xp.tensordot(Ai, theta, axes=(2, 0))              # pair (i, j)
            Ai, S, Bj = self.evolve_pair(i, 0.5*dt, theta)          # forward
            self.psi.tensors[j] = Bj
            self.grow_right_environment(j, Bj)
            theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))
        # The left boundary bond is trivial, so the remaining center tensor of site 0 is
        # already right-canonical.
        self.psi.tensors[0] = theta
        return psi

    # ------------------------------------------------------------------ local updates
    def evolve_bond(self, i, dt, theta):
        """Return exp(-i H_bond dt) applied to the bond matrix between sites ``i`` and ``i+1``."""
        Heff = EffectiveBondOperator(self.left_envs[i + 1], self.right_envs[i])
        theta = xp.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        return xp.reshape(theta, Heff.theta_shape)

    def evolve_site(self, i, dt, theta):
        """Return exp(-i H_site dt) applied to the tensor of site ``i`` (no truncation)."""
        Heff = EffectiveSiteOperator(self.left_envs[i], self.right_envs[i], self.H_mpo[i])
        theta = xp.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        return xp.reshape(theta, Heff.theta_shape)

    def evolve_pair(self, i, dt, theta):
        """Evolve the pair of sites ``(i, i+1)`` by ``dt`` and split it by a truncated SVD.

        Returns the left-canonical tensor of site ``i``, the normalized singular values on the
        bond (also stored in ``psi.singular_values[i+1]``), and the right-canonical tensor of
        site ``i+1``.
        """
        j = i + 1
        Heff = EffectivePairOperator(self.left_envs[i], self.right_envs[j], self.H_mpo[i], self.H_mpo[j])
        flat = xp.reshape(theta, [Heff.shape[0]])
        flat = to_device(self.expm_multiply(Heff, flat, dt))
        theta = xp.reshape(flat, Heff.theta_shape)
        Ai, S, Bj = mps.split_pair(theta, self.chi_max, self.eps)
        self.psi.singular_values[j] = S
        return Ai, S, Bj

    def split_site_tensor(self, i, theta, move_right=True):
        """Split the center tensor of site ``i`` by an SVD (no truncation).

        With ``move_right=True`` the result is a left-canonical tensor of site ``i`` and the bond
        matrix to its right; otherwise the bond matrix to its left and a right-canonical tensor.
        The normalized singular values are stored on the corresponding bond.
        """
        chivL, d, chivR = theta.shape
        if move_right:
            theta = xp.reshape(theta, [chivL * d, chivR])
            A, S, V = mps.svd(theta, full_matrices=False)
            S = S / xp.linalg.norm(S)
            self.psi.singular_values[i + 1] = S
            chivC = len(S)
            A = xp.reshape(A, [chivL, d, chivC])
            theta = xp.tensordot(xp.diag(S), V, axes=(1, 0))
            return A, theta
        else:
            theta = xp.reshape(theta, [chivL, d * chivR])
            U, S, B = mps.svd(theta, full_matrices=False)
            S = S / xp.linalg.norm(S)
            self.psi.singular_values[i] = S
            chivC = len(S)
            B = xp.reshape(B, [chivC, d, chivR])
            theta = xp.tensordot(U, xp.diag(S), axes=(1, 0))
            return theta, B

    # ------------------------------------------------------------------ environments
    def grow_right_environment(self, i, B):
        """Extend ``right_envs[i]`` by the right-canonical tensor ``B`` of site ``i``; store it as
        ``right_envs[i-1]``."""
        j = (i - 1) % self.psi.L
        right_env = self.right_envs[i]
        Bc = B.conj()
        W = self.H_mpo[i]
        right_env = xp.tensordot(B, right_env, axes=(2, 0))
        right_env = xp.tensordot(right_env, W, axes=([1, 2], [3, 1]))
        right_env = xp.tensordot(right_env, Bc, axes=([1, 3], [2, 1]))
        self.right_envs[j] = right_env

    def grow_left_environment(self, i, A):
        """Extend ``left_envs[i]`` by the left-canonical tensor ``A`` of site ``i``; store it as
        ``left_envs[i+1]``."""
        j = (i + 1) % self.psi.L
        left_env = self.left_envs[i]
        Ac = A.conj()
        W = self.H_mpo[i]
        left_env = xp.tensordot(left_env, A, axes=(2, 0))
        left_env = xp.tensordot(W, left_env, axes=([0, 3], [1, 2]))
        left_env = xp.tensordot(Ac, left_env, axes=([0, 1], [2, 1]))
        self.left_envs[j] = left_env

    def expm_multiply(self, H, psi0, dt):
        """Return exp(-i H dt) psi0, computed with a Lanczos/Krylov approximation."""
        return lanczos_expm_multiply(H, psi0, dt)


class EffectiveSiteOperator(scipy.sparse.linalg.LinearOperator):
    """Effective Hamiltonian of one site.

    Acts on the flattened tensor of a single site by contracting it with the left environment,
    the MPO tensor of that site, and the right environment. Used for the one-site TDVP updates.

    Parameters
    ----------
    left_env, right_env : ndarray
        Environments of the site.
    W1 : ndarray
        MPO tensor of the site, legs (left bond, right bond, physical out, physical in).
    prefactor : float, optional
        Kept for interface compatibility; not used.
    """

    def __init__(self, left_env, right_env, W1, prefactor=1.):
        self.left_env = left_env
        self.right_env = right_env
        self.W1 = W1
        chi1, chi2 = left_env.shape[0], right_env.shape[2]
        d1 = W1.shape[2]
        self.theta_shape = (chi1, d1, chi2)
        self.shape = (chi1 * d1 * chi2, chi1 * d1 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        x = xp.reshape(xp.asarray(theta), self.theta_shape)
        x = xp.tensordot(self.left_env, x, axes=(2, 0))
        x = xp.tensordot(x, self.W1, axes=([1, 2], [0, 3]))
        x = xp.tensordot(x, self.right_env, axes=([1, 2], [0, 1]))
        return xp.reshape(x, self.shape[0])


class EffectiveBondOperator(scipy.sparse.linalg.LinearOperator):
    """Effective Hamiltonian of a bond.

    Acts on the flattened bond matrix between two neighbouring sites by contracting it with the
    left and right environments. Used for the backward bond updates of one-site TDVP.

    Parameters
    ----------
    left_env, right_env : ndarray
        Environments to the left and right of the bond.
    prefactor : float, optional
        Kept for interface compatibility; not used.
    """

    def __init__(self, left_env, right_env, prefactor=1.):
        self.left_env = left_env
        self.right_env = right_env
        chi1, chi2 = left_env.shape[0], right_env.shape[2]
        self.theta_shape = (chi1, chi2)
        self.shape = (chi1 * chi2, chi1 * chi2)
        self.dtype = left_env.dtype

    def _matvec(self, theta):
        x = xp.reshape(xp.asarray(theta), self.theta_shape)
        x = xp.tensordot(self.left_env, x, axes=(2, 0))
        x = xp.tensordot(x, self.right_env, axes=([1, 2], [1, 0]))
        return xp.reshape(x, self.shape[0])
