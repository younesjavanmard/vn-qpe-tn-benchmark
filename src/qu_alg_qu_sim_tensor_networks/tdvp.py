import numpy as np
import scipy.sparse.linalg
from qu_alg_qu_sim_tensor_networks.lanczos import lanczos_expm_multiply

from qu_alg_qu_sim_tensor_networks.dmrg import Heff2
from qu_alg_qu_sim_tensor_networks import mps
from qu_alg_qu_sim_tensor_networks.backend import xp, to_device, to_numpy


class TDVPEngine:
    """TDVP algorithm for finite systems, implemented as class holding the necessary data.
    Note that this class is very similar to `dmrg.DMRGEngine`.
    We could use a common base class; but to keep things maximally simple and readable,
    we rather duplicate the code for the `__init__`, `absorb_left_site`, and `absorb_right_site` methods.
    Also, here we generalize the sweep to temporarily change the MPS to a mixed canonical form
    and directly save `A` tensors in it. This means that the MPS methods (which *assume*
    that the tensors are all right-canonical) would give wrong results *during* the sweep; yet
    we recover the all-right-canonical B form on each site at the end of the sweep.
    Parameters
    ----------
    psi, chi_max, eps:
        See attributes below.
    model :
        The model with the Hamiltonian for time evolution as `model.H_mpo`.
    Attributes
    ----------
    psi : MPS
        The current state to be evolved.
    H_mpo : list of W tensors with legs ``wL wR i i*``
        The Hamiltonian as an MPO.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.svd_split_truncate`.
        Only used when we evolve two-site wave functions!
    left_envs, right_envs : list of np.Array[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        ``left_envs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``right_envs[i]`` for all parts right of site `i`.
        Each ``left_envs[i]`` has legs ``vL wL* vL*``, ``right_envs[i]`` has legs ``vR* wR* vR``
    """
    def __init__(self, psi, model, chi_max, eps):
        """Initialize the TDVP engine and build all right environments.

        Parameters
        ----------
        psi : MPS
            Initial state (must be finite and right-canonical).
        model : model object
            Model providing ``model.H_mpo`` and ``model.L``.
        chi_max : int
            Maximum bond dimension for SVD truncation (two-site sweep only).
        eps : float
            Singular-value cutoff for SVD truncation (two-site sweep only).
        """
        assert psi.L == model.L   # ensure compatibility
        if psi.bc != 'finite':
            raise ValueError("This TDVP implementation works only for finite MPS.")
        # move MPS tensors and MPO to compute device
        psi.Bs = [to_device(B) for B in psi.Bs]
        psi.Ss = [to_device(S) for S in psi.Ss]
        self.H_mpo = [to_device(W) for W in model.H_mpo]
        self.psi = psi
        self.left_envs = [None] * psi.L
        self.right_envs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        # initialize left and right environment on device
        D = self.H_mpo[0].shape[0]
        chi = psi.Bs[0].shape[0]
        left_env = xp.zeros([chi, D, chi], dtype=float)  # vL wL* vL*
        right_env = xp.zeros([chi, D, chi], dtype=float)  # vR* wR* vR
        left_env[:, 0, :] = xp.eye(chi)
        right_env[:, D - 1, :] = xp.eye(chi)
        self.left_envs[0] = left_env
        self.right_envs[-1] = right_env
        # initialize necessary right_envs
        for i in range(psi.L - 1, 0, -1):
            self.absorb_right_site(i, psi.Bs[i])

    def sweep_one_site(self, dt):
        """Perform one one-site TDVP sweep to evolve |psi⟩ → exp(-i H dt)|psi⟩.

        This is the strictly variational TDVP algorithm: it does *not* grow the
        bond dimension. Each site is updated forward in time (dt/2), the zero-site
        bond is evolved backward (-dt/2), and the sweep is mirrored right-to-left.

        Parameters
        ----------
        dt : float or complex
            Time step. Use ``dt = -1j * tau`` for imaginary-time evolution.

        Returns
        -------
        psi : MPS
            The updated MPS in right-canonical form.
        """
        psi = self.psi
        L = self.psi.L
        # sweep from left to right
        theta = self.psi.one_site_center(0)
        for i in range(L - 1):
            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            Ai, theta = self.split_one_site_theta(i, theta, move_right=True)
            # here theta is zero-site between site i and i+1
            psi.Bs[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.absorb_left_site(i, Ai)
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            j = i + 1
            Bj = self.psi.Bs[j]
            theta = xp.tensordot(theta, Bj, axes=(1, 0))  # vL [vL'], [vL] j vR
            # here theta is one-site on site j = i + 1
        # right boundary
        i = L - 1
        theta = self.evolve_one_site(i, dt, theta)  # forward
        theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
        self.psi.Bs[i] = Bi
        self.absorb_right_site(i, Bi)
        # sweep from right to left
        for i in reversed(range(L - 1)):
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            Ai = self.psi.Bs[i]  # still in left-canonical A form from the above right-sweep!
            theta = xp.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
            self.psi.Bs[i] = Bi
            self.absorb_right_site(i, Bi)
        # The last `evolve_one_site` brought the tensor on site 0 in right-canonical B form,
        # recovering the right-canonical form on each MPS tensor (as the MPS assumes).
        # It splitted the very left, trivial leg off theta,
        # which should only have an arbitrary phase for the left, trivial singular vector,
        # and a singular value 1 (if the state is normalized).
        assert theta.shape == (1, 1)
        assert abs(abs(theta[0]) - 1.) < 1.e-10
        # To keep track of the phase, we put it back into the tensor.
        self.psi.Bs[0] *= theta[0, 0]
        return psi

    def sweep_two_site(self, dt):
        """Perform one two-site TDVP sweep to evolve |psi⟩ → exp(-i H dt)|psi⟩.

        Unlike the one-site variant, this sweep can grow the bond dimension via
        two-site SVD truncation, but is not strictly variational (not TDVP).

        Parameters
        ----------
        dt : float or complex
            Time step. Use ``dt = -1j * tau`` for imaginary-time evolution.

        Returns
        -------
        psi : MPS
            The updated MPS in right-canonical form.
        """
        psi = self.psi
        L = self.psi.L
        # sweep from left to right
        theta = self.psi.two_site_center(0)
        for i in range(L - 2):
            j = i + 1
            k = i + 2
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            psi.Bs[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.absorb_left_site(i, Ai)
            theta = xp.tensordot(xp.diag(S), Bj, axes=(1, 0))  # vL [vL'], [vL] j vC
            # here theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Bk = self.psi.Bs[k]
            theta = xp.tensordot(theta, Bk, axes=(2, 0))  # vL j [vC], [vC] k vR
            # here theta is two-site on sites j, k = i + 1, i + 2
        # right boundary
        i = L - 2
        j = L - 1
        Ai, S, Bj = self.evolve_split_two_site(i, dt, theta)  # forward
        theta = xp.tensordot(Ai, xp.diag(S), axes=(2, 0))  # vL i [vC], [vC'] vC
        self.psi.Bs[j] = Bj
        self.absorb_right_site(j, Bj)
        # sweep from right to left
        for i in reversed(range(L - 2)):
            j = i + 1
            # here, theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Ai = self.psi.Bs[i]  # still in left-canonical A form from the above right-sweep!
            theta = xp.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            # here, theta is two-site on sites i, j = i, i + 1
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            self.psi.Bs[j] = Bj
            self.absorb_right_site(j, Bj)
            theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))  # vL i vC, [vC'] vC
        self.psi.Bs[0] = theta  # this is right-canonical, because for a finite system
        # the left-most virtual bond is trivial, so `theta` and `B` are the same on site 0.
        # So we recovered the right-canonical form on each MPS tensor (as the MPS assumes).
        return psi

    def evolve_zero_site(self, i, dt, theta):
        """Evolve the zero-site tensor between sites `i` and `i+1`.

        Applies exp(-i H_eff dt) to the bond tensor using ``SimpleHeff0``.
        No truncation is needed since no physical index is involved.

        Parameters
        ----------
        i : int
            Left site index; the bond is between sites `i` and `i+1`.
        dt : float or complex
            Time step (use complex for imaginary-time evolution).
        theta : np.ndarray, shape (χL, χR)
            Zero-site bond tensor.

        Returns
        -------
        np.ndarray, shape (χL, χR)
            Time-evolved bond tensor.
        """
        Heff = SimpleHeff0(self.left_envs[i + 1], self.right_envs[i])
        theta = xp.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        return xp.reshape(theta, Heff.theta_shape)

    def evolve_one_site(self, i, dt, theta):
        """Evolve the one-site tensor on site `i`.

        Applies exp(-i H_eff dt) to the single-site tensor using ``SimpleHeff1``.
        No truncation is performed.

        Parameters
        ----------
        i : int
            Site index.
        dt : float or complex
            Time step.
        theta : np.ndarray, shape (χL, d, χR)
            One-site wave function tensor.

        Returns
        -------
        np.ndarray, shape (χL, d, χR)
            Time-evolved one-site tensor.
        """
        # get effective Hamiltonian
        Heff = SimpleHeff1(self.left_envs[i], self.right_envs[i], self.H_mpo[i])
        theta = xp.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        return xp.reshape(theta, Heff.theta_shape)

    def evolve_split_two_site(self, i, dt, theta):
        """Evolve the two-site tensor on sites `i` and `i+1`, then split via SVD.

        Applies exp(-i H_eff dt) using ``Heff2``, then performs a truncated SVD
        to split the result into left-canonical A, singular values S, and
        right-canonical B tensors. Updates ``psi.Ss[i+1]`` in place.

        Parameters
        ----------
        i : int
            Left site index.
        dt : float or complex
            Time step.
        theta : np.ndarray, shape (χL, d, d, χR)
            Two-site wave function tensor.

        Returns
        -------
        Ai : np.ndarray, shape (χL, d, χC)
            Left-canonical tensor on site `i`.
        S : np.ndarray, shape (χC,)
            Singular values (Schmidt coefficients).
        Bj : np.ndarray, shape (χC, d, χR)
            Right-canonical tensor on site `i+1`.
        """
        j = i + 1
        # get effective Hamiltonian
        Heff = Heff2(self.left_envs[i], self.right_envs[j], self.H_mpo[i], self.H_mpo[j])
        flat = xp.reshape(theta, [Heff.shape[0]])
        flat = to_device(self.expm_multiply(Heff, flat, dt))
        theta = xp.reshape(flat, Heff.theta_shape)
        # truncation necessary!
        Ai, S, Bj = mps.svd_split_truncate(theta, self.chi_max, self.eps)
        self.psi.Ss[j] = S
        return Ai, S, Bj

    def split_one_site_theta(self, i, theta, move_right=True):
        """Split a one-site wave function tensor via SVD and update Schmidt values.

        Parameters
        ----------
        i : int
            Site index.
        theta : np.ndarray, shape (χL, d, χR)
            One-site wave function in mixed canonical form.
        move_right : bool, optional
            If True (default), produce a left-canonical A and push singular values
            right (returns A, theta_right). If False, produce a right-canonical B
            and push singular values left (returns theta_left, B).

        Returns
        -------
        A, theta : (np.ndarray, np.ndarray) if move_right is True
        theta, B : (np.ndarray, np.ndarray) if move_right is False
        """
        chivL, d, chivR = theta.shape
        if move_right:
            theta = xp.reshape(theta, [chivL * d, chivR])
            A, S, V = mps.svd(theta, full_matrices=False)
            S = S / xp.linalg.norm(S)
            self.psi.Ss[i + 1] = S
            chivC = len(S)
            A = xp.reshape(A, [chivL, d, chivC])
            theta = xp.tensordot(xp.diag(S), V, axes=(1, 0))
            return A, theta
        else:
            theta = xp.reshape(theta, [chivL, d * chivR])
            U, S, B = mps.svd(theta, full_matrices=False)
            S = S / xp.linalg.norm(S)
            self.psi.Ss[i] = S
            chivC = len(S)
            B = xp.reshape(B, [chivC, d, chivR])
            theta = xp.tensordot(U, xp.diag(S), axes=(1, 0))
            return theta, B

    def absorb_right_site(self, i, B):
        """Calculate and store the right environment left of site `i-1`.

        Contracts the existing ``right_envs[i]`` with the right-canonical tensor `B`
        on site `i` and the MPO tensor ``H_mpo[i]`` to produce ``right_envs[i-1]``.

        Parameters
        ----------
        i : int
            Site index whose right environment is used as input.
        B : np.ndarray, shape (χL, d, χR)
            Right-canonical MPS tensor on site `i`.
        """
        j = (i - 1) % self.psi.L
        right_env = self.right_envs[i]  # vR* wR* vR
        # B has legs     vL i vR
        Bc = B.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        right_env = xp.tensordot(B, right_env, axes=(2, 0))
        right_env = xp.tensordot(right_env, W, axes=([1, 2], [3, 1]))
        right_env = xp.tensordot(right_env, Bc, axes=([1, 3], [2, 1]))
        self.right_envs[j] = right_env

    def absorb_left_site(self, i, A):
        """Calculate and store the left environment right of site `i+1`.

        Contracts the existing ``left_envs[i]`` with the left-canonical tensor `A`
        on site `i` and the MPO tensor ``H_mpo[i]`` to produce ``left_envs[i+1]``.

        Parameters
        ----------
        i : int
            Site index whose left environment is used as input.
        A : np.ndarray, shape (χL, d, χR)
            Left-canonical MPS tensor on site `i`.
        """
        j = (i + 1) % self.psi.L
        left_env = self.left_envs[i]  # vL wL vL*
        # A has legs    vL i vR
        Ac = A.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        left_env = xp.tensordot(left_env, A, axes=(2, 0))
        left_env = xp.tensordot(W, left_env, axes=([0, 3], [1, 2]))
        left_env = xp.tensordot(Ac, left_env, axes=([0, 1], [2, 1]))
        self.left_envs[j] = left_env


    def expm_multiply(self, H, psi0, dt):
        """Apply exp(-i H dt) to psi0 via the Lanczos algorithm.

        Parameters
        ----------
        H : LinearOperator
            Effective Hamiltonian.
        psi0 : np.ndarray
            Initial state vector (flattened).
        dt : float or complex
            Time step.

        Returns
        -------
        np.ndarray
            Evolved state vector exp(-i H dt)|psi0⟩.
        """
        return lanczos_expm_multiply(H, psi0, dt)

class SimpleHeff1(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian on 1 site.
    Basically the same as d_dmrg.SimpleHeff2, but acts on a single site::
        .--vL*     vR*--.
        |       i*      |
        |       |       |
       (left_env)----(W1)----(right_env)
        |       |       |
        |       i       |
        .--vL       vR--.
    """
    def __init__(self, left_env, right_env, W1, prefactor=1.):
        """Initialize the one-site effective Hamiltonian.

        Parameters
        ----------
        left_env : np.ndarray, shape (χL, wL, χL*)
            Left environment tensor with legs ``vL wL* vL*``.
        right_env : np.ndarray, shape (χR*, wR, χR)
            Right environment tensor with legs ``vR* wR* vR``.
        W1 : np.ndarray, shape (wL, wR, d, d*)
            MPO tensor on this site with legs ``wL wR i i*``.
        prefactor : float, optional
            Overall prefactor applied to the matrix-vector product (default 1).
        """
        self.left_env = left_env  # vL wL* vL*
        self.right_env = right_env  # vR* wR* vR
        self.W1 = W1  # wL wR i i*
        chi1, chi2 = left_env.shape[0], right_env.shape[2]
        d1 = W1.shape[2]
        self.theta_shape = (chi1, d1, chi2)  # vL i vR
        self.shape = (chi1 * d1 * chi2, chi1 * d1 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = xp.reshape(xp.asarray(theta), self.theta_shape)
        x = xp.tensordot(self.left_env, x, axes=(2, 0))
        x = xp.tensordot(x, self.W1, axes=([1, 2], [0, 3]))
        x = xp.tensordot(x, self.right_env, axes=([1, 2], [0, 1]))
        return xp.reshape(x, self.shape[0])


class SimpleHeff0(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian.
    Basically the same as d_dmrg.SimpleHeff1, but acts on the zero-site wave function::
        .--vL*   vR*--.
        |             |
        |             |
       (left_env)----------(right_env)
        |             |
        |             |
        .--vL     vR--.
    """
    def __init__(self, left_env, right_env, prefactor=1.):
        """Initialize the zero-site effective Hamiltonian.

        Parameters
        ----------
        left_env : np.ndarray, shape (χL, wL, χL*)
            Left environment tensor with legs ``vL wL* vL*``.
        right_env : np.ndarray, shape (χR*, wR, χR)
            Right environment tensor with legs ``vR* wR* vR``.
        prefactor : float, optional
            Overall prefactor applied to the matrix-vector product (default 1).
        """
        self.left_env = left_env  # vL wL* vL*
        self.right_env = right_env  # vR* wR* vR
        chi1, chi2 = left_env.shape[0], right_env.shape[2]
        self.theta_shape = (chi1, chi2)  # vL vR
        self.shape = (chi1 * chi2, chi1 * chi2)
        self.dtype = left_env.dtype

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = xp.reshape(xp.asarray(theta), self.theta_shape)
        x = xp.tensordot(self.left_env, x, axes=(2, 0))
        x = xp.tensordot(x, self.right_env, axes=([1, 2], [1, 0]))
        return xp.reshape(x, self.shape[0])


def example_TDVP_tf_ising_lightcone(L, g, tmax, dt, one_site=True, chi_max=50):
    """Run TDVP real-time evolution on the transverse-field Ising model and plot entanglement.

    Prepares the ground state via DMRG, applies σ^z on the central site to create
    a local excitation, then evolves in real time and plots the resulting
    entanglement-entropy light cone.

    Parameters
    ----------
    L : int
        Number of sites.
    g : float
        Transverse field strength.
    tmax : float
        Total evolution time.
    dt : float
        Time step size.
    one_site : bool, optional
        If True (default), use the one-site TDVP sweep; otherwise use two-site.
    chi_max : int, optional
        Maximum bond dimension (default 50, only relevant for two-site sweep).
    """
    # compare this code to c_tebd.example_TEBD_tf_ising_lightcone - it's almost the same.
    print("finite TEBD, real time evolution, transverse field Ising")
    print("L={L:d}, g={g:.2f}, tmax={tmax:.2f}, dt={dt:.3f}".format(L=L, g=g, tmax=tmax, dt=dt))
    # find ground state with TEBD or DMRG
    #  E, psi, model = example_TEBD_gs_tf_ising_finite(L, g)
    from .dmrg import example_DMRG_tf_ising_finite
    E, psi, model = example_DMRG_tf_ising_finite(L, g)
    i0 = L // 2
    # apply sigmax on site i0
    SxB = np.tensordot(model.sigmaz, psi.Bs[i0], axes=(1, 1))  # i [i*], vL [i] vR
    psi.Bs[i0] = np.transpose(SxB, [1, 0, 2])  # vL i vR
    E = np.sum(psi.bond_expectation_value(model.H_bonds))
    print("E after applying Sz = {E:.13f}".format(E=E))
    eng = TDVPEngine(psi, model, chi_max=chi_max, eps=1.e-7)
    S = [psi.entanglement_entropy()]
    Nsteps = int(tmax / dt + 0.5)
    for n in range(Nsteps):
        if abs((n * dt + 0.1) % 0.2 - 0.1) < 1.e-10:
            print("t = {t:.2f}, chi =".format(t=n * dt), psi.get_chi())
        if one_site:
            eng.sweep_one_site(dt)
        else:
            eng.sweep_two_site(dt)
        S.append(psi.entanglement_entropy())
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(S[::-1],
               vmin=0.,
               aspect='auto',
               interpolation='nearest',
               extent=(0, L - 1., -0.5 * dt, (Nsteps + 0.5) * dt))
    plt.xlabel('site $i$')
    plt.ylabel('time $t/J$')
    plt.ylim(0., tmax)
    plt.colorbar().set_label('entropy $S$')
    E = np.sum(psi.bond_expectation_value(model.H_bonds))
    print("final E = {E:.13f}".format(E=E))


if __name__ == "__main__":
    # this code is not called if you import this module from another file
    example_TDVP_tf_ising_lightcone(L=20, g=1.5, tmax=3., dt=0.05, one_site=True)
