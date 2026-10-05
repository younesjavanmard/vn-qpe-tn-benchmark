from scipy.sparse.linalg import expm
import numpy as np
import warnings

from qu_alg_qu_sim_tensor_networks.backend import get_array_module, to_device, to_numpy

default_k = 20


def _apply(H, x):
    """H @ x without SciPy's LinearOperator wrapper, which rejects CuPy arrays.

    The effective Hamiltonians implement `_matvec` with the active array module
    (NumPy or CuPy), so calling it directly keeps the vector on the device.
    """
    if isinstance(x, np.ndarray) or not hasattr(H, "_matvec"):
        return H @ x  # NumPy path: unchanged
    return H._matvec(x)


def lanczos_ground_state(H, psi0, k=None):
    """Use lanczos to calculate the ground state of a hermitian H.

    Builds an orthogonal basis in the Krylov space spanned by {H^i |psi0> for i < k}
    and finds the ground state in this subspace. Works on CPU and GPU.
    """
    T, vecs = lanczos_iterations(H, psi0, k)
    E, v = np.linalg.eigh(T)  # T is small CPU matrix
    result = vecs @ to_device(v[:, 0].astype(vecs.dtype))
    if abs(float(get_array_module(vecs).linalg.norm(result)) - 1.) > 1.e-5:
        warnings.warn("poorly conditioned lanczos. Maybe a non-hermitian H?")
    return E[0], result


def lanczos_expm_multiply(H, psi0, dt, k=None):
    """Use lanczos to compute ``expm(-i H dt)|psi0>`` for sufficiently small dt and hermitian H.

    Builds the Krylov basis and evolves in that subspace. Works on CPU and GPU.
    """
    T, vecs = lanczos_iterations(H, psi0, k)
    # expm on the small k×k CPU matrix
    v0 = np.zeros(T.shape[0])
    v0[0] = 1.
    vt = expm(-1.j * dt * T) @ v0
    result = vecs @ to_device(vt.astype(complex))
    xp = get_array_module(vecs)
    if abs(float(xp.linalg.norm(result)) - 1.) > 1.e-5:
        warnings.warn("poorly conditioned lanczos. Maybe a non-hermitian H?")
    return result


def lanczos_iterations(H, psi0, k):
    """Perform `k` Lanczos iterations building tridiagonal matrix T and ONB of the Krylov space."""
    if k is None:
        k = default_k
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")

    xp = get_array_module(psi0)
    psi0 = psi0 / xp.linalg.norm(psi0)
    vecs = [psi0]
    T = np.zeros((k, k))  # small CPU matrix — used with scipy expm
    psi = _apply(H, psi0)
    alpha = T[0, 0] = float(xp.dot(psi0.conj(), psi).real)
    psi = psi - alpha * vecs[-1]
    for i in range(1, k):
        beta = float(xp.linalg.norm(psi))
        if beta < 1.e-13:
            T = T[:i, :i]
            break
        psi = psi / beta
        vecs.append(psi)
        psi = _apply(H, psi) - beta * vecs[-2]
        alpha = float(xp.dot(vecs[-1].conj(), psi).real)
        psi = psi - alpha * vecs[-1]
        T[i, i] = alpha
        T[i - 1, i] = T[i, i - 1] = beta
    return T, xp.stack(vecs).T
