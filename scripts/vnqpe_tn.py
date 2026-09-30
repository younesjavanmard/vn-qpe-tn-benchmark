"""Core functions of vn_qpe_tn_benchmark_clean.ipynb (Sec. 2), as a module."""
import os, time, copy
import numpy as np
from scipy.optimize import least_squares
from qu_alg_qu_sim_tensor_networks import hamiltonians as hm, mps, dmrg, mps_helper
from qu_alg_qu_sim_tensor_networks import mpo_builder as qmpo
from qu_alg_qu_sim_tensor_networks.tdvp import TDVPEngine
from qu_alg_qu_sim_tensor_networks.backend import to_numpy
DT, CHI_TDVP, EPS_TDVP, MPO_EPS, SEED = float(os.environ.get("DT", 0.05)), 128, 1e-6, 1e-8, 0
I2, Z2 = np.eye(2), np.diag([1.0, -1.0])

# ---------------------------------------------------------------- truncation-error bookkeeping
TRUNC = {"step": 0.0}
_svd_orig = mps.svd_split_truncate
def _svd_logged(theta, chi_max, eps, return_norm=False):
    out = _svd_orig(theta, chi_max, eps, return_norm=True)
    kept = float(to_numpy(out[3])) ** 2                       # kept weight (norm^2 of kept singular values)
    total = float(np.linalg.norm(to_numpy(theta)) ** 2)
    TRUNC["step"] += max(total - kept, 0.0) / max(total, 1e-300)
    return out if return_norm else out[:3]
mps.svd_split_truncate = _svd_logged                           # used by tdvp.TDVPEngine

# ---------------------------------------------------------------- MPOs
def system_mpo(H, epsilon=MPO_EPS, chi_max=256):
    # graph-based MPO (left/right fragment graphs), lossless SVD compression
    wmpo, _, _ = hm.get_mpo(H.n, H, epsilon=epsilon, chi_max=chi_max)
    return wmpo, mps_helper.vvdd_ddvv_mpo_swap(wmpo)

def pointer_mpo_tensors(r):
    # p = sum_j 2^{-j}(1-Z_j)/2 ; bulk [[I, p_j],[0, I]] ; first tensor = row 0, last = column 1
    Ws = []
    for j in range(1, r + 1):
        pj = 2.0 ** (-j) * (I2 - Z2) / 2
        W = np.zeros((2, 2, 2, 2), dtype=complex)
        W[0, 0], W[0, 1], W[1, 1] = I2, pj, I2
        if j == 1: W = W[0:1]
        if j == r: W = W[:, 1:2]
        Ws.append(W)
    return Ws

def coupled_model(sys_model, r):
    # MPO of H (x) p on N + r sites: system tensors followed by pointer tensors, D_{H(x)p} = D_H
    assert sys_model.H_mpo[-1].shape[1] == 1
    return qmpo.MPO([np.asarray(W) for W in sys_model.H_mpo] + pointer_mpo_tensors(r), Ss=None, bonds=None)

# ---------------------------------------------------------------- DMRG
def run_dmrg(sys_model, wmpo, chi, sweeps=30, eps=1e-10, seed=SEED, tol=1e-10, verbose=False):
    psi = mps.init_random_bond_chi_MPS(sys_model.L, 2, min(chi, 2), sys_model.bc, seed=seed)
    eng = dmrg.DMRGEngine(psi, sys_model, chi_max=chi, eps=eps)
    E_prev = None
    for k in range(sweeps):
        eng.sweep()
        E = float(np.real(mps_helper.expectation_value(wmpo, psi)))
        if verbose: print(f"  sweep {k:2d}: E = {E:.10f}  chi = {max(psi.get_chi())}")
        if E_prev is not None and abs(E - E_prev) < tol * max(1.0, abs(E)): break
        E_prev = E
    psi.Bs = [to_numpy(B) for B in psi.Bs]; psi.Ss = [to_numpy(S) for S in psi.Ss]
    return psi, E

def attach_pointer(psi_sys, r):
    # |Psi(0)> = |psi_DMRG> (x) |+>^r   (|x=0> of the pointer is the product state |+>^r)
    plus = np.full((1, 2, 1), 1 / np.sqrt(2), dtype=complex)
    Bs = [np.asarray(B, dtype=complex) for B in psi_sys.Bs] + [plus.copy() for _ in range(r)]
    Ss = [np.asarray(S) for S in psi_sys.Ss] + [np.ones(1) for _ in range(r)]
    return mps.MPS(Bs, Ss, bc="finite")

# ---------------------------------------------------------------- readout
def qft_matrix(r):
    z = np.arange(2 ** r)
    return np.exp(2j * np.pi * np.outer(z, z) / 2 ** r) / np.sqrt(2 ** r)

def pointer_rdm(psi, n_sys, r):
    # contract the system into a left environment, then the r pointer tensors (index = big-endian bitstring)
    Bs = [to_numpy(B) for B in psi.Bs]
    E = np.ones((1, 1), dtype=complex)
    for B in Bs[:n_sys]:
        E = np.tensordot(E, B, axes=(0, 0))
        E = np.tensordot(E, B.conj(), axes=([0, 1], [0, 1]))
    theta = Bs[n_sys]
    for B in Bs[n_sys + 1:]:
        theta = np.tensordot(theta, B, axes=(-1, 0))
    theta = theta.reshape(theta.shape[0], 2 ** r)
    rho = np.einsum("ab,ax,by->xy", E, theta, theta.conj())
    return rho / np.trace(rho).real

def pointer_distribution(rho, r):
    F = qft_matrix(r)
    return np.real(np.diag(F.conj().T @ rho @ F))          # inverse QFT, then computational-basis readout

def pointer_entropy(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)))

# ---------------------------------------------------------------- TDVP
def evolve(psi0, model, n_sys, r, tmax, dt=DT, chi_max=CHI_TDVP, eps=EPS_TDVP, verbose=False, checkpoint=None):
    # dynamic scheme: 2TDVP while chi < chi_max, then 1TDVP
    psi = copy.deepcopy(psi0)
    ts, Ps, Ss, chis, DW, EG = [], [], [], [], [], []
    E0 = float(np.real(mps_helper.expectation_value(qmpo.MPO(list(map(lambda W: np.transpose(np.asarray(W), (2, 3, 0, 1)), model.H_mpo))), psi)))
    TRUNC["step"] = 0.0                                          # discard DMRG truncations counted before
    eng = TDVPEngine(psi, model, chi_max=chi_max, eps=eps)
    nsteps = int(round(tmax / dt)); t0 = time.time()
    for n in range(1, nsteps + 1):
        if os.environ.get("TDVP_2SITE_ONLY") or max(psi.get_chi()) < chi_max:
            psi = eng.sweep_two_site(dt)       # truncated at chi_max
        else:
            psi = eng.sweep_one_site(dt)
        if not np.all(np.isfinite(to_numpy(psi.Bs[0]))):
            raise FloatingPointError(f"non-finite MPS at step {n}")
        DW.append(TRUNC["step"]); TRUNC["step"] = 0.0
        rho = pointer_rdm(psi, n_sys, r)
        ts.append(n * dt); Ps.append(pointer_distribution(rho, r)); Ss.append(pointer_entropy(rho))
        chis.append(max(psi.get_chi()))
        if n % 10 == 0 or n == nsteps:
            EG.append((n * dt, float(np.real(mps_helper.expectation_value(qmpo.MPO(list(map(lambda W: np.transpose(np.asarray(W), (2, 3, 0, 1)), model.H_mpo))), psi))) - E0))
        if verbose and n % max(1, nsteps // 20) == 0:
            print(f"  t = {n*dt:7.2f}  chi = {chis[-1]:4d}  S = {Ss[-1]:.3f}  sum_dw = {sum(DW):.2e}  dE_gen = {EG[-1][1] if EG else 0:.2e}  ({time.time()-t0:.0f} s)", flush=True)
        if checkpoint and n % 50 == 0:
            np.savez(checkpoint, ts=ts, P=Ps, S=Ss, chi=chis, dw=DW, dE_gen=EG)
    evolve.last_errors = dict(dw=np.array(DW), dE_gen=np.array(EG))
    return np.array(ts), np.array(Ps), np.array(Ss), np.array(chis)

# ---------------------------------------------------------------- energy estimation
def fejer(t, E, x, r):
    # |f(E,x)|^2 (paper Eq. 8), inverse-QFT convention: peak at x = -E t / 2pi (mod 2^r)
    a = x + E * t / (2 * np.pi)
    num, den = np.sin(np.pi * a) ** 2, np.sin(np.pi * a / 2 ** r) ** 2
    small = den < 1e-14
    return np.where(small, 1.0, num / (4 ** r * np.where(small, 1.0, den)))

def fit_row(ts, Px, x, r, E_lo, E_hi, n_grid=4001):
    # global grid scan followed by least-squares refinement (avoids fits that stop on a bound)
    grid = np.linspace(E_lo, E_hi, n_grid)
    cost = [np.sum((fejer(ts, E, x, r) - Px) ** 2) for E in grid]
    E0 = grid[int(np.argmin(cost))]
    res = least_squares(lambda E: fejer(ts, E[0], x, r) - Px, x0=[E0], bounds=([E_lo], [E_hi]))
    return float(res.x[0])

def estimate_energy(ts, P, r, E_lo, E_hi):
    rows = np.array([fit_row(ts, P[:, x], x, r, E_lo, E_hi) for x in range(2 ** r)])
    return rows

# ---------------------------------------------------------------- exact reference (small systems)
def exact_pointer_distribution(H, psi_vec, r, ts):
    E, V = np.linalg.eigh(H.to_dense())
    w = np.abs(V.conj().T @ psi_vec) ** 2
    F, z = qft_matrix(r), np.arange(2 ** r)
    return np.array([w @ np.abs((np.exp(-1j * np.outer(E, z) * t / 2 ** r) / np.sqrt(2 ** r)) @ F.conj()) ** 2
                     for t in ts])

def mps_to_vector(psi):
    v = np.ones((1, 1), dtype=complex)
    for B in psi.Bs: v = np.tensordot(v, to_numpy(B), axes=(-1, 0))
    return v.reshape(-1)
