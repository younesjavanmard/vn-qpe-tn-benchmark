"""Generate vn_qpe_tn_benchmark_clean.ipynb (run: python make_notebook.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Tensor-network benchmarking of QPE in the von Neumann measurement formulation

Companion notebook to *"Tensor-network benchmarking of quantum phase estimation in the von Neumann measurement
formulation, with pre-trained DMRG initial states"* (Y. Javanmard).

The quantum protocol is that of Childs *et al.*, PRA **66**, 032314 (2002) and Novo *et al.*, Quantum **5**, 465 (2021):
the system is coupled to an $r$-qubit pointer through $e^{-it\hat H\otimes\hat p}$, $\hat p=\sum_{j=1}^r 2^{-j}(1-\hat Z_j)/2$,
and the pointer is read out after an inverse QFT. This notebook simulates it with tensor networks:

| Step | What | Paper |
|---|---|---|
| 1 | Graph-based MPO of $\hat H$, lossless SVD compression | Sec. III B |
| 2 | Pointer MPO (bond dim. 2) concatenated with the system MPO $\Rightarrow$ MPO of $\hat H\otimes\hat p$ | Sec. III C |
| 3 | DMRG initial state with small $\chi_{\rm DMRG}$, pointer in $\lvert+\rangle^{\otimes r}$ | Sec. III D |
| 4 | Dynamic 2TDVP/1TDVP evolution of the composite MPS | Sec. III E |
| 5 | Pointer reduced density matrix $\to$ $P(x;t)=\langle x\lvert F^\dagger\rho_p F\rvert x\rangle$, entropy $S(t)$ | Sec. III F |
| 6 | Per-row least-squares fit of $P(x;t)$ to $\lvert f(E,x)\rvert^2$ $\Rightarrow$ $E_{\rm es}(x)$ | Sec. II C |

**Contents**

- **Part A** reproduces the published figures and numbers from the saved simulation data (`Ps_*.npy`, fast).
- **Part B** validates the tensor-network pipeline against an exact state-vector simulation (small lattice, minutes).
- **Part C** re-runs the Heisenberg $4\times3$ benchmark and the entanglement-vs-$\chi$ study (slower).
Parts B and C are switched on in the configuration cell.
""")

md("## 0. Setup (local or Google Colab)")
code(r"""
import os, sys, time, json, copy
IN_COLAB = "google.colab" in sys.modules

# --- Colab: install the package from GitHub (edit REPO_URL / use a token for a private repo) ---
REPO_URL = "https://github.com/younesjavanmard/vn-qpe-tn-benchmark.git"
if IN_COLAB:
    if not os.path.isdir("vn-qpe-tn-benchmark"):
        !git clone -q {REPO_URL}
    !pip -q install -e vn-qpe-tn-benchmark
    REPO_DIR = os.path.abspath("vn-qpe-tn-benchmark")
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))
else:
    REPO_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_DIR = os.path.join(REPO_DIR, "data")                    # saved Ps_*.npy of the paper
OUT_DIR = os.path.join(os.getcwd(), "outputs")               # figures + json written here
os.makedirs(OUT_DIR, exist_ok=True)

os.environ.setdefault("DMRG_BACKEND", "numpy")                # "auto" to use CuPy on a GPU runtime
print("repo:", REPO_DIR, "| colab:", IN_COLAB, "| backend:", os.environ["DMRG_BACKEND"])
""")

code(r"""
import numpy as np
import scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import qu_alg_qu_sim_tensor_networks as qa
from qu_alg_qu_sim_tensor_networks import hamiltonians as hm, mps, dmrg, mps_helper
from qu_alg_qu_sim_tensor_networks import mpo_builder as qmpo
from qu_alg_qu_sim_tensor_networks.tdvp import TDVPEngine
from qu_alg_qu_sim_tensor_networks.backend import to_numpy
from qu_alg_qu_sim_tensor_networks.lattice.geometry import TriangularLattice
from qu_alg_qu_sim_tensor_networks.lattice import models

print(f"numpy {np.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__} | package {qa.__version__}")
plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "savefig.bbox": "tight"})
""")

md("## 1. Configuration")
code(r"""
R = 4                  # pointer qubits (all benchmarks of the paper)
DT = 0.05              # TDVP time step (1/J or 1/Ha)
CHI_TDVP = 128         # maximal TDVP bond dimension (paper)
EPS_TDVP = 1e-6        # SVD cutoff in 2TDVP
MPO_EPS = 1e-8         # lossless MPO compression threshold
SEED = 0

RUN_PART_B = True      # validation against exact state vector (6 spins + 3 pointer qubits, ~2 min)
RUN_PART_C = False     # re-run Heisenberg 4x3 (chi=20) and entanglement vs chi (~hours on a laptop)

""")

md(r"""
## 2. Core functions

MPS tensors use the package convention `B[i] : (vL, i, vR)` (right-canonical); engine MPO tensors are `W[i] : (wL, wR, i, i*)`
with start state `0` and final state `D-1`.
""")
code(r"""
I2, Z2 = np.eye(2), np.diag([1.0, -1.0])

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
    ts, Ps, Ss, chis = [], [], [], []
    eng = TDVPEngine(psi, model, chi_max=chi_max, eps=eps)
    nsteps = int(round(tmax / dt)); t0 = time.time()
    for n in range(1, nsteps + 1):
        psi = eng.sweep_two_site(dt) if max(psi.get_chi()) < chi_max else eng.sweep_one_site(dt)
        rho = pointer_rdm(psi, n_sys, r)
        ts.append(n * dt); Ps.append(pointer_distribution(rho, r)); Ss.append(pointer_entropy(rho))
        chis.append(max(psi.get_chi()))
        if verbose and n % max(1, nsteps // 20) == 0:
            print(f"  t = {n*dt:7.2f}  chi = {chis[-1]:4d}  S = {Ss[-1]:.3f}  ({time.time()-t0:.0f} s)")
        if checkpoint and n % 50 == 0:
            np.savez(checkpoint, ts=ts, P=Ps, S=Ss, chi=chis)
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
""")

code(r"""
def plot_pointer_stack(ts, P, E_rows, title, xlabel, fname, t_show=None, n_top=None):
    # time-resolved pointer distribution: one row per outcome |x>, fitted curve in black
    r = int(np.log2(P.shape[1])); m = ts <= (t_show if t_show else ts[-1])
    fig, axs = plt.subplots(2 ** r, 1, sharex=True, figsize=(7, 0.5 * 2 ** r + 1.2),
                            gridspec_kw={"hspace": 0.05})
    colors = plt.cm.viridis(np.linspace(0, 0.9, 2 ** r))
    for x in range(2 ** r):
        ax = axs[x]
        ax.fill_between(ts[m], P[m, x], color=colors[x], alpha=0.6, lw=0)
        tt = np.linspace(ts[m][0], ts[m][-1], 4000)
        ax.plot(tt, fejer(tt, E_rows[x], x, r), "k", lw=0.5)
        ax.set_ylim(0, 1.05); ax.set_yticks([])
        ax.text(1.01, 0.3, rf"$|{x}\rangle$  $E_{{es}}$={E_rows[x]:.4f}", transform=ax.transAxes, fontsize=7)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    axs[0].set_title(title, fontsize=9); axs[-1].set_xlabel(xlabel)
    fig.savefig(os.path.join(OUT_DIR, fname)); plt.show()
""")

md(r"""
# Part A — Reproduce the published results from the saved data

The arrays `Ps_*.npy` contain $P(x;t)$ from the TDVP simulations of the paper (rows: $t=n\,\delta t$, $\delta t=0.05$; columns: $x=0,\dots,15$).
Energies and DMRG input energies are those printed in the paper figures. The molecular Hamiltonians were simulated on the
$\mathbb{Z}_2$-tapered 13-qubit representation (see the file names `*_13_*` of the figures).
""")
code(r"""
t_mol = np.load(os.path.join(DATA_DIR, "tlist.npy"))
BENCH = {
    "heisenberg_4x3": dict(P=np.load(os.path.join(DATA_DIR, "Ps_Heisenberg_triangular_4_3.npy")),
                           E_init=-26.7333, E_ref=-27.7232, chi=20, unit="J", t_show=None),
    "H8_sto3g":       dict(P=np.load(os.path.join(DATA_DIR, "Ps_H8_stgo3g.npy")),
                           E_init=-4.18513790, E_ref=-4.271824120916049, chi=4, unit="Ha", t_show=None),
    "pyridine_sto3g": dict(P=np.load(os.path.join(DATA_DIR, "Ps_pyridine_stgo3g.npy")),
                           E_init=-243.66072968, E_ref=-243.6967896844996, chi=2, unit="Ha", t_show=6.0),
}
for k, b in BENCH.items():
    b["ts"] = DT * np.arange(1, len(b["P"]) + 1)
    print(f"{k:16s} P shape {b['P'].shape}, t_max = {b['ts'][-1]:.2f}, sum_x P = {b['P'].sum(1).min():.6f}..{b['P'].sum(1).max():.6f}")
""")
code(r"""
RESULTS_A = {}
for k, b in BENCH.items():
    rows = estimate_energy(b["ts"], b["P"], R, b["E_init"] - 1.0, b["E_init"])     # window used in the paper
    b["rows"] = rows
    RESULTS_A[k] = dict(E_init=b["E_init"], E_ref=b["E_ref"], chi_DMRG=b["chi"], t_max=float(b["ts"][-1]),
                        E_rows=rows.tolist(), E_min=float(rows.min()), E_median=float(np.median(rows)),
                        dE_min=float(abs(rows.min() - b["E_ref"])), dE_median=float(abs(np.median(rows) - b["E_ref"])))
    print(f"{k:16s} E_init={b['E_init']:.4f}  min_x E_es={rows.min():.4f}  median={np.median(rows):.4f}  "
          f"E_ref={b['E_ref']:.4f}  |dE|(min)={abs(rows.min()-b['E_ref']):.2e}")
    print("   rows:", np.round(rows, 4))
json.dump(RESULTS_A, open(os.path.join(OUT_DIR, "partA_results.json"), "w"), indent=1)
""")
code(r"""
for k, b in BENCH.items():
    title = (rf"{k}: $E^{{init}}_{{dmrg}}$={b['E_init']:.4f}, min$_x E_{{es}}$={b['rows'].min():.4f}, "
             rf"$E_{{ref}}$={b['E_ref']:.4f} {b['unit']}")
    plot_pointer_stack(b["ts"], b["P"], b["rows"], title, rf"$t$ [1/{b['unit']}]", f"partA_{k}_stack.pdf",
                       t_show=b["t_show"])
""")
md(r"""
### Convergence of the estimate with the evolution window (paper Fig. 5 and pyridine text)
The estimate at time $t$ is obtained by fitting only the data up to $t$.
""")
code(r"""
CHEM = 1.5936e-3
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
for k, T_list in (("H8_sto3g", [0.5, 2.5, 3.75, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30]),
                  ("pyridine_sto3g", [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30])):
    b = BENCH[k]; est = []
    for T in T_list:
        m = b["ts"] <= T + 1e-9
        est.append(estimate_energy(b["ts"][m], b["P"][m], R, b["E_init"] - 1.0, b["E_init"]).min())
    est = np.array(est); b["conv"] = (T_list, est)
    axs[0].plot(T_list, est - b["E_ref"], "o-", label=k)
    axs[1].semilogy(T_list, np.abs(est - b["E_ref"]), "o-", label=k)
    print(k, [f"t={T}: dE={abs(e-b['E_ref'])*1e3:.2f} mHa" for T, e in zip(T_list, est)])
axs[1].axhspan(1e-5, CHEM, color="purple", alpha=0.15, label="chemical accuracy")
axs[0].set(xlabel=r"$t$ [1/Ha]", ylabel=r"$E(t)-E_{\rm ref}$ [Ha]"); axs[1].set(xlabel=r"$t$ [1/Ha]", ylabel=r"$|\Delta E|$ [Ha]")
axs[1].legend(fontsize=7); plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "partA_convergence.pdf")); plt.show()
""")

md(r"""
# Part B — Validation against an exact state-vector simulation

$2\times3$ periodic triangular Heisenberg model (6 spins) with $r=3$ pointer qubits. The TDVP distribution $P(x;t)$ is compared with
the exact distribution of the ideal circuit, and both are fitted with the same procedure.

**Observed (local run, $\delta t=0.05$):** $\max|P_{\rm TN}-P_{\rm exact}|\approx1.1\times10^{-2}$ and per-row energies differ by up to $5\times10^{-2}$ for this strongly mixed $\chi=3$ input over $t\le4$. The TDVP error in $P$ decreases only linearly with $\delta t$ (first-order global error of the package's sweep), so check $\delta t$-convergence (e.g. `DT=0.025`) before quoting new results.
""")
code(r"""
if RUN_PART_B:
    latB = TriangularLattice(2, 3, boundary="periodic"); HB = models.heisenberg(latB); nB, rB = latB.num_sites, 3
    E0B = np.linalg.eigvalsh(HB.to_dense())[0]
    wB, smB = system_mpo(HB)
    psiB, EdB = run_dmrg(smB, wB, chi=3)
    print(f"D_H = {max(wB.bonds)}, E0 = {E0B:.6f}, E_DMRG(chi=3) = {EdB:.6f}")
    tsB, PB, SB, chiB = evolve(attach_pointer(psiB, rB), coupled_model(smB, rB), nB, rB, tmax=4.0, eps=1e-12)
    PexB = exact_pointer_distribution(HB, mps_to_vector(psiB), rB, tsB)
    rows_tn = estimate_energy(tsB, PB, rB, E0B - 3, EdB); rows_ex = estimate_energy(tsB, PexB, rB, E0B - 3, EdB)
    print(f"max |P_TN - P_exact| = {np.abs(PB - PexB).max():.2e}")
    print("E_es (TN)   :", np.round(rows_tn, 5)); print("E_es (exact):", np.round(rows_ex, 5))
    print(f"max |E_es(TN) - E_es(exact)| = {np.abs(rows_tn - rows_ex).max():.2e}")
""")

md(r"""
# Part C — Re-run the Heisenberg $4\times3$ benchmark and the entanglement study

Uses the lattice class of the package; if its site ordering or bond convention differs from the one used for the paper
data, $E_{\rm ref}$ printed here identifies the lattice (paper: $E_0=-27.7232\,J$).
""")
code(r"""
if RUN_PART_C:
    latC = TriangularLattice(4, 3, boundary="periodic"); HC = models.heisenberg(latC); nC = latC.num_sites
    E0C = np.linalg.eigvalsh(HC.to_dense())[0]
    wC, smC = system_mpo(HC); mC = coupled_model(smC, R)
    print(f"sites {nC}, bonds {len(latC.bonds)}, D_H = {max(wC.bonds)}, E0 = {E0C:.6f}")
    RESULTS_C = {}
    for chi in (2, 20, 32):
        psiC, EdC = run_dmrg(smC, wC, chi=chi)
        tsC, PC, SC, chiC = evolve(attach_pointer(psiC, R), mC, nC, R, tmax=10.0, verbose=True,
                                   checkpoint=os.path.join(OUT_DIR, f"partC_chi{chi}.npz"))
        rows = estimate_energy(tsC, PC, R, EdC - 2, EdC)
        RESULTS_C[chi] = dict(E_init=EdC, E_rows=rows.tolist(), S=SC.tolist(), chi_tdvp=chiC.tolist())
        print(f"chi={chi}: E_init={EdC:.4f}, min E_es={rows.min():.4f}, E0={E0C:.4f}")
        if chi == 20:
            plot_pointer_stack(tsC, PC, rows, rf"Heisenberg 4x3, $\chi$=20", r"$t$ [1/J]", "partC_heis_stack.pdf")
    json.dump(RESULTS_C, open(os.path.join(OUT_DIR, "partC_results.json"), "w"))
    for chi, d in RESULTS_C.items(): plt.plot(tsC, d["S"], label=rf"$\chi_{{DMRG}}={chi}$")
    plt.axhline(R * np.log(2), ls=":", c="k", lw=0.8, label=r"$r\ln 2$")
    plt.xlabel(r"$t$ [1/J]"); plt.ylabel(r"$S(t)$"); plt.legend(); plt.savefig(os.path.join(OUT_DIR, "partC_entropy.pdf")); plt.show()
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbf.write(nb, "vn_qpe_tn_benchmark_clean.ipynb")
print("written", len(cells), "cells")
