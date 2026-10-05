"""Finite-shot pointer readout for the Heisenberg 4x3 benchmark (paper Fig. 5, Appendix "Finite-shot readout").

Two parts.
* Shot noise of the estimator (published data): N_s outcomes per time step are drawn from the published TDVP
  distribution P(x,t) of Fig. 5 (data/Ps_Heisenberg_triangular_4_3.npy); for an r-qubit pointer this has exactly the
  statistics of perfect sampling of the MPS. The estimator of the paper (Fejer fit per row, median over rows) is
  applied to the histograms and compared with the estimate from the exact P(x,t).
* Check of the MPS sampler on a time-evolved state: DMRG (chi = 20) + TDVP of H (x) p; at every step the inverse
  QFT is applied to the pointer sites and N_MAX shots are drawn from the MPS with mps_helper.sample_bitstrings
  (perfect sampling, system traced out exactly through its left environment); the histograms are compared with
  the exact pointer distribution of the same MPS (chi-squared test).

usage: python sample_pointer.py published                 # -> ../results/sampling_heisenberg_published.json
       python sample_pointer.py run     # -> ../results/sampling_heisenberg.npz
       python sample_pointer.py analyze                   # sampler check -> ../results/sampling_heisenberg_mps.json
       python sample_pointer.py plot                      # -> ../figures/fig_sampling_heisenberg.pdf
"""
import os, sys, json, time, pickle
import numpy as np
import vnqpe_tn as V
from qu_alg_qu_sim_tensor_networks.hamiltonians import Hamiltonian
from qu_alg_qu_sim_tensor_networks.mps_helper import sample_bitstrings
from qu_alg_qu_sim_tensor_networks.backend import to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
NPZ = os.path.join(RES, "sampling_heisenberg.npz")
JSN_PUB, JSN_MPS = (os.path.join(RES, f"sampling_heisenberg_{k}.json") for k in ("published", "mps"))
R, CHI_DMRG, T_MAX, N_MAX, SEED = 4, 20, 10.0, 100_000, 11
N_S, REPS = [10, 30, 100, 300, 1000, 3000, 10000], 20
E_INIT_PUB = -26.7333                        # DMRG input energy of the published run (Fig. 5)
E_EXACT = -27.723249451503534


def heisenberg_torus_4x3():
    # triangular lattice 4x3 on a torus, 36 bonds, H = sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j)
    L1, L2 = 4, 3
    idx = lambda x, y: (y % L2) * L1 + (x % L1)
    bonds = {tuple(sorted((idx(x, y), idx(x + dx, y + dy)))) for x in range(L1) for y in range(L2)
             for dx, dy in ((1, 0), (0, 1), (1, 1))}
    H = Hamiltonian(L1 * L2)
    for i, j in sorted(bonds):
        for a in "XYZ":
            H.add_term([(i, a), (j, a)], 1.0)
    return H


def pointer_after_iqft(psi, n_sys, r):
    """Left environment of the system and the r pointer tensors after the inverse QFT (exact SVD split)."""
    tensors = [to_numpy(B) for B in psi.tensors]
    E = np.ones((1, 1), dtype=complex)
    for B in tensors[:n_sys]:
        E = np.tensordot(E, B, axes=(0, 0))
        E = np.tensordot(E, B.conj(), axes=([0, 1], [0, 1]))
    theta = tensors[n_sys]
    for B in tensors[n_sys + 1:]:
        theta = np.tensordot(theta, B, axes=(-1, 0))
    chi = theta.shape[0]
    phi = theta.reshape(chi, 2 ** r) @ V.qft_matrix(r).conj()           # amplitude of |x> after the inverse QFT
    As, M = [], phi.reshape(chi, 2 ** r)
    for _ in range(r - 1):
        a = M.shape[0]
        U, S, Vh = np.linalg.svd(M.reshape(a * 2, -1), full_matrices=False)
        k = int(np.sum(S > 1e-14 * S[0]))
        As.append(U[:, :k].reshape(a, 2, k)); M = S[:k, None] * Vh[:k]
    As.append(M.reshape(M.shape[0], 2, 1))
    return E / np.trace(E).real, As


def run():
    H = heisenberg_torus_4x3(); n = H.n
    w, sm = V.system_mpo(H)
    cache = os.path.join(RES, f"dmrg_heisenberg_chi{CHI_DMRG}.pkl")
    if os.path.exists(cache):
        psi, E_init = pickle.load(open(cache, "rb"))
    else:
        psi, E_init = V.run_dmrg(sm, w, chi=CHI_DMRG)
        os.makedirs(RES, exist_ok=True); pickle.dump((psi, E_init), open(cache, "wb"))
    print(f"D_H = {max(w.bonds)}, E_DMRG(chi={CHI_DMRG}) = {E_init:.6f}, E_exact = {E_EXACT:.6f}", flush=True)
    rng, shots, P_mps = np.random.default_rng(SEED), [], []

    def draw(step, psi_t):
        E, As = pointer_after_iqft(psi_t, n, R)
        bits = sample_bitstrings(As, N_MAX, left_env=E, rng=rng)
        shots.append((bits.astype(np.int64) << np.arange(R - 1, -1, -1)).sum(1).astype(np.uint8))
        th = As[0]
        for A in As[1:]:
            th = np.tensordot(th, A, axes=(-1, 0))
        th = th.reshape(th.shape[0], 2 ** R)
        P_mps.append(np.real(np.einsum("ab,ax,bx->x", E, th, th.conj())))

    t0 = time.time()
    ts, P, S, chis = V.evolve(V.attach_pointer(psi, R), V.coupled_model(sm, R), n, R, T_MAX, verbose=True, callback=draw)
    print(f"TDVP + sampling: {time.time() - t0:.0f} s, max chi = {chis.max()}", flush=True)
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(NPZ, ts=ts, P=P, P_mps=np.array(P_mps), shots=np.array(shots), E_init=E_init, chi=chis)


def published():
    """Energy estimate from N_s shots per time step, drawn from the published P(x,t) of Fig. 5."""
    P = np.load(os.path.join(HERE, "..", "data", "Ps_Heisenberg_triangular_4_3.npy"))
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True)
    ts = V.DT * np.arange(1, len(P) + 1)
    lo, hi = E_INIT_PUB - 1.0, E_INIT_PUB
    rows = V.estimate_energy(ts, P, R, lo, hi)
    out = dict(E_init=E_INIT_PUB, E_exact=E_EXACT, E_median_exactP=float(np.median(rows)), n_t=len(ts), reps=REPS, by_Ns={})
    rng = np.random.default_rng(SEED)
    for Ns in N_S:
        E_med, tv = [], []
        for _ in range(REPS):
            Ph = np.stack([rng.multinomial(Ns, p) for p in P]) / Ns
            E_med.append(float(np.median(V.estimate_energy(ts, Ph, R, lo, hi))))
            tv.append(float(0.5 * np.abs(Ph - P).sum(1).mean()))
        E_med = np.array(E_med)
        out["by_Ns"][Ns] = dict(E_median=E_med.tolist(), mean=float(E_med.mean()), std=float(E_med.std(ddof=1)),
                                rmse_vs_exactP=float(np.sqrt(np.mean((E_med - out["E_median_exactP"]) ** 2))),
                                abs_err_vs_exact=float(np.mean(np.abs(E_med - E_EXACT))), tv_mean=float(np.mean(tv)))
        print(f"N_s={Ns:6d}  E_med = {E_med.mean():.5f} +- {E_med.std(ddof=1):.5f}  rmse vs exact-P estimate = "
              f"{out['by_Ns'][Ns]['rmse_vs_exactP']:.2e}  mean TV = {np.mean(tv):.3f}", flush=True)
    print(f"exact P: E_med = {out['E_median_exactP']:.5f};  E_exact = {E_EXACT:.5f}")
    os.makedirs(RES, exist_ok=True); json.dump(out, open(JSN_PUB, "w"), indent=1)


def analyze():
    """Chi-squared test of the MPS samples against the exact pointer distribution of the same MPS."""
    from scipy.stats import chi2
    d = np.load(NPZ); shots, P_mps = d["shots"], d["P_mps"]
    P_mps = np.clip(P_mps, 0, None); P_mps /= P_mps.sum(1, keepdims=True)
    N = shots.shape[1]; nx = 2 ** R
    stats, pvals = [], []
    for b, p in zip(shots, P_mps):
        h = np.bincount(b, minlength=nx)
        m = p * N > 5                                           # cells with enough expected counts
        e = p[m] * N; o = h[m]
        rest_o, rest_e = h[~m].sum(), p[~m].sum() * N
        o, e = np.append(o, rest_o), np.append(e, rest_e)
        keep = e > 0
        c = float(np.sum((o[keep] - e[keep]) ** 2 / e[keep])); dof = int(keep.sum() - 1)
        stats.append(c / dof); pvals.append(float(chi2.sf(c, dof)))
    pvals = np.array(pvals)
    tv_max = float(max(0.5 * np.abs(np.bincount(b, minlength=nx) / N - p).sum() for b, p in zip(shots, P_mps)))
    out = dict(E_init=float(d["E_init"]), n_t=len(shots), shots_per_step=int(N), chi2_per_dof_mean=float(np.mean(stats)),
               frac_p_below_0p01=float(np.mean(pvals < 0.01)), ks_uniform_p=None, tv_max=tv_max,
               max_abs_P_mps_vs_P_readout=float(np.abs(P_mps - d["P"]).max()))
    from scipy.stats import kstest
    out["ks_uniform_p"] = float(kstest(pvals, "uniform").pvalue)
    print(json.dumps(out, indent=1))
    json.dump(out, open(JSN_MPS, "w"), indent=1)


def plot(t_show=(1.0, 2.5, 5.0, 10.0), n_bars=(100, 1000)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    res = json.load(open(JSN_PUB))
    P = np.load(os.path.join(HERE, "..", "data", "Ps_Heisenberg_triangular_4_3.npy"))
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True)
    ts, nx = V.DT * np.arange(1, len(P) + 1), 2 ** R
    xs, rng = np.arange(nx), np.random.default_rng(SEED + 1)
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.1), hspace=0.6, wspace=0.12)
    cols = ("#9ecae1", "#2171b5")
    for i, t in enumerate(t_show):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        k = int(np.argmin(np.abs(ts - t))); w = 0.38
        for j, (Ns, c) in enumerate(zip(n_bars, cols)):
            ax.bar(xs + (j - 0.5) * w, rng.multinomial(Ns, P[k]) / Ns, width=w, color=c, label=f"{Ns} shots", zorder=2)
        ax.plot(xs, P[k], "k_", ms=9, mew=1.6, label=r"exact $P(x;t)$", zorder=3)
        ax.axvline((-E_EXACT * ts[k] / (2 * np.pi)) % nx, color="0.35", ls="--", lw=0.8, zorder=1)
        ax.set_xticks(xs); ax.set_xlim(-0.7, nx - 0.3); ax.set_ylim(0, 1); ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(rf"$t={ts[k]:.1f}\,J^{{-1}}$", fontsize=10)
        ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
        if i % 2 == 0: ax.set_ylabel("frequency")
        else: ax.set_yticklabels([])
        if i // 2 == 1: ax.set_xlabel(r"pointer outcome $x$")
        if i == 0: ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax = fig.add_subplot(gs[2, :])
    Ns = np.array(sorted(int(k) for k in res["by_Ns"]))
    m = np.array([res["by_Ns"][str(k)]["mean"] for k in Ns]); s = np.array([res["by_Ns"][str(k)]["std"] for k in Ns])
    ax.errorbar(Ns, m, yerr=s, fmt="o-", color=cols[1], ms=4, capsize=3,
                label=r"$E_{\rm estimated}$ from $N_s$ shots per time step (mean $\pm$ std, " + f"{res['reps']} repetitions)")
    ax.axhline(res["E_median_exactP"], color="k", lw=0.9, label=r"$E_{\rm estimated}$ from exact $P(x;t)$")
    ax.axhline(E_EXACT, color="purple", lw=0.9, ls="--", label=r"$E_0$ (exact)")
    ax.set_xscale("log"); ax.set_xlabel(r"shots per time step $N_s$ (" + f"{res['n_t']}" + r" time steps, $\delta t=0.05/J$)")
    ax.set_ylabel(r"energy [$J$]"); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(lw=0.4, alpha=0.5)
    fig.text(0.015, 0.92, "(a)", fontsize=11); fig.text(0.015, 0.345, "(b)", fontsize=11)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_sampling_heisenberg.{ext}"), bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    {"published": published, "run": run, "analyze": analyze, "plot": plot}[sys.argv[1]]()
