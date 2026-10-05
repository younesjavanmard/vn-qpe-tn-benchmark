"""System-pointer entanglement S(t) for DMRG inputs of different bond dimension (paper Fig. 9).

Heisenberg model, 4x3 triangular lattice on a torus (36 bonds, J = 1), r = 4 pointer. For each chi_DMRG:
DMRG with the protocol of the paper (random product start, truncation 1e-7, at most 6 sweeps, stop when the
relative energy change is below 1e-6), then two-site TDVP of H (x) p (dt = 0.05, chi_max = 128, eps = 1e-6) up to
t = 10. At every step: S(t) = -Tr rho_p ln rho_p of the pointer reduced density matrix (system traced out),
the pointer distribution P(x;t), the bond dimension, the discarded weight, and the drift of <H (x) p>.

usage: python entanglement_vs_chi.py CHI [SEED]   -> ../results/entanglement_chi{CHI}_seed{SEED}.npz
       python entanglement_vs_chi.py plot                          -> ../figures/fig_entanglement_vs_chi.pdf
"""
import os, sys, glob, time, json
import numpy as np
import vnqpe_tn as V
from qu_alg_qu_sim_tensor_networks import mps, dmrg, mps_helper
from qu_alg_qu_sim_tensor_networks.backend import to_numpy
from sample_pointer import heisenberg_torus_4x3, E_EXACT

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
R, T_MAX = 4, 10.0


def dmrg_paper_protocol(sm, wmpo, chi, seed, sweeps=6, eps=1e-7, tol=1e-6):
    psi = mps.random_mps_state(sm.L, 2, 1, sm.bc, seed=seed)
    eng = dmrg.DMRGSolver(psi, sm, chi_max=chi, eps=eps)
    Es = [0.0]
    for k in range(sweeps):
        eng.sweep()
        Es.append(float(np.real(mps_helper.expectation_value(wmpo, psi))))
        print(f"  sweep {k}: E = {Es[-1]:.8f}  chi = {max(psi.bond_dimensions())}", flush=True)
        if abs((Es[-1] - Es[-2]) / Es[-1]) < tol:
            break
    psi.tensors = [to_numpy(B) for B in psi.tensors]; psi.singular_values = [to_numpy(S) for S in psi.singular_values]
    return psi, Es[-1]


def run(chi, seed=0):
    H = heisenberg_torus_4x3(); n = H.n
    w, sm = V.system_mpo(H)
    psi, E_init = dmrg_paper_protocol(sm, w, chi, seed)
    print(f"chi_DMRG = {chi}, seed = {seed}: E_init = {E_init:.6f} (E0 = {E_EXACT:.6f}), D_H = {max(w.bonds)}", flush=True)
    t0 = time.time()
    out = os.path.join(RES, f"entanglement_chi{chi}_seed{seed}.npz")
    ts, P, S, chis = V.evolve(V.attach_pointer(psi, R), V.coupled_model(sm, R), n, R, T_MAX, chi_max=128, eps=1e-6,
                              verbose=True, checkpoint=out.replace(".npz", "_partial.npz"))
    err = V.evolve.last_errors
    np.savez(out, ts=ts, P=P, S=S, chi=chis, dw=err["dw"], dE_gen=err["dE_gen"], E_init=E_init, chi_dmrg=chi,
             seed=seed, seconds=time.time() - t0)
    rows = V.estimate_energy(ts, P, R, E_init - 1.0, E_init)
    print(f"done in {time.time() - t0:.0f} s: S(t_max) = {S[-1]:.3f}, max S = {S.max():.3f}, "
          f"sum dw = {err['dw'].sum():.2e}, E_estimated = {np.median(rows):.5f}", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6, 4))
    summary = {}
    for f in sorted(glob.glob(os.path.join(RES, "entanglement_chi*_seed*.npz")), key=lambda f: -int(np.load(f)["chi_dmrg"])):
        if "_partial" in f:
            continue
        d = np.load(f); chi = int(d["chi_dmrg"])
        ax.plot(d["ts"], d["S"], label=rf"$\chi_{{\rm DMRG}}={chi}$ ($E_{{\rm init}}={float(d['E_init']):.3f}\,J$)")
        summary[chi] = dict(E_init=float(d["E_init"]), S_max=float(d["S"].max()), S_end=float(d["S"][-1]),
                            sum_dw=float(d["dw"].sum()))
    ax.axhline(R * np.log(2), color="0.5", ls=":", lw=0.8)
    ax.text(0.2, R * np.log(2) - 0.15, r"$r\ln 2$", color="0.4", fontsize=9)
    ax.set_xlabel(r"$t\,J$"); ax.set_ylabel(r"$S(t)$"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_entanglement_vs_chi.{ext}"), dpi=150)
    json.dump(summary, open(os.path.join(RES, "entanglement_vs_chi.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "plot":
        plot()
    else:
        run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
