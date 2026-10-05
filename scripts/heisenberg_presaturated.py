"""Heisenberg 4x3 benchmark (paper Fig. 5) with and without saturating the bond dimension before the evolution.

Two-site TDVP grows the bond dimension only gradually from the low-chi initial state |psi_DMRG> (x) |+>^r; while it
grows, the two-site tangent space does not contain the full action of H (x) p, which has long-range couplings
(paper Sec. III H). With --presaturate the bond dimension is first grown by N_PRE two-site steps of negligible
duration DT_PRE; the elapsed time N_PRE * DT_PRE is included in the time axis.

usage: python heisenberg_presaturated.py standard|presaturated   -> ../results/heisenberg_{mode}.npz
       python heisenberg_presaturated.py analyze                 -> ../results/heisenberg_presaturated.json
"""
import os, sys, time, json, pickle
import numpy as np
import vnqpe_tn as V
from qu_alg_qu_sim_tensor_networks.tdvp import TDVPIntegrator
from qu_alg_qu_sim_tensor_networks import mps_helper, mpo_builder as qmpo
from sample_pointer import heisenberg_torus_4x3, E_EXACT
from entanglement_vs_chi import dmrg_paper_protocol

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R, CHI_DMRG, T_MAX, DT, CHI_MAX, EPS = 4, 20, 10.0, 0.05, 128, 1e-10
N_PRE, DT_PRE = 8, 1e-4


def run(mode):
    H = heisenberg_torus_4x3(); n = H.n
    w, sm = V.system_mpo(H)
    cache = os.path.join(RES, f"dmrg_heisenberg_paperprotocol_chi{CHI_DMRG}.pkl")
    if os.path.exists(cache):
        psi, E_init = pickle.load(open(cache, "rb"))
    else:
        psi, E_init = dmrg_paper_protocol(sm, w, CHI_DMRG, seed=0)
        pickle.dump((psi, E_init), open(cache, "wb"))
    print(f"{mode}: E_init = {E_init:.6f}, E0 = {E_EXACT:.6f}", flush=True)
    model = V.coupled_model(sm, R)
    Hmpo = qmpo.MPO([np.transpose(np.asarray(W), (2, 3, 0, 1)) for W in model.H_mpo])
    p = V.attach_pointer(psi, R)
    E_start = float(np.real(mps_helper.expectation_value(Hmpo, p)))
    eng = TDVPIntegrator(p, model, chi_max=CHI_MAX, eps=EPS)
    t0 = 0.0
    if mode == "presaturated":
        for _ in range(N_PRE):
            p = eng.step_two_site(DT_PRE); t0 += DT_PRE
        print(f"  after pre-saturation: chi = {p.bond_dimensions()}", flush=True)
    ts, P, S, chis, DW, EG = [], [], [], [], [], []
    V.TRUNC["step"] = 0.0
    nsteps = int(round(T_MAX / DT)); tic = time.time()
    for k in range(1, nsteps + 1):
        p = eng.step_two_site(DT)
        DW.append(V.TRUNC["step"]); V.TRUNC["step"] = 0.0
        rho = V.pointer_rdm(p, n, R)
        ts.append(t0 + k * DT); P.append(V.pointer_distribution(rho, R)); S.append(V.pointer_entropy(rho))
        chis.append(max(p.bond_dimensions()))
        if k % 10 == 0:
            EG.append(float(np.real(mps_helper.expectation_value(Hmpo, p))) - E_start)
            print(f"  t = {ts[-1]:6.3f}  chi = {chis[-1]}  S = {S[-1]:.3f}  sum dw = {sum(DW):.2e}  "
                  f"dE_gen = {EG[-1]:.2e}  ({time.time() - tic:.0f} s)", flush=True)
    np.savez(os.path.join(RES, f"heisenberg_{mode}.npz"), ts=np.array(ts), P=np.array(P), S=np.array(S),
             chi=np.array(chis), dw=np.array(DW), dE_gen=np.array(EG), E_init=E_init, t0=t0)


def analyze():
    out = {}
    for mode in ("standard", "presaturated"):
        d = np.load(os.path.join(RES, f"heisenberg_{mode}.npz"))
        ts, P, Ei = d["ts"], d["P"], float(d["E_init"])
        rows = V.estimate_energy(ts, P, R, Ei - 1.0, Ei)
        rows_wide = V.estimate_energy(ts, P, R, Ei - 3.0, Ei)
        out[mode] = dict(E_init=Ei, E_median=float(np.median(rows)), E_median_wide=float(np.median(rows_wide)),
                         dE=float(abs(np.median(rows) - E_EXACT)), rows=rows.tolist(), spread=float(np.ptp(rows)),
                         sum_dw=float(d["dw"].sum()), dE_gen_end=float(d["dE_gen"][-1]), S_end=float(d["S"][-1]))
        print(f"{mode:13s} E_estimated = {np.median(rows):.5f} (wide window {np.median(rows_wide):.5f})  "
              f"|E - E0| = {abs(np.median(rows) - E_EXACT):.2e}  spread = {np.ptp(rows):.4f}  "
              f"sum dw = {d['dw'].sum():.1e}  drift = {d['dE_gen'][-1]:.1e}")
    a, b = np.load(os.path.join(RES, "heisenberg_standard.npz")), np.load(os.path.join(RES, "heisenberg_presaturated.npz"))
    out["max_abs_dP_between_runs"] = float(np.abs(a["P"] - b["P"]).max())
    print(f"max |P_standard - P_presaturated| = {out['max_abs_dP_between_runs']:.2e}")
    json.dump(out, open(os.path.join(RES, "heisenberg_presaturated.json"), "w"), indent=1)


if __name__ == "__main__":
    analyze() if sys.argv[1] == "analyze" else run(sys.argv[1])
