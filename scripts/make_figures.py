"""Regenerate the time-resolved pointer figures from the saved TDVP data (Ps_*.npy).

Each row x: P(x;t) (filled) and the one-parameter fit |f(E,x)|^2 (black line); E_es(x) printed per row.
E_estimated = median over the 2^r row estimates (robust to single failed fits).
usage: python make_figures.py   (writes figures/*.pdf and scripts/figure_numbers.json)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vnqpe_tn as V

DATA = os.environ.get("DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figures")); os.makedirs(FIG, exist_ok=True)
R, DT, CHEM = 4, 0.05, 1.5936e-3
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

BENCH = {
    "heisenberg": dict(file="Ps_Heisenberg_triangular_4_3.npy", E_init=-26.7333, E_ref=-27.7232, unit="J",
                       ref_label=r"E_{\rm exact}", t_show=None, trotter_axis=False),
    "H8": dict(file="Ps_H8_stgo3g.npy", E_init=-4.18513790, E_ref=-4.271824120916049, unit="Ha",
               ref_label=r"E_{\rm FCI}", t_show=None, trotter_axis=True),
    "pyridine": dict(file="Ps_pyridine_stgo3g.npy", E_init=-243.66072968, E_ref=-243.6967896844996, unit="Ha",
                     ref_label=r"E_{\rm ref}", t_show=6.0, trotter_axis=True),
}


def stack_plot(ts, P, rows, b, fname):
    nx = P.shape[1]
    m = ts <= (b["t_show"] or ts[-1]) + 1e-9
    fig, ax = plt.subplots(figsize=(6.8, 7.6))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, nx))
    tt = np.linspace(ts[m][0], ts[m][-1], 6000)
    for x in range(nx):
        y0 = (nx - 1 - x) * 1.0                                  # |0> on top
        ax.fill_between(ts[m], y0, y0 + 0.95 * P[m, x], color=cols[x], alpha=0.65, lw=0)
        ax.plot(tt, y0 + 0.95 * V.fejer(tt, rows[x], x, R), "k", lw=0.45)
        ax.text(1.01, (y0 + 0.2) / nx, rf"$|{x}\rangle$  $E_{{\rm es}}={rows[x]:.4f}$", transform=ax.transAxes,
                fontsize=7.5, va="bottom")
    ax.set_yticks([]); ax.set_ylim(-0.1, nx + 0.1)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    unit = "J" if b["unit"] == "J" else r"{\rm Ha}"
    ax.set_xlabel(rf"total evolution time $t$ [$1/{unit}$]")
    ax.set_xlim(ts[m][0] - 0.02 * ts[m][-1], ts[m][-1])
    head = (rf"$E^{{\rm init}}_{{\rm DMRG}}={b['E_init']:.4f}$;  $E_{{\rm estimated}}={np.median(rows):.4f}$;  "
            rf"${b['ref_label']}={b['E_ref']:.4f}$ [${unit}$]")
    if b["trotter_axis"]:
        top = ax.secondary_xaxis("top", functions=(lambda t: t / DT, lambda n: n * DT))
        top.set_xlabel(r"Trotter steps $n=t/\delta t$")
        ax.set_title(head, fontsize=9, pad=28)
    else:
        ax.set_title(head, fontsize=9)
    ax.plot([ts[m][0]] * 2, [nx - 1, nx - 1 + 0.95], "k", lw=0.8)
    ax.text(ts[m][0] - 0.015 * ts[m][-1], nx - 0.05, "1", ha="right", fontsize=7)
    ax.text(ts[m][0] - 0.015 * ts[m][-1], nx - 1, "0", ha="right", fontsize=7)
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)


numbers = {}
for key, b in BENCH.items():
    P = np.load(os.path.join(DATA, b["file"])); ts = DT * np.arange(1, len(P) + 1)
    rows = V.estimate_energy(ts, P, R, b["E_init"] - 1.0, b["E_init"])
    numbers[key] = dict(t_max=float(ts[-1]), E_init=b["E_init"], E_ref=b["E_ref"], rows=rows.tolist(),
                        E_median=float(np.median(rows)), E_min=float(rows.min()), spread=float(rows.max() - rows.min()),
                        dE_median=float(abs(np.median(rows) - b["E_ref"])))
    stack_plot(ts, P, rows, b, f"fig_{key}_stack.pdf")
    print(f"{key:10s} t_max={ts[-1]:5.1f} median={np.median(rows):.5f} min={rows.min():.5f} "
          f"|dE|(median)={abs(np.median(rows)-b['E_ref']):.2e} spread={rows.max()-rows.min():.4f}")

# convergence with the fitted window (estimate at t uses data up to t), H8 and pyridine
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for key, T_list, c in (("H8", [0.5, 1.5, 2.5, 3.75, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30], "C0"),
                       ("pyridine", [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30], "C1")):
    b = BENCH[key]; P = np.load(os.path.join(DATA, b["file"])); ts = DT * np.arange(1, len(P) + 1)
    est = []
    for T in T_list:
        mm = ts <= T + 1e-9
        est.append(np.median(V.estimate_energy(ts[mm], P[mm], R, b["E_init"] - 1.0, b["E_init"])))
    est = np.array(est); numbers[key]["convergence"] = dict(t=T_list, E=est.tolist())
    lab = r"H$_8$" if key == "H8" else "pyridine"
    axs[0].plot(T_list, est - b["E_ref"], "o-", color=c, ms=3.5, label=lab)
    axs[1].semilogy(T_list, np.abs(est - b["E_ref"]), "o-", color=c, ms=3.5, label=lab)
    print(key, "convergence:", [f"t={T}: {abs(e-b['E_ref'])*1e3:.2f} mHa" for T, e in zip(T_list, est)])
axs[0].axhline(0, color="k", lw=0.6)
axs[0].set(xlabel=r"$t$ [1/Ha]", ylabel=r"$E_{\rm estimated}(t)-E_{\rm ref}$ [Ha]")
axs[1].axhspan(1e-4, CHEM, color="purple", alpha=0.12); axs[1].axhline(CHEM, color="purple", lw=0.6, ls="--")
axs[1].text(0.5, CHEM * 1.15, "chemical accuracy", color="purple", fontsize=7)
axs[1].set(xlabel=r"$t$ [1/Ha]", ylabel=r"$|\Delta E|$ [Ha]", ylim=(3e-4, 0.15))
for a in axs:
    a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_convergence_molecules.pdf")); plt.close(fig)
json.dump(numbers, open(os.path.join(HERE, "..", "results", "figure_numbers.json"), "w"), indent=1)
print("figures written to", os.path.abspath(FIG))
