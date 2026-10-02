"""Graph-based MPO construction for two molecules of this work (appendix figure).

Pyridine, STO-3G, (8e,8o) active space (HOMO-3 ... LUMO+3), and BH3, STO-6G, both Jordan-Wigner (16 qubits).
For each molecule:
  * full Pauli decomposition (OpenFermion + PySCF), number of terms K,
  * MPO bond dimensions of the full Hamiltonian from the graph-based builder before/after lossless compression,
  * drawing of the left/right fragment graphs for a random subset of n_show terms.
Output: figures/fig_mpo_graph_molecules.pdf and results/mpo_molecules.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from openfermion import MolecularData, get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"I": "0.55", "X": "red", "Y": "green", "Z": "blue", "c": "olive"}

PYRIDINE = [("C", (1.3603, 0.0256, 0.0)), ("C", (0.6971, -1.2020, 0.0)), ("C", (-0.6944, -1.2184, 0.0)),
            ("C", (-1.3895, -0.0129, 0.0)), ("C", (-0.6712, 1.1834, 0.0)), ("N", (0.6816, 1.1960, 0.0)),
            ("H", (2.4530, 0.1083, 0.0)), ("H", (1.2665, -2.1365, 0.0)), ("H", (-1.2365, -2.1696, 0.0)),
            ("H", (-2.4837, 0.0011, 0.0)), ("H", (-1.1569, 2.1657, 0.0))]
BH3 = [("B", (0.0, 0.0, 0.0)), ("H", (0.0, 1.0, 1.0)), ("H", (1.0, 0.0, 1.0)), ("H", (1.0, 1.0, 0.0))]


def pauli_terms(geom, basis, active=None, occupied=None):
    mol = run_pyscf(MolecularData(geom, basis, 1, 0), run_scf=True)
    H = mol.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    q = jordan_wigner(get_fermion_operator(H)); q.compress(1e-10)
    n = 2 * (len(active) if active else mol.n_orbitals)
    out = {}
    for term, c in q.terms.items():
        s = ["I"] * n
        for i, a in term:
            s[i] = a
        out["".join(s)] = float(np.real(c))
    return out, n


def mpo_bonds(terms, n):
    from qu_alg_qu_sim_tensor_networks.hamiltonians import Hamiltonian, get_mpo
    from qu_alg_qu_sim_tensor_networks import graph_builder_new_version_just_to_keep_it as gb
    from qu_alg_qu_sim_tensor_networks import mpo_builder as qmpo
    H = Hamiltonian(n)
    for s, c in terms.items():
        H.add_term([(i, a) for i, a in enumerate(s) if a != "I"], c)
    raw = qmpo.MPO(gb.build_mpo_tensors(H.ham_terms, n, H.as_dict, H.term_indices), Ss=None, bonds=None)
    comp, _, _ = get_mpo(n, H)
    return list(raw.bonds), list(comp.bonds)


def draw(ax, strings, n, title):
    m = n // 2
    nodes, edges = {}, []
    def layer(keys, x):
        keys = sorted(keys); k = len(keys)
        for i, key in enumerate(keys):
            nodes[key] = (x, (i - (k - 1) / 2))
    for i in range(m + 1):
        layer({("L", t[:i]) for t in strings}, i)
    for i in range(m, n + 1):
        layer({("R", t[i:]) for t in strings}, i + 1)
    for t in strings:
        for i in range(m):
            edges.append((("L", t[:i]), ("L", t[:i + 1]), t[i]))
        edges.append((("L", t[:m]), ("R", t[m:]), "c"))
        for i in range(m, n):
            edges.append((("R", t[i:]), ("R", t[i + 1:]), t[i]))
    for u, v, lab in dict.fromkeys(edges):
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        ax.plot([x1, x2], [y1, y2], color=COL[lab], lw=0.6 if lab != "c" else 0.45, alpha=0.85, zorder=1)
    xs, ys = zip(*nodes.values())
    ax.scatter(xs, ys, s=7, c="darkorange", edgecolors="0.25", linewidths=0.3, zorder=2)
    ax.set_title(title, fontsize=9); ax.axis("off")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    mols = {
        "pyridine": dict(geom=PYRIDINE, basis="sto-3g", active=list(range(17, 25)), occupied=list(range(17)),
                         label=r"pyridine, STO-3G, (8e,8o)"),
        "BH3": dict(geom=BH3, basis="sto-6g", active=None, occupied=None, label=r"$\mathrm{BH}_3$, STO-6G"),
    }
    n_show, res = 40, {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (key, d) in zip(axs, mols.items()):
        terms, n = pauli_terms(d["geom"], d["basis"], d["active"], d["occupied"])
        raw, comp = mpo_bonds(terms, n)
        strings = list(rng.choice(sorted(terms), size=n_show, replace=False))
        res[key] = dict(qubits=n, K=len(terms), D_raw_max=max(raw), D_compressed_max=max(comp),
                        bonds_raw=raw, bonds_compressed=comp, n_shown=n_show)
        print(f"{key}: N={n}, K={len(terms)}, max D raw={max(raw)}, compressed={max(comp)}")
        draw(ax, strings, n, f"{d['label']}: {n} qubits, $K={len(terms)}$ terms (showing {n_show})")
    handles = [Line2D([0], [0], color=COL[k], lw=2) for k in ("I", "X", "Y", "Z", "c")]
    fig.legend(handles, [r"$\hat I$", r"$\hat X$", r"$\hat Y$", r"$\hat Z$", r"$c_k$"], loc="lower center", ncol=5,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, "..", "figures", "fig_mpo_graph_molecules.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "..", "figures", "fig_mpo_graph_molecules.png"), dpi=100, bbox_inches="tight")
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "..", "results", "mpo_molecules.json"), "w"), indent=1)
