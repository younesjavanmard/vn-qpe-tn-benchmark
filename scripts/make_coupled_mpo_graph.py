"""Graph of the MPO of H (x) p: molecular Hamiltonian (H2, STO-3G, Jordan-Wigner) followed by the pointer.

System part: left trie of Pauli prefixes (sites 0..m-1) and right trie of suffixes (sites m..N-1), joined at
the cut m by coefficient edges c_k (as in Ref. [MPO-LCU], Fig. 4 of the paper). All suffixes end in one
terminal vertex = "H completed" (MPO bond dimension 1 between system and pointer).
Pointer part: p = sum_j p_j with p_j = 2^{-j}(1 - Z_j)/2; two states per bond ("p not yet placed", "placed"),
edges labelled I or p_j. Output: figures/fig_mpo_graph_coupled.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 15 Pauli strings of the H2/STO-3G Hamiltonian (Jordan-Wigner, 4 qubits); coefficients do not affect the graph
TERMS = ["IIII", "ZIII", "IZII", "IIZI", "IIIZ", "ZZII", "ZIZI", "ZIIZ", "IZZI", "IZIZ", "IIZZ",
         "XXYY", "XYYX", "YXXY", "YYXX"]
N, m, r = 4, 2, 3
COL = {"I": "0.55", "X": "red", "Y": "green", "Z": "blue", "c": "olive", "p": "purple"}

nodes, edges = {}, []                   # nodes: key -> (x, y); edges: (u, v, label)
def layer_positions(keys, x, spread=1.0):
    keys = sorted(keys)
    n = len(keys)
    for i, k in enumerate(keys):
        nodes[k] = (x, (i - (n - 1) / 2) * spread)

# left trie: prefixes of length 0..m at x = 0..m
for i in range(m + 1):
    layer_positions({("L", t[:i]) for t in TERMS}, i)
for t in TERMS:
    for i in range(m):
        edges.append((("L", t[:i]), ("L", t[:i + 1]), t[i]))
# right trie: suffixes starting at site i (i = m..N) at x = i+1 ; suffix "" is the terminal vertex
for i in range(m, N + 1):
    layer_positions({("R", t[i:]) for t in TERMS}, i + 1)
for t in TERMS:
    edges.append((("L", t[:m]), ("R", t[m:]), "c"))                    # coefficient bridge at the cut
    for i in range(m, N):
        edges.append((("R", t[i:]), ("R", t[i + 1:]), t[i]))
# pointer chain: A_j = "p not yet placed", B_j = "placed" ; A_0 = terminal system vertex
x0 = N + 1
term = ("R", "")
for j in range(1, r + 1):
    last = j == r                               # right boundary keeps only the "placed" state
    nodes[("B", j)] = (x0 + j, 0.0 if last else -0.6)
    a_prev = term if j == 1 else ("A", j - 1)
    if not last:
        nodes[("A", j)] = (x0 + j, 0.6)
        edges.append((a_prev, ("A", j), "I"))
    edges.append((a_prev, ("B", j), "p"))
    if j > 1:
        edges.append((("B", j - 1), ("B", j), "I"))
edges = list(dict.fromkeys(edges))
used = {e[0] for e in edges} | {e[1] for e in edges}

fig, ax = plt.subplots(figsize=(9.0, 4.2))
for u, v, lab in edges:
    (x1, y1), (x2, y2) = nodes[u], nodes[v]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COL[lab], lw=1.1 if lab in "cp" else 1.0,
                                shrinkA=4, shrinkB=4, mutation_scale=8, alpha=0.9))
    if lab == "p":
        j = nodes[v][0] - x0
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 - 0.15, rf"$p_{{{int(j)}}}$", color="purple", fontsize=8)
for k in used:
    x, y = nodes[k]
    face = "darkorange" if k[0] in ("L", "R") else "plum"
    if k == term:
        face = "black"
    ax.plot(x, y, "o", ms=6, mfc=face, mec="0.2", mew=0.5, zorder=3)
xt = nodes[term]
ax.annotate(r"$\hat H$ complete" + "\n(bond dim. 1)", xy=xt, xytext=(xt[0] - 0.2, xt[1] - 2.6), fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
ymax = max(y for _, y in nodes.values()) + 0.9
ax.axvspan(-0.4, N + 1.4, color="orange", alpha=0.06, lw=0)
ax.axvspan(N + 1.4, x0 + r + 0.4, color="purple", alpha=0.06, lw=0)
ax.text((N + 1) / 2, ymax, r"system: $\mathrm{H}_2$ (STO-3G), $N=4$ qubits", ha="center", fontsize=9)
ax.text(x0 + (r + 1) / 2 - 0.3, ymax, rf"pointer: $r={r}$ qubits", ha="center", fontsize=9)
handles = [Line2D([0], [0], color=COL[k], lw=2) for k in ("I", "X", "Y", "Z", "c", "p")]
ax.legend(handles, [r"$\hat I$", r"$\hat X$", r"$\hat Y$", r"$\hat Z$", r"$c_k$", r"$\hat p_j$"], loc="lower left",
          fontsize=7, frameon=False, ncol=6, bbox_to_anchor=(0.0, -0.13))
ax.set_xlim(-0.5, x0 + r + 0.5); ax.set_ylim(min(y for _, y in nodes.values()) - 1.0, ymax + 0.6)
ax.axis("off")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig_mpo_graph_coupled.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")
bond = [len({k for k in used if nodes[k][0] == x}) for x in sorted({nodes[k][0] for k in used})]
print("vertices per layer (MPO bond dimensions before compression):", bond)
