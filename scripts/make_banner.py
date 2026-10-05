"""Banner for the README: graph of the MPO of H (x) p for HeH+ (STO-3G, 4 qubits) coupled to an r = 3 pointer.

Writes figures/banner_light.png and figures/banner_dark.png (and .svg). Same graph as paper Fig. 4.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

TERMS = ["IIII", "IIIZ", "IIZI", "IIZZ", "IXIX", "IXZX", "IYIY", "IYZY", "IZII", "IZIZ", "IZZI", "XIXI", "XXYY",
         "XYYX", "XZXI", "XZXZ", "YIYI", "YXXY", "YYXX", "YZYI", "YZYZ", "ZIII", "ZIIZ", "ZIZI", "ZXZX", "ZYZY", "ZZII"]
N, m, r = 4, 2, 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# ---------------------------------------------------------------- graph (prefix/suffix tries + pointer chain)
nodes, edges = {}, []
def layer(keys, x, spread=0.62):
    keys = sorted(keys)
    for i, k in enumerate(keys):
        nodes[k] = (x, (i - (len(keys) - 1) / 2) * spread)
for i in range(m + 1):
    layer({("L", t[:i]) for t in TERMS}, i)
for i in range(m, N + 1):
    layer({("R", t[i:]) for t in TERMS}, i + 1)
for t in TERMS:
    for i in range(m):
        edges.append((("L", t[:i]), ("L", t[:i + 1]), t[i]))
    edges.append((("L", t[:m]), ("R", t[m:]), "c"))
    for i in range(m, N):
        edges.append((("R", t[i:]), ("R", t[i + 1:]), t[i]))
x0, term = N + 1, ("R", "")
for j in range(1, r + 1):
    last = j == r
    nodes[("B", j)] = (x0 + j, 0.0 if last else -0.75)
    prev = term if j == 1 else ("A", j - 1)
    if not last:
        nodes[("A", j)] = (x0 + j, 0.75)
        edges.append((prev, ("A", j), "I"))
    edges.append((prev, ("B", j), "p"))
    if j > 1:
        edges.append((("B", j - 1), ("B", j), "I"))
edges = list(dict.fromkeys(edges))
# MPO bond dimensions after lossless compression (get_mpo, eps = 1e-8) of H (x) p with the HeH+ coefficients
COMPRESSED = [1, 4, 16, 4, 1, 2, 2, 1]
used = {e[0] for e in edges} | {e[1] for e in edges}
bond = [len({k for k in used if nodes[k][0] == x}) for x in sorted({nodes[k][0] for k in used})]


def draw(theme):
    dark = theme == "dark"
    fg, sub = ("#e6edf3", "#9aa7b4") if dark else ("#1f2328", "#57606a")
    col = {"I": "#8b949e" if dark else "#9aa0a6", "X": "#ff6b6b", "Y": "#51cf66", "Z": "#4dabf7",
           "c": "#fcc419", "p": "#cc5de8"}
    node_sys, node_ptr, node_term = ("#ffa94d", "#e599f7", "#ffffff") if dark else ("#fd7e14", "#be4bdb", "#1f2328")

    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=200)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ybot = min(y for _, y in nodes.values())
    ytop = max(y for _, y in nodes.values())
    box_lo, box_hi = ybot - 1.05, ytop + 0.45          # panel boxes; text lives outside the node area

    # panels
    for (xa, xb, c, title) in ((-0.55, N + 1.45, "#fd7e14", r"molecule  $\hat H$  (HeH$^+$, STO-3G, 4 qubits)"),
                               (N + 1.6, x0 + r + 0.55, "#be4bdb", r"pointer  $\hat p$  ($r=3$ qubits)")):
        ax.add_patch(FancyBboxPatch((xa, box_lo), xb - xa, box_hi - box_lo,
                                    boxstyle="round,pad=0.02,rounding_size=0.35", lw=1.2,
                                    ec=c, fc=c, alpha=0.07 if not dark else 0.10, zorder=0))
        ax.add_patch(FancyBboxPatch((xa, box_lo), xb - xa, box_hi - box_lo,
                                    boxstyle="round,pad=0.02,rounding_size=0.35", lw=1.2,
                                    ec=c, fc="none", alpha=0.55, zorder=0))
        ax.text((xa + xb) / 2, box_hi + 0.3, title, ha="center", va="bottom", fontsize=12.5, color=fg)

    # edges: soft glow + curved stroke
    rng = np.random.default_rng(2)
    for u, v, lab in edges:
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        rad = 0.0 if lab in "p" or y1 == y2 else float(rng.uniform(-0.12, 0.12))
        for lw, a in ((4.5, 0.10), (1.3, 0.85)):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-", lw=lw, color=col[lab], alpha=a,
                                         shrinkA=3, shrinkB=3, zorder=1))
        if lab == "p":
            j = int(round(x2 - x0))
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy / np.hypot(dx, dy), dx / np.hypot(dx, dy)     # unit normal, pointing up
            if ny < 0:
                nx, ny = -nx, -ny
            ax.text((x1 + x2) / 2 + 0.34 * nx, (y1 + y2) / 2 + 0.34 * ny, rf"$\hat p_{{{j}}}$",
                    color=col["p"], fontsize=11, ha="center", va="center")

    # nodes with glow
    for k in used:
        x, y = nodes[k]
        c = node_term if k == term else (node_sys if k[0] in "LR" else node_ptr)
        ax.scatter([x], [y], s=230, color=c, alpha=0.18, lw=0, zorder=2)
        ax.scatter([x], [y], s=46, color=c, edgecolors=fg, linewidths=0.4, zorder=3)
    xt, yt = nodes[term]
    ax.annotate(r"$\hat H$ complete" + "\nbond dim. 1", xy=(xt, yt), xytext=(x0 + 1.6, yt - 2.9),
                fontsize=10.5, ha="center", va="center", color=fg,
                arrowprops=dict(arrowstyle="-|>", color=sub, lw=0.9, shrinkB=6))
    ax.text(m + 0.5, ybot - 0.62, r"coefficients $c_k$ at the cut", ha="center", va="center", fontsize=10, color=fg)

    # bond dimensions along the bottom: graph vertices per layer, and the MPO after lossless SVD compression
    # (the two vertex layers at the cut are one MPO bond, joined by the K-coefficient matrix)
    xs = sorted({nodes[k][0] for k in used})
    y1, y2 = box_lo - 0.5, box_lo - 1.15
    for x, D in zip(xs, bond):
        ax.text(x, y1, str(D), ha="center", va="center", fontsize=11, color=sub)
    for x, D in zip(xs[:m] + [m + 0.5] + xs[m + 2:], COMPRESSED):
        ax.text(x, y2, str(D), ha="center", va="center", fontsize=11, color=fg, fontweight="bold")
    ax.text(xs[0] - 0.45, y1, "graph vertices", ha="right", va="center", fontsize=10, color=sub)
    ax.text(xs[0] - 0.45, y2, "compressed MPO", ha="right", va="center", fontsize=10, color=fg)

    handles = [Line2D([0], [0], color=col[k], lw=3) for k in ("I", "X", "Y", "Z", "c", "p")]
    leg = ax.legend(handles, [r"$\hat I$", r"$\hat X$", r"$\hat Y$", r"$\hat Z$", r"$c_k$", r"$\hat p_j$"],
                    loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6, frameon=False, fontsize=11,
                    labelcolor=fg, handlelength=1.6)
    ax.set_xlim(-1.6, x0 + r + 0.75); ax.set_ylim(box_lo - 1.5, box_hi + 1.05); ax.axis("off")
    fig.suptitle(r"MPO of $\hat H\otimes\hat p$: a molecular Hamiltonian coupled to a von Neumann pointer",
                 fontsize=14.5, color=fg, y=0.995)
    fig.text(0.5, 0.0, r"lossless SVD compression ($\epsilon=10^{-8}$) of the graph MPO: HeH$^+$ is already minimal;"
             r"  pyridine (16 qubits, $K=2765$): $D_{\max}$ 437 $\to$ 138;  BH$_3$ (16 qubits, $K=5437$): 529 $\to$ 133",
             ha="center", va="top", fontsize=10.5, color=sub)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"banner_{theme}.{ext}"), bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    for theme in ("light", "dark"):
        draw(theme)
    print("bond dimensions per layer:", bond)
