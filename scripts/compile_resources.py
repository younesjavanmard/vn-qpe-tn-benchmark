"""Two-qubit gate counts of one first-order Trotter step for the systems of the paper.

usage: python compile_resources.py SYSTEM STRATEGY [--pointer]
  SYSTEM   : heisenberg | H8 | pyridine | BH3 | H8_tapered | pyridine_tapered
             (*_tapered: the Z2-tapered 13-qubit Hamiltonians of the paper, data/tapered_hamiltonians.json)
  STRATEGY : baseline | qiskit | rustiq | tket | routed_line
  --pointer: compile the step of H (x) p with an r = 4 pointer instead of H
Environment variables: ANGLE_SCALE (multiply all rotation angles, e.g. 1000, to separate structural gate
reductions from small-angle approximations made by the compilers), SCREEN (drop terms with |c| < SCREEN).
Appends one JSON line to ../results/compile_resources.jsonl. Needs qiskit >= 2.5 (Rustiq is built in),
pytket and pytket-qiskit (requirements-compile.txt); openfermion + openfermionpyscf for untapered molecules.
"""
import sys, json, time, os
import numpy as np

R = 4
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- Hamiltonians as {pauli_string: coeff}
def heisenberg_triangular_4x3():
    L1, L2 = 4, 3
    idx = lambda x, y: (y % L2) * L1 + (x % L1)
    bonds = set()
    for x in range(L1):
        for y in range(L2):
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                e = tuple(sorted((idx(x, y), idx(x + dx, y + dy))))
                bonds.add(e)
    n = L1 * L2
    H = {}
    for i, j in sorted(bonds):
        for a in "XYZ":
            s = ["I"] * n; s[i] = a; s[j] = a
            H["".join(s)] = 1.0
    return H, n


def molecule(name):
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf
    if name == "H8":
        xyz = [(1.6180339887, 0, 0), (1.3090169944, 0.9510565163, 0), (0.5, 1.5388417686, 0), (-0.5, 1.5388417686, 0),
               (-1.3090169944, 0.9510565163, 0), (-1.6180339887, 0, 0), (-1.3090169944, -0.9510565163, 0),
               (-0.5, -1.5388417686, 0)]
        geom, basis, act, occ = [("H", p) for p in xyz], "sto-3g", None, None
    elif name == "pyridine":
        geom = [("C", (1.3603, 0.0256, 0.0)), ("C", (0.6971, -1.2020, 0.0)), ("C", (-0.6944, -1.2184, 0.0)),
                ("C", (-1.3895, -0.0129, 0.0)), ("C", (-0.6712, 1.1834, 0.0)), ("N", (0.6816, 1.1960, 0.0)),
                ("H", (2.4530, 0.1083, 0.0)), ("H", (1.2665, -2.1365, 0.0)), ("H", (-1.2365, -2.1696, 0.0)),
                ("H", (-2.4837, 0.0011, 0.0)), ("H", (-1.1569, 2.1657, 0.0))]
        basis, act, occ = "sto-3g", list(range(17, 25)), list(range(17))
    elif name == "BH3":
        geom = [("B", (0.0, 0.0, 0.0)), ("H", (0.0, 1.0, 1.0)), ("H", (1.0, 0.0, 1.0)), ("H", (1.0, 1.0, 0.0))]
        basis, act, occ = "sto-6g", None, None
    mol = run_pyscf(MolecularData(geom, basis, 1, 0), run_scf=True)
    q = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian(occupied_indices=occ, active_indices=act)))
    q.compress(1e-10)
    n = 2 * (len(act) if act else mol.n_orbitals)
    H = {}
    for term, c in q.terms.items():
        s = ["I"] * n
        for i, a in term:
            s[i] = a
        H["".join(s)] = float(np.real(c))
    return H, n


SCREEN = float(os.environ.get("SCREEN", 0.0))     # drop Pauli terms with |c| < SCREEN (Ha or J)
ANGLE_SCALE = float(os.environ.get("ANGLE_SCALE", 1.0))   # >1: O(1) angles, removes small-angle approximations


def rotations(H, n, pointer):
    """List of (pauli_string, angle) for one Trotter step (qubit 0 = leftmost character)."""
    dt = 0.05
    rots = []
    for s, c in H.items():
        if set(s) == {"I"} or abs(c) < SCREEN:
            continue
        if not pointer:
            rots.append((s, 2 * c * dt))
        else:
            # c P (x) p_j , p_j = 2^{-j}(1 - Z_j)/2  ->  rotation of P and of P Z_j (j = 1..r)
            rots.append((s + "I" * R, 2 * c * dt * sum(2.0 ** (-j) / 2 for j in range(1, R + 1))))
            for j in range(1, R + 1):
                z = ["I"] * R; z[j - 1] = "Z"
                rots.append((s + "".join(z), -2 * c * dt * 2.0 ** (-j) / 2))
    rots = [(p_, th * ANGLE_SCALE) for p_, th in rots]
    return rots, n + (R if pointer else 0)


# ---------------------------------------------------------------- strategies
def baseline(rots, nq, pointer=False):
    """Paper model (Sec. V): a CNOT ladder of 2(w-1) CNOTs per system string of weight w; with the
    pointer, the ladder is shared and each pointer qubit costs G_ctrl = 2 extra CNOTs."""
    w = lambda s: sum(a != "I" for a in s)
    if not pointer:
        return dict(cx=int(sum(2 * (w(s) - 1) for s, _ in rots if w(s) > 1)), depth=None)
    sys_rots = [s for s, _ in rots if set(s[-R:]) == {"I"}]
    return dict(cx=int(sum(2 * (w(s) - 1) for s in sys_rots if w(s) > 1) + 2 * R * len(sys_rots)), depth=None)


def qiskit_circuit(rots, nq, synthesis=None):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    qc = QuantumCircuit(nq)
    for s, th in rots:            # qiskit labels are little-endian: reverse the string
        op = SparsePauliOp(s[::-1])
        qc.append(PauliEvolutionGate(op, time=th / 2), range(nq))
    return qc


def qiskit_opt(rots, nq, plugin=None, coupling=None):
    from qiskit import transpile
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
    qc = qiskit_circuit(rots, nq)
    kw = dict(basis_gates=["cx", "rz", "sx", "x"], optimization_level=3, seed_transpiler=1)
    if plugin:
        kw["hls_config"] = HLSConfig(PauliEvolution=[(plugin, {})])
    if coupling is not None:
        kw["coupling_map"] = coupling
    out = transpile(qc, **kw)
    res = dict(cx=int(out.count_ops().get("cx", 0)), depth=int(out.depth(lambda g: g.operation.num_qubits == 2)))
    if coupling is None:
        res["infidelity"] = verify(rots, nq, out)
    return res


def apply_pauli_rotations(psi, rots, nq):
    """Exact exp(-i th/2 P) for each (P, th) in order, on a state vector in qiskit's little-endian
    ordering (bit q of the index = qubit q = character q of the Pauli string)."""
    idx = np.arange(2**nq)
    for s, th in rots:
        x = sum(1 << q for q, a in enumerate(s) if a in "XY")
        z = sum(1 << q for q, a in enumerate(s) if a in "YZ")
        ny = s.count("Y")
        src = idx ^ x
        sign = 1 - 2 * (np.bitwise_count((src & z).astype(np.uint64)) & 1).astype(np.int8)
        Ppsi = (1j ** ny) * sign * psi[src]
        psi = np.cos(th / 2) * psi - 1j * np.sin(th / 2) * Ppsi
    return psi


def verify(rots, nq, compiled, seed=3):
    """Infidelity 1 - |<psi_exact|psi_compiled>|^2 for a random input state."""
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    psi0 = rng.normal(size=2**nq) + 1j * rng.normal(size=2**nq)
    psi0 /= np.linalg.norm(psi0)
    a = apply_pauli_rotations(psi0.copy(), rots, nq)
    b = Statevector(psi0).evolve(compiled).data
    return float(1 - abs(np.vdot(a, b)) ** 2)


def tket(rots, nq):
    from pytket import Circuit
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli
    from pytket.passes import GreedyPauliSimp, FullPeepholeOptimise, AutoRebase
    from pytket import OpType
    P = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    c = Circuit(nq)
    for s, th in rots:
        c.add_pauliexpbox(PauliExpBox([P[a] for a in s], th / np.pi), list(range(nq)))
    GreedyPauliSimp().apply(c)
    FullPeepholeOptimise().apply(c)
    AutoRebase({OpType.CX, OpType.Rz, OpType.Rx}).apply(c)
    from pytket.extensions.qiskit import tk_to_qiskit
    res = dict(cx=int(c.n_gates_of_type(OpType.CX)), depth=int(c.depth_by_type(OpType.CX)))
    res["infidelity"] = verify(rots, nq, tk_to_qiskit(c, replace_implicit_swaps=True))
    return res


if __name__ == "__main__":
    system, strategy, pointer = sys.argv[1], sys.argv[2], "--pointer" in sys.argv
    if system.endswith("_tapered"):
        d = json.load(open(os.path.join(HERE, "..", "data", "tapered_hamiltonians.json")))[system[:-8]]
        H, n = d["terms"], d["n"]
    else:
        H, n = heisenberg_triangular_4x3() if system == "heisenberg" else molecule(system)
    rots, nq = rotations(H, n, pointer)
    t0 = time.time()
    if strategy == "baseline":
        res = baseline(rots, nq, pointer)
    elif strategy == "qiskit":
        res = qiskit_opt(rots, nq)
    elif strategy == "rustiq":
        res = qiskit_opt(rots, nq, plugin="rustiq")
    elif strategy == "tket":
        res = tket(rots, nq)
    elif strategy == "routed_line":
        from qiskit.transpiler import CouplingMap
        res = qiskit_opt(rots, nq, plugin="rustiq", coupling=CouplingMap.from_line(nq))
    rec = dict(system=system, pointer=pointer, strategy=strategy, screen=SCREEN, angle_scale=ANGLE_SCALE, qubits=nq, K=len([s for s in H if set(s) != {"I"}]),
               rotations=len(rots), seconds=round(time.time() - t0, 1), **res)
    print(json.dumps(rec), flush=True)
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    with open(os.path.join(HERE, "..", "results", "compile_resources.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
