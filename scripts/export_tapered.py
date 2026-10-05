"""Export the Z2-tapered (13-qubit) JW Hamiltonians of H8 and pyridine used in the simulations (Tangelo) -> ../data/tapered_hamiltonians.json."""
import json, os
from tangelo import SecondQuantizedMolecule
from tangelo.toolboxes.molecular_computation.frozen_orbitals import get_orbitals_excluding_homo_lumo
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.operators.taper_qubits import QubitTapering
from tangelo.toolboxes.operators import count_qubits
H8 = [("H", p) for p in [(1.6180339887, 0, 0), (1.3090169944, 0.9510565163, 0), (0.5, 1.5388417686, 0),
      (-0.5, 1.5388417686, 0), (-1.3090169944, 0.9510565163, 0), (-1.6180339887, 0, 0),
      (-1.3090169944, -0.9510565163, 0), (-0.5, -1.5388417686, 0)]]
PYR = [("C", (1.3603, 0.0256, 0.0)), ("C", (0.6971, -1.2020, 0.0)), ("C", (-0.6944, -1.2184, 0.0)),
       ("C", (-1.3895, -0.0129, 0.0)), ("C", (-0.6712, 1.1834, 0.0)), ("N", (0.6816, 1.1960, 0.0)),
       ("H", (2.4530, 0.1083, 0.0)), ("H", (1.2665, -2.1365, 0.0)), ("H", (-1.2365, -2.1696, 0.0)),
       ("H", (-2.4837, 0.0011, 0.0)), ("H", (-1.1569, 2.1657, 0.0))]
out = {}
for name, geo, homo_lumo in (("H8", H8, None), ("pyridine", PYR, (3, 3))):
    full = SecondQuantizedMolecule(geo, q=0, spin=0, basis="sto-3g")
    mol = full if homo_lumo is None else SecondQuantizedMolecule(
        geo, q=0, spin=0, basis="sto-3g",
        frozen_orbitals=get_orbitals_excluding_homo_lumo(full, homo_minus_n=homo_lumo[0], lumo_plus_n=homo_lumo[1]))
    h = fermion_to_qubit_mapping(mol.fermionic_hamiltonian, mapping="JW")
    t = QubitTapering(h, count_qubits(h), n_electrons=mol.n_active_electrons).z2_tapered_op
    n = count_qubits(t)
    H = {}
    for term, c in t.terms.items():
        s = ["I"] * n
        for i, a in term:
            s[i] = a
        H["".join(s)] = float(c.real)
    out[name] = dict(n=n, terms=H, n_active_mos=mol.n_active_mos, n_active_electrons=mol.n_active_electrons)
    print(name, "active MOs", mol.n_active_mos, "electrons", mol.n_active_electrons, "| JW qubits", count_qubits(h),
          "terms", len(h.terms), "| tapered qubits", n, "terms", len(H))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tapered_hamiltonians.json"), "w"))
