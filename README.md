# Tensor-network benchmarking of QPE in the von Neumann measurement formulation — code

This repository is the reproducibility package for the paper

> **Tensor-network benchmarking of quantum phase estimation in the von Neumann
> measurement formulation, with DMRG initial states**
> Younes Javanmard
> arXiv: https://arxiv.org/abs/2407.19348

The quantum protocol studied in the paper is the von Neumann measurement formulation of
phase estimation of Childs *et al.*, Phys. Rev. A **66**, 032314 (2002), and Novo *et al.*,
Quantum **5**, 465 (2021): the system is coupled to an *r*-qubit pointer through
exp(−i t H ⊗ p), p = Σ_j 2^{−j}(1 − Z_j)/2, and the pointer is read out after an inverse QFT.
This repository contains the tensor-network simulation of that protocol with DMRG-pretrained
matrix-product-state inputs, the simulation data behind the paper figures, and the scripts
that regenerate them.

## Layout
```
src/qu_alg_qu_sim_tensor_networks/   core library (paper subset)
  hamiltonians.py      Pauli-sum Hamiltonian, get_mpo (graph-based MPO, lossless SVD compression)
  graph_builder_new_version_just_to_keep_it.py
                       left/right fragment graphs used by get_mpo
  mpo_builder.py       MPO class and compression
  mps.py, mps_helper.py, dmrg.py, tdvp.py, lanczos.py, backend.py
                       MPS, two-site DMRG, TDVP (serial), Krylov exponential
  p2tdvp.py            two-site TDVP in the inverse canonical gauge, serial or parallel
                       (Secular et al., Phys. Rev. B 101, 235123 (2020)), adaptive bond dimension
  lattice/             lattices (triangular, square, ...) and spin models
scripts/
  vnqpe_tn.py          protocol-level functions: pointer MPO, coupled MPO of H (x) p, DMRG input,
                       TDVP evolution, pointer readout P(x;t), entropy, energy fit
  make_figures.py      regenerates Figs. 5-8 and results/figure_numbers.json from data/
  validate_tdvp.py     validation against exact state-vector evolution (paper Sec. III G)
  make_coupled_mpo_graph.py  graph of the MPO of H (x) p for HeH+ + pointer (paper Fig. 4)
  make_mpo_graphs_molecules.py  MPO graphs and bond dimensions of pyridine and BH3 (paper Appendix B, Fig. 11;
                       needs openfermion and openfermionpyscf)
notebooks/
  vn_qpe_tn_benchmark_clean.ipynb   end-to-end walkthrough (Part A: paper figures from data;
                                    Part B: validation; Part C: re-run of the 4x3 Heisenberg benchmark)
  make_notebook.py                  generates the notebook
data/                  TDVP pointer distributions P(x;t) used in the paper
  Ps_Heisenberg_triangular_4_3.npy  Heisenberg model, 4x3 triangular lattice (200 steps, dt = 0.05)
  Ps_H8_stgo3g.npy                  H8, STO-3G, FCI (600 steps)
  Ps_pyridine_stgo3g.npy            pyridine, STO-3G, (8e,8o) active space (600 steps)
  tlist.npy                         time grid of the molecular runs
results/figure_numbers.json         fitted energies (per pointer outcome and median) behind the figures
figures/                            figures produced by make_figures.py
```
Each `Ps_*.npy` array has shape `(n_steps, 2**r)` with `r = 4`: row `n` is the pointer
distribution at time `t = (n+1) * 0.05`.

## Install
```bash
conda env create -f environment.yml
conda activate vn-qpe-tn
pip install -e .
```
or, in an existing environment with Python >= 3.9: `pip install -e .`

## Reproduce the paper figures and numbers
```bash
cd scripts
python make_figures.py      # Figs. 5-8 -> ../figures, energies -> ../results/figure_numbers.json (~1 min)
python validate_tdvp.py     # TDVP vs exact state-vector evolution (~5 min)
```
Set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` explicitly (e.g. to 1-4). The tensors in these
simulations are small, and letting BLAS use all cores slows DMRG and TDVP down considerably.

The energy of each pointer outcome `x` is obtained by a least-squares fit of the time trace
`P(x;t)` to the single-eigenvalue form of the pointer distribution (paper Eq. 8), after a global
scan of `E`; the reported energy is the median over the 16 outcomes.

## Citation
```bibtex
@misc{javanmard2024vnqpe,
  title  = {Tensor-network benchmarking of quantum phase estimation in the von Neumann
            measurement formulation, with DMRG initial states},
  author = {Javanmard, Younes},
  year   = {2024},
  eprint = {2407.19348},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## License
MIT, see `LICENSE`.
