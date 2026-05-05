# TriangularPlateaux.jl

[![Build Status](https://github.com/Hao-Phys/TriangularPlateaux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Hao-Phys/TriangularPlateaux.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements the self-consistent one-loop spin-wave theory calculations presented in:

> **$J_1-J_2$ Triangular Lattice Antiferromagnet in a Magnetic Field**  
> Anna Keselman, Xinyuan Xu, Hao Zhang, Cristian D. Batista, Oleg A. Starykh  
> [arXiv:2512.02150](https://arxiv.org/abs/2512.02150) (2025)

## Overview

The package computes the magnon excitation spectra and magnetization plateau phase boundaries of the spin-1/2 $J_1-J_2$ triangular-lattice Heisenberg antiferromagnet in an applied magnetic field. It combines linear spin-wave theory (LSWT) with a self-consistent normal-ordering correction to stabilize the up-up-down (UUD) and up-up-up-down (UUUD) magnetization plateau phases, which are induced by quantum fluctuations.

The implementation is built on the [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) platform for spin dynamics calculations.

## Features

- LSWT Hamiltonian for the UUD and UUUD plateau phases on the triangular lattice
- Normal-ordering correction $H_{\rm NO}^{(2)}(\bm{q})$ from quartic Holstein-Primakoff interactions, including both $J_1$ (nearest-neighbor) and $J_2$ (next-nearest-neighbor) contributions
- One-loop perturbative correction to the LSWT spectrum
- Self-consistent one-loop (SCOL) theory with iterative updating of bosonic bilinears
- Phase boundary determination via bisection on the magnon gap

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Hao-Phys/TriangularPlateaux.jl")
```

## Usage

```julia
using TriangularPlateaux

# Set model parameters
```
See the `examples/` directory for scripts reproducing the figures in the paper.

## Citation

If you use this package, please cite:

```bibtex
@article{Keselman2025,
  title   = {$J_1$-$J_2$ Triangular Lattice Antiferromagnet in a Magnetic Field},
  author  = {Keselman, Anna and Xu, Xinyuan and Zhang, Hao and Batista, Cristian D. and Starykh, Oleg A.},
  journal = {arXiv:2512.02150},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.02150}
}
```

## License

MIT