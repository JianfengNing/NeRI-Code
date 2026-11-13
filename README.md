# NeRI

This repository contains the numerical experiments for **Neural Residual Iteration for Elliptic Operator Equations and Variational Inequalities**, including NeRI and the comparison methods.

## Repository Structure

| Directory | Experiment |
| --- | --- |
| `6.1_Corner_singular` | Corner singularity |
| `6.2_Discontinuous_Coffe` | Discontinuous coefficients |
| `6.3_P_Laplace` | p-Laplace equation |
| `6.4_Biharmonic` | Biharmonic equation |
| `6.5_Viariational Inequalities` | Variational inequalities |

Each directory contains `methods.py` for training or numerical solution, `create_figure.py` for plotting saved results, and a `data/` folder for data and outputs.

## Requirements

Use Python 3.10 or later and install the dependencies:

```bash
pip install torch numpy scipy matplotlib
```

## Usage

For the corner singularity example, run the following from the repository root:

```bash
cd 6.1_Corner_singular
python methods.py --method neri
python create_figure.py
```

Run other experiments in their respective directories in the same way. Use `python methods.py --help` to view available methods. Training settings are defined in `Parameters` in `methods.py`. Training may overwrite existing results with the same filenames in `data/`.
