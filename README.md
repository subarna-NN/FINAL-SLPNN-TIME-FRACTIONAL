# SLPNN: Shifted Legendre Polynomial Neural Network

Code and data for:
**"Shifted Legendre Polynomial Neural Network for the Time-Fractional
Suspended Sediment Advection-Diffusion Equation"**

## Overview
This repository implements the SLPNN — a physics-informed polynomial
neural network using shifted Legendre basis functions and exact
initial-condition enforcement; for the time-fractional suspended
sediment advection-diffusion equation of Kumar et al. (2025), and
reproduces all results in the manuscript.

## Requirements
- Python 3.9+
- numpy, scipy, matplotlib, mpmath

Install with:
```bash
pip install numpy scipy matplotlib mpmath
```

## Repository Structure
| File | Description |
| `baseline.py` | Experiment 1: SLPNN vs semi-analytical reference (α = 0.75) |
| `alpha_effect.py` | Experiment 2: Effect of fractional order, α ∈ {0.50, 0.75, 0.90} |
| `comparison.py` | Experiment 3: Three-way comparison — SLPNN vs Chopra FNN vs MLP-PINN |

## Usage
Each script is self-contained and reproduces the corresponding figures
and tables in the manuscript:
```bash
python baseline.py
python alpha_effect.py
python comparison.py
```


## Citation
If you use this code, please cite.
