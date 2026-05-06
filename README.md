# Adaptive Norm-Based Regularization for Neural Networks — Pure Python Reproduction

This repository contains a faithful end-to-end reproduction of the arXiv paper
**"Adaptive Norm-Based Regularization for Neural Networks"** by Muhammad Qasim
and Farrukh Javed (Lund University), implemented entirely in Python with NumPy.

## Reproduction Summary

The paper proposes two data-adaptive regularizers for feedforward neural
networks:

- **Covridge**: a covariance-weighted ridge penalty
  `λ₁‖C_{δ,n}^{1/2} W‖_F² + λ₂‖W‖_F²`.
- **Sparridge**: the same covariance-weighted quadratic term plus an ℓ₁ sparsity
  penalty `λ₁‖C_{δ,n}^{1/2} W‖_F² + γ‖W‖_1`.

Both penalties adapt shrinkage to the empirical geometry of the inputs,
outperforming standard ridge, lasso, and elastic net in correlated or
high-dimensional settings.

### What was implemented

- All regularizers (No Reg, Ridge, Lasso, Elastic Net, Covridge, Sparridge).
- Feedforward ReLU network with manual forward/backward propagation.
- Adam optimizer from scratch (NumPy).
- MSE and softmax cross-entropy losses with analytical derivatives.
- DGP generators matching the paper’s three simulation designs (DGP1/2/3).
- UCI Energy Efficiency and GSE9476 leukemia data loaders.
- k-fold cross-validation with the paper’s hyperparameter grids.
- Evaluation metrics: MSE, MAE, RMSE, R², balanced accuracy.
- End-to-end demo scripts for simulations and real-data experiments.
- Comprehensive unit and integration tests (81 tests, all passing).
- Full documentation in `docs/` covering mathematics, architecture, API, usage, and fidelity.

### What remains uncertain / assumed

- **Initialization**: Paper does not specify weight initialization. We use
  Xavier uniform (standard practice) and label it as an assumption.
- **Learning rate**: Paper says "Adam default settings" but does not state the
  learning rate. We use `1e-3` (TensorFlow/Keras default) and label it as an
  assumption.
- **δ stabilization**: Paper defines `C_{δ,n} = C_n + δ I` but does not specify
  `δ`. We use `δ = 1e-4` and label it as an assumption.
- **GSE9476**: The paper does not provide a public download URL. We attempt to
  fetch a surrogate from OpenML; if unavailable, the demo prints instructions.
- **Theoretical sections (Theorems 5.1, 5.2)**: not implemented; only the
  empirical training algorithm is reproduced.

## Fidelity Report

| Paper component | Status | Notes |
|---|---|---|
| Covridge penalty definition | **Exact** | Equation 3.2.1 reproduced verbatim |
| Sparridge penalty definition | **Exact** | Equation 3.2.2 reproduced verbatim |
| Gram matrix `C_{δ,n}` | **Exact** | Computed as `(1/n) HᵀH + δ I_p` |
| Network architecture (64/32) | **Exact** | Two hidden layers, ReLU, linear output |
| Network architecture (8/4) | **Exact** | Two hidden layers, ReLU, softmax output |
| Adam optimizer | **Algorithm exact** | Default hyperparameters assumed |
| MSE loss | **Exact** | Matches paper’s regression objective |
| Cross-entropy loss | **Exact** | Softmax + NLL for classification |
| DGP1/2/3 designs | **Exact** | `n, p, k`, correlation, noise, linear/nonlinear |
| 75/25 train/test split | **Exact** | Used in simulations |
| Hyperparameter grid `{0.001,0.01,0.1,0.5,0.9}` | **Exact** | Matches paper |
| Batch size 32 (regression) | **Exact** | Matches paper |
| Batch size 16 (classification) | **Exact** | Matches paper |
| 500 epochs | **Exact** | Matches paper |
| Early stopping (patience=10) | **Exact** | Used for classification |
| Feature standardization | **Exact** | Training stats only |
| Evaluation metrics | **Exact** | MSE, MAE, RMSE, R², balanced accuracy |
| UCI Energy experiment | **Approximate** | Dataset fetched from OpenML; should match |
| GSE9476 experiment | **Approximate** | OpenML surrogate; may differ slightly |

## Setup

Install dependencies in a clean environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Training

### Quick integration test

```bash
pytest tests/test_integration.py -v
```

### Run a single simulation replicate

```bash
cd demo
python run_simulation.py
```

Use `--full` for 100 replications (slow):

```bash
python run_simulation.py --full
```

### Run real-data experiments

```bash
cd demo
python run_real_data.py
```

## Evaluation

Tests cover:
- Regularizer penalties and gradients against analytical formulas.
- Network forward/backward shapes.
- Finite-difference gradient checks.
- End-to-end training on tiny problems (loss must decrease).
- Cross-validation grid search execution.

Run all tests:

```bash
pytest tests/ -v
```

## Project Structure

```
anbr/
├── anbr/
│   ├── regularizers.py     # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── losses.py           # MSE and softmax cross-entropy
│   ├── network.py          # Manual feedforward ReLU network
│   ├── optimizer.py        # NumPy Adam
│   ├── data.py             # DGPs and real-data loaders
│   ├── metrics.py          # MSE, MAE, RMSE, R², balanced accuracy
│   ├── trainer.py          # Training loop with early stopping
│   └── cv.py               # k-fold CV and hyperparameter grids
tests/
├── test_regularizers.py
├── test_network.py
├── test_optimizer.py
├── test_losses.py
├── test_metrics.py
├── test_data.py
└── test_integration.py
docs/
├── math.md                 # Mathematical foundations and derivations
├── architecture.md         # System architecture and design decisions
├── api.md                  # Complete API reference
├── usage.md                # Setup, training, evaluation guide
└── fidelity.md             # Paper-to-code fidelity report
demo/
├── run_simulation.py
└── run_real_data.py
```

## Extensions (not part of the paper)

The following are isolated in the codebase and do not alter baseline behavior:

- Optional PyTorch reference implementation for gradient verification.
- Optional learning-rate decay schedule.
- Optional vectorized batching for large-scale speedups.
- Optional sparse-matrix `C_{δ,n}` for very high-dimensional inputs.

These are not implemented by default to preserve fidelity to the original paper.
