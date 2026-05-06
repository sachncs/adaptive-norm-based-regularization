# Design: Adaptive Norm-Based Regularization for Neural Networks — Pure Python Reproduction

## Overview

Faithful end-to-end reproduction of the arXiv paper "Adaptive Norm-Based Regularization for Neural Networks" (Qasim & Javed) in pure Python with NumPy. The reproduction will implement the proposed Covridge and Sparridge penalties, a feedforward ReLU network with manual backpropagation, the Adam optimizer, simulation DGPs, real-data experiments, tests, and a demo CLI.

## Architecture

```
anbr/
├── anbr/
│   ├── __init__.py
│   ├── network.py          # Feedforward ReLU network, forward/backward, parameter updates
│   ├── regularizers.py     # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── optimizer.py        # Adam optimizer (pure NumPy)
│   ├── losses.py           # MSE and cross-entropy with manual backprop derivatives
│   ├── trainer.py          # Training loop, mini-batching, early stopping
│   ├── data.py             # DGP generators (DGP1/2/3) and real-data loaders
│   ├── metrics.py          # MSE, MAE, RMSE, R², balanced accuracy
│   └── cv.py               # k-fold cross-validation and hyperparameter grid search
├── tests/
│   ├── __init__.py
│   ├── test_regularizers.py
│   ├── test_network.py
│   ├── test_optimizer.py
│   ├── test_data.py
│   └── test_integration.py
├── demo/
│   ├── run_simulation.py   # End-to-end simulation replication
│   └── run_real_data.py  # UCI energy + leukemia classification
├── setup.py / pyproject.toml
└── README.md
```

## Components

### 1. Network (`network.py`)
- Manual fully connected layers with ReLU activation.
- Xavier/He-style initialization (exact scheme not in paper; will use standard Xavier and document as assumption).
- Forward pass caches pre-activations for backprop.
- Backprop computes gradients of loss w.r.t. each weight matrix and bias vector.
- Regularizer gradients are added to parameter gradients before optimizer step.

### 2. Regularizers (`regularizers.py`)
- **Ridge**: `λ ‖W‖_F²`
- **Lasso**: `γ ‖W‖_1`
- **Elastic Net**: `α γ ‖W‖_1 + (1-α)/2 ‖W‖_F²`
- **Covridge**: `λ₁ ‖C_{δ,n}^{1/2} W‖_F² + λ₂ ‖W‖_F²`
- **Sparridge**: `λ₁ ‖C_{δ,n}^{1/2} W‖_F² + γ ‖W‖_1`
- `C_{δ,n}` is computed from a representation matrix `H` (by default the design matrix `X` itself, as implied by the paper) as `(1/n) HᵀH + δ I_p`.
- Each regularizer exposes `penalty(weights)` and `gradient(weights)`.

### 3. Optimizer (`optimizer.py`)
- Pure NumPy Adam with first and second moment estimates.
- Default hyperparameters: `β₁=0.9`, `β₂=0.999`, `ε=1e-8` (standard defaults; paper says "Adam default settings" without specifying).
- Learning rate is a tunable hyperparameter; default `1e-3` if not specified by paper.

### 4. Losses (`losses.py`)
- **MSE** for regression: `1/n Σ (y - ŷ)²` with gradient `2/n (ŷ - y)`.
- **Cross-entropy** for classification (softmax + negative log-likelihood) with gradient `(ŷ - y_onehot)`.

### 5. Trainer (`trainer.py`)
- Epoch-based training loop with mini-batch sampling.
- Early stopping with patience (used for classification; patience=10).
- Feature standardization on training set only; statistics applied to validation/test.
- Training/validation split (75/25 for simulations).

### 6. Data (`data.py`)
- **DGP1**: `n=200, p=20, k=10`
- **DGP2**: `n=1000, p=200, k=100`
- **DGP3**: `n=500, p=2000, k=100`
- Informative predictors drawn from multivariate normal with covariance `Σ` (diagonal 1, off-diagonal `ρ`).
- Coefficients `θ_j⋆ ~ N(0, τ²)` for `j ≤ k`.
- Linear response: `y = Xθ + ε`.
- Nonlinear response: `y = Σ θ_j sin(x_j) + ε`.
- Noise `ε ~ N(0, σ²)`.
- **UCI Energy**: fetch UCI "Energy efficiency" dataset, use cooling load as target.
- **GSE9476**: fetch from GEO or provide instructions; ANOVA feature selection to 2000 genes; 5-class leukemia labels.

### 7. Metrics (`metrics.py`)
- MSE, MAE, RMSE, R² for regression.
- Balanced accuracy for classification.

### 8. Cross-Validation (`cv.py`)
- k-fold CV (default k=5).
- Hyperparameter grid search over `{0.001, 0.01, 0.1, 0.5, 0.9}` for simulation; `[0.0001, 1.0]` for high-dimensional classification.
- Grid evaluates all combinations for two-parameter methods (Elastic Net, Covridge, Sparridge).

## Data Flow

1. **Data generation/loading** → standardized train/validation/test splits.
2. **Gram matrix computation** → `C_{δ,n}` from training features (`H = X_train`), with default `δ = 1e-4` (not specified in paper; assumption).
3. **Model construction** → instantiate `Network` with layer sizes and `Regularizer`.
4. **Training loop** → for each epoch, sample mini-batches, compute forward pass, loss + regularizer, backward pass, Adam step.
5. **Evaluation** → compute metrics on held-out test set.
6. **CV** → repeat 3–5 for each hyperparameter combo, select best, retrain on full training set, evaluate on test set.

## Error Handling & Defensive Checks
- Type hints on all public functions.
- Shape assertions in forward/backward passes.
- Validation of hyperparameter grids (non-negative).
- Graceful fallback if real-data download fails.

## Testing Strategy
- **Unit tests**: regularizer values and gradients against analytical formulas.
- **Shape tests**: forward/backward output shapes match expected dimensions.
- **Gradient tests**: finite-difference checks on network gradients.
- **Integration tests**: train a tiny network for a few steps and verify loss decreases.
- **Data tests**: DGP generators produce correct shapes and covariance structure.

## Fidelity Notes
- The paper does not specify initialization scheme, learning rate, or `δ` value. These will be assumptions clearly labeled in code and docs.
- The original paper used TensorFlow. We replace TF with manual NumPy backprop, preserving the algorithmic intent.
- Real-data experiments depend on external datasets. Download/fetch scripts will be provided; if datasets are unavailable, synthetic substitutes will be noted.
- The theoretical sections (Theorems 5.1, 5.2) are not implemented; only the empirical algorithm is reproduced.

## Extensions (isolated)
- Optional PyTorch reference implementation for verification.
- Optional learning-rate decay.
- Optional vectorized batching improvements.
- Optional sparse matrix support for very large `p`.

## Tech Stack
- Python 3.10+
- `numpy` (core computation)
- `scipy` (eigendecomposition for `C^{1/2}`)
- `scikit-learn` (metrics, StandardScaler, ANOVA F-test, dataset helpers)
- `pandas` (real-data loading)
- `pytest` (testing)
