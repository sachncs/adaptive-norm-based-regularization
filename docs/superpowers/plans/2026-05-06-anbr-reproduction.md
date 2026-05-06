# ANBR Pure Python Reproduction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully reproduce the paper "Adaptive Norm-Based Regularization for Neural Networks" in pure Python (NumPy), including all regularizers, a manual feedforward ReLU network, Adam optimizer, DGP generators, real-data experiments, tests, and demos.

**Architecture:** Modular design with separate files for regularizers, network layers, losses, optimizer, data generation, metrics, cross-validation, and training loops. All core computation uses NumPy with manual backpropagation.

**Tech Stack:** Python 3.10+, numpy, scipy, scikit-learn, pandas, pytest.

---

## File Structure

```
anbr/
├── anbr/
│   ├── __init__.py
│   ├── regularizers.py     # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── losses.py           # MSE and softmax cross-entropy with derivatives
│   ├── network.py          # Feedforward ReLU network, forward/backward
│   ├── optimizer.py        # NumPy Adam
│   ├── data.py             # DGP1/2/3 and real-data loaders
│   ├── metrics.py          # MSE, MAE, RMSE, R², balanced accuracy
│   ├── trainer.py          # Epoch loop, early stopping, standardization
│   └── cv.py               # k-fold CV and hyperparameter grid search
tests/
├── __init__.py
├── test_regularizers.py
├── test_network.py
├── test_optimizer.py
├── test_data.py
└── test_integration.py
demo/
├── run_simulation.py
└── run_real_data.py
pyproject.toml
README.md
```

---

### Task 1: Core Regularizers

**Files:**
- Create: `anbr/regularizers.py`
- Test: `tests/test_regularizers.py`

- [ ] **Step 1: Write failing tests** for each regularizer verifying analytical penalty values and gradient formulas.
- [ ] **Step 2: Run tests — expect failures (module not found).**
- [ ] **Step 3: Implement `BaseRegularizer`, `Ridge`, `Lasso`, `ElasticNet`, `Covridge`, `Sparridge`.**
  - `Covridge`: `lambda_1 * ||C_delta_n^{1/2} W||_F^2 + lambda_2 * ||W||_F^2`
  - `Sparridge`: `lambda_1 * ||C_delta_n^{1/2} W||_F^2 + gamma * ||W||_1`
  - `C_delta_n` computed via `scipy.linalg.sqrtm` or eigendecomposition.
- [ ] **Step 4: Run tests — expect pass.**
- [ ] **Step 5: Commit.**

---

### Task 2: Loss Functions

**Files:**
- Create: `anbr/losses.py`
- Test: `tests/test_losses.py` (implicit in integration)

- [ ] **Step 1: Implement `MSELoss` and `CrossEntropyLoss` with `forward(y_pred, y_true)` and `backward(y_pred, y_true)`.**
- [ ] **Step 2: Write shape and value tests.**
- [ ] **Step 3: Run tests.**
- [ ] **Step 4: Commit.**

---

### Task 3: Network Forward/Backward

**Files:**
- Create: `anbr/network.py`
- Test: `tests/test_network.py`

- [ ] **Step 1: Write failing tests for forward pass shape and backward gradient finite-difference check.**
- [ ] **Step 2: Implement `FullyConnectedNetwork` with Xavier uniform initialization.**
  - `forward(x)` returns output and caches pre-activations.
  - `backward(dloss_dy)` returns gradient dict for all weights/biases.
- [ ] **Step 3: Run shape tests and finite-difference gradient tests.**
- [ ] **Step 4: Commit.**

---

### Task 4: Adam Optimizer

**Files:**
- Create: `anbr/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test for Adam step updating parameters.**
- [ ] **Step 2: Implement `Adam` with `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.**
- [ ] **Step 3: Run tests.**
- [ ] **Step 4: Commit.**

---

### Task 5: Data Generators and Loaders

**Files:**
- Create: `anbr/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing tests for DGP shape, covariance structure, and label generation.**
- [ ] **Step 2: Implement `make_dgp1/2/3` with linear and nonlinear variants.**
  - Informative block covariance: `rho` on off-diagonals.
  - Coefficients `~ N(0, tau^2)`.
  - Noise `~ N(0, sigma^2)`.
- [ ] **Step 3: Implement UCI Energy loader (`load_energy_data`).**
- [ ] **Step 4: Implement GSE9476 loader (`load_leukemia_data`) with ANOVA selection.**
- [ ] **Step 5: Run tests.**
- [ ] **Step 6: Commit.**

---

### Task 6: Metrics

**Files:**
- Create: `anbr/metrics.py`
- Test: `tests/test_metrics.py` (inline)

- [ ] **Step 1: Implement `mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`, `r2_score`, `balanced_accuracy_score`.**
- [ ] **Step 2: Write tests comparing against sklearn equivalents.**
- [ ] **Step 3: Run tests.**
- [ ] **Step 4: Commit.**

---

### Task 7: Trainer

**Files:**
- Create: `anbr/trainer.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Implement `Trainer` class.**
  - `fit(X_train, y_train, X_val, y_val)` with mini-batching.
  - Feature standardization using training stats.
  - Early stopping with patience.
  - History tracking (train/val loss per epoch).
- [ ] **Step 2: Write integration test: train a tiny network for 10 epochs and assert loss decreases.**
- [ ] **Step 3: Run tests.**
- [ ] **Step 4: Commit.**

---

### Task 8: Cross-Validation

**Files:**
- Create: `anbr/cv.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Implement `GridSearchCV` for neural-network hyperparameters.**
  - k-fold split (default 5).
  - Grid over `{0.001, 0.01, 0.1, 0.5, 0.9}` for simulations.
  - For two-parameter methods, evaluate all combinations.
  - `C_delta_n` computed from full training fold (not mini-batch).
- [ ] **Step 2: Write integration test on DGP1 with tiny grid.**
- [ ] **Step 3: Run tests.**
- [ ] **Step 4: Commit.**

---

### Task 9: Demos

**Files:**
- Create: `demo/run_simulation.py`
- Create: `demo/run_real_data.py`

- [ ] **Step 1: `run_simulation.py` replicates Tables 1–3 for a reduced number of runs (e.g., 5 instead of 100) to keep runtime reasonable, with a flag for full 100 runs.**
- [ ] **Step 2: `run_real_data.py` runs UCI Energy regression and prints Table 4 metrics; runs leukemia classification and prints Table 5 balanced accuracy.**
- [ ] **Step 3: Verify scripts run without errors on small data subsets.**
- [ ] **Step 4: Commit.**

---

### Task 10: Documentation and Packaging

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`

- [ ] **Step 1: Write `pyproject.toml` with dependencies and metadata.**
- [ ] **Step 2: Write `README.md` with setup, training, evaluation, and reproduction fidelity notes.**
- [ ] **Step 3: Add fidelity report inline in README flagging exact vs assumed details.**
- [ ] **Step 4: Final lint/format check (adhere to Google Python Style Guide).**
- [ ] **Step 5: Commit.**

---

## Self-Review

- **Spec coverage:** All paper sections (regularizers, network, DGPs, real data, CV, metrics) map to tasks above.
- **Placeholder scan:** No TBDs or incomplete steps.
- **Type consistency:** Regularizer interface used consistently across Ridge/Lasso/ElasticNet/Covridge/Sparridge.
