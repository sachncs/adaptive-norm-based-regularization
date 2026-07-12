<p align="center">
  <h1 align="center">ANBR — Adaptive Norm-Based Regularization</h1>
  <p align="center">Pure Python (NumPy-only) reproduction of Qasim &amp; Javed (Lund University).</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/adaptive-norm-based-regularization/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/adaptive-norm-based-regularization/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/adaptive-norm-based-regularization/stargazers"><img src="https://img.shields.io/github/stars/sachncs/adaptive-norm-based-regularization" alt="Stars"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  </p>
</p>

**anbr** is a faithful, end-to-end reproduction of the paper *"Adaptive
Norm-Based Regularization for Neural Networks"* by Muhammad Qasim and
Farrukh Javed (Lund University). Every gradient flows through hand-written
back-propagation, every regularizer exposes analytical penalties and
gradients, and there are no hidden deep-learning framework defaults.

---

## Features

- **All 6 regularizers** from the paper — NoReg, Ridge, Lasso, Elastic Net,
  Covridge, Sparridge
- **Pure NumPy** — no deep learning frameworks required; every weight update
  is inspectable
- **Manual back-propagation** — feedforward ReLU network with explicit
  forward / backward passes
- **Adam optimizer** implemented from scratch with bias-corrected moment
  estimates
- **MSE and softmax cross-entropy** losses with analytical derivatives and
  `1/n` scaling
- **3 DGP generators** matching the paper's simulation designs (DGP1/2/3)
- **Real-data loaders** for UCI Energy Efficiency and GSE9476 Leukemia
- **k-fold cross-validation** with the paper's hyperparameter grids
- **Evaluation metrics** — MSE, MAE, RMSE, R², balanced accuracy
  (no sklearn runtime dependency)
- **End-to-end demo scripts** for simulation and real-data experiments
- **Comprehensive test suite** — 81 unit and integration tests (87% coverage)
- **Paper-to-code fidelity report** documenting equation-level reproductions

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/adaptive-norm-based-regularization.git
cd adaptive-norm-based-regularization
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.23, SciPy ≥ 1.9, scikit-learn ≥ 1.1,
pandas ≥ 1.5.

---

## Quick Start

### CLI

```bash
# Run a single simulation replication (quick mode, 5 reps)
python demo/run_simulation.py

# Run full simulation (100 replications)
python demo/run_simulation.py --full

# Run real-data experiments (UCI Energy + GSE9476 Leukemia)
python demo/run_real_data.py
```

### Python API

```python
import numpy as np
from anbr.network import FullyConnectedNetwork
from anbr.losses import MSELoss
from anbr.optimizer import Adam
from anbr.regularizers import Covridge
from anbr.trainer import Trainer

# Generate synthetic data
X = np.random.randn(200, 10)
y = X @ np.random.randn(10, 1) + 0.1 * np.random.randn(200, 1)

# Build model
net = FullyConnectedNetwork([10, 64, 32, 1])

# Configure training
loss_fn = MSELoss()
optimizer = Adam(learning_rate=1e-3)
reg = Covridge(lambda1=0.01, lambda2=0.001, c_delta_n=np.eye(10))

# Train
trainer = Trainer(net, loss_fn, reg, optimizer, batch_size=32, epochs=500)
trainer.fit(X, y)

# Predict
predictions = trainer.predict(X[:20])
```

### Hyperparameter Search

```python
from anbr.cv import build_regularizer, grid_search_cv
from anbr.data import make_dgp1
from anbr.losses import MSELoss

x, y = make_dgp1(rho=0.25, sigma_noise=0.10, random_state=0)
best_params, best_score = grid_search_cv(
    x, y,
    layer_sizes=[20, 64, 32, 1],
    method="covridge",
    param_grid=[
        {"lambda1": a, "lambda2": b}
        for a in [0.01, 0.1, 0.5]
        for b in [0.001, 0.01, 0.1]
    ],
    loss_fn=MSELoss(),
    n_splits=5,
)
```

---

## Configuration

This project does not require environment variables for standard usage.
All hyperparameters are passed programmatically or via CLI arguments to the
demo scripts.

### Network Architecture

| Parameter      | Default     | Description                              |
|----------------|-------------|------------------------------------------|
| `layer_sizes`  | `[10, 64, 32, 1]` | Layer widths including input and output |
| `batch_size`   | `32`        | Mini-batch size for training             |
| `epochs`       | `500`       | Maximum number of training epochs        |
| `early_stopping` | `False`   | Stop on validation-loss plateau         |
| `patience`     | `10`        | Epochs without improvement before stop   |

### Regularizers

| Method         | Penalty                                                 | Hyperparameters                  |
|----------------|---------------------------------------------------------|----------------------------------|
| `none`         | `0`                                                     | none                             |
| `ridge`        | `λ ‖W‖_F²`                                             | `lambda_`                        |
| `lasso`        | `γ ‖W‖_1`                                              | `gamma`                          |
| `elastic_net`  | `α γ ‖W‖_1 + (1 − α)/2 ‖W‖_F²`                          | `alpha ∈ [0,1]`, `gamma`         |
| `covridge`     | `λ₁ ‖C^{1/2} W‖_F² + λ₂ ‖W‖_F²`                         | `lambda1`, `lambda2`, `C_{δ,n}`  |
| `sparridge`    | `λ₁ ‖C^{1/2} W‖_F² + γ ‖W‖_1`                          | `lambda1`, `gamma`, `C_{δ,n}`    |

See [docs/api.md](docs/api.md) for the complete regularizer reference.

### Optimizer (Adam)

| Parameter      | Default   | Description                                  |
|----------------|-----------|----------------------------------------------|
| `learning_rate`| `1e-3`    | Step size `η`                                |
| `beta1`        | `0.9`     | Decay rate for the first moment              |
| `beta2`        | `0.999`   | Decay rate for the second moment             |
| `epsilon`      | `1e-8`    | Numerical stability term                     |

---

## Project Structure

```
adaptive-norm-based-regularization/
├── anbr/                  # Core library
│   ├── __init__.py        # Package overview and quick start
│   ├── regularizers.py    # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── losses.py          # MSE and softmax cross-entropy
│   ├── network.py         # Feedforward ReLU network with manual backprop
│   ├── optimizer.py       # NumPy Adam optimizer
│   ├── data.py            # DGP generators and real-data loaders
│   ├── metrics.py         # MSE, MAE, RMSE, R², balanced accuracy
│   ├── trainer.py         # Training loop with early stopping
│   └── cv.py              # k-fold CV and hyperparameter grid search
├── tests/                 # Test suite (81 tests)
│   ├── test_regularizers.py
│   ├── test_network.py
│   ├── test_optimizer.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_integration.py
├── demo/                  # Experiment reproduction scripts
│   ├── run_simulation.py  # Replicates Tables 1–3 from the paper
│   └── run_real_data.py   # UCI Energy + GSE9476 Leukemia experiments
├── docs/                  # Documentation
│   ├── math.md            # Mathematical foundations
│   ├── architecture.md    # System architecture
│   ├── api.md             # Complete API reference
│   ├── usage.md           # Setup and usage guide
│   ├── fidelity.md        # Paper-to-code fidelity report
│   ├── getting-started.md # Quick start guide
│   └── faq.md             # Frequently asked questions
└── pyproject.toml         # Build & tool config
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Format code
black --target-version py310 anbr tests demo
isort anbr tests demo

# Check formatting (CI mode)
black --check --target-version py310 anbr tests demo
isort --check-only anbr tests demo

# Type check
mypy anbr tests demo

# All checks
pytest && black --check --target-version py310 anbr tests demo && isort --check-only anbr tests demo && mypy anbr tests demo
```

### Code Style

- Line length: 80 (configured in `pyproject.toml`)
- Formatting: [black](https://github.com/psf/black) with `--target-version py310`
- Import sorting: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- Type hints: required on all public signatures
- Docstrings: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  with `Args`, `Returns`, `Raises` sections; explain *why*, not *what*

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Sparridge regularizer with geometry-aware penalty
fix: clamp negative eigenvalues in Covridge eigendecomposition
docs: add comprehensive docstrings across all modules
refactor: remove unused imports from trainer and cv
test: add finite-difference gradient verification
chore: update CHANGELOG with commit references
```

---

## Architecture

The implementation separates concerns into isolated, transparently-testable
modules. Each layer has a single responsibility, and every gradient flows
through hand-written code rather than an autograd engine.

### Gradient Flow

```
loss.backward  →  network.backward  →  regularizer.gradient  →  optimizer.step
      ↓                  ↓                     ↓                      ↓
 1/n scaling      raw parameter       analytic penalty        Adam update
 into gradient    gradients (sum)    → weight gradient       (bias-corrected)
```

The `1/N` scaling lives in the **loss backward** (not the network
backward), matching the standard deep-learning convention. Covridge and
Sparridge are applied **only to the first weight matrix** because
`C_{δ,n}` is defined over the input dimension.

### Mathematical Guarantees

1. **Frobenius-norm penalty is differentiable** at every weight
   configuration (squared L2 norm is smooth).
2. **Subgradients are well-defined** at the origin: `numpy.sign(0) == 0`,
   so the L1 subgradient exists at `W = 0`.
3. **Covridge / Sparridge square root is real**: `C_{δ,n}` is symmetric
   positive-semidefinite by construction, so `numpy.linalg.eigh` returns
   a real eigenbasis and the square root is taken elementwise on the
   (clipped) eigenvalues.
4. **Symmetric eigendecomposition is preferred** over `scipy.linalg.sqrtm`
   to avoid its complex-arithmetic round-trip.
5. **Numerical stability**: tiny negative eigenvalues from floating-point
   error are clamped to zero before the square root.

See [docs/architecture.md](docs/architecture.md) for the full design
rationale, [docs/math.md](docs/math.md) for equation-by-equation
mappings to the paper, and [docs/fidelity.md](docs/fidelity.md) for the
paper-to-code audit.

### Lifecycle

A `Trainer` is intended for a **single training run** on a single
network. Re-using the same trainer for a second independent optimization
is not supported; construct a fresh `Trainer` per run.

---

## Tech Stack

| Category       | Technology                                       |
|----------------|--------------------------------------------------|
| Language       | Python 3.10+ (3.10, 3.11, 3.12, 3.13 in CI)       |
| Numerical      | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Data           | [pandas](https://pandas.pydata.org/), scikit-learn datasets |
| ML Utilities   | [scikit-learn](https://scikit-learn.org/) (CV splits, scaling) |
| Build          | [setuptools](https://setuptools.pypa.io/)        |
| Lint / Format  | [black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/), [pydocstyle](https://pydocstyle.org/) |
| Type Check     | [mypy](https://mypy-lang.org/)                   |
| Testing        | [pytest](https://docs.pytest.org/)               |
| CI / CD        | [GitHub Actions](https://github.com/features/actions) |

---

## Roadmap

- **v0.1.0** — Current release: core implementation, 6 regularizers, all
  demo scripts, 81 tests
- **v0.2.0** — Sparse-matrix optimization for high-dimensional `C_{δ,n}`
- **v0.3.0** — Learning-rate decay schedules; PyTorch reference
  implementation for gradient verification
- **v1.0.0** — Additional UCI benchmarks, PyPI release, GPU acceleration
  via CuPy (optional)

---

## Paper-to-Code Fidelity

| Component                          | Status      | Notes                                  |
|------------------------------------|-------------|----------------------------------------|
| Covridge penalty                   | **Exact**   | Equation 3.2.1 reproduced verbatim     |
| Sparridge penalty                  | **Exact**   | Equation 3.2.2 reproduced verbatim     |
| Gram matrix `C_{δ,n}`              | **Exact**   | `(1/n) X^T X + δ I_p`                  |
| Network architecture (64/32)       | **Exact**   | Two hidden layers, ReLU, linear output |
| Network architecture (8/4)         | **Exact**   | Two hidden layers, ReLU, softmax output |
| Adam optimizer                     | **Exact**   | Default hyperparameters assumed        |
| DGP1/2/3 designs                   | **Exact**   | `n, p, k`, correlation, noise          |
| 75/25 train/test split             | **Exact**   | Used in simulations                    |
| Hyperparameter grid                | **Exact**   | `{0.001, 0.01, 0.1, 0.5, 0.9}`         |
| 500 epochs                         | **Exact**   | Matches paper                          |
| Early stopping (patience=10)       | **Exact**   | Used for classification                |

See [docs/fidelity.md](docs/fidelity.md) for the full fidelity report.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines on how to contribute to this project, including:

- Development setup
- Pull request process
- Coding standards (80-char line, black, Google docstrings)
- Test expectations (every public function has a test)

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to abide by its terms.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin
