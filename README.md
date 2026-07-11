<p align="center">
  <h1 align="center">Adaptive Norm-Based Regularization for Neural Networks</h1>
  <p align="center">Pure Python (NumPy-only) reproduction of Qasim &amp; Javed (Lund University).</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachn-cs/adaptive-norm-based-regularization/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachn-cs/adaptive-norm-based-regularization/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachn-cs/adaptive-norm-based-regularization/blob/master/pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-green" alt="Version"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  </p>
</p>

A faithful end-to-end reproduction of the arXiv paper **"Adaptive Norm-Based
Regularization for Neural Networks"** by Muhammad Qasim and Farrukh Javed
(Lund University). Every gradient flows through hand-written back-propagation,
every regularizer exposes analytical penalties and gradients, and there are no
hidden deep-learning framework defaults.

---

## Features

- **All 6 regularizers** from the paper — NoReg, Ridge, Lasso, Elastic Net, Covridge, Sparridge
- **Pure NumPy** — no deep learning frameworks required; every weight update is inspectable
- **Manual back-propagation** — feedforward ReLU network with explicit forward/backward passes
- **Adam optimizer** implemented from scratch with bias-corrected moment estimates
- **MSE and softmax cross-entropy** losses with analytical derivatives and `1/n` scaling
- **3 DGP generators** matching the paper's simulation designs (DGP1/2/3)
- **Real-data loaders** for UCI Energy Efficiency and GSE9476 leukemia datasets
- **k-fold cross-validation** with the paper's hyperparameter grids
- **Evaluation metrics** — MSE, MAE, RMSE, R², balanced accuracy (no sklearn runtime dependency)
- **End-to-end demo scripts** for simulation and real-data experiments
- **Comprehensive test suite** — 81 unit and integration tests (all passing)
- **Paper-to-code fidelity report** documenting equation-level reproductions

---

## Installation

### From source

```bash
git clone https://github.com/sachn-cs/adaptive-norm-based-regularization.git
cd adaptive-norm-based-regularization
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### CLI

```bash
# Run a single simulation replication (quick mode)
python demo/run_simulation.py

# Run full simulation (100 replications)
python demo/run_simulation.py --full

# Run real-data experiments (UCI Energy + Leukemia)
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

---

## Configuration

This project does not require environment variables or configuration files for
standard usage. All hyperparameters are passed programmatically or via CLI
arguments to the demo scripts.

### CLI Options

| Script | Flag | Description |
|--------|------|-------------|
| `run_simulation.py` | `--full` | Run 100 replications (default: 5) |
| `run_real_data.py` | *(none)* | Runs both Energy and Leukemia benchmarks |

---

## Project Structure

```
adaptive-norm-based-regularization/
├── anbr/                              # Core library
│   ├── __init__.py                    # Package overview and quick start
│   ├── regularizers.py                # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── losses.py                      # MSE and softmax cross-entropy
│   ├── network.py                     # Feedforward ReLU network with manual backprop
│   ├── optimizer.py                   # NumPy Adam optimizer
│   ├── data.py                        # DGP generators and real-data loaders
│   ├── metrics.py                     # MSE, MAE, RMSE, R², balanced accuracy
│   ├── trainer.py                     # Training loop with early stopping
│   └── cv.py                          # k-fold CV and hyperparameter grid search
├── tests/                             # Test suite (81 tests)
│   ├── test_regularizers.py
│   ├── test_network.py
│   ├── test_optimizer.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_integration.py
├── demo/                              # Experiment reproduction scripts
│   ├── run_simulation.py              # Replicates Tables 1–3 from the paper
│   └── run_real_data.py               # UCI Energy + GSE9476 experiments
├── docs/                              # Documentation
│   ├── math.md                        # Mathematical foundations
│   ├── architecture.md                # System architecture
│   ├── api.md                         # Complete API reference
│   ├── usage.md                       # Setup and usage guide
│   ├── fidelity.md                    # Paper-to-code fidelity report
│   ├── getting-started.md             # Quick start guide
│   └── faq.md                         # Frequently asked questions
└── pyproject.toml                     # Build configuration and dependencies
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Format code
black --target-version py310 anbr tests demo
isort anbr tests demo

# Check formatting (CI mode)
black --check --target-version py310 anbr tests demo
isort --check-only anbr tests demo

# Lint docstrings
pydocstyle anbr tests demo

# Type check
mypy anbr tests demo

# All checks
pytest && black --check --target-version py310 anbr tests demo && isort --check-only anbr tests demo && mypy anbr tests demo
```

### Code Style

- Line length: 88 (black default)
- Formatter: [black](https://github.com/psf/black) (`--target-version py310`)
- Import sorting: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- Type hints: required on all public signatures
- Docstrings: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) with `Args`, `Returns`, `Raises` sections
- No inline comments on obvious code; comments explain *why*, not *what*

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

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Core computation | [NumPy](https://numpy.org/) |
| ML utilities | [scikit-learn](https://scikit-learn.org/) (data loading, scaling, CV splits) |
| Testing | [pytest](https://docs.pytest.org/) |
| Formatting | [black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/) |
| Type checking | [mypy](https://mypy-lang.org/) |
| Lint | [pydocstyle](https://pydocstyle.org/) |
| CI/CD | [GitHub Actions](https://github.com/features/actions) |

---

## Roadmap

- [ ] Add sparse-matrix optimization for high-dimensional `C_{δ,n}`
- [ ] Implement optional learning-rate decay schedules
- [ ] Add PyTorch reference implementation for gradient verification
- [ ] Expand real-data benchmarks (additional UCI datasets)
- [ ] Add comprehensive benchmarks against scikit-learn regularizers
- [ ] Support for custom loss functions
- [ ] GPU acceleration via CuPy (optional)
- [ ] Published package on PyPI

---

## Paper-to-Code Fidelity

| Component | Status | Notes |
|-----------|--------|-------|
| Covridge penalty | **Exact** | Equation 3.2.1 reproduced verbatim |
| Sparridge penalty | **Exact** | Equation 3.2.2 reproduced verbatim |
| Gram matrix `C_{δ,n}` | **Exact** | `(1/n) HᵀH + δ I_p` |
| Network architecture (64/32) | **Exact** | Two hidden layers, ReLU, linear output |
| Network architecture (8/4) | **Exact** | Two hidden layers, ReLU, softmax output |
| Adam optimizer | **Exact** | Default hyperparameters assumed |
| DGP1/2/3 designs | **Exact** | `n, p, k`, correlation, noise |
| 75/25 train/test split | **Exact** | Used in simulations |
| Hyperparameter grid | **Exact** | `{0.001, 0.01, 0.1, 0.5, 0.9}` |
| 500 epochs | **Exact** | Matches paper |
| Early stopping (patience=10) | **Exact** | Used for classification |

See [`docs/fidelity.md`](docs/fidelity.md) for the full fidelity report.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on how to contribute to this project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin
