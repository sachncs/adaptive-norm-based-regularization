# Adaptive Norm-Based Regularization for Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/sachn-cs/adaptive-norm-based-regularization/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/adaptive-norm-based-regularization/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A faithful end-to-end **pure Python reproduction** of the arXiv paper
**"Adaptive Norm-Based Regularization for Neural Networks"** by Muhammad Qasim
and Farrukh Javed (Lund University), implemented entirely with NumPy.

## Features

- **All 6 regularizers** from the paper: NoReg, Ridge, Lasso, Elastic Net, Covridge, Sparridge
- **Pure NumPy implementation** — no deep learning frameworks required
- **Manual backpropagation** — feedforward ReLU network with explicit forward/backward passes
- **Adam optimizer** from scratch
- **MSE and softmax cross-entropy** losses with analytical derivatives
- **3 DGP generators** matching the paper's simulation designs (DGP1/2/3)
- **Real-data loaders** for UCI Energy Efficiency and GSE9476 leukemia datasets
- **k-fold cross-validation** with the paper's hyperparameter grids
- **Evaluation metrics**: MSE, MAE, RMSE, R², balanced accuracy
- **End-to-end demo scripts** for simulations and real-data experiments
- **Comprehensive test suite** — 81 unit and integration tests

## Installation

```bash
# Clone the repository
git clone https://github.com/sachn-cs/adaptive-norm-based-regularization.git
cd adaptive-norm-based-regularization

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install code quality tools
pip install black isort mypy
```

## Usage

### Run a single simulation replicate

```bash
python demo/run_simulation.py
```

### Run full simulation (100 replications)

```bash
python demo/run_simulation.py --full
```

### Run real-data experiments

```bash
python demo/run_real_data.py
```

### Programmatic usage

```python
import numpy as np
from anbr.network import Network
from anbr.losses import MSELoss
from anbr.optimizer import Adam
from anbr.trainer import Trainer
from anbr.regularizers import Covridge

# Generate data
X = np.random.randn(200, 10)
y = X @ np.random.randn(10, 1) + 0.1 * np.random.randn(200, 1)

# Build model
layer_sizes = [10, 64, 32, 1]
activations = ["relu", "relu", "linear"]
net = Network(layer_sizes, activations)

# Configure training
loss_fn = MSELoss()
optimizer = Adam(net.parameters(), lr=1e-3)
reg = Covridge(lam1=0.01, lam2=0.001)

# Train
trainer = Trainer(net, loss_fn, optimizer, regularizer=reg)
history = trainer.fit(X, y, epochs=500, batch_size=32)

# Evaluate
predictions = net.forward(X[:20])
```

## Configuration

This project does not require environment variables or configuration files for
standard usage. All hyperparameters are passed programmatically or via CLI
arguments to the demo scripts.

### CLI Options

| Script | Flag | Description |
|--------|------|-------------|
| `run_simulation.py` | `--full` | Run 100 replications (default: 1) |
| `run_real_data.py` | `--full` | Run full cross-validation (default: quick mode) |

## Project Structure

```
adaptive-norm-based-regularization/
├── anbr/                          # Core library
│   ├── __init__.py                # Package initialization
│   ├── regularizers.py            # Ridge, Lasso, ElasticNet, Covridge, Sparridge
│   ├── losses.py                  # MSE and softmax cross-entropy
│   ├── network.py                 # Feedforward ReLU network with manual backprop
│   ├── optimizer.py               # NumPy Adam optimizer
│   ├── data.py                    # DGP generators and real-data loaders
│   ├── metrics.py                 # MSE, MAE, RMSE, R², balanced accuracy
│   ├── trainer.py                 # Training loop with early stopping
│   └── cv.py                      # k-fold CV and hyperparameter grid search
├── tests/                         # Test suite (81 tests)
│   ├── test_regularizers.py
│   ├── test_network.py
│   ├── test_optimizer.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_integration.py
├── demo/                          # Demo and experiment scripts
│   ├── run_simulation.py          # Replicates Tables 1-3 from the paper
│   └── run_real_data.py           # UCI Energy + GSE9476 experiments
├── docs/                          # Documentation
│   ├── math.md                    # Mathematical foundations
│   ├── architecture.md            # System architecture
│   ├── api.md                     # Complete API reference
│   ├── usage.md                   # Setup and usage guide
│   ├── fidelity.md                # Paper-to-code fidelity report
│   ├── getting-started.md         # Quick start guide
│   └── faq.md                     # Frequently asked questions
└── pyproject.toml                 # Build configuration and dependencies
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"
pip install black isort mypy

# Run tests
pytest tests/ -v

# Format code
black .
isort .

# Check formatting (CI mode)
black --check --diff .
isort --check-only --diff .

# Type check
mypy anbr tests demo
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Core computation | NumPy |
| Eigendecomposition | SciPy |
| ML utilities | scikit-learn |
| Data loading | pandas |
| Testing | pytest |
| Formatting | black, isort |
| Type checking | mypy |
| CI/CD | GitHub Actions |

## Roadmap

- [ ] Add sparse-matrix optimization for high-dimensional `C_{δ,n}`
- [ ] Implement optional learning-rate decay schedules
- [ ] Add PyTorch reference implementation for gradient verification
- [ ] Expand real-data benchmarks (additional UCI datasets)
- [ ] Add comprehensive benchmarks against scikit-learn regularizers
- [ ] Support for custom loss functions
- [ ] GPU acceleration via CuPy (optional)
- [ ] Published package on PyPI

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on how to contribute to this project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

- Muhammad Qasim and Farrukh Javed (Lund University) for the original paper
- The authors of the NumPy, SciPy, scikit-learn, and pandas projects
