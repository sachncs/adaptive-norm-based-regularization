# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Open source community files (CODE_OF_CONDUCT, SECURITY, CONTRIBUTING expansion)
- GitHub issue and pull request templates
- Dependabot configuration for automated dependency updates
- EditorConfig for consistent formatting across editors
- GitAttributes for line ending normalization
- Getting started and FAQ documentation
- CHANGELOG.md following Keep a Changelog format
- Comprehensive README with badges, usage examples, and project structure
- FUNDING.yml for sponsorship configuration

### Changed

- Expanded CONTRIBUTING.md with detailed guidelines (branching, commits, PRs)
- Updated pyproject.toml with full project metadata
- Enhanced CI workflow with build verification step
- Improved .gitignore with additional patterns

## [0.1.0] - 2026-01-01

### Added

- All 6 regularizers: NoReg, Ridge, Lasso, Elastic Net, Covridge, Sparridge
- Feedforward ReLU network with manual forward/backward propagation
- Adam optimizer implemented from scratch in NumPy
- MSE and softmax cross-entropy losses with analytical derivatives
- 3 DGP generators matching paper's simulation designs (DGP1/2/3)
- UCI Energy Efficiency data loader
- GSE9476 leukemia data loader
- k-fold cross-validation with hyperparameter grid search
- Evaluation metrics: MSE, MAE, RMSE, R², balanced accuracy
- Training loop with mini-batching and early stopping
- Demo scripts for simulations (`run_simulation.py`) and real data (`run_real_data.py`)
- 81 unit and integration tests (all passing)
- Full documentation: math foundations, architecture, API reference, usage guide
- Paper-to-code fidelity report
- GitHub Actions CI with multi-Python-version matrix testing (3.10-3.13)
- Code formatting with black and isort
- Type checking with mypy

[Unreleased]: https://github.com/sachn-cs/adaptive-norm-based-regularization/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sachn-cs/adaptive-norm-based-regularization/releases/tag/v0.1.0
