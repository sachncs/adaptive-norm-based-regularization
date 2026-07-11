# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive docstrings across all modules, classes, and public methods
  with Args/Returns/Raises sections, edge-case documentation, and
  numerical stability notes (`e07df44` — 2026-07-11)
- Rebuilt README with centered header, Quick Start section, Code Style,
  and Commit Conventions subsections (`ecb68ad` — 2026-07-11)

### Changed

- Removed unused imports: `Tuple` from network, `Any`/`Tuple`/`StandardScaler`
  from trainer, `Callable` from cv, `StandardScaler` from run_real_data,
  `pytest` from test_network, `build_regularizer` from test_integration
  (`468d98b` — 2026-07-11)
- Applied black formatting to test files for style consistency
  (`2b8316d` — 2026-07-11)
- Updated CHANGELOG with commit IDs and dates
  (`9e973b8` — 2026-07-11)

## [0.1.1] - 2026-06-19

### Added

- Open source community files: CODE_OF_CONDUCT.md, SECURITY.md
  (`5fd67e7`..`e519252`)
- Expanded CONTRIBUTING.md with detailed guidelines (branching, commits, PRs)
  (`5fd67e7`..`e519252`)
- GitHub issue templates (bug report, feature request) and pull request
  template (`5fd67e7`..`e519252`)
- Dependabot configuration for automated GitHub Actions dependency updates
  (`5fd67e7`..`e519252`)
- EditorConfig for consistent formatting across editors
  (`5fd67e7`..`e519252`)
- GitAttributes for line-ending normalization
  (`5fd67e7`..`e519252`)
- Getting started guide (`docs/getting-started.md`) and FAQ (`docs/faq.md`)
  (`5fd67e7`..`e519252`)
- CHANGELOG.md following Keep a Changelog format
  (`5fd67e7`..`e519252`)
- Comprehensive README with badges, usage examples, and project structure
  (`5fd67e7`..`e519252`)
- FUNDING.yml for sponsorship configuration
  (`5fd67e7`..`e519252`)
- `.env.example` with documented environment variables
  (`5fd67e7`..`e519252`)

### Changed

- Updated pyproject.toml with full project metadata
  (`5fd67e7`..`e519252`)
- Enhanced CI workflow with build verification step
  (`5fd67e7`..`e519252`)
- Improved .gitignore with additional patterns
  (`5fd67e7`..`e519252`)

### CI

- Bump `actions/checkout` from v4 to v7
  (`8929311` — 2026-06-19, merged `0e07d96` — 2026-07-07)
- Bump `actions/setup-python` from v5 to v6
  (`04336c8` — 2026-06-19, merged `46484b8` — 2026-07-07)

## [0.1.0] - 2026-05-06

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
- Demo scripts for simulations (`run_simulation.py`) and real data
  (`run_real_data.py`)
- 81 unit and integration tests (all passing)
- Full documentation: math foundations, architecture, API reference,
  usage guide
- Paper-to-code fidelity report
- GitHub Actions CI with multi-Python-version matrix testing (3.10–3.13)
- Code formatting with black and isort
- Type checking with mypy

[Unreleased]: https://github.com/sachncs/adaptive-norm-based-regularization/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sachncs/adaptive-norm-based-regularization/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachncs/adaptive-norm-based-regularization/releases/tag/v0.1.0
