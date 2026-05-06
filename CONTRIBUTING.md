# Contributing

This repository reproduces the paper "Adaptive Norm-Based Regularization for Neural Networks". All code is pure Python with NumPy.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8 with 80-character line limit.
- Use type hints on all public APIs.
- Write docstrings for all public classes and functions.
- Do not introduce placeholders, TODOs, or stubs.
