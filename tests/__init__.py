"""Test suite for the ``anbr`` package.

Contains unit tests for individual modules (network, regularizers, losses,
optimizer, metrics, data generators) and integration tests that exercise
the full train-evaluate pipeline through :class:`~anbr.trainer.Trainer`
and :func:`~anbr.cv.grid_search_cv`.

Test layout
-----------
* :mod:`tests.test_network` -- forward / backward shapes, single-layer
  and multi-output behaviour, gradient agreement with central finite
  differences.
* :mod:`tests.test_regularizers` -- penalty/gradient correctness for
  every regularizer, parameter-validation errors, special-case
  reductions (Elastic-Net at ``alpha in {0, 1}``, identity
  Covridge/Sparridge).
* :mod:`tests.test_losses` -- known-value forward scores and
  gradients, numerical stability for the cross-entropy softmax.
* :mod:`tests.test_optimizer` -- Adam updates, bias correction,
  reset, multi-group parameter handling.
* :mod:`tests.test_metrics` -- MSE / MAE / RMSE / R-squared /
  balanced accuracy including constant-target and class-imbalance
  edge cases.
* :mod:`tests.test_data` -- DGP shapes, equi-correlation structure,
  zero-noise determinism, seed independence.
* :mod:`tests.test_integration` -- end-to-end training for
  regression and classification, every regularizer invoked,
  early stopping and cross-validation.

Running
-------
``pytest`` from the repository root discovers all 81 tests
automatically; coverage is reported via ``pytest --cov=anbr``.
"""
