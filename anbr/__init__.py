"""Adaptive Norm-Based Regularization for Neural Networks.

Pure Python (NumPy-only) reproduction of the empirical methodology from
Qasim & Javed (Lund University).  Designed for *transparency* and
*reproducibility* rather than throughput: every gradient flows through
hand-written back-propagation, every regularizer exposes analytical
penalties and gradients, and there are no hidden deep-learning framework
defaults.

Quick start
-----------
>>> from anbr.network import FullyConnectedNetwork
>>> from anbr.losses import MSELoss
>>> from anbr.regularizers import Ridge
>>> from anbr.optimizer import Adam
>>> from anbr.trainer import Trainer
>>> net = FullyConnectedNetwork([10, 32, 1])
>>> trainer = Trainer(net, MSELoss(), Ridge(0.01), Adam())
>>> trainer.fit(X_train, y_train)
>>> preds = trainer.predict(X_test)

Modules
-------
regularizers
    Abstract :class:`~anbr.regularizers.Regularizer` and concrete penalties
    (Ridge, Lasso, ElasticNet, Covridge, Sparridge, plus a no-op baseline).
    Covridge and Sparridge use a precomputed matrix square root of the
    empirical Gram matrix ``C_{delta,n}`` via symmetric eigendecomposition.
losses
    :class:`~anbr.losses.MSELoss` and :class:`~anbr.losses.CrossEntropyLoss`
    with analytical backward passes.  The ``1 / n_samples`` scaling lives
    here, not in the network backward.
network
    :class:`~anbr.network.FullyConnectedNetwork` -- a feedforward ReLU MLP
    with Xavier-uniform initialization, manual forward/backward, and
    parameter snapshots for early-stopping restore.
optimizer
    :class:`~anbr.optimizer.Adam` with per-parameter-group first and second
    moment estimates, bias-corrected step, and an explicit :meth:`reset`.
data
    Synthetic data generators (``make_dgp``, ``make_dgp1/2/3``) and real-data
    loaders for the UCI Energy Efficiency and GSE9476 leukemia datasets.
metrics
    Pure-NumPy regression and classification metrics (MSE, MAE, RMSE,
    R-squared, balanced accuracy).  No sklearn dependency at runtime.
trainer
    :class:`~anbr.trainer.Trainer` -- mini-batch epoch loop that stitches
    the network, loss, regularizer, and optimizer together; supports
    validation, history tracking, and early stopping.
cv
    :func:`~anbr.cv.build_regularizer` factory and
    :func:`~anbr.cv.grid_search_cv` driver for k-fold cross-validation
    over hyperparameter grids.

Design rationale
----------------
* **Pure NumPy.** Removes framework-induced ambiguity and lets every weight
  update be inspected.
* **Layer-wise Covridge/Sparridge.** The empirical Gram matrix
  ``C_{delta,n}`` is defined over the input dimension; in a multi-layer
  network it therefore only matches the first weight matrix.  Covridge and
  Sparridge are applied only to ``W^{(1)}`` -- see :mod:`anbr.trainer`
  for the exact branch.
* **Loss-side scaling.** Gradients are summed across samples in the
  network backward and divided by ``n`` in the loss backward.  This
  matches standard deep-learning convention and keeps the optimizer
  step transparent.

References
----------
Qasim, M. & Javed, F. (Lund University).  "Adaptive Norm-Based
Regularization for Neural Networks."  See ``docs/math.md`` for
equation-by-equation mappings to the implementation.
"""
