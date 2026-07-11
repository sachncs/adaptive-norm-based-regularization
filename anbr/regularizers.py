"""Regularizer penalties and analytical gradients.

This module defines the regularizers used throughout the ANBR
reproduction. Each regularizer is a stateless callable object exposing
two methods:

* :meth:`Regularizer.penalty` — scalar penalty ``Ω(W)`` added to the
  empirical loss.
* :meth:`Regularizer.gradient` — analytical gradient ``∇_W Ω(W)`` of
  the penalty with respect to the weight matrix, used by
  :class:`~anbr.trainer.Trainer` during backpropagation.

The concrete penalties map directly to equations in Qasim & Javed:

==========================  ============================================
Class                       Paper equation
==========================  ============================================
:class:`Ridge`              ``λ ‖W‖_F^2``
:class:`Lasso`              ``γ ‖W‖_1``
:class:`ElasticNet`         ``α γ ‖W‖_1 + (1 - α)/2 · ‖W‖_F^2``
:class:`Covridge`           ``λ₁ ‖C_{δ,n}^{1/2} W‖_F^2 + λ₂ ‖W‖_F^2``
:class:`Sparridge`          ``λ₁ ‖C_{δ,n}^{1/2} W‖_F^2 + γ ‖W‖_1``
==========================  ============================================

Numerical stability
-------------------
:class:`Covridge` and :class:`Sparridge` factor ``C_{δ,n}`` once at
construction using a symmetric eigendecomposition (``numpy.linalg.eigh``).
The decomposition is symmetric by construction (``C_n`` and ``I`` are
symmetric and so is their sum), which is why ``eigh`` is preferred over
``scipy.linalg.sqrtm``: ``eigh`` returns real eigenvalues and avoids the
complex round-trip that ``sqrtm`` performs. Tiny negative eigenvalues
introduced by floating-point error are clamped to zero before the
square root is taken, guaranteeing a real PSD square root.

Complexity
----------
All penalty/gradient evaluations are linear in the number of weight
entries. The eigendecomposition performed once per ``Covridge``/
``Sparridge`` instance is ``O(p³)`` where ``p`` is the input dimension.
After construction, the per-step cost is identical to that of Ridge.
"""

from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    """Abstract base class for weight-matrix regularizers.

    All regularizers share the same interface so that
    :class:`~anbr.trainer.Trainer` can dispatch them uniformly. Subclasses
    must implement :meth:`penalty` and :meth:`gradient`; both are pure
    functions of the weight matrix and have no side effects.

    Thread-safety
    -------------
    Instances are immutable after construction (hyperparameters are
    stored once and only read), so they are safe to share across
    threads provided the underlying numpy arrays are not mutated by
    callers. The network weights passed to :meth:`penalty` and
    :meth:`gradient` are never copied.
    """

    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        """Compute the scalar regularization penalty.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Scalar penalty value ``Ω(W)`` to be added to the empirical
            loss.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute the gradient of the penalty w.r.t. ``weights``.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Gradient array with the same shape as ``weights``. Caller is
            responsible for accumulating this with the data-gradient from
            backpropagation before invoking the optimizer.
        """
        raise NotImplementedError


class NoRegularizer(Regularizer):
    """No-op regularizer used as the unregularized baseline.

    Always returns zero penalty and zero gradient so that
    :class:`~anbr.trainer.Trainer` can iterate over regularizer instances
    without special-casing the absence of regularization.
    """

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``0.0``.

        Args:
            weights: Ignored. Present to satisfy the abstract interface.

        Returns:
            ``0.0`` regardless of input.
        """
        return 0.0

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return a zero array shaped like ``weights``.

        Args:
            weights: Reference shape used to construct the result.

        Returns:
            Array of zeros with the same shape and dtype as ``weights``.
        """
        return np.zeros_like(weights)


class Ridge(Regularizer):
    """Ridge (``ℓ₂`` / Tikhonov) penalty: ``λ ‖W‖_F²``.

    Penalty: ``Ω(W) = λ · Σ_{i,j} w_{ij}²``.
    Gradient: ``∇_W Ω(W) = 2 λ W``.

    Equivalent to a Gaussian prior on the weights. Smooth and strongly
    convex in ``W``; encourages small magnitudes without inducing
    sparsity. Setting ``lambda_ = 0`` reduces this to
    :class:`NoRegularizer` (the penalty and gradient are still computed
    numerically rather than short-circuited).
    """

    def __init__(self, lambda_: float) -> None:
        """Store the non-negative regularization strength.

        Args:
            lambda_: Penalty weight ``λ``. Must be ``>= 0``; negative
                values are mathematically meaningless and rejected.

        Raises:
            ValueError: If ``lambda_ < 0``.
        """
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        self.lambda_ = lambda_

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``λ · ‖W‖_F²``.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Scalar Frobenius-norm-squared penalty.
        """
        return self.lambda_ * float(np.sum(weights**2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return ``2 λ W``.

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        return 2.0 * self.lambda_ * weights


class Lasso(Regularizer):
    """Lasso (``ℓ₁``) penalty: ``γ ‖W‖_1``.

    Penalty: ``Ω(W) = γ · Σ_{i,j} |w_{ij}|``.
    Subgradient: ``∇_W Ω(W) = γ · sign(W)`` with ``sign(0) := 0``.

    Promotes sparsity by driving individual weights exactly to zero.
    Non-smooth at the origin, so the implementation uses a subgradient
    rather than a true gradient.
    """

    def __init__(self, gamma: float) -> None:
        """Store the non-negative sparsity weight.

        Args:
            gamma: Penalty weight ``γ``. Must be ``>= 0``.

        Raises:
            ValueError: If ``gamma < 0``.
        """
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``γ · ‖W‖_1``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar ``ℓ₁`` penalty.
        """
        return self.gamma * float(np.sum(np.abs(weights)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return the subgradient ``γ · sign(W)``.

        ``numpy.sign`` already returns ``0`` for zero inputs, so the
        subgradient is well-defined at ``W = 0`` and requires no
        additional clipping.

        Args:
            weights: Weight matrix.

        Returns:
            Subgradient array, same shape as ``weights``.
        """
        # Subgradient: sign(0) = 0 is provided by ``numpy.sign`` directly.
        return self.gamma * np.sign(weights)


class ElasticNet(Regularizer):
    """Elastic Net: ``α γ ‖W‖_1 + (1 - α)/2 · ‖W‖_F²``.

    Convex combination of :class:`Lasso` (sparsity) and
    :class:`Ridge` (shrinkage). At the endpoints it degenerates to:

    * ``α = 0`` → ``(1/2) ‖W‖_F²`` — equivalent to :class:`Ridge` with
      ``lambda_ = 0.5``.
    * ``α = 1`` → ``γ ‖W‖_1`` — equivalent to :class:`Lasso` with the
      same ``γ``.

    Penalty: ``Ω(W) = α γ ‖W‖_1 + (1 - α)/2 · ‖W‖_F²``.
    Gradient: ``∇_W Ω(W) = α γ · sign(W) + (1 - α) · W``.
    """

    def __init__(self, alpha: float, gamma: float) -> None:
        """Store mixing parameter and penalty weight.

        Args:
            alpha: Mixing parameter in ``[0, 1]``. ``0`` reduces to a
                pure quadratic penalty, ``1`` reduces to a pure ``ℓ₁``
                penalty.
            gamma: Penalty weight ``γ``. Must be ``>= 0``.

        Raises:
            ValueError: If ``alpha ∉ [0, 1]`` or ``gamma < 0``.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        """Return the elastic-net penalty.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar ``α γ ‖W‖_1 + (1 - α)/2 · ‖W‖_F²``.
        """
        l1 = self.alpha * self.gamma * float(np.sum(np.abs(weights)))
        l2 = (1.0 - self.alpha) * 0.5 * float(np.sum(weights**2))
        return l1 + l2

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return ``α γ · sign(W) + (1 - α) · W``.

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        l1_grad = self.alpha * self.gamma * np.sign(weights)
        l2_grad = (1.0 - self.alpha) * weights
        return l1_grad + l2_grad


class Covridge(Regularizer):
    """Covridge penalty: ``λ₁ ‖C_{δ,n}^{1/2} W‖_F² + λ₂ ‖W‖_F²``.

    The first term shrinks weights along the eigenvectors of the
    empirical Gram matrix ``C_n = (1/n) X^T X`` (stabilized by
    ``δ I_p`` to form ``C_{δ,n}``); the second term is a standard
    isotropic ridge. Together they implement the geometry-aware
    shrinkage of the paper.

    Penalty: ``Ω(W) = λ₁ ‖C_{δ,n}^{1/2} W‖_F² + λ₂ ‖W‖_F²``.
    Gradient: ``∇_W Ω(W) = 2 λ₁ C_{δ,n} W + 2 λ₂ W``.

    Special cases
    -------------
    * ``C_{δ,n} = I`` reduces Covridge to
      :class:`Ridge` with ``lambda_ = λ₁ + λ₂``.
    * ``λ₂ = 0`` keeps only the geometry-aware term; ``λ₁ = 0`` keeps
      only the isotropic term.

    Construction cost
    -----------------
    A single symmetric eigendecomposition of the ``p × p`` matrix is
    performed once and cached as ``_c_sqrt``. The per-step penalty and
    gradient then cost ``O(p² q)`` and ``O(p² q)`` respectively, where
    ``q`` is the output dimension of the layer being regularized.
    """

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        c_delta_n: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{δ,n}^{1/2}``.

        Args:
            lambda1: Geometry-aware penalty weight ``λ₁``. Must be
                ``>= 0``.
            lambda2: Isotropic penalty weight ``λ₂``. Must be ``>= 0``.
            c_delta_n: Stabilized Gram matrix of shape ``(p, p)``,
                typically ``(1/n) X^T X + δ I_p``.

        Raises:
            ValueError: If ``lambda1`` or ``lambda2`` is negative.

        Notes:
            The eigendecomposition uses ``numpy.linalg.eigh``, which is
            both faster and more numerically stable than
            ``scipy.linalg.sqrtm`` because ``C_{δ,n}`` is symmetric
            positive-semidefinite by construction.
        """
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be non-negative.")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # Compute the matrix square root via symmetric eigendecomposition:
        # ``eigh`` is preferred over ``sqrtm`` because ``C_{δ,n}`` is
        # symmetric by construction, which makes the eigenbasis real and
        # avoids the complex-arithmetic detour of ``sqrtm``.
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        # Guard against tiny negative eigenvalues introduced by floating-
        # point error; clipping to 0 keeps the square root real-valued
        # without materially affecting the penalty for non-degenerate
        # ``C_{δ,n}``.
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``λ₁ ‖C^{1/2} W‖_F² + λ₂ ‖W‖_F²``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar penalty.
        """
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.lambda2 * float(
            np.sum(weights**2)
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return ``2 λ₁ C_{δ,n} W + 2 λ₂ W``.

        The first term is derived as
        ``∇_W ‖C^{1/2} W‖_F² = 2 C^{1/2} (C^{1/2} W) = 2 C W``.
        The cached ``_c_sqrt`` is multiplied with itself rather than
        storing ``c_delta_n`` separately so that the gradient operates
        on the same matrix that defines the penalty (including any
        numerical clipping performed at construction time).

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        # ∇_W λ₁ ‖C^{1/2} W‖_F² = 2 λ₁ C W
        # ∇_W λ₂ ‖W‖_F²        = 2 λ₂ W
        c_mat = self._c_sqrt @ self._c_sqrt
        return (
            2.0 * self.lambda1 * (c_mat @ weights)
            + 2.0 * self.lambda2 * weights
        )


class Sparridge(Regularizer):
    """Sparridge penalty: ``λ₁ ‖C_{δ,n}^{1/2} W‖_F² + γ ‖W‖_1``.

    Combines the geometry-aware shrinkage of :class:`Covridge` with
    the sparsity-inducing ``ℓ₁`` term of :class:`Lasso`. Useful when the
    data exhibit both correlated predictors and a sparse subset of
    informative features.

    Penalty: ``Ω(W) = λ₁ ‖C_{δ,n}^{1/2} W‖_F² + γ ‖W‖_1``.
    Subgradient: ``∇_W Ω(W) = 2 λ₁ C_{δ,n} W + γ · sign(W)``.

    Special cases
    -------------
    * ``C_{δ,n} = I`` reduces Sparridge to an elastic-net-like penalty
      with ``λ₁`` playing the role of the ridge weight.
    * ``γ = 0`` removes the sparsity term and recovers the geometry-aware
      ridge of :class:`Covridge`.
    """

    def __init__(
        self,
        lambda1: float,
        gamma: float,
        c_delta_n: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{δ,n}^{1/2}``.

        Args:
            lambda1: Geometry-aware penalty weight ``λ₁``. Must be
                ``>= 0``.
            gamma: Sparsity weight ``γ``. Must be ``>= 0``.
            c_delta_n: Stabilized Gram matrix of shape ``(p, p)``.

        Raises:
            ValueError: If ``lambda1`` or ``gamma`` is negative.
        """
        if lambda1 < 0 or gamma < 0:
            raise ValueError("lambda1 and gamma must be non-negative.")
        self.lambda1 = lambda1
        self.gamma = gamma
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        # Symmetry guarantees a real eigenbasis; clamp tiny negative
        # eigenvalues to 0 to keep the square root real-valued.
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``λ₁ ‖C^{1/2} W‖_F² + γ ‖W‖_1``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar penalty.
        """
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.gamma * float(
            np.sum(np.abs(weights))
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return ``2 λ₁ C_{δ,n} W + γ · sign(W)``.

        Args:
            weights: Weight matrix.

        Returns:
            Subgradient array, same shape as ``weights``.
        """
        c_mat = self._c_sqrt @ self._c_sqrt
        l2_grad = 2.0 * self.lambda1 * (c_mat @ weights)
        l1_grad = self.gamma * np.sign(weights)
        return l2_grad + l1_grad
