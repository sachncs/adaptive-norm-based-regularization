"""Regularizers: Ridge, Lasso, Elastic Net, Covridge, Sparridge."""

from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    """Abstract base class for weight regularizers."""

    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        """Compute the regularization penalty.

        Args:
            weights: Weight matrix of shape (in_features, out_features).

        Returns:
            Scalar penalty value.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute the gradient of the penalty w.r.t. weights.

        Args:
            weights: Weight matrix of shape (in_features, out_features).

        Returns:
            Gradient matrix of the same shape as weights.
        """
        raise NotImplementedError


class NoRegularizer(Regularizer):
    """No-op regularizer for unregularized baseline."""

    def penalty(self, weights: np.ndarray) -> float:
        return 0.0

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return np.zeros_like(weights)


class Ridge(Regularizer):
    """Ridge (ℓ2) regularization: λ ‖W‖_F²."""

    def __init__(self, lambda_: float) -> None:
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        self.lambda_ = lambda_

    def penalty(self, weights: np.ndarray) -> float:
        return self.lambda_ * float(np.sum(weights**2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2.0 * self.lambda_ * weights


class Lasso(Regularizer):
    """Lasso (ℓ1) regularization: γ ‖W‖_1."""

    def __init__(self, gamma: float) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        return self.gamma * float(np.sum(np.abs(weights)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # Subgradient: sign(0) = 0.
        return self.gamma * np.sign(weights)


class ElasticNet(Regularizer):
    """Elastic Net: α γ ‖W‖_1 + (1 - α)/2 ‖W‖_F²."""

    def __init__(self, alpha: float, gamma: float) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        l1 = self.alpha * self.gamma * float(np.sum(np.abs(weights)))
        l2 = (1.0 - self.alpha) * 0.5 * float(np.sum(weights**2))
        return l1 + l2

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        l1_grad = self.alpha * self.gamma * np.sign(weights)
        l2_grad = (1.0 - self.alpha) * weights
        return l1_grad + l2_grad


class Covridge(Regularizer):
    """Covridge: λ₁ ‖C^{1/2} W‖_F² + λ₂ ‖W‖_F².

    The matrix square root of C_{δ,n} is precomputed at initialization.
    """

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        c_delta_n: np.ndarray,
    ) -> None:
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be non-negative.")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # Compute matrix square root via eigendecomposition for stability.
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        # Guard against tiny negative eigenvalues from numerical error.
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.lambda2 * float(
            np.sum(weights**2)
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # ∇_W λ₁ ‖C^{1/2} W‖_F² = 2 λ₁ C W
        # ∇_W λ₂ ‖W‖_F²        = 2 λ₂ W
        c_mat = self._c_sqrt @ self._c_sqrt
        return 2.0 * self.lambda1 * (c_mat @ weights) + 2.0 * self.lambda2 * weights


class Sparridge(Regularizer):
    """Sparridge: λ₁ ‖C^{1/2} W‖_F² + γ ‖W‖_1."""

    def __init__(
        self,
        lambda1: float,
        gamma: float,
        c_delta_n: np.ndarray,
    ) -> None:
        if lambda1 < 0 or gamma < 0:
            raise ValueError("lambda1 and gamma must be non-negative.")
        self.lambda1 = lambda1
        self.gamma = gamma
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.gamma * float(
            np.sum(np.abs(weights))
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        c_mat = self._c_sqrt @ self._c_sqrt
        l2_grad = 2.0 * self.lambda1 * (c_mat @ weights)
        l1_grad = self.gamma * np.sign(weights)
        return l2_grad + l1_grad
