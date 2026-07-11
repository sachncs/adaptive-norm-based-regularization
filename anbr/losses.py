"""Loss functions with forward and backward methods.

Provides differentiable loss functions used during training.  Each loss
exposes a :meth:`forward` (scalar evaluation) and a :meth:`backward`
(analytical gradient w.r.t. the prediction).  The ``1 / n_samples``
averaging convention is applied *inside* the backward pass so that
gradients can be directly consumed by the optimizer.

Supported losses
----------------
* :class:`MSELoss` -- mean squared error for regression tasks.
* :class:`CrossEntropyLoss` -- softmax cross-entropy for multi-class
  classification with integer labels.

Numerical stability
-------------------
:class:`CrossEntropyLoss` subtracts the row-max logit before exponentiation
to avoid overflow, and clips log-probabilities with a floor of ``1e-15``
to prevent ``log(0)``.
"""

import numpy as np


class MSELoss:
    """Mean squared error loss for regression.

    Computes ``(1/n) * sum((y_pred - y_true)^2)``.  The gradient is
    ``2(y_pred - y_true) / n`` where ``n`` equals the total number of
    scalar elements (``y_pred.size``), not just the number of rows.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar MSE loss.

        Args:
            y_pred: Predictions of shape ``(n_samples, n_outputs)``.
            y_true: Targets of shape ``(n_samples, n_outputs)``.

        Returns:
            Mean squared error as a Python float.
        """
        diff = y_pred - y_true
        return float(np.mean(diff**2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE w.r.t. ``y_pred``.

        The gradient is ``(2 / n_elements) * (y_pred - y_true)`` where
        ``n_elements = y_pred.size`` (total scalar entries).

        Args:
            y_pred: Predictions of shape ``(n_samples, n_outputs)``.
            y_true: Targets of shape ``(n_samples, n_outputs)``.

        Returns:
            Gradient array with the same shape as ``y_pred``.
        """
        n_elements = y_pred.size
        return 2.0 * (y_pred - y_true) / n_elements


class CrossEntropyLoss:
    """Softmax cross-entropy loss for multi-class classification.

    Applies a numerically stable softmax internally and computes
    ``-log(p[y_true])`` averaged over samples.  Labels must be integer
    class indices in ``[0, n_classes)``.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar cross-entropy loss.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
            y_true: Integer class labels of shape ``(n_samples,)``.

        Returns:
            Mean cross-entropy as a Python float.

        Raises:
            IndexError: If any label in ``y_true`` is outside
                ``[0, n_classes)``.
        """
        # Numerically stable softmax: subtract row max.
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        # Cross-entropy: -log(p[y_true]) with floor for numerical safety.
        log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-15)
        return float(-np.mean(log_probs))

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of cross-entropy w.r.t. ``logits``.

        The gradient equals ``(softmax(logits) - one_hot(y_true)) / n``
        where ``n = logits.shape[0]``.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
            y_true: Integer class labels of shape ``(n_samples,)``.

        Returns:
            Gradient array with the same shape as ``logits``.
        """
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        n_samples = logits.shape[0]
        grad = probs.copy()
        grad[np.arange(len(y_true)), y_true] -= 1.0
        return grad / n_samples
