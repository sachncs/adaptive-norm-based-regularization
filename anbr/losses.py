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

Gradient scaling convention
---------------------------
Gradients are **summed** across samples in the network backward pass
(``FullyConnectedNetwork.backward``) and **divided by n** in the loss
backward pass.  This matches standard deep-learning convention: the
optimizer receives ``(1/n) * sum_i nabla L_i``.

Numerical stability
-------------------
:class:`CrossEntropyLoss` subtracts the row-max logit before
exponentiation to avoid overflow, and clips log-probabilities with a
floor of ``1e-15`` to prevent ``log(0)``.
"""

import numpy as np


class MSELoss:
    """Mean squared error loss for regression.

    Computes ``(1/n) * sum((y_pred - y_true)^2)``.  The gradient is
    ``2(y_pred - y_true) / n`` where ``n`` equals the total number of
    scalar elements (``y_pred.size``), not just the number of rows.

    This distinction matters for multi-output regression: a prediction
    of shape ``(100, 3)`` has ``n = 300``, not ``n = 100``.

    When to use
    -----------
    Use for continuous-valued targets.  For classification tasks use
    :class:`CrossEntropyLoss` instead.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar MSE loss.

        Args:
            y_pred: Predictions of shape ``(n_samples, n_outputs)``.
            y_true: Targets of shape ``(n_samples, n_outputs)``.

        Returns:
            Mean squared error as a Python float.

        Raises:
            ValueError: If ``y_pred`` and ``y_true`` have different shapes
                (implicit via broadcasting).
        """
        diff = y_pred - y_true
        return float(np.mean(diff**2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE w.r.t. ``y_pred``.

        The gradient is ``(2 / n_elements) * (y_pred - y_true)`` where
        ``n_elements = y_pred.size`` (total scalar entries).

        This gradient is already scaled by ``1/n`` and ready to be passed
        to ``FullyConnectedNetwork.backward``.

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

    The softmax is embedded in the forward pass (not exposed separately)
    so that numerical stability optimizations can be applied.  The
    backward pass computes ``(softmax(logits) - one_hot(y_true)) / n``,
    which is the well-known closed-form gradient of softmax cross-entropy.

    When to use
    -----------
    Use for classification with integer labels.  For regression use
    :class:`MSELoss` instead.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar cross-entropy loss.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
                These are pre-softmax values; the softmax is applied
                internally.
            y_true: Integer class labels of shape ``(n_samples,)``.

        Returns:
            Mean cross-entropy as a Python float.

        Raises:
            IndexError: If any label in ``y_true`` is outside
                ``[0, n_classes)``.
        """
        # Numerically stable softmax: subtract row max to prevent overflow.
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        # Cross-entropy: -log(p[y_true]) with floor for numerical safety.
        # The 1e-15 floor prevents log(0) when the model is extremely
        # confident but wrong.
        log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-15)
        return float(-np.mean(log_probs))

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of cross-entropy w.r.t. ``logits``.

        The gradient equals ``(softmax(logits) - one_hot(y_true)) / n``
        where ``n = logits.shape[0]``.

        This is the standard closed-form result: the cross-entropy
        gradient w.r.t. logits simplifies to the difference between the
        predicted probability distribution and the one-hot target.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
            y_true: Integer class labels of shape ``(n_samples,)``.

        Returns:
            Gradient array with the same shape as ``logits``.
        """
        # Recompute softmax (no caching across forward/backward).
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        n_samples = logits.shape[0]
        # Subtract 1.0 at the true class position: softmax - one_hot.
        grad = probs.copy()
        grad[np.arange(len(y_true)), y_true] -= 1.0
        return grad / n_samples
