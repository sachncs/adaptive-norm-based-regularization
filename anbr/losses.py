"""Loss functions with forward and backward methods."""

import numpy as np


class MSELoss:
    """Mean squared error for regression."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE.

        Args:
            y_pred: Predictions of shape (n_samples, n_outputs).
            y_true: Targets of shape (n_samples, n_outputs).

        Returns:
            Scalar MSE.
        """
        diff = y_pred - y_true
        return float(np.mean(diff**2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE w.r.t. y_pred.

        Args:
            y_pred: Predictions of shape (n_samples, n_outputs).
            y_true: Targets of shape (n_samples, n_outputs).

        Returns:
            Gradient of shape (n_samples, n_outputs).
        """
        n_elements = y_pred.size
        return 2.0 * (y_pred - y_true) / n_elements


class CrossEntropyLoss:
    """Softmax cross-entropy for multi-class classification."""

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss.

        Args:
            logits: Logits of shape (n_samples, n_classes).
            y_true: Integer class labels of shape (n_samples,).

        Returns:
            Scalar cross-entropy.
        """
        # Numerically stable softmax.
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        # Cross-entropy: -log(p[y_true])
        log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-15)
        return float(-np.mean(log_probs))

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy w.r.t. logits.

        Args:
            logits: Logits of shape (n_samples, n_classes).
            y_true: Integer class labels of shape (n_samples,).

        Returns:
            Gradient of shape (n_samples, n_classes).
        """
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        n_samples = logits.shape[0]
        grad = probs.copy()
        grad[np.arange(len(y_true)), y_true] -= 1.0
        return grad / n_samples
