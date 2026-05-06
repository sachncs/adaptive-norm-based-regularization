"""Manual feedforward ReLU network with backpropagation."""

from typing import Dict, List, Tuple

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier uniform initialization.

    Args:
        fan_in: Input dimension.
        fan_out: Output dimension.

    Returns:
        Initialized weight matrix.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


class FullyConnectedNetwork:
    """Fully connected feedforward network with ReLU activations."""

    def __init__(self, layer_sizes: List[int]) -> None:
        """Initialize the network.

        Args:
            layer_sizes: List of layer dimensions, including input and output.
                e.g., [p, 64, 32, 1] for a 2-hidden-layer regression net.
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            w = xavier_uniform(layer_sizes[i], layer_sizes[i + 1])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Caches for forward pass.
        self._zs: List[np.ndarray] = []
        self._as: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input of shape (n_samples, n_features).

        Returns:
            Output of shape (n_samples, n_outputs).
        """
        self._zs = []
        self._as = [x]
        a = x
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self._zs.append(z)
            if i < self.n_layers - 1:
                a = np.maximum(z, 0.0)  # ReLU
            else:
                a = z  # linear output
            self._as.append(a)
        return a

    def backward(self, dloss_dy: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Backward pass.

        Args:
            dloss_dy: Gradient of loss w.r.t. network output,
                shape (n_samples, n_outputs).

        Returns:
            Dictionary with keys 'weights' and 'biases', each a list
            of gradients matching the parameter shapes.
        """
        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        # Start from the last layer.
        delta = dloss_dy
        for i in reversed(range(self.n_layers)):
            a_prev = self._as[i]
            # Gradient w.r.t. weights and biases.
            # delta already contains the 1/n_samples factor from the loss.
            dw = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                # Backprop through linear transformation.
                delta = delta @ self.weights[i].T
                # Backprop through ReLU.
                z = self._zs[i - 1]
                delta *= (z > 0.0).astype(delta.dtype)

        return {"weights": grads_w, "biases": grads_b}

    def get_params(self) -> Dict[str, List[np.ndarray]]:
        """Return a copy of current parameters."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params: Dict[str, List[np.ndarray]]) -> None:
        """Set parameters from a dictionary."""
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]
