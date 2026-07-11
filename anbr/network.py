"""Manual feedforward ReLU network with back-propagation.

Implements a fully connected multi-layer perceptron using only NumPy.
All forward and backward passes are explicit -- there is no autograd
engine.  Weight initialization follows the Xavier (Glorot) uniform
scheme.

Typical usage
-------------
>>> net = FullyConnectedNetwork([p, 64, 32, 1])
>>> out = net.forward(x)
>>> grads = net.backward(dloss)
"""

from typing import Dict, List

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Sample a weight matrix from the Xavier uniform distribution.

    Draws from ``Uniform(-limit, limit)`` where
    ``limit = sqrt(6 / (fan_in + fan_out))``.

    Args:
        fan_in: Number of input features.
        fan_out: Number of output features.

    Returns:
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


class FullyConnectedNetwork:
    """Feedforward MLP with ReLU hidden activations and a linear output.

    The network stores its own forward-pass activations and
    pre-activation values in ``_as`` and ``_zs`` respectively so that
    :meth:`backward` can compute exact gradients without re-running the
    forward pass.

    Attributes:
        layer_sizes: List of layer widths ``[input, hidden_1, ..., output]``.
        n_layers: Number of weight matrices (``len(layer_sizes) - 1``).
        weights: List of weight matrices, one per layer.
        biases: List of row-vector biases, one per layer.
    """

    def __init__(self, layer_sizes: List[int]) -> None:
        """Initialise weights with Xavier-uniform and biases to zero.

        Args:
            layer_sizes: Layer widths including input and output, e.g.
                ``[p, 64, 32, 1]`` for a 2-hidden-layer regression net.
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

        # Forward-pass caches: pre-activations (_zs) and activations (_as).
        self._zs: List[np.ndarray] = []
        self._as: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass, caching intermediates for :meth:`backward`.

        Hidden layers use ``ReLU(z) = max(z, 0)``; the output layer is
        linear (no activation).

        Args:
            x: Input batch of shape ``(n_samples, n_features)``.

        Returns:
            Network output of shape ``(n_samples, n_outputs)``.
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
        """Compute parameter gradients via back-propagation.

        Assumes the loss gradient ``dloss_dy`` already contains the
        ``1 / n_samples`` scaling from the loss backward pass.

        Args:
            dloss_dy: Upstream gradient w.r.t. the network output, shape
                ``(n_samples, n_outputs)``.

        Returns:
            Dictionary with two keys:

            * ``"weights"`` -- list of weight-gradient arrays, one per
              layer, in input-to-output order.
            * ``"biases"`` -- list of bias-gradient arrays (row vectors).
        """
        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        # Propagate from the output layer backwards.
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
                # Back-prop through the linear transformation.
                delta = delta @ self.weights[i].T
                # Back-prop through ReLU: gradient is 0 where pre-activation
                # was non-positive.
                z = self._zs[i - 1]
                delta *= (z > 0.0).astype(delta.dtype)

        return {"weights": grads_w, "biases": grads_b}

    def get_params(self) -> Dict[str, List[np.ndarray]]:
        """Return deep copies of all current parameters.

        Returns:
            Dictionary with ``"weights"`` and ``"biases"`` lists.
        """
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params: Dict[str, List[np.ndarray]]) -> None:
        """Replace all parameters (deep-copied from *params*).

        Args:
            params: Dictionary with ``"weights"`` and ``"biases"`` lists
                matching the network architecture.
        """
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]
