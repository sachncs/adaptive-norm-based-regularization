"""Manual feedforward ReLU network with back-propagation.

Implements a fully connected multi-layer perceptron using only NumPy.
All forward and backward passes are explicit -- there is no autograd
engine.  Weight initialization follows the Xavier (Glorot) uniform
scheme.

Architecture
------------
The network is defined by a list of layer widths
``[input, hidden_1, ..., output]``.  Hidden layers use ReLU activations;
the output layer is linear (no activation).  This is the standard
architecture for regression; for classification the output is passed
through a softmax in the loss function (see
:class:`~anbr.losses.CrossEntropyLoss`).

Back-propagation
----------------
The backward pass computes exact gradients w.r.t. all weights and biases
using the chain rule.  It requires cached activations from the forward
pass (stored in ``_as`` and ``_zs``), so :meth:`forward` must be called
before :meth:`backward`.

Typical usage
-------------
>>> net = FullyConnectedNetwork([p, 64, 32, 1])
>>> out = net.forward(x)
>>> grads = net.backward(dloss)

References
----------
Glorot, X. & Bengio, Y. (2010).  "Understanding the difficulty of
training deep feedforward neural networks."  *AISTATS*.
"""

from typing import Dict, List

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Sample a weight matrix from the Xavier uniform distribution.

    Draws from ``Uniform(-limit, limit)`` where
    ``limit = sqrt(6 / (fan_in + fan_out))``.  This variance scaling
    keeps the variance of activations and gradients roughly constant
    across layers, which stabilizes training for deep networks.

    References
    ----------
    Glorot, X. & Bengio, Y. (2010).  Section 4.2, equation (12).

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

    Thread safety
    -------------
    The network is not thread-safe: :meth:`forward` mutates internal
    caches (``_zs``, ``_as``).  Do not call ``forward`` and ``backward``
    concurrently from different threads.
    """

    def __init__(self, layer_sizes: List[int]) -> None:
        """Initialise weights with Xavier-uniform and biases to zero.

        Args:
            layer_sizes: Layer widths including input and output, e.g.
                ``[p, 64, 32, 1]`` for a 2-hidden-layer regression net.
                Must have at least 2 elements (input + output).

        Raises:
            ValueError: If ``layer_sizes`` has fewer than 2 elements.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements.")
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
        # These are populated by forward() and consumed by backward().
        self._zs: List[np.ndarray] = []
        self._as: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass, caching intermediates for :meth:`backward`.

        Applies ``ReLU(z) = max(z, 0)`` to hidden layers and a linear
        (identity) activation to the output layer.  The cached values in
        ``_zs`` and ``_as`` are replaced on each call.

        Side effects
        ------------
        Mutates ``self._zs`` and ``self._as`` -- these caches are
        required by :meth:`backward`.

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
                # ReLU: zero-out negative pre-activations.
                # The gradient of ReLU is 0 for z <= 0 and 1 for z > 0;
                # this is applied in backward via a mask.
                a = np.maximum(z, 0.0)
            else:
                # Linear output: no activation for regression.
                # For classification, softmax is applied in the loss.
                a = z
            self._as.append(a)
        return a

    def backward(self, dloss_dy: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Compute parameter gradients via back-propagation.

        Assumes the loss gradient ``dloss_dy`` already contains the
        ``1 / n_samples`` scaling from the loss backward pass.  The
        returned gradients are **not** scaled by the learning rate --
        that is the optimizer's job.

        Algorithm
        ---------
        Starting from the output layer, for each layer ``i`` (in reverse
        order):

        1. Compute ``dw = a_prev^T @ delta`` (weight gradient).
        2. Compute ``db = sum(delta, axis=0)`` (bias gradient).
        3. Propagate delta: ``delta = delta @ W^T``.
        4. Apply ReLU mask: ``delta *= (z > 0)``.

        Args:
            dloss_dy: Upstream gradient w.r.t. the network output, shape
                ``(n_samples, n_outputs)``.

        Returns:
            Dictionary with two keys:

            * ``"weights"`` -- list of weight-gradient arrays, one per
              layer, in input-to-output order.
            * ``"biases"`` -- list of bias-gradient arrays (row vectors).

        Raises:
            RuntimeError: If :meth:`forward` has not been called (empty
                caches).
        """
        if not self._as:
            raise RuntimeError("forward() must be called before backward().")
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
                # was non-positive.  The mask is computed from the cached
                # pre-activation z (not from a, because ReLU(a) = a for a > 0).
                z = self._zs[i - 1]
                delta *= (z > 0.0).astype(delta.dtype)

        return {"weights": grads_w, "biases": grads_b}

    def get_params(self) -> Dict[str, List[np.ndarray]]:
        """Return deep copies of all current parameters.

        Used by :class:`~anbr.trainer.Trainer` to snapshot parameters
        for early-stopping restore.  Deep copies ensure that subsequent
        training does not mutate the saved snapshot.

        Returns:
            Dictionary with ``"weights"`` and ``"biases"`` lists.
        """
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params: Dict[str, List[np.ndarray]]) -> None:
        """Replace all parameters (deep-copied from *params*).

        Used by :class:`~anbr.trainer.Trainer` to restore the best
        parameters during early stopping, and by
        :class:`~anbr.optimizer.Adam` to apply parameter updates.

        Args:
            params: Dictionary with ``"weights"`` and ``"biases"`` lists
                matching the network architecture.
        """
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]
