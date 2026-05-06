"""Adam optimizer implemented in pure NumPy."""

from typing import Dict, List, Optional

import numpy as np


class Adam:
    """Adam optimizer with first and second moment estimates."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Adam.

        Args:
            learning_rate: Step size.
            beta1: Exponential decay rate for first moment.
            beta2: Exponential decay rate for second moment.
            epsilon: Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self._m: Dict[str, List[Optional[np.ndarray]]] = {}
        self._v: Dict[str, List[Optional[np.ndarray]]] = {}

    def step(
        self,
        params: Dict[str, List[np.ndarray]],
        grads: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        """Perform one Adam update.

        Args:
            params: Dictionary of parameter arrays (e.g., 'weights', 'biases').
            grads: Dictionary of gradient arrays matching params.

        Returns:
            Updated parameters (same structure).
        """
        self.t += 1
        updated: Dict[str, List[np.ndarray]] = {}
        for key in params:
            updated[key] = []
            if key not in self._m:
                self._m[key] = []
                self._v[key] = []
            # Ensure moment lists are long enough.
            while len(self._m[key]) < len(params[key]):
                self._m[key].append(None)
                self._v[key].append(None)
            for i, (p, g) in enumerate(zip(params[key], grads[key])):
                if self._m[key][i] is None:
                    self._m[key][i] = np.zeros_like(p)
                    self._v[key][i] = np.zeros_like(p)
                m = self._m[key][i]
                v = self._v[key][i]
                assert m is not None
                assert v is not None
                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g**2)
                self._m[key][i] = m
                self._v[key][i] = v
                m_hat = m / (1.0 - self.beta1**self.t)
                v_hat = v / (1.0 - self.beta2**self.t)
                p_new = p - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                updated[key].append(p_new)
        return updated

    def reset(self) -> None:
        """Reset internal state."""
        self.t = 0
        self._m = {}
        self._v = {}
