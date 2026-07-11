"""Adam optimizer implemented in pure NumPy.

Implements the Adam algorithm (Kingma & Ba, 2014) with per-parameter
first- and second-moment estimates, bias-corrected step sizes, and an
explicit :meth:`reset` for re-training from scratch.

The optimizer operates on dictionaries of parameter lists (matching the
format returned by :meth:`~anbr.network.FullyConnectedNetwork.get_params`)
so it is agnostic to network architecture.
"""

from typing import Dict, List, Optional

import numpy as np


class Adam:
    r"""Adam optimizer with adaptive learning rates.

    Maintains exponential moving averages of past gradients (first
    moment, ``m``) and past squared gradients (second moment, ``v``).
    At each step the bias-corrected estimates are used:

    .. math::

        \hat{m}_t &= m_t / (1 - \beta_1^t) \\
        \hat{v}_t &= v_t / (1 - \beta_2^t) \\
        \theta_t   &= \theta_{t-1} - \eta \, \hat{m}_t /
                       (\sqrt{\hat{v}_t} + \epsilon)

    Attributes:
        learning_rate: Step size ``eta``.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Small constant added to the denominator for numerical
            stability.
        t: Internal step counter (incremented on each :meth:`step` call).
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialise Adam hyper-parameters and empty moment buffers.

        Args:
            learning_rate: Step size ``eta`` (default ``1e-3``).
            beta1: Decay rate for the first moment (default ``0.9``).
            beta2: Decay rate for the second moment (default ``0.999``).
            epsilon: Numerical stability term (default ``1e-8``).
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
        """Perform one Adam update in place and return new parameters.

        First-time parameters are initialised to zero moments.  The step
        counter ``t`` is incremented *before* the bias correction is
        computed so that ``t = 1`` on the very first call.

        Args:
            params: Current parameter arrays keyed by group name
                (``"weights"``, ``"biases"``).
            grads: Gradient arrays with the same structure as *params*.

        Returns:
            New parameter dictionary (same structure, new arrays).
        """
        self.t += 1
        updated: Dict[str, List[np.ndarray]] = {}
        for key in params:
            updated[key] = []
            if key not in self._m:
                self._m[key] = []
                self._v[key] = []
            # Grow moment lists to match the number of parameter arrays.
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
                # Update biased moments.
                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g**2)
                self._m[key][i] = m
                self._v[key][i] = v
                # Bias-corrected estimates.
                m_hat = m / (1.0 - self.beta1**self.t)
                v_hat = v / (1.0 - self.beta2**self.t)
                p_new = p - self.learning_rate * m_hat / (
                    np.sqrt(v_hat) + self.epsilon
                )
                updated[key].append(p_new)
        return updated

    def reset(self) -> None:
        """Reset all internal state to the initial condition.

        Clears the step counter and all moment buffers so the optimizer
        can be reused for a fresh training run.
        """
        self.t = 0
        self._m = {}
        self._v = {}
