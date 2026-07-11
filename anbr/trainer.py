"""Training loop with mini-batching, early stopping, and standardization.

The :class:`Trainer` class orchestrates the epoch-level training loop,
including:

* Mini-batch sampling with per-epoch shuffling.
* Forward pass, loss computation, and regularizer penalty.
* Back-propagation and parameter update via :class:`~anbr.optimizer.Adam`.
* Optional validation loss tracking and early stopping.
* Verbose progress printing.

Covridge / Sparridge layer handling
------------------------------------
:class:`~anbr.regularizers.Covridge` and :class:`~anbr.regularizers.Sparridge`
are applied **only to the first weight matrix** because the empirical Gram
matrix ``C_{delta,n}`` is defined over the input dimension and therefore
only matches ``W^{(1)}``.  This is a design assumption: the paper defines
``C_{delta,n} = (1/n) X^T X + delta I_p`` where ``X`` is the input
matrix, so applying it to deeper layers would require computing a
different Gram matrix for each layer's activations.

All other regularizers (Ridge, Lasso, ElasticNet, NoRegularizer) are
applied to **every** weight matrix, which is the standard approach.

Gradient flow
-------------
The gradient path is: loss.backward -> network.backward -> regularizer
gradient -> optimizer.step.  The ``1/n`` scaling lives in the loss
backward, not in the network backward.  This means the network backward
computes raw gradients (summed across samples), and the loss backward
divides by the total number of scalar elements.
"""

from typing import Dict, List, Optional

import numpy as np

import anbr.losses as losses
from anbr.network import FullyConnectedNetwork
from anbr.optimizer import Adam
from anbr.regularizers import Covridge, Regularizer, Sparridge


class Trainer:
    """End-to-end trainer for the manual NumPy network.

    Ties together a :class:`~anbr.network.FullyConnectedNetwork`, a loss
    function, a regularizer, and an :class:`~anbr.optimizer.Adam`
    optimizer into a single :meth:`fit` / :meth:`predict` interface.

    The trainer is **not** reusable for multiple independent runs without
    calling :meth:`predict` (which does not reset state) -- create a new
    ``Trainer`` instance for each independent experiment.

    Attributes:
        history: Dictionary with ``"train_loss"`` and ``"val_loss"`` lists
            populated during :meth:`fit``.

    Thread safety
    -------------
    The trainer is **not** thread-safe: it mutates the network, optimizer,
            and history during :meth:`fit`.
    """

    def __init__(
        self,
        network: FullyConnectedNetwork,
        loss_fn: losses.MSELoss | losses.CrossEntropyLoss,
        regularizer: Regularizer,
        optimizer: Adam,
        batch_size: int = 32,
        epochs: int = 500,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialise the trainer with all component objects.

        Args:
            network: The network to train.  Will be mutated in place
                during :meth:`fit`.
            loss_fn: Loss function with ``forward`` / ``backward`` methods.
            regularizer: Regularizer applied to weight matrices.
            optimizer: Adam optimizer instance.
            batch_size: Mini-batch size (default ``32``).  The last batch
                may be smaller if ``n_samples`` is not divisible by
                ``batch_size``.
            epochs: Maximum number of training epochs (default ``500``).
            early_stopping: Enable early stopping on validation loss.
                Requires ``x_val`` and ``y_val`` to be passed to
                :meth:`fit`.
            patience: Number of epochs without improvement before
                stopping (default ``10``).
            verbose: Print progress every 50 epochs.
        """
        self.network = network
        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the network on *x_train* with optional validation.

        The training loop shuffles the data each epoch, iterates over
        mini-batches, and updates parameters via the optimizer.  When
        ``early_stopping`` is enabled and validation data is provided,
        the best parameters (lowest validation loss) are restored at the
        end of training.

        Side effects
        ------------
        Mutates ``self.network`` weights, ``self.optimizer`` state, and
        ``self.history``.

        Complexity
        ----------
        O(epochs * n_samples * (forward + backward + update)).  The
        forward and backward passes are each O(sum of layer sizes * batch
        size).

        Args:
            x_train: Training inputs of shape ``(n, p)``.
            y_train: Training targets of shape ``(n, 1)`` or ``(n,)``.
            x_val: Validation inputs (optional).  Required for early
                stopping and validation loss tracking.
            y_val: Validation targets (optional).

        Raises:
            ValueError: If ``early_stopping`` is ``True`` but ``x_val``
                or ``y_val`` is ``None``.
        """
        if self.early_stopping and (x_val is None or y_val is None):
            raise ValueError(
                "early_stopping requires x_val and y_val to be provided."
            )
        n_samples = x_train.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0
        best_params: Optional[Dict[str, List[np.ndarray]]] = None

        for epoch in range(self.epochs):
            # Shuffle indices each epoch for stochasticity.
            indices = np.random.permutation(n_samples)
            epoch_losses: List[float] = []
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass and loss.
                y_pred = self.network.forward(x_batch)
                loss = self.loss_fn.forward(y_pred, y_batch)

                # Regularization penalty.
                # NOTE: Covridge and Sparridge are applied only to the first
                # layer because C_{delta,n} is computed from the input
                # dimension.  This is a design assumption not explicitly
                # stated in the paper.
                reg_penalty = 0.0
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    reg_penalty += self.regularizer.penalty(
                        self.network.weights[0]
                    )
                else:
                    for w in self.network.weights:
                        reg_penalty += self.regularizer.penalty(w)
                total_loss = loss + reg_penalty
                epoch_losses.append(total_loss)

                # Backward pass.
                dloss = self.loss_fn.backward(y_pred, y_batch)
                grads = self.network.backward(dloss)

                # Add regularizer gradients.
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    grads["weights"][0] += self.regularizer.gradient(
                        self.network.weights[0]
                    )
                else:
                    for i, w in enumerate(self.network.weights):
                        grads["weights"][i] += self.regularizer.gradient(w)

                # Optimizer step.
                params = {
                    "weights": self.network.weights,
                    "biases": self.network.biases,
                }
                new_params = self.optimizer.step(params, grads)
                self.network.set_params(new_params)

            avg_train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(avg_train_loss)

            # Validation evaluation.
            if x_val is not None and y_val is not None:
                val_pred = self.network.forward(x_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    val_loss += self.regularizer.penalty(
                        self.network.weights[0]
                    )
                else:
                    for w in self.network.weights:
                        val_loss += self.regularizer.penalty(w)
                self.history["val_loss"].append(float(val_loss))

                # Early stopping logic.
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_params = self.network.get_params()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            if best_params is not None:
                                self.network.set_params(best_params)
                            break

            if self.verbose and (epoch + 1) % 50 == 0:
                msg = (
                    f"Epoch {epoch + 1}/{self.epochs}"
                    f" -- train loss: {avg_train_loss:.4f}"
                )
                if x_val is not None:
                    msg += f", val loss: {self.history['val_loss'][-1]:.4f}"
                print(msg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input.

        Runs a single forward pass through the network.  This does not
        modify the network weights or optimizer state.

        Args:
            x: Input array of shape ``(n_samples, n_features)``.

        Returns:
            Network output of shape ``(n_samples, n_outputs)``.
        """
        return self.network.forward(x)
