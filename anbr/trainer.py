"""Training loop with mini-batching, early stopping, and standardization."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

import anbr.losses as losses
import anbr.metrics as metrics
from anbr.network import FullyConnectedNetwork
from anbr.optimizer import Adam
from anbr.regularizers import Covridge, Regularizer, Sparridge


class Trainer:
    """Trainer for the manual NumPy network."""

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
        """Initialize trainer.

        Args:
            network: The network to train.
            loss_fn: Loss function with forward/backward.
            regularizer: Regularizer applied to all weight matrices.
            optimizer: Optimizer instance.
            batch_size: Mini-batch size.
            epochs: Number of training epochs.
            early_stopping: Whether to stop when validation loss plateaus.
            patience: Epochs to wait for improvement before stopping.
            verbose: Print progress.
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
        """Fit the model.

        Args:
            x_train: Training inputs.
            y_train: Training targets.
            x_val: Validation inputs (optional).
            y_val: Validation targets (optional).
        """
        n_samples = x_train.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0
        best_params: Optional[Dict[str, List[np.ndarray]]] = None

        for epoch in range(self.epochs):
            # Shuffle indices each epoch.
            indices = np.random.permutation(n_samples)
            epoch_losses: List[float] = []
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward.
                y_pred = self.network.forward(x_batch)
                loss = self.loss_fn.forward(y_pred, y_batch)

                # Regularization penalty.
                # NOTE: Covridge and Sparridge are applied only to the first
                # layer because C_{δ,n} is computed from the input dimension.
                # This is an assumption not explicitly stated in the paper.
                reg_penalty = 0.0
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    reg_penalty += self.regularizer.penalty(self.network.weights[0])
                else:
                    for w in self.network.weights:
                        reg_penalty += self.regularizer.penalty(w)
                total_loss = loss + reg_penalty
                epoch_losses.append(total_loss)

                # Backward.
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

            # Validation.
            if x_val is not None and y_val is not None:
                val_pred = self.network.forward(x_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    val_loss += self.regularizer.penalty(self.network.weights[0])
                else:
                    for w in self.network.weights:
                        val_loss += self.regularizer.penalty(w)
                self.history["val_loss"].append(float(val_loss))

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
                msg = f"Epoch {epoch + 1}/{self.epochs} — train loss: {avg_train_loss:.4f}"
                if x_val is not None:
                    msg += f", val loss: {self.history['val_loss'][-1]:.4f}"
                print(msg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            x: Input array.

        Returns:
            Predictions.
        """
        return self.network.forward(x)
