"""Cross-validation and hyperparameter grid search utilities."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import anbr.losses as losses
import anbr.metrics as metrics
from anbr.network import FullyConnectedNetwork
from anbr.optimizer import Adam
from anbr.regularizers import (
    Covridge,
    ElasticNet,
    Lasso,
    NoRegularizer,
    Regularizer,
    Ridge,
    Sparridge,
)
from anbr.trainer import Trainer


def build_regularizer(
    method: str,
    hp: Dict[str, float],
    x_train: np.ndarray,
    delta: float = 1e-4,
) -> Regularizer:
    """Instantiate a regularizer from method name and hyperparameters.

    Args:
        method: One of 'none', 'ridge', 'lasso', 'elastic_net',
            'covridge', 'sparridge'.
        hp: Hyperparameter dict (e.g., {'lambda_': 0.01}).
        x_train: Training features used to compute C_{δ,n}.
        delta: Stabilization constant for Gram matrix.

    Returns:
        Regularizer instance.
    """
    if method == "none":
        return NoRegularizer()
    if method == "ridge":
        return Ridge(lambda_=hp["lambda_"])
    if method == "lasso":
        return Lasso(gamma=hp["gamma"])
    if method == "elastic_net":
        return ElasticNet(alpha=hp["alpha"], gamma=hp["gamma"])

    n, p = x_train.shape
    c_n = (x_train.T @ x_train) / n
    c_delta_n = c_n + delta * np.eye(p)

    if method == "covridge":
        return Covridge(
            lambda1=hp["lambda1"], lambda2=hp["lambda2"], c_delta_n=c_delta_n
        )
    if method == "sparridge":
        return Sparridge(lambda1=hp["lambda1"], gamma=hp["gamma"], c_delta_n=c_delta_n)
    raise ValueError(f"Unknown method: {method}")


def grid_search_cv(
    x: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    method: str,
    param_grid: List[Dict[str, float]],
    loss_fn: losses.MSELoss | losses.CrossEntropyLoss,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    early_stopping: bool = False,
    patience: int = 10,
    task: str = "regression",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, float], float]:
    """Run k-fold CV over a hyperparameter grid.

    Args:
        x: Features.
        y: Targets.
        layer_sizes: Network architecture.
        method: Regularization method name.
        param_grid: List of hyperparameter dicts to evaluate.
        loss_fn: Loss function.
        n_splits: Number of CV folds.
        batch_size: Mini-batch size.
        epochs: Training epochs.
        learning_rate: Adam learning rate.
        early_stopping: Enable early stopping.
        patience: Early stopping patience.
        task: 'regression' or 'classification'.
        random_state: RNG seed for KFold.

    Returns:
        Tuple of (best_params, best_score).
        For regression best_score is negative MSE (higher is better);
        for classification it is balanced accuracy.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -float("inf")
    best_params: Optional[Dict[str, float]] = None

    for params in param_grid:
        scores: List[float] = []
        for train_idx, val_idx in kf.split(x):
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            x_train_fold = scaler.fit_transform(x_train_fold)
            x_val_fold = scaler.transform(x_val_fold)

            regularizer = build_regularizer(method, params, x_train_fold)
            net = FullyConnectedNetwork(layer_sizes)
            opt = Adam(learning_rate=learning_rate)
            trainer = Trainer(
                net,
                loss_fn,
                regularizer,
                opt,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            trainer.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            preds = trainer.predict(x_val_fold)

            if task == "regression":
                score = -metrics.mean_squared_error(y_val_fold, preds)
            else:
                class_preds = np.argmax(preds, axis=1)
                score = metrics.balanced_accuracy_score(y_val_fold, class_preds)
            scores.append(score)

        avg_score = float(np.mean(scores))
        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

    assert best_params is not None
    return best_params, best_score
