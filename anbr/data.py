"""Data generators for simulations and real-data loaders."""

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _make_covariance_matrix(k: int, rho: float) -> np.ndarray:
    """Create a k×k covariance with 1 on diagonal and rho off-diagonal."""
    sigma = np.full((k, k), rho)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def make_dgp(
    n: int,
    p: int,
    k: int,
    rho: float,
    sigma_noise: float,
    tau: float = 1.0,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data according to the paper's DGP.

    Args:
        n: Number of samples.
        p: Number of features.
        k: Number of informative features.
        rho: Correlation among informative features.
        sigma_noise: Standard deviation of additive Gaussian noise.
        tau: Standard deviation of true coefficients.
        nonlinear: If True, use sinusoidal signal.
        random_state: RNG seed.

    Returns:
        Tuple of (X, y) with shapes (n, p) and (n,) or (n, 1).
    """
    rng = np.random.default_rng(random_state)
    # Informative block.
    if k > 0:
        cov = _make_covariance_matrix(k, rho)
        x_informative = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        x_informative = np.empty((n, 0))
    # Noise features.
    x_noise = rng.standard_normal(size=(n, p - k))
    x = np.hstack([x_informative, x_noise])
    # Coefficients.
    theta = rng.normal(0.0, tau, size=k)
    if k == 0:
        y = np.zeros(n)
    elif nonlinear:
        y = np.sum(theta * np.sin(x_informative), axis=1)
    else:
        y = x_informative @ theta
    y += rng.normal(0.0, sigma_noise, size=n)
    return x, y.reshape(-1, 1)


def make_dgp1(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP1: n=200, p=20, k=10."""
    return make_dgp(
        200, 20, 10, rho, sigma_noise, nonlinear=nonlinear, random_state=random_state
    )


def make_dgp2(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP2: n=1000, p=200, k=100."""
    return make_dgp(
        1000, 200, 100, rho, sigma_noise, nonlinear=nonlinear, random_state=random_state
    )


def make_dgp3(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP3: n=500, p=2000, k=100."""
    return make_dgp(
        500, 2000, 100, rho, sigma_noise, nonlinear=nonlinear, random_state=random_state
    )


def load_energy_data(
    test_size: float = 0.25,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load UCI Energy Efficiency dataset (cooling load).

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    data = fetch_openml(name="energy-efficiency", as_frame=True, parser="auto")
    df = data.frame
    # Targets: 'y1' (heating), 'y2' (cooling). We use y2.
    feature_cols = [c for c in df.columns if c not in ("y1", "y2")]
    x = df[feature_cols].astype(float).to_numpy()
    y = df["y2"].astype(float).to_numpy().reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler


def load_leukemia_data(
    n_features: int = 2000,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load GSE9476 leukemia microarray data.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Attempt to fetch from OpenML (GSE9476 surrogate).
    # If unavailable, raise with instructions.
    try:
        data = fetch_openml(data_id=1120, as_frame=True, parser="auto")
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch GSE9476 from OpenML. "
            "Please download manually and place in data/gse9476.csv."
        ) from exc
    df = data.frame
    y_raw = df[data.target_names[0]].to_numpy()
    x = df.drop(columns=data.target_names).astype(float).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # ANOVA feature selection to top n_features.
    f_values, _ = f_classif(x, y)
    top_idx = np.argsort(f_values)[-n_features:]
    x = x[:, top_idx]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler
