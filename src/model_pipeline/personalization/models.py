"""
Model factory for the personalization point-valuation regression task.

Registered models:
- ``mean``           — predict training-set mean (baseline)
- ``random_forest``  — scikit-learn RandomForestRegressor
- ``xgboost``        — XGBRegressor
- ``mlp``            — scikit-learn MLPRegressor

All models expose a unified ``fit`` / ``predict`` / ``get_params`` interface
via a thin wrapper so they integrate cleanly with the trainer and MLflow.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean for every sample."""

    def __init__(self) -> None:
        self.mean_: Optional[float] = None

    def fit(self, X: Any, y: Any) -> "MeanBaselineRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model not fitted")
        return np.full(len(X), self.mean_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {}


# ── Registry ──────────────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model builder in the factory."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


@register_model("mean")
class _MeanWrapper(MeanBaselineRegressor):
    pass


def _make_random_forest(params: Dict[str, Any]) -> BaseEstimator:
    from sklearn.ensemble import RandomForestRegressor

    defaults = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(params)
    return RandomForestRegressor(**defaults)


def _make_xgboost(params: Dict[str, Any]) -> BaseEstimator:
    from xgboost import XGBRegressor

    defaults = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    defaults.update(params)
    return XGBRegressor(**defaults)


def _make_mlp(params: Dict[str, Any]) -> BaseEstimator:
    from sklearn.neural_network import MLPRegressor

    defaults = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "max_iter": 300,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
    }
    defaults.update(params)
    return MLPRegressor(**defaults)


_FACTORY_FUNCS = {
    "random_forest": _make_random_forest,
    "xgboost": _make_xgboost,
    "mlp": _make_mlp,
}


def create_model(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """Create a model by name.

    Parameters
    ----------
    name : str
        One of ``mean``, ``random_forest``, ``xgboost``, ``mlp``.
    params : dict or None
        Hyperparameters to override defaults.

    Returns
    -------
    sklearn-compatible estimator with ``fit`` and ``predict``.
    """
    params = params or {}

    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name](**params)

    if name in _FACTORY_FUNCS:
        return _FACTORY_FUNCS[name](params)

    available = sorted(set(list(_MODEL_REGISTRY) + list(_FACTORY_FUNCS)))
    raise ValueError(f"Unknown model '{name}'. Available: {available}")


def list_models():
    """Return names of all registered models."""
    return sorted(set(list(_MODEL_REGISTRY) + list(_FACTORY_FUNCS)))
