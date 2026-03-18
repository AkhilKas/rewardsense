"""
Optuna-based hyperparameter tuning for the personalization model.

Tuning flow:
1. Define search space from config or defaults
2. Run N trials with cross-validation on the training set
3. Log each trial to MLflow
4. Return the best hyperparameters
5. Retrain final model with best params on full training set
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import cross_val_score

from model_pipeline.personalization.models import create_model

# ── Default search space for XGBoost ──────────────────────────────────

DEFAULT_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 0.0, "high": 1.0},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
}


def _sample_params(trial, search_space: Dict[str, Dict]) -> Dict[str, Any]:
    """Use an Optuna trial to sample hyperparameters from the search space."""
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            step = spec.get("step", 1)
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=step)
        elif ptype == "float":
            if spec.get("log", False):
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=True
                )
            else:
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


class HyperparameterTuner:
    """Optuna-powered tuner for the personalization model.

    Parameters
    ----------
    model_name : str
        Model to tune (typically ``"xgboost"``).
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    search_space : dict or None
        Override default search space. Keys are param names, values are
        dicts with ``type``, ``low``, ``high``, etc.
    n_trials : int
        Number of Optuna trials.
    cv_folds : int
        Cross-validation folds.
    metric : str
        Scoring metric (``"rmse"``).
    random_seed : int
        Reproducibility seed.
    use_mlflow : bool
        Log trials to MLflow.
    experiment_name : str
        MLflow experiment name.
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        search_space: Optional[Dict[str, Dict]] = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        metric: str = "rmse",
        random_seed: int = 42,
        use_mlflow: bool = True,
        experiment_name: str = "personalization-tuning",
    ) -> None:
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_seed = random_seed
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.study: Optional[Any] = None

    def _objective(self, trial) -> float:
        """Optuna objective: CV-RMSE for sampled hyperparams."""
        params = _sample_params(trial, self.search_space)
        params["random_state"] = self.random_seed

        model = create_model(self.model_name, params)

        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=self.cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse = float(-scores.mean())

        if self.use_mlflow:
            self._log_trial(trial.number, params, cv_rmse)

        return cv_rmse

    def tune(self) -> Dict[str, Any]:
        """Run the full tuning loop. Returns best hyperparams.

        Returns
        -------
        dict
            Best hyperparameters found by Optuna.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"personalization_{self.model_name}",
        )

        logger.info(
            "Starting Optuna tuning: {} trials, {}-fold CV",
            self.n_trials,
            self.cv_folds,
        )

        self.study.optimize(
            self._objective, n_trials=self.n_trials, show_progress_bar=False
        )

        self.best_params = self.study.best_params
        self.best_params["random_state"] = self.random_seed
        self.best_score = self.study.best_value

        logger.info(
            "Tuning complete — best CV-RMSE: {:.6f}, params: {}",
            self.best_score,
            self.best_params,
        )
        return self.best_params

    def retrain_best(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ):
        """Retrain the model with best params on the full training set.

        Returns the fitted model.
        """
        if self.best_params is None:
            raise RuntimeError("Must call tune() before retrain_best()")

        X = X_train if X_train is not None else self.X_train
        y = y_train if y_train is not None else self.y_train

        model = create_model(self.model_name, self.best_params)
        model.fit(X, y)
        logger.info(
            "Retrained {} with best params on {} samples", self.model_name, len(X)
        )
        return model

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Return the Optuna study trials as a DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        return self.study.trials_dataframe()

    def _log_trial(self, trial_number: int, params: Dict, cv_rmse: float) -> None:
        """Log a single trial to MLflow (nested run)."""
        try:
            import mlflow

            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=f"trial_{trial_number}", nested=True):
                safe_params = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                mlflow.log_params(safe_params)
                mlflow.log_metric("cv_rmse", cv_rmse)
        except Exception as exc:
            logger.debug("MLflow trial logging failed: {}", exc)
