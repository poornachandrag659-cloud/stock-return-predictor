from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .features import FEATURE_COLUMNS

def make_model(name: str = "random_forest", seed: int = 42):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, max_depth=5, min_samples_leaf=15,
            max_features=0.8, n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=400, max_depth=5, min_samples_leaf=15,
            max_features=0.8, n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    if name == "baseline":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model {name!r}; choose random_forest, extra_trees, hist_gradient_boosting, baseline.")

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model_name: str, seed: int = 42):
    model = make_model(model_name, seed)
    model.fit(train[FEATURE_COLUMNS], train["target_return"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    return model, predictions

def prediction_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    direction = np.mean((actual > 0) == (predicted > 0))
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
            "directional_accuracy": float(direction),
            "actual_mean_return": float(np.mean(actual)),
            "predicted_mean_return": float(np.mean(predicted))}
