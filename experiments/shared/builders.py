# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Factories for the unfitted regressors and classifiers used in the experiments."""

from __future__ import annotations

from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def make_regressor(
    model_family: str,
    *,
    seed: int,
) -> Any:
    """Build a regressor instance for the specified model family.

    Args:
        model_family (str): Name of the model family ('linear', 'random_forest', 'mlp', or
            'xgboost').
        seed (int): Random state seed for reproducibility.

    Returns:
        Any: An un-fitted scikit-learn compatible regressor object.

    Raises:
        ValueError: If model_family is not supported.
    """
    if model_family == "linear":
        return LinearRegression()
    if model_family == "random_forest":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=1,
        )
    if model_family == "mlp":
        reg = MLPRegressor(random_state=seed)

        return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    if model_family == "xgboost":
        return XGBRegressor(
            random_state=seed,
            n_jobs=1,
        )
    msg = f"Unsupported regression model family '{model_family}'."
    raise ValueError(msg)


def make_classifier(
    model_family: str,
    *,
    seed: int,
) -> Any:
    """Build a classifier instance for the specified model family.

    Args:
        model_family (str): Name of the model family ('logistic', 'random_forest', 'mlp',
            or 'xgboost').
        seed (int): Random state seed for reproducibility.

    Returns:
        Any: An un-fitted scikit-learn compatible classifier object.

    Raises:
        ValueError: If model_family is not supported.
    """
    if model_family == "logistic":
        return LogisticRegression(max_iter=5000, random_state=seed)
    if model_family == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
        )
    if model_family == "mlp":
        return MLPClassifier(random_state=seed)
    if model_family == "xgboost":
        return XGBClassifier(
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
        )
    msg = f"Unsupported classification model family '{model_family}'."
    raise ValueError(msg)
