# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Shared type aliases, constants, and noise/risk configuration for the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DimensionBucket = Literal["low", "medium", "high"]
TaskType = Literal["regression", "classification"]
ModelFamily = Literal["linear", "logistic", "random_forest", "xgboost", "mlp"]

DIMENSION_BUCKET_ORDER: tuple[DimensionBucket, ...] = ("low", "medium", "high")
# Query percentiles used in the experiments.
DEFAULT_QUERY_PERCENTILES: tuple[float, ...] = (0.10, 0.50)

REGRESSION_MODEL_FAMILIES: tuple[ModelFamily, ...] = ("linear", "xgboost", "random_forest", "mlp")
CLASSIFICATION_MODEL_FAMILIES: tuple[ModelFamily, ...] = ("logistic", "xgboost", "random_forest", "mlp")


@dataclass(frozen=True)
class NoiseRiskConfig:
    """One fixed physical-noise and risk configuration.

    Attributes:
        mu (float): Perturbation step size.
        sigma (float): Standard deviation of action noise.
        env_std (float): Standard deviation of environmental noise.
        alpha (float): Confidence level for CVaR (risk level).
    """

    mu: float
    sigma: float
    env_std: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        """Return a flat serialization-friendly payload.

        Returns:
            dict[str, float]: Dictionary representation of the noise configuration parameters.
        """
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "env_std": self.env_std,
            "alpha": self.alpha,
        }


def model_families_for_task(task: TaskType) -> tuple[ModelFamily, ...]:
    """Return the configured model families for one task type.

    Args:
        task (TaskType): The task type ('regression' or 'classification').

    Returns:
        tuple[ModelFamily, ...]: Tuple of ModelFamily names.
    """
    return REGRESSION_MODEL_FAMILIES if task == "regression" else CLASSIFICATION_MODEL_FAMILIES
