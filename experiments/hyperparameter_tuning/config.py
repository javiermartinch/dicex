# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Configuration profiles for gold-standard generation and Optuna hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from experiments.shared.specs import NoiseRiskConfig

if TYPE_CHECKING:
    from experiments.shared.specs import DimensionBucket


@dataclass(frozen=True)
class GoldBudget:
    """Budget policy for gold standard generation in one dimension bucket."""

    min_runtime_seconds: float


@dataclass(frozen=True)
class GoldProfile:
    """Resolved configuration profile for gold standard generation."""

    query_percentiles: tuple[float, ...]
    parameter_grid: tuple[NoiseRiskConfig, ...]
    budgets_by_bucket: dict[DimensionBucket, GoldBudget]

    # Gold standard optimizer settings
    angular_schedule_deg: tuple[float, ...]
    eta: float
    grid_t_len: int
    b_samples: int
    n_exploration: int
    n_test: int
    n_evaluation: int
    n_bootstrap: int


@dataclass(frozen=True)
class OptunaProfile:
    """Resolved configuration profile for Optuna tuning."""

    query_percentiles: tuple[float, ...]
    trials_by_bucket: dict[DimensionBucket, int]
    timeouts_by_bucket: dict[DimensionBucket, float]

    # Optimizer settings held fixed during tuning
    n_evaluation: int
    n_bootstrap: int

    search_space: dict[str, Any]


def default_gold_profile() -> GoldProfile:
    """Return the default configuration profile for gold standard generation.

    Returns:
        GoldProfile: Default configuration profile for gold standard generation.
    """
    return GoldProfile(
        query_percentiles=(0.50,),
        parameter_grid=(
            NoiseRiskConfig(mu=0.25, sigma=0.25, env_std=0.05, alpha=0.25),
            NoiseRiskConfig(mu=0.25, sigma=0.25, env_std=0.05, alpha=0.75),
            NoiseRiskConfig(mu=0.25, sigma=0.35, env_std=0.07, alpha=0.25),
            NoiseRiskConfig(mu=0.25, sigma=0.35, env_std=0.07, alpha=0.75),
        ),
        budgets_by_bucket={
            "low": GoldBudget(min_runtime_seconds=240.0),
            "medium": GoldBudget(min_runtime_seconds=420.0),
            "high": GoldBudget(min_runtime_seconds=600.0),
        },
        angular_schedule_deg=(85.0, 70.0, 55.0, 35.0, 20.0, 10.0, 5.0, 1.0),
        eta=0.05,
        grid_t_len=201,
        b_samples=256,
        n_exploration=3000,
        n_test=5000,
        n_evaluation=20000,
        n_bootstrap=5000,
    )


def default_optuna_profile() -> OptunaProfile:
    """Return the default configuration profile for Optuna tuning.

    Returns:
        OptunaProfile: Default configuration profile for Optuna tuning.
    """
    return OptunaProfile(
        query_percentiles=(0.50,),
        trials_by_bucket={"low": 100, "medium": 100, "high": 100},
        timeouts_by_bucket={
            "low": 240.0,
            "medium": 420.0,
            "high": 600.0,
        },
        n_evaluation=5000,
        n_bootstrap=2500,
        search_space={
            "n_exploration": (500, 2000, 100),
            "test_multiplier": (1, 3, 1),
            "b_samples": (50, 400, 50),
            "eta": [0.05],
            "grid_t_len": (20, 100, 20),
        },
    )
