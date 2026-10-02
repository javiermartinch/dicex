# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Preset hyperparameter configurations for DiCEx optimizers."""

from typing import Any

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "low": {
        "low": {
            "n_exploration": 500,
            "b_samples": 250,
            "eta": 0.05,
            "grid_t_len": 20,
            "n_test": 1000,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "mid": {
            "n_exploration": 700,
            "b_samples": 150,
            "eta": 0.05,
            "grid_t_len": 40,
            "n_test": 2100,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "high": {
            "n_exploration": 1000,
            "b_samples": 150,
            "eta": 0.05,
            "grid_t_len": 100,
            "n_test": 2500,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
    },
    "medium": {
        "low": {
            "n_exploration": 500,
            "b_samples": 250,
            "eta": 0.05,
            "grid_t_len": 20,
            "n_test": 1000,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "mid": {
            "n_exploration": 800,
            "b_samples": 100,
            "eta": 0.05,
            "grid_t_len": 80,
            "n_test": 2400,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "high": {
            "n_exploration": 1800,
            "b_samples": 350,
            "eta": 0.05,
            "grid_t_len": 100,
            "n_test": 2500,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
    },
    "high": {
        "low": {
            "n_exploration": 500,
            "b_samples": 250,
            "eta": 0.05,
            "grid_t_len": 20,
            "n_test": 1000,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "mid": {
            "n_exploration": 900,
            "b_samples": 350,
            "eta": 0.05,
            "grid_t_len": 60,
            "n_test": 1800,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
        "high": {
            "n_exploration": 1400,
            "b_samples": 400,
            "eta": 0.05,
            "grid_t_len": 40,
            "n_test": 2500,
            "n_evaluation": 5000,
            "n_bootstrap": 2500,
        },
    },
}
