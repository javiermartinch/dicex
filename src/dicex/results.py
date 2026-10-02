# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Explanation result dataclass."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _default_metadata() -> dict[str, Any]:
    """Create an empty metadata dictionary.

    Returns:
        dict[str, Any]: Empty metadata dictionary.
    """
    return {}


@dataclass
class ExplanationResult:
    """Result of a DiCEx explanation.

    Attributes:
        direction (np.ndarray): Recommended direction of change, a unit vector in the
            original feature space. The zero vector means that no direction reliably
            beats not acting.
        robust_value (float): Lower-tail CVaR, at level ``alpha``, of the gain in the
            prediction when moving along the direction with a random step.
        alpha (float): Risk level used.
        metadata (dict[str, Any]): Details of the search, among them ``beats_baseline``
            (whether acting beats not acting), ``baseline_cvar`` (the value of not
            acting), ``direction_scaled`` (the direction in standardized units, when a
            scaler is used), ``n_model_evals``, ``stop_reason`` and ``wall_time_seconds``.
    """

    direction: np.ndarray
    robust_value: float
    alpha: float
    metadata: dict[str, Any] = field(default_factory=_default_metadata)
