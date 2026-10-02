# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Integration tests for the DiCEx facade."""

from typing import Any, cast

import numpy as np
from sklearn.linear_model import Ridge

from dicex import Dicex, GaussianPerturbation


def test_dicex_facade_end_to_end() -> None:
    """Verify the full DiCEx flow using a real sklearn model."""
    # Training data for f(x1, x2) = x1 + 0.1 * x2, where x1 is the dominant feature.
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((500, 2))
    y_train = 1.0 * x_train[:, 0] + 0.1 * x_train[:, 1]

    estimator = cast("Any", Ridge())
    model = estimator.fit(x_train, y_train)

    # A small perturbation scale makes the optimal direction close to the gradient.
    perturbation = GaussianPerturbation(mu=0.1, sigma=0.01)

    explainer = Dicex(
        model,
        task="regression",
        perturbation=perturbation,
        alpha=1.0,
        n_exploration=200,
        n_test=200,
        n_evaluation=150,
        n_bootstrap=128,
        kappas=[1.0, 10.0, 100.0],
        trace_level="summary",
        seed=42,
    )
    explainer.fit(x_train)

    # Explain a single point.
    x0 = np.array([0.0, 0.0])
    result = explainer.explain(x0)

    # The best direction to increase f(x) is close to [1, 0].
    assert result.direction.shape == (2,)
    assert np.isclose(np.linalg.norm(result.direction), 1.0)

    # The first coordinate dominates the second.
    assert abs(result.direction[0]) > abs(result.direction[1])
    assert result.direction[0] > 0

    # Metadata.
    assert "n_model_evals" in result.metadata
    assert "direction_scaled" in result.metadata
    assert result.metadata["n_test"] == 200
    assert result.metadata["n_evaluation"] == 150
    assert result.metadata["n_bootstrap"] == 128
    assert result.metadata["trace_level"] == "summary"
    assert result.metadata["phase_eval_counts"]["baseline_evaluation"] > 0
    assert "anytime_trace" in result.metadata
