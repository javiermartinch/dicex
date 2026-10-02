# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Integration tests for the high-level classification explainer."""

from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from dicex import Dicex, GaussianPerturbation


def test_dicex_explainer_classification_end_to_end() -> None:
    """Verify the fit/explain pipeline for a simple classification model."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((700, 2))
    logits = 3.0 * x_train[:, 0] + 0.2 * x_train[:, 1]
    y_train = (logits > 0.0).astype(int)

    estimator = cast("Any", LogisticRegression(max_iter=2000))
    model = estimator.fit(x_train, y_train)
    explainer = Dicex(
        model,
        task="classification",
        target_class=1,
        perturbation=GaussianPerturbation(mu=0.1, sigma=0.01),
        alpha=1.0,
        n_exploration=200,
        kappas=[1.0, 10.0, 100.0],
        seed=42,
    ).fit(x_train)

    result = explainer.explain(np.array([0.0, 0.0]))

    assert result.direction.shape == (2,)
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.direction[0] > 0.0
    assert abs(result.direction[0]) > abs(result.direction[1])
    assert "direction_scaled" in result.metadata
