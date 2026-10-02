# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for explainer options: dimension buckets, verbosity, batching, and scalers."""

import numpy as np

from dicex.explainer import Dicex
from dicex.preprocessing.scaling import DicexScaler


class _DummyRegressor:
    """Minimal regressor that returns row sums (fast to evaluate)."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)


# Parameters are written inline in each test because unpacking a shared dictionary
# of keyword arguments would discard the parameter types checked by pyright.


def test_explainer_dimension_buckets() -> None:
    """Verify that the problem dimension maps to the low, medium, and high preset buckets."""
    explainer = Dicex(model=_DummyRegressor(), task="regression", scaler=None)

    explainer._d = 5  # type: ignore[reportPrivateUsage]
    assert explainer._get_dimension_bucket() == "low"  # type: ignore[reportPrivateUsage]

    explainer._d = 30  # type: ignore[reportPrivateUsage]
    assert explainer._get_dimension_bucket() == "medium"  # type: ignore[reportPrivateUsage]

    explainer._d = 100  # type: ignore[reportPrivateUsage]
    assert explainer._get_dimension_bucket() == "high"  # type: ignore[reportPrivateUsage]


def test_explainer_verbose_config_paths() -> None:
    """Verify that every supported verbosity setting yields a valid explainer."""
    for v in (True, False, "info", "debug", "none"):
        Dicex(model=_DummyRegressor(), task="regression", scaler=None, verbose=v)


def test_explainer_d_sets_from_x0() -> None:
    """Verify that ``explain`` infers the dimension from ``x0`` when ``fit`` was not called."""
    explainer = Dicex(
        model=_DummyRegressor(),
        task="regression",
        seed=0,
        scaler=None,
        n_exploration=10,
        kappas=[1.0],
        alpha=1.0,
    )
    # Force _d to None so explain infers it from x0
    explainer._d = None  # type: ignore[reportPrivateUsage]
    assert explainer._is_fitted  # type: ignore[reportPrivateUsage]

    result = explainer.explain(np.array([0.0, 0.0, 0.0]))
    assert result.direction.shape == (3,)


def test_explainer_batch() -> None:
    """Verify that ``explain_batch`` returns one explanation per input row."""
    explainer = Dicex(
        model=_DummyRegressor(),
        task="regression",
        seed=0,
        scaler=None,
        n_exploration=10,
        kappas=[1.0],
        alpha=1.0,
    )
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    results = explainer.explain_batch(x)
    assert len(results) == 2
    for r in results:
        assert r.direction.shape == (2,)


def test_explainer_prefitted_scaler() -> None:
    """Verify that a pre-fitted ``DicexScaler`` marks the explainer as fitted."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((10, 2))

    scaler = DicexScaler()
    scaler.fit(x_train)

    explainer = Dicex(
        model=_DummyRegressor(),
        task="regression",
        seed=0,
        scaler=scaler,
        n_exploration=10,
        kappas=[1.0],
        alpha=1.0,
    )
    assert explainer._is_fitted  # type: ignore[reportPrivateUsage]

    result = explainer.explain(np.array([0.0, 0.0]))
    assert result.direction.shape == (2,)
