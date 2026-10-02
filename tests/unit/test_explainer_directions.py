# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests that the returned direction is the optimized direction expressed in original units."""

from typing import Literal

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from dicex import Dicex, GaussianPerturbation


@pytest.fixture(scope="module")
def anisotropic_linear_model() -> tuple[LinearRegression, np.ndarray]:
    """Linear model y = x1 + 0.01 x2 with feature standard deviations 1 and 100."""
    rng = np.random.default_rng(0)
    x = np.c_[rng.normal(0.0, 1.0, 2000), rng.normal(0.0, 100.0, 2000)]
    y = x[:, 0] + 0.01 * x[:, 1]
    model = LinearRegression()
    model.fit(x, y)  # pyright: ignore[reportUnknownMemberType]
    return model, x


def _direction(model: LinearRegression, x: np.ndarray, scaler: Literal["auto"] | None) -> np.ndarray:
    explainer = Dicex(
        model,
        task="regression",
        perturbation=GaussianPerturbation(mu=1.0, sigma=0.01),
        alpha=0.5,
        scaler=scaler,
        n_exploration=300,
        kappas=[1.0, 10.0, 100.0],
        seed=0,
        verbose=False,
    ).fit(x)
    c = np.asarray(explainer.explain(np.zeros(2)).direction, dtype=float)
    return c / np.linalg.norm(c)


def test_direction_without_scaler_follows_gradient(
    anisotropic_linear_model: tuple[LinearRegression, np.ndarray],
) -> None:
    """Moves of size T in original units: the optimal direction is the gradient (1, 0.01)."""
    model, x = anisotropic_linear_model
    expected = np.array([1.0, 0.01]) / np.linalg.norm([1.0, 0.01])
    assert np.degrees(np.arccos(np.clip(_direction(model, x, None) @ expected, -1, 1))) < 2.0


def test_direction_with_scaler_is_mapped_to_original_units(
    anisotropic_linear_model: tuple[LinearRegression, np.ndarray],
) -> None:
    """Moves of T standard deviations: optimum (1, 1)/sqrt(2) in scaled space, i.e. ~(0.01, 1) in original units."""
    model, x = anisotropic_linear_model
    std = x.std(axis=0)
    expected = std * np.array([1.0, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.degrees(np.arccos(np.clip(_direction(model, x, "auto") @ expected, -1, 1))) < 2.0
