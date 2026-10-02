# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for SLERP and geodesic refinement."""

import numpy as np

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.scoring import ModelScorer
from dicex.optim.slerp import geodesic_refinement, slerp


def test_slerp_endpoints() -> None:
    """Verify that slerp(0) = m and slerp(1) = u."""
    m = np.array([1.0, 0.0, 0.0])
    u = np.array([0.0, 1.0, 0.0])

    assert np.allclose(slerp(m, u, 0.0), m)
    assert np.allclose(slerp(m, u, 1.0), u)


def test_slerp_unit_norm() -> None:
    """Verify that slerp outputs always have unit norm."""
    m = np.array([1.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 1.0])
    ts = np.linspace(0, 1, 10)

    for t in ts:
        res = slerp(m, u, t)
        assert np.isclose(np.linalg.norm(res), 1.0, atol=1e-10)


def test_slerp_vectorized() -> None:
    """Verify that slerp works with an array of t."""
    m = np.array([1.0, 0.0])
    u = np.array([0.0, 1.0])
    ts = np.array([0.0, 0.5, 1.0])

    res = slerp(m, u, ts)
    assert res.shape == (3, 2)
    assert np.allclose(res[0], m)
    assert np.allclose(res[-1], u)
    assert np.isclose(np.linalg.norm(res[1]), 1.0)


def test_slerp_degenerate_small_angle() -> None:
    """Verify stability when m and u are very close."""
    m = np.array([1.0, 0.0])
    u = np.array([1.0, 1e-8])
    u = u / np.linalg.norm(u)

    res = slerp(m, u, 0.5)
    assert np.isclose(np.linalg.norm(res), 1.0)
    assert not np.any(np.isnan(res))


def test_slerp_antipodal_vectors() -> None:
    """Verify SLERP remains finite for antipodal endpoints."""
    m = np.array([1.0, 0.0])
    u = np.array([-1.0, 0.0])

    res = slerp(m, u, 0.5)

    assert np.isclose(np.linalg.norm(res), 1.0)
    assert np.isclose(np.dot(res, m), 0.0, atol=1e-10)
    assert not np.any(np.isnan(res))


def test_geodesic_refinement_improvement() -> None:
    """Verify that geodesic refinement finds a direction at least as good as m."""

    class MockModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            # f(x) = x[0] + x[1] - simple linear model
            return np.sum(x, axis=1)

    mock_model = MockModel()
    x0 = np.array([0.0, 0.0])
    m = np.array([1.0, 0.0])
    u_tilde = np.array([0.0, 1.0])
    # For a linear model, the best direction between (1,0) and (0,1) is (1/sqrt(2), 1/sqrt(2))

    noise_samples = np.array([[0.1, 0.0], [-0.1, 0.0]])  # Very simple noise
    alpha = 0.5
    grid_t = np.linspace(0, 1, 5)

    scorer = ModelScorer(mock_model)
    best_dir, best_val = geodesic_refinement(scorer, x0, m, u_tilde, noise_samples, alpha, grid_t)

    # Evaluate at m for comparison
    y0 = mock_model.predict(x0.reshape(1, -1))[0]
    m_improvements = mock_model.predict(x0 + noise_samples @ np.diag(m)) - y0
    m_val = lower_tail_cvar_empirical(m_improvements, alpha)

    assert best_val >= m_val
    assert np.isclose(np.linalg.norm(best_dir), 1.0)


def test_geodesic_refinement_classification() -> None:
    """Verify that geodesic refinement works with a classifier and target_class."""

    class MockClassifier:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # Three classes; the probability of class 1 increases with the sum of the coordinates.
            n = x.shape[0]
            probs = np.zeros((n, 3))
            scores = np.sum(x, axis=1)
            probs[:, 1] = np.clip(0.5 + scores, 0, 1)
            probs[:, 0] = (1.0 - probs[:, 1]) / 2.0
            probs[:, 2] = (1.0 - probs[:, 1]) / 2.0
            return probs

    mock_cls = MockClassifier()
    x0 = np.array([0.0, 0.0])
    m = np.array([1.0, 0.0])
    u_tilde = np.array([0.0, 1.0])
    noise_samples = np.array([[0.1, 0.1]])

    # Target class 1
    scorer = ModelScorer(mock_cls, target_class=1)
    best_dir, best_val = geodesic_refinement(scorer, x0, m, u_tilde, noise_samples, 1.0, np.linspace(0, 1, 5))

    assert np.isclose(np.linalg.norm(best_dir), 1.0)
    assert best_val > 0.1  # Baseline prob 0.5, best improvement is sqrt(2)*0.1 approx 0.1414


def test_slerp_binary_scalar() -> None:
    """Verify _binary_slerp with scalar t for 1D vectors (line 33)."""
    m = np.array([1.0])
    u = np.array([-1.0])
    # t < 0.5 → returns m
    assert np.allclose(slerp(m, u, 0.3), m)
    # t >= 0.5 → returns u
    assert np.allclose(slerp(m, u, 0.5), u)
    assert np.allclose(slerp(m, u, 0.7), u)


def test_slerp_antipodal_vectorized() -> None:
    """Verify _antipodal_slerp with array t (lines 43-45)."""
    m = np.array([1.0, 0.0])
    u = np.array([-1.0, 0.0])
    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = slerp(m, u, ts)
    assert res.shape == (5, 2)
    assert np.allclose(res[0], m)
    assert np.allclose(res[-1], u)
    for r in res:
        assert np.isclose(np.linalg.norm(r), 1.0)


def test_geodesic_refinement_1d_noise() -> None:
    """Verify geodesic_refinement with 1D noise_samples (line 134 else branch)."""

    class MockModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=1)

    mock_model = MockModel()
    x0 = np.array([0.0, 0.0])
    m = np.array([1.0, 0.0])
    u_tilde = np.array([0.0, 1.0])
    noise_samples = np.array([0.1, -0.1])  # 1D, shape (2,)
    alpha = 0.5
    grid_t = np.linspace(0, 1, 5)

    scorer = ModelScorer(mock_model)
    best_dir, best_val = geodesic_refinement(scorer, x0, m, u_tilde, noise_samples, alpha, grid_t)
    assert np.isclose(np.linalg.norm(best_dir), 1.0)
    assert np.isfinite(best_val)
