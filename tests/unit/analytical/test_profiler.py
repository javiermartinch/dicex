# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the unified variance profiler module."""

import numpy as np

from dicex.analytical.profiler import build_variance_profiler


def test_profiler_isotropic_boundary() -> None:
    """Verify profiler at boundary t = ||beta||."""
    beta = np.array([3.0, 4.0])
    beta_norm = float(np.linalg.norm(beta))
    profiler = build_variance_profiler(beta)

    smin, smax, cmin, cmax = profiler(beta_norm)

    expected_w = (3.0**4 + 4.0**4) / (3.0**2 + 4.0**2)
    assert np.isclose(smin, expected_w)
    assert np.isclose(smax, expected_w)
    assert np.allclose(cmin, beta / beta_norm)
    assert np.allclose(cmax, beta / beta_norm)


def test_profiler_extrema_ordering_and_constraints() -> None:
    """Verify that smin <= smax and constraints hold for intermediate t."""
    beta = np.array([2.0, -1.0, 0.5])
    beta_norm = float(np.linalg.norm(beta))
    profiler = build_variance_profiler(beta)

    for t in np.linspace(-0.8 * beta_norm, 0.8 * beta_norm, 15):
        smin, smax, cmin, cmax = profiler(float(t))
        assert smin <= smax + 1e-10
        # Check projection constraint beta^T c = t
        assert np.isclose(np.dot(beta, cmin), t, atol=1e-4)
        assert np.isclose(np.dot(beta, cmax), t, atol=1e-4)
        # Check unit norm constraint ||c|| = 1
        assert np.isclose(np.linalg.norm(cmin), 1.0, atol=1e-4)
        assert np.isclose(np.linalg.norm(cmax), 1.0, atol=1e-4)


def test_profiler_zero_coefficient() -> None:
    """Verify profiler behavior when beta contains zero coordinates."""
    beta = np.array([1.5, 0.0, -2.0])
    _beta_norm = float(np.linalg.norm(beta))
    profiler = build_variance_profiler(beta)

    smin, smax, cmin, cmax = profiler(0.0)
    assert smin <= smax
    assert np.isclose(np.linalg.norm(cmin), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(cmax), 1.0, atol=1e-4)
