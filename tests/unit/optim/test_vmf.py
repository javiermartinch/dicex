# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for von Mises-Fisher sampling and utilities."""

import numpy as np
import pytest

from dicex.optim.vmf import (
    angular_radius,
    kappa_from_angular_radius,
    sample_uniform_sphere,
    sample_vmf,
    vmf_A_d,
)


def test_sample_uniform_sphere_unit_norm() -> None:
    """Verify that samples from uniform sphere have unit norm."""
    d, n = 10, 1000
    samples = sample_uniform_sphere(d, n)
    norms = np.linalg.norm(samples, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_sample_vmf_unit_norm() -> None:
    """Verify that samples from vMF have unit norm."""
    mu = np.array([1.0, 0.0, 0.0])
    kappa = 10.0
    n = 500
    samples = sample_vmf(mu, kappa, n)
    norms = np.linalg.norm(samples, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_vmf_empirical_mean() -> None:
    """Verify that the empirical mean matches the theoretical mean direction and magnitude."""
    mu = np.array([0.0, 1.0, 0.0])
    kappa = 50.0
    d = 3
    n = 100_000  # High sample size for low variance

    samples = sample_vmf(mu, kappa, n, rng=np.random.default_rng(42))
    empirical_mean_vec = np.mean(samples, axis=0)

    # Direction should match mu
    direction = empirical_mean_vec / np.linalg.norm(empirical_mean_vec)
    assert np.allclose(direction, mu, atol=0.05)

    # Magnitude should match A_d(kappa)
    expected_magnitude = vmf_A_d(kappa, d)
    assert np.isclose(np.linalg.norm(empirical_mean_vec), expected_magnitude, rtol=0.05)


def test_kappa_rho_roundtrip() -> None:
    """Verify that kappa -> rho -> kappa is a round-trip."""
    d = 5
    kappa_orig = 25.0
    rho = angular_radius(kappa_orig, d)
    kappa_new = kappa_from_angular_radius(rho, d)

    assert np.isclose(kappa_orig, kappa_new, rtol=1e-4)


def test_vmf_kappa_zero_uniformity() -> None:
    """Verify that kappa=0 yields a uniform distribution (empirical mean approx 0)."""
    mu = np.array([1.0, 0.0])
    kappa = 0.0
    n = 50_000

    samples = sample_vmf(mu, kappa, n, rng=np.random.default_rng(42))
    empirical_mean_vec = np.mean(samples, axis=0)

    # For a uniform distribution on the circle, the mean should be zero
    assert np.allclose(empirical_mean_vec, 0.0, atol=0.02)


def test_sample_vmf_dimension_one() -> None:
    """Verify the 1D vMF sampler returns points on S^0 without crashing."""
    mu = np.array([1.0])
    samples = sample_vmf(mu, 10.0, 10_000, rng=np.random.default_rng(42))

    assert samples.shape == (10_000, 1)
    assert np.all(np.isin(samples[:, 0], [-1.0, 1.0]))
    assert np.mean(samples[:, 0]) > 0.95


@pytest.mark.parametrize("d", [2, 3, 5, 10])
@pytest.mark.parametrize("kappa", [0.1, 1.0, 10.0, 100.0])
def test_vmf_A_d_monotonicity(d: int, kappa: float) -> None:  # noqa: N802
    """A_d(kappa) should be between 0 and 1 and increasing with kappa."""
    a1 = vmf_A_d(kappa, d)
    a2 = vmf_A_d(kappa * 2, d)

    assert 0.0 < a1 < 1.0
    assert a2 > a1


def test_angular_radius_range() -> None:
    """Angular radius should be in [0, pi/2]."""
    assert np.isclose(angular_radius(1e6, 3), 0.0, atol=2e-3)
    assert angular_radius(1e-6, 3) <= np.pi / 2.0


def test_kappa_from_angular_radius_large_rho() -> None:
    """kappa_from_angular_radius returns 0 for rho >= pi/2."""
    assert kappa_from_angular_radius(np.pi / 2.0, 3) == 0.0
    assert kappa_from_angular_radius(np.pi, 3) == 0.0


def test_kappa_from_angular_radius_brentq_failure() -> None:
    """kappa_from_angular_radius returns 0 when brentq cannot find a root."""
    # rho=0 gives target_a_d=1.0, which A_d(kappa) approaches asymptotically
    # but never reaches for finite kappa, causing brentq to fail.
    result = kappa_from_angular_radius(0.0, 2)
    assert result == 0.0


def test_sample_vmf_zero_mu_error() -> None:
    """sample_vmf raises ValueError for zero mu vector."""
    with pytest.raises(ValueError, match="non-zero"):
        sample_vmf(np.array([0.0, 0.0, 0.0]), 10.0, 10)
