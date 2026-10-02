# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for bootstrap acceptance testing."""

import numpy as np
import pytest

from dicex.optim.bootstrap import acceptance_test, paired_percentile_bootstrap_lci


def test_bootstrap_positive_difference() -> None:
    """Verify that a clearly better direction is accepted."""
    rng = np.random.default_rng(42)
    # z_c is clearly better than z_cprime
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime + 0.5

    # With eta=0.05 (95% confidence), a shift of 0.5 should be very significant
    accepted = acceptance_test(z_c, z_cprime, alpha=1.0, eta=0.05, n_bootstrap=500, rng=rng)
    assert accepted is True


def test_bootstrap_null_difference() -> None:
    """Verify that identical distributions are rejected."""
    rng = np.random.default_rng(42)
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime.copy()  # Identical

    # Difference is exactly zero, so LCI will be 0, and 0 > 0 is False
    accepted = acceptance_test(z_c, z_cprime, alpha=1.0, eta=0.05, n_bootstrap=500, rng=rng)
    assert accepted is False


def test_bootstrap_negative_difference() -> None:
    """Verify that a worse direction is rejected."""
    rng = np.random.default_rng(42)
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime - 0.2

    accepted = acceptance_test(z_c, z_cprime, alpha=1.0, eta=0.05, n_bootstrap=500, rng=rng)
    assert accepted is False


def test_bootstrap_determinism() -> None:
    """Verify that fixed seed yields same result."""
    rng = np.random.default_rng(7)
    z_cprime = rng.normal(0, 1, 50)
    z_c = z_cprime + 0.1

    lci1 = paired_percentile_bootstrap_lci(
        z_c, z_cprime, alpha=0.5, eta=0.1, n_bootstrap=100, rng=np.random.default_rng(42)
    )
    lci2 = paired_percentile_bootstrap_lci(
        z_c, z_cprime, alpha=0.5, eta=0.1, n_bootstrap=100, rng=np.random.default_rng(42)
    )

    assert lci1 == lci2


def test_bootstrap_mismatched_length() -> None:
    """Verify that different lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        paired_percentile_bootstrap_lci(np.zeros(10), np.zeros(5), 1.0, 0.05)


def test_paired_percentile_bootstrap_lci_no_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback when Rust extension is missing."""
    import dicex.optim.bootstrap  # noqa: PLC0415

    monkeypatch.setattr(dicex.optim.bootstrap, "HAS_RUST_EXT", False)

    rng = np.random.default_rng(42)
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime + 0.5

    lci, deltas = paired_percentile_bootstrap_lci(
        z_c, z_cprime, alpha=0.5, eta=0.05, n_bootstrap=100, rng=rng, return_deltas=True
    )
    assert lci > 0
    assert len(deltas) == 100


def test_paired_percentile_bootstrap_lci_no_rust_no_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Python fallback without return_deltas (covers line 122)."""
    import dicex.optim.bootstrap  # noqa: PLC0415

    monkeypatch.setattr(dicex.optim.bootstrap, "HAS_RUST_EXT", False)

    rng = np.random.default_rng(42)
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime + 0.5

    lci = paired_percentile_bootstrap_lci(z_c, z_cprime, alpha=0.5, eta=0.05, n_bootstrap=100, rng=rng)
    assert isinstance(lci, float)
    assert lci > 0


def test_acceptance_test_return_details() -> None:
    """Verify return_details flag."""
    rng = np.random.default_rng(42)
    z_cprime = rng.normal(0, 1, 100)
    z_c = z_cprime + 0.5

    accepted, lci, deltas = acceptance_test(
        z_c, z_cprime, alpha=0.5, eta=0.05, n_bootstrap=10, rng=rng, return_details=True
    )
    assert accepted is True
    assert lci > 0
    assert len(deltas) == 10
