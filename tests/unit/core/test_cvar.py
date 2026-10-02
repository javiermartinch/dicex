# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the empirical CVaR estimator and validations."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from dicex.core.cvar import lower_tail_cvar_empirical


def test_empirical_constant() -> None:
    """Test empirical CVaR of a constant sample is the constant."""
    samples = np.array([5.0, 5.0, 5.0, 5.0])
    alpha = 0.5
    cvar = lower_tail_cvar_empirical(samples, alpha)
    assert np.isclose(cvar, 5.0)


def test_empirical_subset() -> None:
    """Test empirical CVaR with sample [1,2,3,4,5] and alpha=0.4."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.4
    # ceil(0.4 * 5) = 2. Smallest two are 1 and 2. Mean is 1.5.
    cvar = lower_tail_cvar_empirical(samples, alpha)
    assert np.isclose(cvar, 1.5)


def test_empirical_alpha_one() -> None:
    """Test empirical CVaR with alpha=1.0 recovers the sample mean."""
    samples = np.array([10.0, -5.0, 3.0, 2.0, 0.0])
    alpha = 1.0
    cvar = lower_tail_cvar_empirical(samples, alpha)
    assert np.isclose(cvar, np.mean(samples))


def test_empirical_invalid_alpha() -> None:
    """Test empirical CVaR raises ValueError for invalid alpha."""
    samples = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Alpha must be strictly greater than 0"):
        lower_tail_cvar_empirical(samples, 0.0)
    with pytest.raises(ValueError, match="Alpha must be strictly greater than 0"):
        lower_tail_cvar_empirical(samples, 1.5)


def test_empirical_empty_samples() -> None:
    """Test empirical CVaR raises ValueError for empty samples."""
    samples = np.array([])
    with pytest.raises(ValueError, match="Samples array cannot be empty"):
        lower_tail_cvar_empirical(samples, 0.5)


def test_empirical_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test empirical CVaR uses Python fallback when Rust is disabled."""
    import dicex.core.cvar  # noqa: PLC0415

    monkeypatch.setattr(dicex.core.cvar, "HAS_RUST_CVAR", False)

    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.4
    cvar = lower_tail_cvar_empirical(samples, alpha)
    assert np.isclose(cvar, 1.5)


def test_empirical_no_rust_import() -> None:
    """Verify the pure-Python fallback when the Rust extension cannot be imported."""
    code = textwrap.dedent("""\
        import sys
        import types

        # Replace so 'from dicex._rust_ext import ...' raises ImportError
        sys.modules["dicex._rust_ext"] = types.ModuleType("dicex._rust_ext")

        # Fresh import forces the except path
        if "dicex.core.cvar" in sys.modules:
            del sys.modules["dicex.core.cvar"]
        if "dicex" in sys.modules:
            del sys.modules["dicex"]

        import dicex.core.cvar
        assert not dicex.core.cvar.HAS_RUST_CVAR

        # Function still works via Python fallback
        import numpy as np
        result = dicex.core.cvar.lower_tail_cvar_empirical(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.4
        )
        assert abs(result - 1.5) < 1e-12
    """)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(sys.prefix),
        check=True,
    )
    assert completed.returncode == 0, (
        f"Subprocess failed with rc={completed.returncode}\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
