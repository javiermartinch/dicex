# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for the vMF-VNS optimizer."""

from contextlib import suppress
import time
import warnings

import numpy as np
import pytest

from dicex.distributions.gaussian import GaussianPerturbation
from dicex.exceptions import ConvergenceWarning
from dicex.optim.vmf_vns import VmfVns


def test_vmf_vns_linear_model() -> None:
    """Verify that VmfVns recovers the correct direction for a linear model."""
    # f(x) = w^T x. Best direction is w/||w||
    w = np.array([0.8, 0.6])  # Unit vector already

    class LinearModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return x @ w

    x0 = np.array([0.0, 0.0])
    perturbation = GaussianPerturbation(mu=0.5, sigma=0.1)

    # Optimizer settings
    kappas = [1.0, 10.0, 100.0]
    optimizer = VmfVns(
        model=LinearModel(),
        x0=x0,
        perturbation=perturbation,
        alpha=1.0,  # Mean improvement
        kappas=kappas,
        b_samples=10,
        n_exploration=100,
        n_test=100,
        n_evaluation=100,
        eta=0.05,
        grid_t=np.linspace(0, 1, 5),
        seed=42,
    )

    result = optimizer.run()

    # Check direction: dot product should be close to 1
    dot_product = np.dot(result.direction, w)
    assert dot_product > 0.9  # Angular tolerance
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert "n_model_evals" in result.metadata
    assert result.metadata["n_model_evals"] > 0
    assert result.metadata["n_iterations"] > 0


def test_vmf_vns_logistic_smoke() -> None:
    """Smoke test with a mock logistic-like model."""

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    class LogisticModel:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # 2 classes
            p1 = sigmoid(x[:, 0] + x[:, 1])
            return np.column_stack([1 - p1, p1])

    x0 = np.array([0.0, 0.0])
    perturbation = GaussianPerturbation(mu=0.5, sigma=0.1)

    optimizer = VmfVns(
        model=LogisticModel(),
        x0=x0,
        perturbation=perturbation,
        alpha=0.5,
        kappas=[10.0],
        b_samples=5,
        n_exploration=50,
        n_test=50,
        n_evaluation=50,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        target_class=1,
        seed=42,
    )

    result = optimizer.run()
    assert len(result.direction) == 2
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.robust_value > -10.0


def test_vmf_vns_one_dimensional_linear_smoke() -> None:
    """Verify the optimizer remains stable in one dimension."""

    class LinearModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return x[:, 0]

    optimizer = VmfVns(
        model=LinearModel(),
        x0=np.array([0.0]),
        perturbation=GaussianPerturbation(mu=0.5, sigma=0.1),
        alpha=1.0,
        kappas=[1.0, 10.0],
        b_samples=5,
        n_exploration=50,
        n_test=50,
        n_evaluation=50,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        seed=42,
    )

    with pytest.warns(ConvergenceWarning), suppress(ConvergenceWarning):
        result = optimizer.run()

    # Rerun with the convergence warning suppressed to inspect the result.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = optimizer.run()
    assert result.direction.shape == (1,)
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.direction[0] in (1.0, -1.0)


def test_vmf_vns_summary_trace_and_phase_metadata() -> None:
    """Verify summary traces and phase metrics are exposed in metadata."""

    class LinearModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return x[:, 0] + 0.25 * x[:, 1]

    optimizer = VmfVns(
        model=LinearModel(),
        x0=np.array([0.0, 0.0]),
        perturbation=GaussianPerturbation(mu=0.5, sigma=0.1),
        alpha=1.0,
        kappas=[1.0, 10.0],
        b_samples=5,
        n_exploration=40,
        n_test=30,
        n_evaluation=45,
        n_bootstrap=32,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        trace_level="summary",
        seed=42,
    )

    result = optimizer.run()
    metadata = result.metadata

    assert metadata["n_test"] == 30
    assert metadata["n_evaluation"] == 45
    assert metadata["n_bootstrap"] == 32
    assert metadata["trace_level"] == "summary"
    assert metadata["stop_reason"] in {"completed_schedule", "no_improvement", "restart_limit"}
    assert metadata["phase_eval_counts"]["shake"] > 0
    assert metadata["phase_eval_counts"]["confirm"] > 0
    assert metadata["phase_eval_counts"]["final_evaluation"] > 0
    assert metadata["phase_wall_times"]["confirm"] >= 0.0
    assert metadata["anytime_trace"]
    assert "detailed_trajectory" not in metadata


def test_vmf_vns_full_trace_and_restart_limit() -> None:
    """Verify full traces and controlled restarts are exposed in metadata."""

    class ConstantModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    optimizer = VmfVns(
        model=ConstantModel(),
        x0=np.array([0.0, 0.0]),
        perturbation=GaussianPerturbation(mu=0.5, sigma=0.1),
        alpha=1.0,
        kappas=[1.0],
        b_samples=3,
        n_exploration=20,
        n_test=20,
        n_evaluation=20,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        n_restarts=1,
        trace_level="full",
        seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = optimizer.run()

    metadata = result.metadata
    assert metadata["trace_level"] == "full"
    assert metadata["n_restarts_performed"] == 1
    assert metadata["stop_reason"] == "restart_limit"
    assert metadata["no_move"] is True
    assert metadata["anytime_trace"]
    assert metadata["detailed_trajectory"]


def test_vmf_vns_early_stopping_max_evals_and_timeout() -> None:
    """Verify that optimizer stops correctly when reaching max_evals or timeout."""

    class SlowModel:
        def predict(self, x: np.ndarray) -> np.ndarray:

            time.sleep(0.01)
            return x[:, 0]

    optimizer_evals = VmfVns(
        model=SlowModel(),
        x0=np.array([0.0, 0.0]),
        perturbation=GaussianPerturbation(mu=0.5, sigma=0.1),
        alpha=1.0,
        kappas=[1.0, 10.0],
        b_samples=5,
        n_exploration=10,
        n_test=10,
        n_evaluation=10,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        max_evals=15,
        seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result_evals = optimizer_evals.run()

    assert result_evals.metadata["stop_reason"] == "max_evals"

    optimizer_timeout = VmfVns(
        model=SlowModel(),
        x0=np.array([0.0, 0.0]),
        perturbation=GaussianPerturbation(mu=0.5, sigma=0.1),
        alpha=1.0,
        kappas=[1.0, 10.0],
        b_samples=10,
        n_exploration=50,
        n_test=50,
        n_evaluation=10,
        eta=0.1,
        grid_t=np.linspace(0, 1, 3),
        timeout_seconds=0.1,
        seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result_timeout = optimizer_timeout.run()

    assert result_timeout.metadata["stop_reason"] == "timeout"
