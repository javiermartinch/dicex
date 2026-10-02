# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the unified model scoring logic."""

import numpy as np
import pytest

from dicex.core.scoring import ModelScorer


class DummyRegressor:
    """Regressor that returns row sums."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)


class DummyClassifier:
    """Binary classifier whose positive-class probability is the clipped row sum divided by 10."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(np.sum(x, axis=1) / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def test_model_scorer_regressor_no_track() -> None:
    """Verify regressor scoring and the ``predict`` alias without evaluation tracking."""
    scorer = ModelScorer(DummyRegressor(), track=False)
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores = scorer.score(pts)
    assert np.allclose(scores, [3.0, 7.0])
    assert scorer.n_evals == 0

    # The predict alias returns the same scores.
    alias_scores = scorer.predict(pts)
    assert np.allclose(alias_scores, scores)


def test_model_scorer_classifier_no_track() -> None:
    """Verify target-class scoring for classifiers without evaluation tracking."""
    scorer = ModelScorer(DummyClassifier(), target_class=1, track=False)
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores = scorer.score(pts)
    assert np.allclose(scores, [0.3, 0.7])

    # predict_proba with track=False does not increment n_evals.
    probs = scorer.predict_proba(pts)
    assert probs.shape == (2, 2)
    assert scorer.n_evals == 0


def test_model_scorer_tracking_and_phases() -> None:
    """Verify that tracked evaluations are counted globally and per phase."""
    scorer = ModelScorer(DummyRegressor(), track=True)
    scorer._current_phase = "init"  # pyright: ignore[reportPrivateUsage]

    pts1 = np.array([[1.0]])
    scorer.score(pts1)
    assert scorer.n_evals == 1
    assert scorer.phase_eval_counts["init"] == 1

    scorer._current_phase = "search"  # pyright: ignore[reportPrivateUsage]
    pts2 = np.array([[2.0], [3.0]])
    scorer.score(pts2)
    assert scorer.n_evals == 3
    assert scorer.phase_eval_counts["init"] == 1
    assert scorer.phase_eval_counts["search"] == 2

    # predict_proba calls are tracked as well.
    cls_scorer = ModelScorer(DummyClassifier(), target_class=1, track=True)
    cls_scorer._current_phase = "test"  # pyright: ignore[reportPrivateUsage]
    cls_scorer.predict_proba(pts2)
    assert cls_scorer.n_evals == 2
    assert cls_scorer.phase_eval_counts["test"] == 2


def test_model_scorer_baseline() -> None:
    """Verify baseline scoring, caching, and cache reset."""
    scorer = ModelScorer(DummyRegressor(), track=True)
    scorer._current_phase = "baseline"  # pyright: ignore[reportPrivateUsage]

    x0 = np.array([2.0, 3.0])
    y0 = scorer.score_baseline(x0)
    assert np.isclose(y0, 5.0)
    # Baseline evaluations do not increment the evaluation counts.
    assert scorer.n_evals == 0
    assert scorer.phase_eval_counts.get("baseline", 0) == 0

    # The second call uses the cached baseline.
    y0_cached = scorer.score_baseline(x0)
    assert np.isclose(y0_cached, 5.0)

    # Reset the cached baseline.
    scorer.reset_baseline()
    y0_reset = scorer.score_baseline(x0)
    assert np.isclose(y0_reset, 5.0)


def test_get_y0_without_baseline() -> None:
    """Verify that ``get_y0`` raises before a baseline is scored."""
    scorer = ModelScorer(DummyRegressor())
    with pytest.raises(RuntimeError, match="Baseline not set"):
        scorer.get_y0()


def test_get_y0_with_baseline() -> None:
    """Verify that ``get_y0`` returns the scored baseline value."""
    scorer = ModelScorer(DummyRegressor())
    scorer.score_baseline(np.array([1.0, 1.0]))
    assert np.isclose(scorer.get_y0(), 2.0)


def test_model_scorer_tracking_no_phase() -> None:
    """Verify that tracking without an active phase counts evaluations but no phase totals."""
    scorer = ModelScorer(DummyRegressor(), track=True)
    assert scorer._current_phase is None  # pyright: ignore[reportPrivateUsage]

    pts = np.array([[1.0], [2.0], [3.0]])
    scorer.score(pts)
    assert scorer.n_evals == 3
    assert len(scorer.phase_eval_counts) == 0

    # score_baseline with tracking enabled and no active phase.
    x0 = np.array([4.0, 5.0])
    y0 = scorer.score_baseline(x0)
    assert np.isclose(y0, 9.0)
    assert scorer.n_evals == 3  # Baseline evaluations are not counted.

    # predict_proba with tracking enabled and no active phase.
    cls_scorer = ModelScorer(DummyClassifier(), target_class=1, track=True)
    cls_scorer.predict_proba(pts)
    assert cls_scorer.n_evals == 3
    assert len(cls_scorer.phase_eval_counts) == 0
