# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""von Mises-Fisher Variable Neighborhood Search (vMF-VNS) optimizer.

Implements the vMF-VNS algorithm for directional robustness optimization:
a variable-neighborhood search on the unit hypersphere that combines
von Mises-Fisher sampling, geodesic refinement, and bootstrap-based
acceptance testing. The optimizer records per-phase evaluation counts and
wall-clock times, optional search traces, and supports controlled restarts.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import TYPE_CHECKING, Any, Literal, cast
import warnings

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    from dicex.distributions.base import BasePerturbation

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.functional import directional_improvement, directional_improvement_batch
from dicex.core.scoring import ModelScorer
from dicex.exceptions import ConvergenceWarning
from dicex.logging import get_logger
from dicex.optim.bootstrap import acceptance_test
from dicex.optim.slerp import geodesic_refinement, slerp
from dicex.optim.vmf import angular_radius, sample_vmf
from dicex.results import ExplanationResult

TraceLevel = Literal["none", "summary", "full"]
PHASE_NAMES = ("shake", "refinement", "confirm", "baseline_0", "final_evaluation")

_logger = get_logger("optim")


class VmfVnsTracker:
    """Tracker for metrics and tracing in the vMF-VNS optimizer."""

    def __init__(self, trace_level: TraceLevel, scorer: ModelScorer) -> None:
        """Initialize the tracker.

        Args:
            trace_level (TraceLevel): The verbosity level for tracing.
            scorer (ModelScorer): The model scorer instance to pull evaluations from.
        """
        self.trace_level = trace_level
        self.scorer = scorer
        self.phase_wall_times: dict[str, float] = dict.fromkeys(PHASE_NAMES, 0.0)
        self._current_phase: str | None = None
        self.cvar_trajectory: list[float] = []
        self.direction_trajectory: list[np.ndarray] = []
        self.detailed_trajectory: list[dict[str, Any]] = []
        self.anytime_trace: list[dict[str, Any]] = []

    @contextmanager
    def phase_timer(self, phase: str) -> Generator[None, None, None]:
        """Time a specific phase of the algorithm.

        Args:
            phase (str): The name of the phase to time.

        Yields:
            None: Used as a context manager.
        """
        previous_phase = self._current_phase
        start = time.perf_counter()
        self._current_phase = phase
        self.scorer._current_phase = phase  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        try:
            yield
        finally:
            self.phase_wall_times[phase] = self.phase_wall_times.get(phase, 0.0) + (time.perf_counter() - start)
            self._current_phase = previous_phase
            self.scorer._current_phase = previous_phase  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    def record_anytime_trace(  # noqa: PLR0913
        self,
        *,
        iteration: int,
        restart_index: int,
        kappa: float,
        rho_deg: float,
        accepted: bool,
        current_value: float,
        best_value: float,
        t_start: float,
    ) -> None:
        """Record a trace entry for the current iteration.

        Args:
            iteration (int): Current neighborhood attempt index.
            restart_index (int): Current restart index.
            kappa (float): Concentration parameter used for this iteration.
            rho_deg (float): Angular radius in degrees for the current kappa.
            accepted (bool): Whether the move was accepted.
            current_value (float): CVaR value after the iteration.
            best_value (float): Best CVaR value observed so far.
            t_start (float): Start time of the optimization (from time.perf_counter()).
        """
        if self.trace_level == "none":
            return
        self.anytime_trace.append(
            {
                "iteration": iteration,
                "restart_index": restart_index,
                "kappa": float(kappa),
                "rho_deg": float(rho_deg),
                "accepted": bool(accepted),
                "current_value": float(current_value),
                "best_value": float(best_value),
                "n_model_evals": self.scorer.n_evals,
                "wall_time_seconds": round(time.perf_counter() - t_start, 6),
            },
        )

    def record_detailed_trajectory(self, details: dict[str, Any]) -> None:
        """Record detailed trajectory information if tracing is enabled.

        Args:
            details (dict[str, Any]): Dictionary with detailed iteration data.
        """
        if self.trace_level == "full":
            self.detailed_trajectory.append(details)

    def record_acceptance(self, move_cvar: float, direction: np.ndarray) -> None:
        """Record an accepted move.

        Args:
            move_cvar (float): CVaR value of the accepted move.
            direction (np.ndarray): Direction vector of the accepted move.
        """
        self.cvar_trajectory.append(move_cvar)
        self.direction_trajectory.append(direction.copy())


class VmfVns:
    """The vMF-VNS optimizer of the robust directional objective.

    A variable neighborhood search on the unit sphere: candidate directions are sampled
    from von Mises-Fisher distributions of decreasing concentration around the
    incumbent, refined along the geodesic, and accepted only when a paired bootstrap
    test shows an improvement. `Dicex` builds it internally; its options can be passed
    to `Dicex` as keyword arguments.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: Any,  # noqa: ANN401
        x0: np.ndarray,
        perturbation: BasePerturbation,
        alpha: float,
        kappas: list[float] | np.ndarray,
        b_samples: int,
        n_exploration: int,
        *,
        n_test: int | None = None,
        n_evaluation: int | None = None,
        eta: float,
        grid_t: np.ndarray,
        target_class: int | None = None,
        seed: int | None = None,
        max_evals: int | None = None,
        timeout_seconds: float | None = None,
        min_runtime_seconds: float | None = None,
        n_bootstrap: int = 1000,
        n_restarts: int = 0,
        trace_level: TraceLevel = "none",
        store_detailed_trajectory: bool = False,
    ) -> None:
        """Initialize the vMF-VNS optimizer.

        Args:
            model (Any): The machine learning model to evaluate.
            x0 (np.ndarray): The baseline point.
            perturbation (BasePerturbation): Directional and environmental noise distribution.
            alpha (float): Risk level for the lower-tail CVaR objective.
            kappas (list[float] | np.ndarray): List of vMF concentration parameters.
            b_samples (int): Number of candidate directions per neighborhood (shake phase).
            n_exploration (int): Number of directional noise samples for shake/refinement.
            n_test (int | None): Number of directional noise samples for the confirm phase.
                Defaults to `n_exploration` when only `n_evaluation` is given.
            n_evaluation (int | None): Number of directional noise samples for the final evaluation.
                Defaults to `n_test` when omitted.
            eta (float): Family-wise significance level for the bootstrap acceptance test.
            grid_t (np.ndarray): Grid of interpolation parameters for geodesic refinement.
            target_class (int | None): Target class index for classification models.
            seed (int | None): Optional seed for the random number generator.
            max_evals (int | None): Maximum number of model evaluations before stopping.
            timeout_seconds (float | None): Maximum wall-clock time before stopping.
            min_runtime_seconds (float | None): Minimum wall-clock time to keep restarting.
            n_bootstrap (int): Number of bootstrap resamples for the acceptance test.
            n_restarts (int): Number of explicit restarts to allow.
            trace_level (TraceLevel): Verbosity level for tracking optimization data.
            store_detailed_trajectory (bool): If True, implies trace_level "full".

        Raises:
            ValueError: If `kappas` contains no positive value, `trace_level` is invalid,
                neither `n_test` nor `n_evaluation` is provided, or `min_runtime_seconds`
                exceeds `timeout_seconds`.
        """
        self.x0 = x0
        self.perturbation = perturbation
        self.alpha = alpha

        valid_kappas = [k for k in kappas if k > 0]
        if not valid_kappas:
            msg = "The 'kappas' list must contain at least one value > 0."
            raise ValueError(msg)
        self.kappas = sorted(valid_kappas, reverse=True)

        if store_detailed_trajectory and trace_level == "none":
            trace_level = "full"
        if trace_level not in ("none", "summary", "full"):
            msg = f"Unsupported trace_level '{trace_level}'."
            raise ValueError(msg)

        if n_test is None and n_evaluation is None:
            msg = "Provide at least one of n_test or n_evaluation."
            raise ValueError(msg)
        if n_test is None:
            n_test = n_exploration
        if n_evaluation is None:
            n_evaluation = n_test

        if timeout_seconds is not None and min_runtime_seconds is not None and min_runtime_seconds > timeout_seconds:
            msg = "min_runtime_seconds cannot exceed timeout_seconds."
            raise ValueError(msg)

        self.b_samples = b_samples
        self.n_exploration = n_exploration
        self.n_test: int = n_test
        self.n_evaluation: int = n_evaluation
        self.n_bootstrap = n_bootstrap
        self.eta = eta
        self.grid_t = grid_t
        self.target_class = target_class
        self.rng = np.random.default_rng(seed)
        self.max_evals = max_evals
        self.timeout_seconds = timeout_seconds
        self.min_runtime_seconds = min_runtime_seconds
        self.n_restarts = n_restarts
        self.trace_level = trace_level

        self.scorer = ModelScorer(model, target_class=target_class, track=True)
        self.tracker = VmfVnsTracker(trace_level, self.scorer)

    def _random_unit_direction(self, d: int) -> np.ndarray:
        """Sample a random unit vector uniformly from the sphere.

        Args:
            d (int): Dimensionality of the sphere.

        Returns:
            np.ndarray: A random unit vector of length d.
        """
        direction = self.rng.standard_normal(d)
        direction /= np.linalg.norm(direction)
        return direction

    def run(self) -> ExplanationResult:  # noqa: C901, PLR0915, PLR0912
        """Run the vMF-VNS optimization algorithm.

        Returns:
            ExplanationResult: Result containing the best direction found, the robust
                value at that direction, and metadata about the optimization run.
        """
        t_start = time.perf_counter()
        d = len(self.x0)
        num_neighborhoods = len(self.kappas)

        _logger.info(
            "VmfVns started: d=%d, alpha=%.3f, L=%d neighborhoods, B=%d proposals, n_explore=%d, n_test=%d, n_eval=%d, eta=%.4f",  # noqa: E501
            d,
            self.alpha,
            num_neighborhoods,
            self.b_samples,
            self.n_exploration,
            self.n_test,
            self.n_evaluation,
            self.eta,
        )

        exploration_noises = self.perturbation.sample(self.n_exploration, d=d, rng=self.rng)
        exploration_env_noises = self.perturbation.sample_env(self.n_exploration, d, rng=self.rng)

        m = self._random_unit_direction(d)
        initial_direction = m.copy()
        best_direction = m.copy()
        best_observed_value = float("-inf")

        z_m_cache: np.ndarray | None = None
        incumbent_val_cache: float | None = None

        neighborhood_attempts = 0
        accepted_moves = 0
        restarts_performed = 0
        explicit_restarts_used = 0
        stop_reason = "completed_schedule"

        while True:
            l_idx = 0
            while l_idx < num_neighborhoods:
                elapsed = time.perf_counter() - t_start
                if self.max_evals is not None and self.scorer.n_evals >= self.max_evals:
                    stop_reason = "max_evals"
                    break
                if self.timeout_seconds is not None and elapsed >= self.timeout_seconds:
                    stop_reason = "timeout"
                    break

                neighborhood_attempts += 1
                kappa = self.kappas[l_idx]
                rho_deg = float(np.degrees(angular_radius(kappa, d)))

                with self.tracker.phase_timer("shake"):
                    u_candidates = sample_vmf(m, kappa, self.b_samples, rng=self.rng)

                    if z_m_cache is None:
                        z_m_cache = directional_improvement(
                            self.scorer,
                            self.x0,
                            m,
                            exploration_noises,
                            exploration_env_noises,
                        )
                        incumbent_val_cache = float(lower_tail_cvar_empirical(z_m_cache, self.alpha))

                    incumbent_val = cast("float", incumbent_val_cache)

                    candidate_improvements = directional_improvement_batch(
                        self.scorer,
                        self.x0,
                        u_candidates,
                        exploration_noises,
                        exploration_env_noises,
                    )
                    candidate_cvar_values = [
                        float(lower_tail_cvar_empirical(imp, self.alpha)) for imp in candidate_improvements
                    ]

                if incumbent_val > best_observed_value:
                    best_observed_value = incumbent_val
                    best_direction = m.copy()

                _logger.debug("[kappa=%.1f, rho=%.1f deg] Shake: CVaR=%.6f", kappa, rho_deg, incumbent_val)

                best_shake_val = incumbent_val
                u_tilde = m
                for _i, (u_i, val_ui) in enumerate(zip(u_candidates, candidate_cvar_values, strict=True), start=1):
                    if val_ui > best_shake_val:
                        best_shake_val = val_ui
                        u_tilde = u_i

                slerp_path_dirs: list[np.ndarray] | None = None
                if self.trace_level == "full":
                    grid = np.unique(np.concatenate([self.grid_t, [0.0, 1.0]]))
                    slerp_path_dirs = [slerp(m, u_tilde, float(t)) for t in grid]

                with self.tracker.phase_timer("refinement"):
                    u_ref, ref_val = geodesic_refinement(
                        self.scorer,
                        self.x0,
                        m,
                        u_tilde,
                        exploration_noises,
                        self.alpha,
                        self.grid_t,
                        env_noise_samples=exploration_env_noises,
                    )

                with self.tracker.phase_timer("confirm"):
                    validation_noises = self.perturbation.sample(self.n_test, d=d, rng=self.rng)
                    validation_env_noises = self.perturbation.sample_env(self.n_test, d, rng=self.rng)

                    val_directions = np.stack([u_ref, m])
                    z_val_batch = directional_improvement_batch(
                        self.scorer,
                        self.x0,
                        val_directions,
                        validation_noises,
                        validation_env_noises,
                    )
                    z_ref, z_m_val = z_val_batch[0], z_val_batch[1]
                    eta_bonf = self.eta / num_neighborhoods

                    if self.trace_level == "full":
                        accepted, lci, bootstrap_deltas = acceptance_test(
                            z_ref,
                            z_m_val,
                            self.alpha,
                            eta_bonf,
                            n_bootstrap=self.n_bootstrap,
                            rng=self.rng,
                            return_details=True,
                        )
                    else:
                        accepted = acceptance_test(
                            z_ref,
                            z_m_val,
                            self.alpha,
                            eta_bonf,
                            n_bootstrap=self.n_bootstrap,
                            rng=self.rng,
                        )
                        lci = None
                        bootstrap_deltas = None

                current_value = incumbent_val
                self.tracker.record_detailed_trajectory(
                    {
                        "kappa": float(kappa),
                        "rho_deg": rho_deg,
                        "incumbent_direction": m.copy(),
                        "candidates": u_candidates.copy(),
                        "candidate_cvar_values": candidate_cvar_values,
                        "best_shake_direction": u_tilde.copy(),
                        "refined_direction": u_ref.copy(),
                        "refined_cvar": float(ref_val),
                        "slerp_path": [direction.copy() for direction in slerp_path_dirs] if slerp_path_dirs else [],
                        "accepted": bool(accepted),
                        "bootstrap_lci": None if lci is None else float(lci),
                        "bootstrap_deltas": [] if bootstrap_deltas is None else bootstrap_deltas.tolist(),
                        "eta_bonf": float(eta_bonf),
                        "restart_index": restarts_performed,
                    },
                )

                if accepted:
                    m = u_ref
                    z_m_cache = None
                    incumbent_val_cache = None
                    self.scorer.reset_baseline()
                    accepted_moves += 1
                    move_cvar = float(lower_tail_cvar_empirical(z_ref, self.alpha))
                    current_value = move_cvar
                    self.tracker.record_acceptance(move_cvar, m)

                    if move_cvar > best_observed_value:
                        best_observed_value = move_cvar
                        best_direction = m.copy()

                    _logger.debug("[kappa=%.1f, rho=%.1f deg] ACCEPTED CVaR=%.6f", kappa, rho_deg, move_cvar)
                    l_idx = 0
                else:
                    _logger.debug("[kappa=%.1f, rho=%.1f deg] REJECTED", kappa, rho_deg)
                    l_idx += 1

                self.tracker.record_anytime_trace(
                    iteration=neighborhood_attempts,
                    restart_index=restarts_performed,
                    kappa=kappa,
                    rho_deg=rho_deg,
                    accepted=bool(accepted),
                    current_value=current_value,
                    best_value=best_observed_value,
                    t_start=t_start,
                )

            if stop_reason in {"max_evals", "timeout"}:
                break

            elapsed = time.perf_counter() - t_start
            needs_time_budget = self.min_runtime_seconds is not None and elapsed < self.min_runtime_seconds
            has_explicit_restart = explicit_restarts_used < self.n_restarts

            if needs_time_budget or has_explicit_restart:
                if has_explicit_restart:
                    explicit_restarts_used += 1
                restarts_performed += 1
                m = self._random_unit_direction(d)
                z_m_cache = None
                incumbent_val_cache = None
                self.scorer.reset_baseline()
                _logger.info("Restarting search: restart=%d", restarts_performed)
                continue

            if (self.n_restarts > 0 and explicit_restarts_used >= self.n_restarts) or (
                self.min_runtime_seconds is not None and elapsed < self.min_runtime_seconds
            ):
                stop_reason = "restart_limit"
            elif accepted_moves > 0:
                stop_reason = "completed_schedule"
            else:
                stop_reason = "no_improvement"
            break

        if accepted_moves == 0:
            warnings.warn("VmfVns terminated without any accepted move.", ConvergenceWarning, stacklevel=2)

        best_sphere_direction = best_direction.copy()
        beats_baseline = True

        with self.tracker.phase_timer("baseline_0"):
            final_test_noises = self.perturbation.sample(self.n_test, d=d, rng=self.rng)
            final_test_env = self.perturbation.sample_env(self.n_test, d, rng=self.rng)
            self.scorer.reset_baseline()

            val_directions_final = np.stack([best_direction, np.zeros(d)])
            z_final_batch = directional_improvement_batch(
                self.scorer,
                self.x0,
                val_directions_final,
                final_test_noises,
                final_test_env,
            )
            z_m, z_0 = z_final_batch[0], z_final_batch[1]

            beats_baseline = acceptance_test(
                z_m,
                z_0,
                self.alpha,
                self.eta,
                n_bootstrap=self.n_bootstrap,
                rng=self.rng,
            )

            if not beats_baseline:
                _logger.info("Best direction did not beat baseline (c=0). Falling back to c=0.")
                best_direction = np.zeros(d)

        with self.tracker.phase_timer("final_evaluation"):
            final_noises = self.perturbation.sample(self.n_evaluation, d=d, rng=self.rng)
            final_env_noises = self.perturbation.sample_env(self.n_evaluation, d, rng=self.rng)
            self.scorer.reset_baseline()

            eval_dirs = np.stack([best_direction, best_sphere_direction, initial_direction])
            z_eval_batch = directional_improvement_batch(
                self.scorer, self.x0, eval_dirs, final_noises, final_env_noises
            )

            z_final = np.asarray(z_eval_batch[0], dtype=float)
            final_val = float(lower_tail_cvar_empirical(z_final, self.alpha))
            final_mean = float(np.mean(z_final))
            final_std = float(np.std(z_final))

            best_sphere_cvar = float(lower_tail_cvar_empirical(np.asarray(z_eval_batch[1], dtype=float), self.alpha))
            initial_cvar = float(lower_tail_cvar_empirical(np.asarray(z_eval_batch[2], dtype=float), self.alpha))

        wall_time = time.perf_counter() - t_start

        metadata: dict[str, Any] = {
            "n_model_evals": self.scorer.n_evals,
            "n_iterations": neighborhood_attempts,
            "n_accepted_moves": accepted_moves,
            "n_restarts_performed": restarts_performed,
            "stop_reason": stop_reason,
            "wall_time_seconds": round(wall_time, 4),
            "phase_eval_counts": dict(self.scorer.phase_eval_counts),
            "phase_wall_times": {phase: round(v, 6) for phase, v in self.tracker.phase_wall_times.items()},
            "d": d,
            "alpha": self.alpha,
            "perturbation_params": self.perturbation.params,
            "kappas": list(self.kappas),
            "n_exploration": self.n_exploration,
            "n_test": self.n_test,
            "n_evaluation": self.n_evaluation,
            "n_bootstrap": self.n_bootstrap,
            "trace_level": self.trace_level,
            "min_runtime_seconds": self.min_runtime_seconds,
            "timeout_seconds": self.timeout_seconds,
            "max_evals": self.max_evals,
            "initial_direction": initial_direction,
            "initial_cvar": initial_cvar,
            "best_sphere_direction": best_sphere_direction,
            "best_sphere_cvar": best_sphere_cvar,
            "beats_baseline": bool(beats_baseline),
            "final_cvar": final_val,
            "final_mean_improvement": final_mean,
            "final_std_improvement": final_std,
            "cvar_trajectory": self.tracker.cvar_trajectory,
            "direction_trajectory": self.tracker.direction_trajectory,
            "no_move": accepted_moves == 0,
            "convergence_warning": accepted_moves == 0,
        }

        if self.trace_level in {"summary", "full"}:
            metadata["anytime_trace"] = self.tracker.anytime_trace
        if self.trace_level == "full":
            metadata["detailed_trajectory"] = self.tracker.detailed_trajectory

        return ExplanationResult(direction=best_direction, robust_value=final_val, alpha=self.alpha, metadata=metadata)
