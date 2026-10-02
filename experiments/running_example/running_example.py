# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Credit Approval Score synthetic running example experiment pipeline."""

import logging
from pathlib import Path
import sys
from typing import Any, Protocol

import cloudpickle
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.geometry import apply_directional_noise
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.explainer import Dicex
from dicex.preprocessing.scaling import DicexScaler
from experiments.shared.helpers import repo_relative
from experiments.shared.log_config import configure_logging, get_logger

EPSILON = 1e-8

N_SAMPLES = 5000
SEED = 42

BASE_SCORE = 50.0
GROWTH_FACTOR = 5.0

STATE_FCI_MIN = -0.5
STATE_FCI_MAX = 3.0
STATE_HVI_MIN = 1.5
STATE_HVI_MAX = 3.5
BONUS_HEIGHT = 100.0

TARGET_IMPROVEMENT = 30.0
X0_TARGET_POS = np.array([-1.0, -0.5])

PERTURBATION_SIGMA = 0.3
PERTURBATION_ENV_MEAN = np.array([0.0, 0.0])
PERTURBATION_ENV_COV = np.array([0.1, 0.1])

DICEX_ALPHAS = [0.1, 0.9]
N_COMPARE_SAMPLES = 100


class RegressionModel(Protocol):
    """Protocol for regression models that have a predict method."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict continuous target values.

        Args:
            x (np.ndarray): Input feature array.

        Returns:
            np.ndarray: Predicted target values.
        """
        ...


class CreditLimitModel:
    """Regression model predicting Credit Approval Score with bank policy + state guarantee."""

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict continuous Credit Approval Score (in points).

        Inside the state guarantee zone, the bank score receives a bonus of
        ``BONUS_HEIGHT`` points. Outside, only the standard bank policy applies.

        Args:
            x_data (np.ndarray): Input feature array of shape (N, 2).

        Returns:
            np.ndarray: Predicted scores of shape (N,).
        """
        x1 = x_data[:, 0]  # FCI
        x2 = x_data[:, 1]  # HVI

        # Standard bank policy score calculation
        bank_score = BASE_SCORE + GROWTH_FACTOR * (x1 - x2)

        # State guarantee bonus applied within designated region
        in_fci_range = (x1 >= STATE_FCI_MIN) & (x1 <= STATE_FCI_MAX)
        in_hvi_range = (x2 >= STATE_HVI_MIN) & (x2 <= STATE_HVI_MAX)
        in_guarantee_box = in_fci_range & in_hvi_range

        premium_score = bank_score + BONUS_HEIGHT
        return np.where(
            in_guarantee_box,
            premium_score,
            bank_score,
        )

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        """Return the predicted score as a single-column array.

        Args:
            x_data (np.ndarray): Input feature array.

        Returns:
            np.ndarray: Predicted score reshaped as 2D array.
        """
        return self.predict(x_data).reshape(-1, 1)


def generate_centered_bank_data(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Credit Approval Score data with a specific distribution and policy.

    Args:
        n_samples (int): Number of synthetic samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (x_data, y_labels).
    """
    rng = np.random.default_rng(seed)
    # Population centered at zero for both indices (reference baselines)
    means = [0.0, 0.0]
    # Larger variance to spread the population: HVI varies more than FCI
    cov = [[4.0, 0.5], [0.5, 6.25]]

    x_data = rng.multivariate_normal(means, cov, n_samples)

    model = CreditLimitModel()
    y_labels = model.predict(x_data)

    return x_data, y_labels


def compute_wachter_cf(
    x0: np.ndarray,
    model: RegressionModel,
    _x_sample_scaled: np.ndarray,
    target_value: float,
    search_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """Compute Wachter counterfactual by minimizing distance in scaled space.

    Finds the closest point to x0 where the predicted score >= target_value.
    Uses a grid search for initialization followed by Nelder-Mead refinement.

    Args:
        x0 (np.ndarray): Query point in scaled space.
        model (RegressionModel): Scaled model predicting in scaled space.
        _x_sample_scaled (np.ndarray): Reference data in scaled space (unused).
        target_value (float): Desired prediction value.
        search_bounds (tuple[float, float, float, float] | None): Bounding box
            (x1_min, x1_max, x2_min, x2_max) for grid search. Defaults to None.

    Returns:
        np.ndarray | None: The optimal counterfactual point, or None if not found.
    """
    # Determine search bounds
    if search_bounds is not None:
        x1_min, x1_max, x2_min, x2_max = search_bounds
    else:
        x1_min = x2_min = -15.0
        x1_max = x2_max = 15.0

    x1_range = np.linspace(x1_min, x1_max, 100)
    x2_range = np.linspace(x2_min, x2_max, 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]

    predictions = model.predict(grid_points).flatten()
    target_grid = grid_points[predictions >= target_value]

    if len(target_grid) == 0:
        return None

    dists = np.linalg.norm(target_grid - x0, axis=1)
    best_idx = np.argmin(dists)
    cf_init = target_grid[best_idx]

    # Refine with local search
    def wachter_penalty(x: np.ndarray, x0_ref: np.ndarray, model_ref: RegressionModel, target: float) -> float:
        d = np.sum((x - x0_ref) ** 2)
        pred = model_ref.predict(x.reshape(1, -1)).flatten()[0]
        penalty_val = 1e6 if pred < target else 0
        return float(d + penalty_val)

    res = minimize(wachter_penalty, cf_init, args=(x0, model, target_value), method="Nelder-Mead")
    return res.x


# Pipeline stages


def setup_environment() -> tuple[Path, logging.Logger]:
    """Create the output directory and configure logging.

    Returns:
        tuple[Path, logging.Logger]: Tuple of (output_dir, logger).
    """
    base_dir = Path(__file__).parent
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(
        script_name="running_example",
        log_dir=base_dir / "results" / "logs",
    )
    logger = get_logger("running_example")
    logger.info("Starting Credit Approval Score Experiment")

    return output_dir, logger


def generate_and_visualize_data(
    _output_dir: Path,
    logger: logging.Logger,
) -> tuple[CreditLimitModel, Any, DicexScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the synthetic data and fit the scaler.

    Args:
        _output_dir (Path): Output directory (unused).
        logger (logging.Logger): Logger instance for recording progress.

    Returns:
        tuple[CreditLimitModel, Any, DicexScaler, np.ndarray, np.ndarray, np.ndarray]:
            Tuple of (model_obj, model_scaled, scaler_obj, x_raw, x_scaled, y_labels_raw).
    """
    model_obj = CreditLimitModel()
    x_raw, y_labels_raw = generate_centered_bank_data(N_SAMPLES, SEED)

    # Statistics before scaling
    mean_raw = np.mean(x_raw, axis=0)
    std_raw = np.std(x_raw, axis=0)
    logger.info("Raw Data Stats - Mean: %s, Std: %s", mean_raw, std_raw)

    # Scale data
    logger.info("Fitting DicexScaler and transforming data...")
    scaler_obj = DicexScaler()
    scaler_obj.fit(x_raw)
    x_scaled = scaler_obj.transform(x_raw)

    mean_scaled = np.mean(x_scaled, axis=0)
    std_scaled = np.std(x_scaled, axis=0)
    logger.info("Scaled Data Stats - Mean: %s, Std: %s", mean_scaled, std_scaled)

    model_scaled = scaler_obj.wrap_model(model_obj)

    return model_obj, model_scaled, scaler_obj, x_raw, x_scaled, y_labels_raw


def analyze_x0_and_wachter(
    _model_obj: CreditLimitModel,
    model_scaled: Any,
    scaler_obj: DicexScaler,
    x_raw: np.ndarray,
    x_scaled: np.ndarray,
    y_labels_raw: np.ndarray,
    _output_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray]:
    """Select the query point and compute the Wachter CF for the target improvement.

    Args:
        _model_obj (CreditLimitModel): Unscaled regression model (unused).
        model_scaled (Any): Scaled wrapper model.
        scaler_obj (DicexScaler): Fitted scaler instance.
        x_raw (np.ndarray): Raw feature array.
        x_scaled (np.ndarray): Scaled feature array.
        y_labels_raw (np.ndarray): Raw target labels.
        _output_dir (Path): Output directory (unused).
        logger (logging.Logger): Logger instance for recording progress.

    Returns:
        tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray]:
            Tuple of (x0_raw, x0_scaled, target_limit, cf_scaled, cf_raw).
    """
    # Select as x0 the sample closest to X0_TARGET_POS: a low-score client near the
    # state guarantee zone, far enough from it for the direction to dominate the noise.
    distances = np.linalg.norm(x_raw - X0_TARGET_POS, axis=1)
    x0_idx = np.argmin(distances)

    x0_raw = x_raw[x0_idx]
    x0_scaled = x_scaled[x0_idx]

    current_limit = y_labels_raw[x0_idx]
    target_limit = current_limit + TARGET_IMPROVEMENT

    logger.info("Selected Query Point x0 (index %s) - Raw: %s, Scaled: %s", x0_idx, x0_raw, x0_scaled)
    logger.info(
        "Current score: %.1f points, Target: %.1f points (+%.1f)", current_limit, target_limit, TARGET_IMPROVEMENT
    )

    # Compute Wachter CF in scaled space
    logger.info("Computing Wachter Counterfactual (target: %.1f points)...", target_limit)
    cf_scaled = compute_wachter_cf(x0_scaled, model_scaled, x_scaled, target_limit)

    if cf_scaled is not None:
        cf_raw = scaler_obj.inverse_transform(cf_scaled.reshape(1, -1)).flatten()
        pred_cf = model_scaled.predict(cf_scaled.reshape(1, -1)).flatten()[0]

        logger.info("Wachter CF found (predicted score: %.1f points)", pred_cf)
        logger.info("CF Scaled: %s, CF Raw: %s", cf_scaled, cf_raw)
    else:
        cf_raw = np.array([])
        logger.warning("No points in the sampling grid meet the target score.")

    return x0_raw, x0_scaled, target_limit, cf_scaled, cf_raw


def analyze_dicex_multiple_alphas(
    x0_raw: np.ndarray,
    model_obj: CreditLimitModel,
    scaler_obj: DicexScaler,
    perturbation: GaussianPerturbation,
    logger: logging.Logger,
) -> tuple[dict[float, Dicex], dict[float, np.ndarray]]:
    """Run the DiCEx explainer for multiple alpha values.

    Args:
        x0_raw (np.ndarray): Query point in raw space.
        model_obj (CreditLimitModel): Unscaled regression model.
        scaler_obj (DicexScaler): Fitted scaler instance.
        perturbation (GaussianPerturbation): Gaussian perturbation instance.
        logger (logging.Logger): Logger instance for recording progress.

    Returns:
        tuple[dict[float, Dicex], dict[float, np.ndarray]]: Tuple of
            (explainers_by_alpha, directions_by_alpha).
    """
    logger.info("Running DiCEx Explainer for alphas: %s", DICEX_ALPHAS)

    _explainers = {}
    directions = {}

    for alpha in DICEX_ALPHAS:
        logger.info("  Running DiCEx with α=%.1f...", alpha)

        explainer = Dicex(
            model=model_obj,
            task="regression",
            perturbation=perturbation,
            alpha=alpha,
            scaler=scaler_obj,
            seed=SEED,
            n_exploration=1000,
            n_test=1000,
            n_evaluation=1000,
        )

        dicex_result = explainer.explain(x0_raw)

        logger.info("    Direction DiCEx (Scaled): %s", dicex_result.metadata["direction_scaled"])
        logger.info("    Robust Value (CVaR α=%.1f): %.4f", alpha, dicex_result.robust_value)

        directions[alpha] = dicex_result.metadata["direction_scaled"]
        _explainers[alpha] = explainer

    return _explainers, directions


GRID_POINTS = 12


def compute_vector_fields(
    model_obj: Any,
    model_scaled: Any,
    scaler_obj: DicexScaler,
    x_scaled: np.ndarray,
    dynamic_mu: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Compute vector field directions for a grid of points using DiCEx and Wachter CF.

    Args:
        model_obj (Any): Unscaled regression model.
        model_scaled (Any): Scaled wrapper model.
        scaler_obj (DicexScaler): Fitted scaler instance.
        x_scaled (np.ndarray): Scaled feature array.
        dynamic_mu (float): Magnitude of perturbation step size.
        logger (logging.Logger): Logger instance for recording progress.

    Returns:
        dict[str, Any]: Dictionary containing grid coordinates, DiCEx directions,
            and Wachter CF directions.
    """
    x_limits = (-2.0, 2.0)
    y_limits = (-2.5, 2.5)
    x1_grid = np.linspace(x_limits[0] + 0.2, x_limits[1] - 0.2, GRID_POINTS)
    x2_grid = np.linspace(y_limits[0] + 0.2, y_limits[1] - 0.2, GRID_POINTS)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    n_total = xx1.size

    logger.info("Computing vector field directions for %d grid points...", n_total)

    perturbation = GaussianPerturbation(
        mu=dynamic_mu,
        sigma=PERTURBATION_SIGMA,
        env_mean=PERTURBATION_ENV_MEAN,
        env_cov=PERTURBATION_ENV_COV,
        seed=SEED,
    )

    alphas = [0.1, 0.9]
    dicex_directions: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for alpha in alphas:
        logger.info("Computing DiCEx directions for α=%.1f...", alpha)
        u = np.zeros_like(xx1)
        v = np.zeros_like(xx2)
        ok = np.zeros(xx1.shape, dtype=bool)
        no_move = np.zeros(xx1.shape, dtype=bool)

        for i in range(GRID_POINTS):
            for j in range(GRID_POINTS):
                pt_scaled = np.array([xx1[i, j], xx2[i, j]])
                pt_raw = scaler_obj.inverse_transform(pt_scaled.reshape(1, -1)).flatten()

                try:
                    explainer = Dicex(
                        model=model_obj,
                        task="regression",
                        perturbation=perturbation,
                        alpha=alpha,
                        scaler=scaler_obj,
                        seed=SEED,
                        n_exploration=500,
                        n_test=500,
                        n_evaluation=500,
                    )
                    result = explainer.explain(pt_raw)
                    c = np.asarray(result.metadata["direction_scaled"], dtype=float)
                    if np.linalg.norm(c) > EPSILON:
                        baseline_cvar = result.metadata.get("baseline_cvar", -np.inf)
                        movement_recommended = result.robust_value > baseline_cvar
                        if movement_recommended:
                            u[i, j] = c[0]
                            v[i, j] = c[1]
                            ok[i, j] = True
                        else:
                            no_move[i, j] = True
                    else:
                        # A zero direction is DiCEx's explicit inaction recommendation.
                        no_move[i, j] = True
                except Exception as e:  # noqa: BLE001
                    logger.debug("DiCEx failed at (%.2f, %.2f) α=%.1f: %s", xx1[i, j], xx2[i, j], alpha, e)

        dicex_directions[alpha] = (u, v, ok, no_move)
        n_ok = int(ok.sum())
        n_no_move = int(no_move.sum())
        logger.info(
            "  α=%.1f: %d arrows, %d no-move crosses, %d/%d total succeeded",
            alpha,
            n_ok,
            n_no_move,
            n_ok + n_no_move,
            n_total,
        )

    logger.info("Computing Wachter CF directions for grid...")
    u_cf = np.zeros_like(xx1)
    v_cf = np.zeros_like(xx2)
    ok_cf = np.zeros(xx1.shape, dtype=bool)

    for i in range(GRID_POINTS):
        for j in range(GRID_POINTS):
            pt_scaled = np.array([xx1[i, j], xx2[i, j]])
            y_pt = model_scaled.predict(pt_scaled.reshape(1, -1)).flatten()[0]
            target_pt = y_pt + TARGET_IMPROVEMENT

            cf_pt = compute_wachter_cf(pt_scaled, model_scaled, x_scaled, target_pt)
            if cf_pt is not None:
                diff = cf_pt - pt_scaled
                norm = np.linalg.norm(diff)
                if norm > EPSILON:
                    c_dir = diff / norm
                    u_cf[i, j] = c_dir[0]
                    v_cf[i, j] = c_dir[1]
                    ok_cf[i, j] = True

    n_ok_cf = int(ok_cf.sum())
    logger.info("  Wachter CF: %d/%d points succeeded", n_ok_cf, n_total)

    return {
        "xx1": xx1,
        "xx2": xx2,
        "dicex_directions": dicex_directions,
        "u_cf": u_cf,
        "v_cf": v_cf,
        "ok_cf": ok_cf,
        "x_limits": x_limits,
        "y_limits": y_limits,
    }


def main() -> None:
    """Run the full Credit Approval Score experiment pipeline (CLI entry point)."""
    output_dir, logger = setup_environment()

    model_obj, model_scaled, scaler_obj, x_raw, x_scaled, y_labels_raw = generate_and_visualize_data(
        output_dir,
        logger,
    )

    x0_raw, x0_scaled, target_limit, cf_scaled, _cf_raw = analyze_x0_and_wachter(
        model_obj,
        model_scaled,
        scaler_obj,
        x_raw,
        x_scaled,
        y_labels_raw,
        output_dir,
        logger,
    )

    y0_val = model_scaled.predict(x0_scaled.reshape(1, -1)).flatten()[0]
    if cf_scaled is not None:
        c_cf = (cf_scaled - x0_scaled) / np.linalg.norm(cf_scaled - x0_scaled)
        dynamic_mu = float(np.linalg.norm(cf_scaled - x0_scaled))
        logger.info(
            "Computed dynamic perturbation step size mu = %.4f from Wachter CF distance.",
            dynamic_mu,
        )
    else:
        c_cf = np.array([0.0, 0.0])
        dynamic_mu = 1.0
        logger.warning("Wachter CF unavailable; using fallback mu = %.1f.", dynamic_mu)

    perturbation = GaussianPerturbation(
        mu=dynamic_mu,
        sigma=PERTURBATION_SIGMA,
        env_mean=PERTURBATION_ENV_MEAN,
        env_cov=PERTURBATION_ENV_COV,
        seed=SEED,
    )

    _explainers, directions = analyze_dicex_multiple_alphas(
        x0_raw,
        model_obj,
        scaler_obj,
        perturbation,
        logger,
    )

    logger.info("Sampling directional noise and computing improvements...")
    noise_comp = perturbation.sample(N_COMPARE_SAMPLES)
    env_comp = perturbation.sample_env(N_COMPARE_SAMPLES, d=2)

    if cf_scaled is not None:
        pts_wachter = apply_directional_noise(x0_scaled, c_cf, noise_comp, env_comp)
        imp_wachter = model_scaled.predict(pts_wachter) - y0_val
        cvar_wachter = float(lower_tail_cvar_empirical(imp_wachter, 0.1))
        failure_rate_wachter = float(np.mean(model_scaled.predict(pts_wachter) < y0_val))
        logger.info("  Wachter CF CVaR(0.1)=%.2f pts, Failure Rate=%.2f%%", cvar_wachter, failure_rate_wachter * 100)
    else:
        pts_wachter = None
        imp_wachter = None
        cvar_wachter = None
        failure_rate_wachter = None

    pts_dicex = {}
    imp_dicex = {}
    cvar_dicex = {}
    failure_rate_dicex = {}
    for alpha in DICEX_ALPHAS:
        c_alpha = directions[alpha]
        pts = apply_directional_noise(x0_scaled, c_alpha, noise_comp, env_comp)
        imp = model_scaled.predict(pts) - y0_val
        pts_dicex[alpha] = pts
        imp_dicex[alpha] = imp
        cvar_dicex[alpha] = lower_tail_cvar_empirical(imp, alpha)
        fail_rate = float(np.mean(model_scaled.predict(pts) < y0_val))
        failure_rate_dicex[alpha] = fail_rate
        logger.info(
            "  DiCEx α=%.1f CVaR=%.2f pts, Failure Rate=%.2f%%, direction=(%.3f, %.3f)",
            alpha,
            cvar_dicex[alpha],
            fail_rate * 100,
            c_alpha[0],
            c_alpha[1],
        )

    running_example_data = {
        "x_raw": x_raw,
        "y_labels_raw": y_labels_raw,
        "x_scaled": x_scaled,
        "model_obj": model_obj,
        "model_scaled": model_scaled,
        "x0_raw": x0_raw,
        "x0_scaled": x0_scaled,
        "target_limit": target_limit,
        "y0_val": y0_val,
        "cf_scaled": cf_scaled,
        "cf_raw": _cf_raw,
        "c_cf": c_cf,
        "pts_wachter": pts_wachter,
        "imp_wachter": imp_wachter,
        "cvar_wachter": cvar_wachter,
        "failure_rate_wachter": failure_rate_wachter,
        "directions": directions,
        "pts_dicex": pts_dicex,
        "imp_dicex": imp_dicex,
        "cvar_dicex": cvar_dicex,
        "failure_rate_dicex": failure_rate_dicex,
    }

    data_path = output_dir / "running_example_data.pkl"
    with open(data_path, "wb") as f:
        cloudpickle.dump(running_example_data, f)

    logger.info("Pipeline data (x0 results) saved to %s.", repo_relative(data_path))

    vector_fields = compute_vector_fields(
        model_obj=model_obj,
        model_scaled=model_scaled,
        scaler_obj=scaler_obj,
        x_scaled=x_scaled,
        dynamic_mu=dynamic_mu,
        logger=logger,
    )
    running_example_data["vector_fields"] = vector_fields

    with open(data_path, "wb") as f:
        cloudpickle.dump(running_example_data, f)

    logger.info(
        "Pipeline data (with vector fields) updated in %s. Run visualize_and_report.py to generate the plots.",
        repo_relative(data_path),
    )
    logger.info("Experiment pipeline completed.")


if __name__ == "__main__":
    main()
