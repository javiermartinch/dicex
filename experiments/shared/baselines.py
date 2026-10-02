# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Baseline direction solvers: random-direction search and the nominal (noise-free) optimum."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from dicex.core.cvar import lower_tail_cvar_empirical

FloatArray = NDArray[np.float64]


def run_random_direction_baseline(
    predict_fn: Callable[[FloatArray], FloatArray],
    x0: FloatArray,
    mu: float,
    sigma: float,
    env_std: float,
    alpha: float,
    n_internal_samples: int = 1000,
    k_directions: int = 50,
    seed: int = 42,
) -> FloatArray:
    """Solve the random-direction baseline (pure random search).

    Generates a set of random directions on the unit hypersphere and evaluates
    their empirical CVaR using simulated action and environment noise. Returns the
    direction that maximizes this robust value.

    Args:
        predict_fn (Callable[[FloatArray], FloatArray]): Prediction function of the target model mapping feature inputs
            to outputs.
        x0 (FloatArray): Baseline query point as a 1D array.
        mu (float): Perturbation step size.
        sigma (float): Standard deviation of the user action noise.
        env_std (float): Standard deviation of the environmental noise.
        alpha (float): Confidence level for CVaR optimization (e.g. 0.10).
        n_internal_samples (int): Number of noise samples to draw for empirical CVaR
            evaluation.
        k_directions (int): Number of random directions to evaluate.
        seed (int): Random seed for reproducible noise generation.

    Returns:
        FloatArray: A 1D numpy array representing the optimal unit direction vector found.
    """
    rng = np.random.default_rng(seed)
    d = len(x0)

    best_cvar = -np.inf
    best_c = np.zeros(d)

    for _ in range(k_directions):
        # Generate random direction on hypersphere
        raw_c = rng.normal(size=d)
        norm = np.linalg.norm(raw_c)
        if norm < 1e-8:
            continue
        c = raw_c / norm

        # Internal CVaR estimation
        action_noise = rng.normal(scale=sigma, size=(n_internal_samples, d))
        env_noise = rng.normal(scale=env_std, size=(n_internal_samples, d))

        # Same perturbation law as the evaluation (x0 + T * c + eps, with T ~ N(mu, sigma^2)).
        perturbed_x = x0 + (mu + action_noise) * c + env_noise
        y_pred = predict_fn(perturbed_x)
        y_base = predict_fn(x0[None, :])[0]
        gain = y_pred - y_base

        cvar_val = float(lower_tail_cvar_empirical(gain, alpha))

        if cvar_val > best_cvar:
            best_cvar = cvar_val
            best_c = c

    return best_c


def _numerical_gradient(
    f: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    h: float = 1e-5,
) -> FloatArray:
    """Compute the gradient of f at x using central finite differences.

    Args:
        f (Callable[[FloatArray], FloatArray]): Target scalar/vector function.
        x (FloatArray): Evaluation point.
        h (float): Finite difference step size.

    Returns:
        FloatArray: Gradient vector.
    """
    d = len(x)
    grad = np.empty(d, dtype=float)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h
        fp = float(f((x + ei)[None, :])[0])
        fm = float(f((x - ei)[None, :])[0])
        grad[i] = (fp - fm) / (2.0 * h)
    return grad


def run_nominal_baseline(
    predict_fn: Callable[[FloatArray], FloatArray],
    x0: FloatArray,
    mu: float,
    *,
    n_restarts: int = 32,
    n_grid: int = 360,
    seed: int = 42,
) -> FloatArray:
    r"""Solve the nominal baseline optimization problem.

    Finds the deterministic direction on the unit hypersphere that maximizes the
    predicted model value at distance mu, assuming perfect execution (zero noise).

    **Four-stage vectorized strategy:**
        0. Numerical gradient — estimate :math:`\nabla M(x_0)` via central finite
           differences and use the normalized gradient as a candidate.  For linear
           models this is the exact optimal direction.
        1. Coarse batch — evaluate ``N`` random directions (or angular grid in 2D)
           in a single batched call and keep the top ``k`` candidates.
        2. Local refinement — for each top candidate, perform a tangent-space
           pattern search on the sphere with geometrically decreasing step sizes.
           All perturbations across all candidates are evaluated in **one batched
           call per iteration**.
        3. Post-optimization — multi-start COBYLA with an equality constraint
           keeping the direction on the unit sphere, started from the best
           refined candidates; a result replaces the incumbent only if it
           improves the prediction.

    Args:
        predict_fn (Callable[[FloatArray], FloatArray]): Prediction function of the target model (accepts 2D batches).
        x0 (FloatArray): Baseline query point as a 1D array.
        mu (float): Step size of the perturbation.
        n_restarts (int): Base number of random restarts; the effective number of
            sampled candidates is ``max(n_restarts, d * 64) * 512`` (ignored in 2D).
        n_grid (int): Number of angular steps for grid search in 2D.
        seed (int): Random seed for restarts.

    Returns:
        FloatArray: A 1D numpy array representing the optimal unit direction vector.
    """
    rng = np.random.default_rng(seed)
    d = len(x0)

    # Stage 0: numerical gradient candidate
    grad_candidates_list: list[FloatArray] = []
    if d >= 2:
        try:
            grad = _numerical_gradient(predict_fn, x0, h=1e-5)
            gn = np.linalg.norm(grad)
            if gn > 1e-12:
                grad_candidates_list.append(grad / gn)
        except Exception:  # noqa: BLE001, S110
            pass

    # Stage 1: coarse batch evaluation
    if d == 2:
        angles = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        candidates = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        effective_restarts = max(n_restarts, d * 64)
        n_candidates = effective_restarts * 512
        raw = rng.normal(size=(n_candidates, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        candidates = raw / norms

    # Prepend gradient-guided candidates so they are not lost in the noise
    if grad_candidates_list and d > 2:
        candidates = np.vstack([np.array(grad_candidates_list), candidates])

    points = x0 + mu * candidates
    preds = np.asarray(predict_fn(points), dtype=float).ravel()

    # Stage 2: local refinement via tangent pattern search
    n_keep = min(128, len(candidates))
    best_indices = np.argpartition(-preds, n_keep)[:n_keep]

    current_c = candidates[best_indices].copy()  # (n_keep, d)
    current_val = -preds[best_indices]  # negated for minimization

    n_tangent = min(1024, max(64, d * 32))  # perturbations per candidate
    n_iter = 50  # number of refinement iterations
    step_sizes = np.geomspace(0.8, 0.0001, n_iter)  # geometrically decreasing step sizes

    for step_size in step_sizes:
        # Perturb with isotropic noise and renormalize.
        # For small step_size this is equivalent to a random walk on S^{d-1}.
        c_pert = current_c[:, None, :] + step_size * rng.normal(size=(n_keep, n_tangent, d))
        cnorm = np.linalg.norm(c_pert, axis=-1, keepdims=True)
        cnorm = np.maximum(cnorm, 1e-12)
        c_pert /= cnorm

        # Single batched evaluation
        flat_candidates = c_pert.reshape(-1, d)  # (k*M, d)
        flat_preds = np.asarray(predict_fn(x0 + mu * flat_candidates), dtype=float).ravel()
        flat_preds_2d = flat_preds.reshape(n_keep, n_tangent)  # (k, M)

        group_best_val = flat_preds_2d.max(axis=1)
        group_best_idx = flat_preds_2d.argmax(axis=1)

        # Update where improved
        improved = group_best_val > -current_val
        for i in range(n_keep):
            if improved[i]:
                current_c[i] = flat_candidates[i * n_tangent + group_best_idx[i]]
                current_val[i] = -group_best_val[i]

    best_c = current_c[int(np.argmin(current_val))]

    # Stage 3: multi-start post-optimization on the sphere
    # Run COBYLA from the top K survivors for non-convex models
    n_cobyla_starts = min(10, n_keep)
    best_candidates = current_c[np.argsort(current_val)][:n_cobyla_starts]

    for start_c in best_candidates:
        try:
            cons = {"type": "eq", "fun": lambda c: float(np.linalg.norm(c) - 1.0)}
            res = minimize(
                lambda c: -float(predict_fn((x0 + mu * c)[None, :])[0]),
                start_c,
                method="COBYLA",
                constraints=cons,
                options={"maxiter": 5000, "tol": 1e-8, "rhobeg": 0.5},
            )
            if res.success:
                c_post = np.asarray(res.x, dtype=float)
                cn = np.linalg.norm(c_post)
                if cn > 1e-12:
                    c_post /= cn
                    f_before = float(predict_fn((x0 + mu * best_c)[None, :])[0])
                    f_after = float(predict_fn((x0 + mu * c_post)[None, :])[0])
                    if f_after > f_before:
                        best_c = c_post
        except Exception:  # noqa: BLE001, S110
            pass

    if np.linalg.norm(best_c) < 1e-8:
        return np.zeros_like(x0)
    return best_c
