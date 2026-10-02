# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Unified variance profiler for directional models on spherical slices.

Evaluates the Lagrangian candidate directions on the slice

    C_t = { c in S^(d-1) : beta^T c = t }

and returns the minimum and maximum profile values (s_min, s_max)
along with their associated direction vectors (c_min, c_max).

The candidate expressions follow the case analysis of the stationarity
conditions on the slice: boundary projections, null projections, and
intermediate projections, each with regular, singular, and zero-coefficient
candidates.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.optimize import root_scalar

# Numerical tolerances for projection checks.
ZERO_PROJECTION_ATOL: float = 1e-12
BOUNDARY_PROJECTION_ATOL: float = 1e-7


def build_variance_profiler(  # noqa: C901, PLR0915
    beta: np.ndarray,
) -> Callable[[float], tuple[float, float, np.ndarray, np.ndarray]]:
    """Build a variance profiler function for a given coefficient vector beta.

    Pre-computes pole bounds and grid evaluations for candidate roots to
    enable fast profiling of variance bounds over spherical slices.

    Args:
        beta (np.ndarray): Model coefficient vector.

    Returns:
        Callable[[float], tuple[float, float, np.ndarray, np.ndarray]]:
            A function taking a projection scalar ``t`` and returning
            ``(s_min, s_max, c_min, c_max)``.
    """
    d = len(beta)
    nz_mask = beta != 0
    zero_idxs = [i for i in range(d) if not nz_mask[i]]

    if not np.any(nz_mask):
        c_zero = np.zeros(d)
        c_zero[0] = 1.0
        return lambda _t: (0.0, 0.0, c_zero, c_zero)

    beta_nz = beta[nz_mask]
    beta_sq_nz = beta_nz**2
    beta_min_sq = float(np.min(beta_sq_nz))
    beta_norm = float(np.linalg.norm(beta_nz))
    w_boundary = float(np.sum(beta_sq_nz**2) / (beta_norm**2))
    c_boundary_positive = np.zeros(d)
    c_boundary_positive[nz_mask] = beta_nz / beta_norm

    epsilon = max(1e-14, beta_min_sq * 1e-12)
    smin_right = max(1000.0, 10.0 * beta_norm)

    # Intervals between poles for non-zero components
    poles = sorted(set(-beta_sq_nz))  # [-beta_max^2, ..., -beta_min^2]
    early_intervals = [(poles[i], poles[i + 1]) for i in range(len(poles) - 1)]

    # Pre-identify singular index sets I_rho where beta_i^2 values are identical
    singular_groups: list[tuple[float, list[int]]] = []
    unique_sq_values = sorted(set(beta_sq_nz))
    for val in unique_sq_values:
        idxs = [i for i in range(d) if nz_mask[i] and np.isclose(beta[i] ** 2, val)]
        singular_groups.append((float(val), idxs))

    def A_beta(lam: np.ndarray | float) -> np.ndarray | float:  # noqa: N802
        """Compute the rational sum A_beta at the multiplier lambda.

        Args:
            lam (np.ndarray | float): Multiplier parameter lambda.

        Returns:
            np.ndarray | float: Sum of ``beta_i^2 / (beta_i^2 + lam)``.
        """
        if isinstance(lam, np.ndarray):
            return np.sum(beta_sq_nz[:, None] / (beta_sq_nz[:, None] + lam[None, :]), axis=0)
        return float(np.sum(beta_sq_nz / (beta_sq_nz + lam)))

    def B_beta(lam: np.ndarray | float) -> np.ndarray | float:  # noqa: N802
        """Compute the rational sum B_beta at the multiplier lambda.

        Args:
            lam (np.ndarray | float): Multiplier parameter lambda.

        Returns:
            np.ndarray | float: Sum of ``beta_i^2 / (beta_i^2 + lam)^2``.
        """
        if isinstance(lam, np.ndarray):
            return np.sum(beta_sq_nz[:, None] / (beta_sq_nz[:, None] + lam[None, :]) ** 2, axis=0)
        return float(np.sum(beta_sq_nz / (beta_sq_nz + lam) ** 2))

    a_right = float(A_beta(smin_right))
    b_sqrt_right = float(np.sqrt(B_beta(smin_right)))

    def _create_lams(a: float, b: float) -> np.ndarray:
        """Create a dense logarithmic sampling grid for a pole interval.

        Args:
            a (float): Lower pole boundary.
            b (float): Upper pole boundary.

        Returns:
            np.ndarray: Grid of sampling points lambda in ``(a, b)``.
        """
        t_mid = (a + b) / 2
        eps_margin = max(1e-14, (b - a) * 1e-12)
        g1 = a + np.logspace(np.log10(eps_margin), np.log10(t_mid - a), 150)
        g2 = b - np.logspace(np.log10(eps_margin), np.log10(b - t_mid), 150)[::-1]
        return np.concatenate([g1, g2])

    early_lams = [_create_lams(a, b) for a, b in early_intervals]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        early_a = [cast("np.ndarray", A_beta(lams)) for lams in early_lams]
        early_b_sqrt = [np.sqrt(B_beta(lams)) for lams in early_lams]

    def _find_root_cached(
        a_vals: np.ndarray,
        b_sqrt: np.ndarray,
        lams: np.ndarray,
        abs_t: float,
        sign: float,
        func: Callable[[float], float],
    ) -> float | None:
        """Find a single root in an interval using pre-computed grid caching.

        Args:
            a_vals (np.ndarray): Pre-evaluated A_beta values on grid.
            b_sqrt (np.ndarray): Pre-evaluated sqrt(B_beta) values on grid.
            lams (np.ndarray): Grid lambda points.
            abs_t (float): Absolute projection value ``|t|``.
            sign (float): Sign multiplier for secular equation branch (+1 or -1).
            func (Callable[[float], float]): Objective secular equation function.

        Returns:
            float | None: Evaluated root lambda, or None if no root exists.
        """
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v_arr = a_vals + sign * abs_t * b_sqrt

        valid_mask = ~np.isnan(v_arr) & ~np.isinf(v_arr)
        if not np.any(valid_mask):
            return None

        valid_lams = lams[valid_mask]
        f_vals = v_arr[valid_mask]

        if np.all(f_vals > 0) or np.all(f_vals < 0):
            return None

        sign_changes = np.where(np.diff(np.sign(f_vals)) != 0)[0]

        if len(sign_changes) > 0:
            idx = int(sign_changes[0])
            bracket_a, bracket_b = valid_lams[idx], valid_lams[idx + 1]
            if f_vals[idx] * f_vals[idx + 1] > 0:
                return None
            try:
                res = cast("Any", root_scalar(func, bracket=(bracket_a, bracket_b), method="brentq"))
                if res.converged:
                    return float(res.root)
            except Exception:  # noqa: BLE001, S110
                pass
        return None

    def _solve_secular_main_branch(abs_t: float, secular_func: Callable[[float], float]) -> float | None:
        """Find the candidate root in (-beta_min^2, inf) for the main branch.

        Args:
            abs_t (float): Absolute projection value ``|t|``.
            secular_func (Callable[[float], float]): Secular equation function.

        Returns:
            float | None: Evaluated root lambda, or None if no root exists.
        """
        cur_eps = epsilon
        left = -beta_min_sq + cur_eps
        f_left = float(A_beta(left)) - abs_t * float(np.sqrt(B_beta(left)))

        if abs_t > np.sqrt(beta_min_sq) + 1e-10:
            attempts = 0
            while f_left > 0 and attempts < 10:  # noqa: PLR2004
                next_eps = cur_eps / 10
                if -beta_min_sq + next_eps == -beta_min_sq:
                    break
                cur_eps = next_eps
                left = -beta_min_sq + cur_eps
                f_left = float(A_beta(left)) - abs_t * float(np.sqrt(B_beta(left)))
                attempts += 1

        local_smin_right = smin_right
        local_a_right = a_right
        local_b_sqrt_right = b_sqrt_right
        f_right = local_a_right - abs_t * local_b_sqrt_right
        if f_right <= 0:
            for _ in range(100):
                local_smin_right *= 10.0
                local_a_right = float(A_beta(local_smin_right))
                local_b_sqrt_right = float(np.sqrt(B_beta(local_smin_right)))
                f_right = local_a_right - abs_t * local_b_sqrt_right
                if f_right > 0:
                    break

        try:
            if f_left * f_right <= 0:
                res = cast("Any", root_scalar(secular_func, bracket=(left, local_smin_right), method="brentq"))
                return float(res.root)
        except Exception:  # noqa: BLE001, S110
            pass
        return None

    def _extract_extrema(
        candidates: list[tuple[float, np.ndarray]], c_default: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Normalize candidates and extract min and max extrema.

        Args:
            candidates (list[tuple[float, np.ndarray]]): List of ``(s_val, c_vec)`` tuples.
            c_default (np.ndarray): Default fallback unit direction vector.

        Returns:
            tuple[float, float, np.ndarray, np.ndarray]:
                Evaluated ``(s_min, s_max, c_min, c_max)``.
        """
        valid_candidates: list[tuple[float, np.ndarray]] = []
        for s_val, c_vec in candidates:
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm > 0:
                c_unit = c_vec / c_norm
                valid_candidates.append((s_val, c_unit))

        if not valid_candidates:
            return w_boundary, w_boundary, c_default, c_default

        min_cand = min(valid_candidates, key=lambda item: item[0])
        max_cand = max(valid_candidates, key=lambda item: item[0])

        return min_cand[0], max_cand[0], min_cand[1], max_cand[1]

    def _make_candidate_vec(c_nz: np.ndarray) -> np.ndarray:
        """Construct the full d-dimensional direction vector from its nonzero-coefficient components.

        Args:
            c_nz (np.ndarray): Candidate sub-vector on non-zero beta indices.

        Returns:
            np.ndarray: Full d-dimensional unit vector padded with zero component slack.
        """
        c_full = np.zeros(d)
        c_full[nz_mask] = c_nz
        if zero_idxs:
            slack_sq = max(0.0, 1.0 - float(np.sum(c_nz**2)))
            c_full[zero_idxs[0]] = np.sqrt(slack_sq)
        return c_full

    def get_extrema(  # noqa: C901, PLR0912, PLR0915
        t: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the Lagrangian candidates on the slice for the projection t.

        Args:
            t (float): Projection parameter scalar ``beta^T c``.

        Returns:
            tuple[float, float, np.ndarray, np.ndarray]:
                Profile extrema tuple ``(s_min, s_max, c_min, c_max)``.
        """
        c_bound = c_boundary_positive if t >= 0 else -c_boundary_positive

        # Case 1: Boundary projection t = +-||beta||
        if abs(t) >= beta_norm - BOUNDARY_PROJECTION_ATOL:
            return w_boundary, w_boundary, c_bound, c_bound

        # Case 2: Null projection t = 0
        if abs(t) < ZERO_PROJECTION_ATOL:
            candidates_t0: list[tuple[float, np.ndarray]] = []

            # 2a. Regular candidates: A_beta(lam) = 0
            def secular_func_t0(lam: float) -> float:
                return float(A_beta(lam))

            lam_main = _solve_secular_main_branch(0.0, secular_func_t0)
            if lam_main is not None:
                b_val = float(B_beta(lam_main))
                c_nz = beta_nz / (np.sqrt(b_val) * (beta_sq_nz + lam_main))
                w_cand = -lam_main
                candidates_t0.append((w_cand, _make_candidate_vec(c_nz)))

            for i in range(len(early_intervals)):
                lams = early_lams[i]
                a_vals = early_a[i]
                b_sqrt = early_b_sqrt[i]
                lam = _find_root_cached(a_vals, b_sqrt, lams, 0.0, 1.0, lambda l_val: float(A_beta(l_val)))
                if lam is not None:
                    b_val = float(B_beta(lam))
                    c_nz = beta_nz / (np.sqrt(b_val) * (beta_sq_nz + lam))
                    w_cand = -lam
                    candidates_t0.append((w_cand, _make_candidate_vec(c_nz)))

            # 2b. Singular candidates: |I_rho| >= 2, beta^T c = 0, ||c|| = 1, w = rho
            for rho_val, idxs in singular_groups:
                if len(idxs) >= 2:  # noqa: PLR2004
                    c_cand = np.zeros(d)
                    i1, i2 = idxs[0], idxs[1]
                    b1, b2 = beta[i1], beta[i2]
                    denom = float(np.sqrt(b1**2 + b2**2))
                    c_cand[i1] = b2 / denom
                    c_cand[i2] = -b1 / denom
                    candidates_t0.append((rho_val, c_cand))

            # 2c. Zero coefficient candidate: beta_i = 0
            if zero_idxs:
                c_zero = np.zeros(d)
                c_zero[zero_idxs[0]] = 1.0
                candidates_t0.append((0.0, c_zero))

            return _extract_extrema(candidates_t0, c_bound)

        # Case 3: General intermediate projection 0 < |t| < ||beta||
        abs_t = abs(t)
        candidates_general: list[tuple[float, np.ndarray]] = []

        # 3a. Regular candidates: t^2 * B_beta(lam) = A_beta(lam)^2
        def secular_func_general(lam: float) -> float:
            return float(A_beta(lam) - abs_t * np.sqrt(B_beta(lam)))

        lam_main = _solve_secular_main_branch(abs_t, secular_func_general)
        if lam_main is not None:
            a_val = float(A_beta(lam_main))
            c_nz = (t * beta_nz) / (a_val * (beta_sq_nz + lam_main))
            w_cand = (t**2 / a_val) - lam_main
            candidates_general.append((w_cand, _make_candidate_vec(c_nz)))

        for i in range(len(early_intervals)):
            lams = early_lams[i]
            a_vals = early_a[i]
            b_sqrt = early_b_sqrt[i]
            for sign in (-1.0, 1.0):
                lam = _find_root_cached(
                    a_vals,
                    b_sqrt,
                    lams,
                    abs_t,
                    sign,
                    lambda l_val, s_sign=sign: float(A_beta(l_val) + s_sign * abs_t * np.sqrt(B_beta(l_val))),
                )
                if lam is not None:
                    a_val = float(A_beta(lam))
                    c_nz = (t * beta_nz) / (a_val * (beta_sq_nz + lam))
                    w_cand = (t**2 / a_val) - lam
                    candidates_general.append((w_cand, _make_candidate_vec(c_nz)))

        # 3b. Singular candidates: |I_rho| = 1 with t^2 = rho, or |I_rho| >= 2 with t^2 <= |I_rho| * rho
        for rho_val, idxs in singular_groups:
            k = len(idxs)
            if k == 1 and np.isclose(abs_t**2, rho_val, rtol=1e-4, atol=1e-6):
                idx = idxs[0]
                c_cand = np.zeros(d)
                c_cand[idx] = t / beta[idx]
                candidates_general.append((rho_val, c_cand))
            elif k >= 2 and abs_t**2 <= k * rho_val + 1e-8:  # noqa: PLR2004
                c_cand = np.zeros(d)
                proj_coeff = t / (k * rho_val)
                for idx in idxs:
                    c_cand[idx] = proj_coeff * beta[idx]
                slack_sq = max(0.0, 1.0 - (abs_t**2 / (k * rho_val)))
                if slack_sq > 1e-12:  # noqa: PLR2004
                    i1, i2 = idxs[0], idxs[1]
                    b1, b2 = beta[i1], beta[i2]
                    denom = float(np.sqrt(b1**2 + b2**2))
                    u1 = b2 / denom
                    u2 = -b1 / denom
                    c_cand[i1] += float(np.sqrt(slack_sq)) * u1
                    c_cand[i2] += float(np.sqrt(slack_sq)) * u2
                candidates_general.append((rho_val, c_cand))

        # 3c. Zero coefficient candidate: beta_i = 0
        if zero_idxs:
            i0 = zero_idxs[0]
            slack_sq = max(0.0, 1.0 - (abs_t / beta_norm) ** 2)
            c_zero = np.zeros(d)
            c_zero[nz_mask] = (t / (beta_norm**2)) * beta_nz
            c_zero[i0] = np.sqrt(slack_sq)
            w_zero = ((t / beta_norm) ** 2) * w_boundary
            candidates_general.append((w_zero, c_zero))

        return _extract_extrema(candidates_general, c_bound)

    return get_extrema
