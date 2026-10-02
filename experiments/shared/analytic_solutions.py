# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Storage and lookup of precomputed analytical solutions."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


def get_solutions_path() -> Path:
    """Return the default path for the precomputed analytical solutions.

    Returns:
        Path: Default path to the analytical solutions pickle file.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "experiments" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "analytic_solutions.pkl"


def load_analytic_solutions(
    path: Path | None = None,
) -> dict[tuple[str, str, float, float, float, float], dict[str, Any]]:
    """Load the precomputed analytical solutions from disk.

    Args:
        path (Path | None): Custom path to load analytical solutions from. If None,
            uses default path.

    Returns:
        dict[tuple[str, str, float, float, float, float], dict[str, Any]]: Dictionary
            mapping solution keys to precomputed solution payloads.
    """
    if path is None:
        path = get_solutions_path()

    if not path.exists():
        msg = (
            f"Precomputed analytical solutions not found at {path}. "
            "Please run `python experiments/shared/precompute_analytic.py` first."
        )
        raise FileNotFoundError(msg)

    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_analytic_solutions(
    solutions: dict[tuple[str, str, float, float, float, float], dict[str, Any]],
    path: Path | None = None,
) -> None:
    """Save the analytical solutions to disk, merging with existing solutions if present.

    Args:
        solutions (dict[tuple[str, str, float, float, float, float], dict[str, Any]]): Dictionary
            of solutions to save.
        path (Path | None): Custom path to save analytical solutions to. If None,
            uses default path.
    """
    if path is None:
        path = get_solutions_path()

    existing_solutions = {}
    if path.exists():
        try:
            with open(path, "rb") as f:
                existing_solutions = pickle.load(f)  # noqa: S301
        except (OSError, pickle.PickleError, EOFError):
            existing_solutions = {}

    existing_solutions.update(solutions)
    with open(path, "wb") as f:
        pickle.dump(existing_solutions, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_solutions_key(
    artifact_id: str, query_id: str, noise_config: Any
) -> tuple[str, str, float, float, float, float]:
    """Create a standardized lookup key for analytical solutions.

    Args:
        artifact_id (str): Identifier of the artifact (e.g. 'dataset__model').
        query_id (str): Identifier of the query point (e.g. 'p50').
        noise_config (Any): Noise configuration object or dictionary containing
            alpha, mu, sigma, and env_std.

    Returns:
        tuple[str, str, float, float, float, float]: Standardized 6-tuple lookup key.
    """
    if isinstance(noise_config, dict):
        alpha = noise_config["alpha"]
        mu = noise_config["mu"]
        sigma = noise_config["sigma"]
        env_std = noise_config["env_std"]
    else:
        alpha = noise_config.alpha
        mu = noise_config.mu
        sigma = noise_config.sigma
        env_std = noise_config.env_std
    return (
        str(artifact_id),
        str(query_id),
        round(float(alpha), 6),
        round(float(mu), 6),
        round(float(sigma), 6),
        round(float(env_std), 6),
    )
