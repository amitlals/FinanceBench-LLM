"""
ELO rating system for model comparison.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import SEED


def compute_elo_ratings(
    comparisons: List[Dict[str, Any]],
    k_factor: float = 32.0,
    initial_rating: float = 1000.0,
) -> Dict[str, float]:
    """
    Compute ELO ratings from pairwise comparisons.

    Args:
        comparisons: List of {"model_a", "model_b", "winner"}
        k_factor: ELO K-factor (higher = more volatile)
        initial_rating: Starting ELO for all models
    """
    models = set()
    for c in comparisons:
        models.add(c["model_a"])
        models.add(c["model_b"])

    ratings = {m: initial_rating for m in models}

    for comp in comparisons:
        a, b = comp["model_a"], comp["model_b"]
        winner = comp["winner"]

        exp_a = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
        exp_b = 1.0 - exp_a

        if winner == a:
            score_a, score_b = 1.0, 0.0
        elif winner == b:
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        ratings[a] += k_factor * (score_a - exp_a)
        ratings[b] += k_factor * (score_b - exp_b)

    return ratings


def generate_pairwise_comparisons(
    results: Dict[str, pd.DataFrame],
    metric_col: str = "correctness_score",
    n_comparisons: int = 1000,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Generate pairwise comparisons between models based on judge scores."""
    rng = np.random.RandomState(seed)
    model_names = list(results.keys())
    comparisons = []

    for _ in range(n_comparisons):
        a, b = rng.choice(model_names, size=2, replace=False)
        min_len = min(len(results[a]), len(results[b]))
        idx = rng.randint(0, min_len)

        score_a = results[a].iloc[idx].get(metric_col, 0)
        score_b = results[b].iloc[idx].get(metric_col, 0)

        if score_a > score_b:
            winner = a
        elif score_b > score_a:
            winner = b
        else:
            winner = "tie"

        comparisons.append(
            {"model_a": a, "model_b": b, "winner": winner}
        )

    return comparisons
