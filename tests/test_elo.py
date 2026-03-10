"""Tests for ELO rating system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.elo import compute_elo_ratings


class TestEloRatings:
    def test_winner_gains_rating(self):
        comparisons = [
            {"model_a": "A", "model_b": "B", "winner": "A"},
        ]
        ratings = compute_elo_ratings(comparisons)
        assert ratings["A"] > ratings["B"]

    def test_initial_ratings(self):
        comparisons = [
            {"model_a": "A", "model_b": "B", "winner": "tie"},
        ]
        ratings = compute_elo_ratings(
            comparisons, initial_rating=1500.0
        )
        # Tie should keep ratings close to initial
        assert abs(ratings["A"] - 1500.0) < 1.0
        assert abs(ratings["B"] - 1500.0) < 1.0

    def test_consistent_winner_dominates(self):
        comparisons = [
            {"model_a": "A", "model_b": "B", "winner": "A"}
            for _ in range(20)
        ]
        ratings = compute_elo_ratings(comparisons)
        assert ratings["A"] > ratings["B"]
        # After 20 wins, gap should be significant
        assert ratings["A"] - ratings["B"] > 200

    def test_three_models_ranking(self):
        comparisons = []
        # A always beats B, B always beats C
        for _ in range(10):
            comparisons.append(
                {"model_a": "A", "model_b": "B", "winner": "A"}
            )
            comparisons.append(
                {"model_a": "B", "model_b": "C", "winner": "B"}
            )
            comparisons.append(
                {"model_a": "A", "model_b": "C", "winner": "A"}
            )
        ratings = compute_elo_ratings(comparisons)
        assert ratings["A"] > ratings["B"] > ratings["C"]

    def test_empty_comparisons(self):
        ratings = compute_elo_ratings([])
        assert ratings == {}

    def test_custom_k_factor(self):
        comparisons = [
            {"model_a": "A", "model_b": "B", "winner": "A"},
        ]
        ratings_low_k = compute_elo_ratings(comparisons, k_factor=8.0)
        ratings_high_k = compute_elo_ratings(comparisons, k_factor=64.0)
        # Higher K means bigger rating changes
        gap_low = ratings_low_k["A"] - ratings_low_k["B"]
        gap_high = ratings_high_k["A"] - ratings_high_k["B"]
        assert gap_high > gap_low
