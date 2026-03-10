"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    _normalize_text,
    _token_f1,
    compute_exact_match,
    compute_f1_score,
)


class TestNormalizeText:
    def test_lowercase_and_strip(self):
        assert _normalize_text("  Hello World  ") == "hello world"

    def test_remove_answer_prefix(self):
        assert _normalize_text("Answer: revenue was $45B") == "revenue was $45b"

    def test_remove_based_on_context_prefix(self):
        result = _normalize_text("Based on the context, revenue was $45B")
        assert result == "revenue was $45b"

    def test_normalize_whitespace(self):
        assert _normalize_text("revenue   was    high") == "revenue was high"

    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_already_normalized(self):
        assert _normalize_text("revenue was $45b") == "revenue was $45b"


class TestTokenF1:
    def test_perfect_match(self):
        assert _token_f1("revenue 45 billion", "revenue 45 billion") == 1.0

    def test_no_overlap(self):
        assert _token_f1("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = _token_f1("revenue was high", "revenue was low")
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert _token_f1("", "reference text") == 0.0

    def test_empty_reference(self):
        assert _token_f1("prediction text", "") == 0.0

    def test_both_empty(self):
        assert _token_f1("", "") == 0.0


class TestExactMatch:
    def test_all_match(self):
        preds = ["revenue was $45b", "net income 12%"]
        refs = ["Revenue was $45B", "Net Income 12%"]
        assert compute_exact_match(preds, refs) == 1.0

    def test_none_match(self):
        preds = ["foo", "bar"]
        refs = ["baz", "qux"]
        assert compute_exact_match(preds, refs) == 0.0

    def test_partial_match(self):
        preds = ["revenue was $45b", "wrong answer"]
        refs = ["Revenue was $45B", "correct answer"]
        assert compute_exact_match(preds, refs) == 0.5

    def test_empty_lists(self):
        assert compute_exact_match([], []) == 0.0


class TestF1Score:
    def test_perfect_f1(self):
        preds = ["revenue was $45 billion"]
        refs = ["revenue was $45 billion"]
        assert compute_f1_score(preds, refs) == 1.0

    def test_zero_f1(self):
        preds = ["hello world"]
        refs = ["foo bar baz"]
        assert compute_f1_score(preds, refs) == 0.0

    def test_empty_lists(self):
        assert compute_f1_score([], []) == 0.0

    def test_with_prefix_normalization(self):
        preds = ["Answer: revenue was $45B"]
        refs = ["Revenue was $45B"]
        score = compute_f1_score(preds, refs)
        assert score > 0.5
