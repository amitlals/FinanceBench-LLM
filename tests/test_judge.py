"""Tests for LLM-as-a-Judge utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.judge import JUDGE_PROMPTS, _extract_score_from_text


class TestExtractScoreFromText:
    def test_json_like(self):
        assert _extract_score_from_text('{"score": 4, "reasoning": "Good"}') == 4

    def test_plain_number(self):
        assert _extract_score_from_text("The score is 3 out of 5") == 3

    def test_no_score(self):
        assert _extract_score_from_text("No numeric score here") == 0

    def test_score_at_boundary_1(self):
        assert _extract_score_from_text("Rating: 1") == 1

    def test_score_at_boundary_5(self):
        assert _extract_score_from_text("Perfect score of 5") == 5

    def test_ignores_out_of_range(self):
        # 7 is outside 1-5 range, should not match
        assert _extract_score_from_text("Score: 7 out of 10") == 0

    def test_first_match_wins(self):
        assert _extract_score_from_text("Between 2 and 4, I give 3") == 2

    def test_empty_string(self):
        assert _extract_score_from_text("") == 0


class TestJudgePrompts:
    def test_all_criteria_exist(self):
        assert "correctness" in JUDGE_PROMPTS
        assert "faithfulness" in JUDGE_PROMPTS
        assert "conciseness" in JUDGE_PROMPTS

    def test_prompts_have_placeholders(self):
        for criterion, prompt in JUDGE_PROMPTS.items():
            assert "{question}" in prompt
            assert "{prediction}" in prompt

    def test_correctness_has_reference(self):
        assert "{reference}" in JUDGE_PROMPTS["correctness"]

    def test_faithfulness_has_evidence(self):
        assert "{evidence}" in JUDGE_PROMPTS["faithfulness"]
