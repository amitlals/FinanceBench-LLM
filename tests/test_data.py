"""Tests for data formatting utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import format_finance_prompt


class TestFormatFinancePrompt:
    def test_basic_question(self):
        result = format_finance_prompt("What was revenue?")
        assert "What was revenue?" in result
        assert "Answer concisely" in result

    def test_with_context(self):
        result = format_finance_prompt(
            "What was revenue?",
            context="Revenue was $10B in 2023.",
        )
        assert "Revenue was $10B" in result

    def test_with_evidence(self):
        result = format_finance_prompt(
            "What was revenue?",
            evidence="Per the 10-K filing, revenue was $10B.",
        )
        assert "Per the 10-K filing" in result
        assert "Evidence from SEC filing" in result

    def test_evidence_takes_priority_over_context(self):
        result = format_finance_prompt(
            "What was revenue?",
            context="Context text",
            evidence="Evidence text",
        )
        assert "Evidence text" in result
        # Context should not appear when evidence is provided
        assert "Context text" not in result

    def test_with_icl_examples(self, sample_icl_examples):
        result = format_finance_prompt(
            "What was revenue?",
            icl_examples=sample_icl_examples,
        )
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "$10 billion" in result

    def test_empty_question(self):
        result = format_finance_prompt("")
        assert "Question:" in result
