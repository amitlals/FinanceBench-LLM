"""Shared test fixtures for FinanceBench-LLM."""

import pytest


@pytest.fixture
def sample_predictions():
    return [
        "revenue was $45.2 billion",
        "net income increased by 12%",
        "the operating margin was 25%",
    ]


@pytest.fixture
def sample_references():
    return [
        "Revenue was $45.2 billion",
        "Net income grew by 12%",
        "Operating margin was 25%",
    ]


@pytest.fixture
def sample_icl_examples():
    return [
        {
            "question": "What was revenue?",
            "answer": "$10 billion",
            "context": "The company reported revenue of $10 billion.",
        },
        {
            "question": "What was net income?",
            "answer": "$2 billion",
            "context": "Net income reached $2 billion.",
        },
    ]
