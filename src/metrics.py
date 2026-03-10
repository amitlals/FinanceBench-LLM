"""
Evaluation metrics for financial QA: Exact Match and token-level F1.
"""

from typing import List

import numpy as np


def compute_exact_match(
    predictions: List[str], references: List[str]
) -> float:
    """Exact match score (normalized)."""
    if not predictions:
        return 0.0
    matches = sum(
        1
        for p, r in zip(predictions, references)
        if _normalize_text(p) == _normalize_text(r)
    )
    return matches / len(predictions)


def compute_f1_score(
    predictions: List[str], references: List[str]
) -> float:
    """Token-level F1 score averaged across examples (GSM8K-style)."""
    if not predictions:
        return 0.0
    scores = [
        _token_f1(_normalize_text(p), _normalize_text(r))
        for p, r in zip(predictions, references)
    ]
    return float(np.mean(scores))


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    text = text.lower().strip()
    for prefix in ["answer:", "the answer is", "based on the context,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = " ".join(text.split())
    return text


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
