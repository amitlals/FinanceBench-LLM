"""
LLM-as-a-Judge evaluation following the NeMo Evaluator pattern.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import JUDGE_MODEL
from .nim_client import NIMInferenceClient

JUDGE_PROMPTS = {
    "correctness": """You are evaluating the correctness of a financial QA answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Rate the model answer's correctness on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially correct but major errors
3 = Mostly correct with minor inaccuracies
4 = Correct with minor differences in wording
5 = Perfectly correct and matches reference

Respond with ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}""",

    "faithfulness": """You are evaluating the faithfulness of a financial QA answer to the provided evidence.

Evidence: {evidence}
Question: {question}
Model Answer: {prediction}

Rate faithfulness on a scale of 1-5:
1 = Answer contradicts or ignores evidence
2 = Answer loosely related to evidence
3 = Answer partially grounded in evidence
4 = Answer well-grounded with minor extrapolation
5 = Answer fully supported by evidence

Respond with ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}""",

    "conciseness": """You are evaluating the conciseness of a financial QA answer.

Question: {question}
Model Answer: {prediction}

Rate conciseness on a scale of 1-5:
1 = Extremely verbose, buries the answer
2 = Too long, includes unnecessary content
3 = Reasonable length but could be shorter
4 = Concise with minimal extra content
5 = Perfectly concise, directly answers the question

Respond with ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}""",
}


class LLMJudge:
    """LLM-as-a-Judge evaluator following the NeMo Evaluator pattern."""

    def __init__(
        self,
        client: NIMInferenceClient,
        judge_model: str = JUDGE_MODEL,
    ):
        self.client = client
        self.original_model = client.model
        self.judge_model = judge_model

    def evaluate_single(
        self,
        question: str,
        prediction: str,
        reference: str,
        evidence: str = "",
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single prediction across all specified criteria."""
        if criteria is None:
            criteria = ["correctness", "faithfulness", "conciseness"]

        results = {}
        self.client.model = self.judge_model

        for criterion in criteria:
            if criterion not in JUDGE_PROMPTS:
                print(f"[WARN] Unknown criterion: {criterion}")
                continue

            prompt = JUDGE_PROMPTS[criterion].format(
                question=question,
                prediction=prediction,
                reference=reference,
                evidence=evidence or "Not provided",
            )

            response = self.client.query(
                prompt,
                system_prompt=(
                    "You are an expert evaluation judge. "
                    "Respond only with valid JSON."
                ),
                temperature=0.0,
                max_tokens=256,
            )

            try:
                parsed = json.loads(response)
                results[criterion] = {
                    "score": int(parsed.get("score", 0)),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except (json.JSONDecodeError, ValueError):
                score = _extract_score_from_text(response)
                results[criterion] = {
                    "score": score,
                    "reasoning": response[:200],
                }

        self.client.model = self.original_model
        return results

    def evaluate_batch(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str],
        evidences: Optional[List[str]] = None,
        criteria: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Evaluate a batch of predictions. Returns a DataFrame."""
        if evidences is None:
            evidences = [""] * len(questions)
        if criteria is None:
            criteria = ["correctness", "faithfulness", "conciseness"]

        all_results = []
        total = len(questions)

        for i in range(total):
            print(f"[INFO] Judging {i + 1}/{total}...", end="\r")
            result = self.evaluate_single(
                question=questions[i],
                prediction=predictions[i],
                reference=references[i],
                evidence=evidences[i],
                criteria=criteria,
            )
            row = {"idx": i, "question": questions[i][:100]}
            for criterion, data in result.items():
                row[f"{criterion}_score"] = data["score"]
                row[f"{criterion}_reasoning"] = data["reasoning"]
            all_results.append(row)
            time.sleep(0.3)

        print(
            f"\n[INFO] Batch evaluation complete: {total} examples judged."
        )
        return pd.DataFrame(all_results)


def _extract_score_from_text(text: str) -> int:
    """Fallback: extract numeric score from judge response text."""
    matches = re.findall(r"\b([1-5])\b", text)
    if matches:
        return int(matches[0])
    return 0
