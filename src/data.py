"""
Dataset loading and formatting utilities for FinanceBench.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import SEED


def load_financebench(
    split_ratio: float = 0.8,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PatronusAI/financebench from Hugging Face and split into train/test.

    Returns:
        (train_df, test_df) with columns: question, answer, evidence, context
    """
    from datasets import load_dataset

    print("[INFO] Loading PatronusAI/financebench from Hugging Face...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    df = ds.to_pandas()

    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if "question" in col_lower or "query" in col_lower:
            column_mapping[col] = "question"
        elif col_lower in ("answer", "ground_truth", "gold_answer"):
            column_mapping[col] = "answer"
        elif "evidence" in col_lower or "passage" in col_lower:
            column_mapping[col] = "evidence"
        elif "context" in col_lower or "doc" in col_lower:
            column_mapping[col] = "context"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    for col in ["question", "answer"]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available: {list(df.columns)}"
            )

    if "evidence" not in df.columns:
        df["evidence"] = ""
    if "context" not in df.columns:
        df["context"] = ""

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(
        f"[INFO] Dataset loaded: {len(df)} total | "
        f"{len(train_df)} train | {len(test_df)} test"
    )
    return train_df, test_df


def format_finance_prompt(
    question: str,
    context: str = "",
    evidence: str = "",
    icl_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Format a financial QA prompt with optional context and ICL examples."""
    parts = []

    if icl_examples:
        parts.append(
            "Here are some examples of financial question answering:\n"
        )
        for i, ex in enumerate(icl_examples, 1):
            parts.append(f"Example {i}:")
            if ex.get("context"):
                parts.append(f"Context: {ex['context'][:500]}")
            parts.append(f"Question: {ex['question']}")
            parts.append(f"Answer: {ex['answer']}\n")

    if evidence:
        parts.append(f"Evidence from SEC filing:\n{evidence}\n")
    elif context:
        parts.append(f"Context:\n{context[:1000]}\n")

    parts.append(f"Question: {question}")
    parts.append(
        "\nAnswer concisely and accurately based on the provided information:"
    )

    return "\n".join(parts)


def format_for_nemo_customizer(
    df: pd.DataFrame,
    output_path: str,
) -> str:
    """Format FinanceBench data into JSONL for NeMo Customizer LoRA training."""
    records = []
    for _, row in df.iterrows():
        prompt = format_finance_prompt(
            question=row["question"],
            context=row.get("context", ""),
            evidence=row.get("evidence", ""),
        )
        records.append({"input": prompt, "output": row["answer"]})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"[INFO] Wrote {len(records)} records to {output_path}")
    return str(output_path)
