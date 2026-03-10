"""
Results I/O helpers for saving and loading evaluation data.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import RESULTS_DIR


def save_results(
    data: Any,
    filename: str,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Save results to JSON or CSV in the results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / filename

    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, (dict, list)):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        with open(filepath, "w") as f:
            f.write(str(data))

    print(f"[INFO] Saved results to {filepath}")
    return str(filepath)


def load_results(
    filename: str, results_dir: Path = RESULTS_DIR
) -> Any:
    """Load results from the results directory."""
    filepath = results_dir / filename
    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".json":
        with open(filepath) as f:
            return json.load(f)
    else:
        with open(filepath) as f:
            return f.read()


def create_comparison_table(
    metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Create a formatted comparison table from metrics dictionaries."""
    df = pd.DataFrame(metrics).T
    df.index.name = "Model"
    return df.round(4)


def print_metrics_summary(
    metrics: Dict[str, float], label: str = "Model"
) -> None:
    """Pretty-print metrics summary."""
    print(f"\n{'=' * 60}")
    print(f"  Metrics Summary: {label}")
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print(f"{'=' * 60}\n")
