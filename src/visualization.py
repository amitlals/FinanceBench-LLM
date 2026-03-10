"""
Visualization helpers for evaluation charts and comparison plots.
"""

from typing import Dict, List, Optional

import pandas as pd


def plot_comparison_bar_chart(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Model Comparison — FinanceBench Evaluation",
    save_path: Optional[str] = None,
) -> None:
    """Generate a grouped bar chart comparing model configurations."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    df = pd.DataFrame(metrics).T
    df.plot(
        kind="bar", figsize=(12, 6), width=0.8,
        edgecolor="black", linewidth=0.5,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model Configuration", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(
        title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Chart saved to {save_path}")
    plt.close()


def plot_training_loss(
    losses: List[float],
    title: str = "LoRA Training Loss — NeMo Customizer",
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss curve."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color="#76b900")  # NVIDIA green
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Chart saved to {save_path}")
    plt.close()


def plot_elo_ratings(
    ratings: Dict[str, float],
    title: str = "ELO-Style Model Ranking",
    save_path: Optional[str] = None,
) -> None:
    """Plot ELO ratings as a horizontal bar chart."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    sorted_ratings = dict(
        sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    )
    colors = ["#76b900", "#667eea", "#f6ad55"][:len(sorted_ratings)]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(
        list(sorted_ratings.keys()),
        list(sorted_ratings.values()),
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, rating in zip(bars, sorted_ratings.values()):
        plt.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{rating:.0f}",
            va="center",
            fontweight="bold",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("ELO Rating", fontsize=12)
    plt.axvline(
        x=1000, color="gray", linestyle="--",
        alpha=0.5, label="Initial Rating",
    )
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Chart saved to {save_path}")
    plt.close()
