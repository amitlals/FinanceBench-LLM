"""
FinanceBench-LLM utilities package.

All public names are re-exported here for backward compatibility.
Notebooks can use: from src.config import set_seed
Or the legacy: from src.utils import set_seed
"""

from .config import (
    DEFAULT_NIM_MODEL,
    EXPORTED_MODELS_DIR,
    JUDGE_MODEL,
    MAX_RETRIES,
    NIM_BASE_URL,
    PROJECT_ROOT,
    RESULTS_DIR,
    RETRY_DELAY,
    SEED,
    set_seed,
)
from .data import (
    format_finance_prompt,
    format_for_nemo_customizer,
    load_financebench,
)
from .elo import compute_elo_ratings, generate_pairwise_comparisons
from .export import export_lora_to_hf_peft
from .io_utils import (
    create_comparison_table,
    load_results,
    print_metrics_summary,
    save_results,
)
from .judge import (
    JUDGE_PROMPTS,
    LLMJudge,
    _extract_score_from_text,
)
from .metrics import (
    _normalize_text,
    _token_f1,
    compute_exact_match,
    compute_f1_score,
)
from .mlflow_utils import log_metrics_to_mlflow
from .nim_client import NIMInferenceClient
from .visualization import (
    plot_comparison_bar_chart,
    plot_elo_ratings,
    plot_training_loss,
)

__all__ = [
    # Config
    "DEFAULT_NIM_MODEL",
    "NIM_BASE_URL",
    "JUDGE_MODEL",
    "SEED",
    "MAX_RETRIES",
    "RETRY_DELAY",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "EXPORTED_MODELS_DIR",
    "set_seed",
    # NIM Client
    "NIMInferenceClient",
    # Data
    "load_financebench",
    "format_finance_prompt",
    "format_for_nemo_customizer",
    # Metrics
    "compute_exact_match",
    "compute_f1_score",
    "_normalize_text",
    "_token_f1",
    # Judge
    "JUDGE_PROMPTS",
    "LLMJudge",
    "_extract_score_from_text",
    # ELO
    "compute_elo_ratings",
    "generate_pairwise_comparisons",
    # I/O
    "save_results",
    "load_results",
    "create_comparison_table",
    "print_metrics_summary",
    # MLflow
    "log_metrics_to_mlflow",
    # Export
    "export_lora_to_hf_peft",
    # Visualization
    "plot_comparison_bar_chart",
    "plot_training_loss",
    "plot_elo_ratings",
]
