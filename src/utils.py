"""
FinanceBench-LLM: Shared Utilities (Backward Compatibility Shim)
================================================================
This file re-exports all utilities from the modularized src/ package.
Existing notebook imports like `from src.utils import X` continue to work.

For new code, import directly from submodules:
    from src.config import set_seed
    from src.nim_client import NIMInferenceClient
    from src.metrics import compute_exact_match

Author: Amit Lal
"""

# Re-export everything from the modularized package
from .config import (  # noqa: F401
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
from .data import (  # noqa: F401
    format_finance_prompt,
    format_for_nemo_customizer,
    load_financebench,
)
from .elo import (  # noqa: F401
    compute_elo_ratings,
    generate_pairwise_comparisons,
)
from .export import export_lora_to_hf_peft  # noqa: F401
from .io_utils import (  # noqa: F401
    create_comparison_table,
    load_results,
    print_metrics_summary,
    save_results,
)
from .judge import (  # noqa: F401
    JUDGE_PROMPTS,
    LLMJudge,
    _extract_score_from_text,
)
from .metrics import (  # noqa: F401
    _normalize_text,
    _token_f1,
    compute_exact_match,
    compute_f1_score,
)
from .mlflow_utils import log_metrics_to_mlflow  # noqa: F401
from .nim_client import NIMInferenceClient  # noqa: F401
from .visualization import (  # noqa: F401
    plot_comparison_bar_chart,
    plot_elo_ratings,
    plot_training_loss,
)
