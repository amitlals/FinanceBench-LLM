"""
Configuration loader for FinanceBench-LLM.
Loads centralized settings from config.yaml.
"""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORTED_MODELS_DIR = PROJECT_ROOT / "exported_models" / "lora_adapter"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Default constants (used if config.yaml is not available)
DEFAULT_NIM_MODEL = "meta/llama-3.1-8b-instruct"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL = "meta/llama-3.1-70b-instruct"
SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 2


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml if available."""
    try:
        import yaml

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f)
    except ImportError:
        pass
    return {}


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"[INFO] Random seed set to {seed}")
