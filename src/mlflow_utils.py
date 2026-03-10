"""
MLflow experiment tracking helpers.
"""

import os
from typing import Any, Dict, List, Optional


def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    run_name: str,
    experiment_name: str = "financebench-llm",
    params: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[str]] = None,
) -> None:
    """Log evaluation metrics to MLflow."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            if artifacts:
                for artifact_path in artifacts:
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path)

            print(
                f"[INFO] Logged to MLflow: run='{run_name}', "
                f"metrics={len(metrics)}"
            )

    except ImportError:
        print("[WARN] MLflow not installed. Skipping logging.")
    except Exception as e:
        print(f"[WARN] MLflow logging failed: {e}")
