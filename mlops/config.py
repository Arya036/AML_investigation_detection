"""
mlops/config.py — MLOps Configuration Constants
"""
from pathlib import Path
import os

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
MLOPS_DIR = PROJECT_ROOT / "mlops"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Modal Volume paths (used when running on Modal)
MODAL_DATA_DIR = Path(os.environ.get("MLOPS_DATA_DIR", "/data/mlops"))
PREDICTION_LOG_DIR = MODAL_DATA_DIR / "predictions"
REFERENCE_PROFILE_DIR = MODAL_DATA_DIR / "reference_profiles"
EVAL_SET_DIR = MODAL_DATA_DIR / "eval_sets"

# When running locally, use project-local paths
if not os.environ.get("MODAL_TASK_ID"):
    PREDICTION_LOG_DIR = PROJECT_ROOT / "mlops" / "predictions"
    REFERENCE_PROFILE_DIR = PROJECT_ROOT / "mlops" / "reference_profiles"
    EVAL_SET_DIR = PROJECT_ROOT / "mlops" / "eval_sets"

# Drift detection
PSI_ALERT_THRESHOLD = 0.25  # PSI > 0.25 = significant drift
PSI_WARNING_THRESHOLD = 0.10  # PSI > 0.10 = moderate drift
MIN_SAMPLES_FOR_DRIFT = 100  # Need at least this many predictions before computing drift
NUM_PSI_BINS = 10  # Number of bins for PSI computation

# Retraining
PROMOTION_AUC_PR_MARGIN = 0.01  # New model must beat old by this margin
RECALL_FLOOR_AT_070 = None  # If set, new recall@0.70 must be >= this value; otherwise must be >= old recall@0.70
DECISION_THRESHOLD = 0.70

# MLflow
MLFLOW_EXPERIMENT_NAME = "sentinel-aml-xgboost"
MLFLOW_CHAMPION_ALIAS = "champion"
MLFLOW_CHALLENGER_ALIAS = "challenger"
MLFLOW_REGISTERED_MODEL_NAME = "sentinel-xgboost-fraud"

# Prediction logging
PREDICTION_LOG_FLUSH_INTERVAL = 1  # Flush to disk on every prediction (immediate dashboard updates)
