"""
mlops/dashboard.py — MLOps dashboard API routes.

Mounted at ``/mlops/`` in the main FastAPI app.  Provides health,
drift, prediction summary, model listing, reload, and flush endpoints.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mlops", tags=["MLOps"])

# Track when the model was loaded into memory
_model_loaded_at: str | None = None


def _get_model_metadata() -> dict:
    """Read model_metadata.json from disk."""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def _model_file_mtime() -> str | None:
    """Get modification time of xgboost_fraud.pkl."""
    model_path = os.path.join(MODEL_DIR, "xgboost_fraud.pkl")
    if not os.path.exists(model_path):
        return None
    mtime = os.path.getmtime(model_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


# ------------------------------------------------------------------
# GET /mlops/status
# ------------------------------------------------------------------

@router.get("/status")
def mlops_status():
    """Current model version, metrics, and load status."""
    global _model_loaded_at
    try:
        meta = _get_model_metadata()
        metrics = meta.get("metrics", {})

        mlflow_version = meta.get("mlflow_version")
        mlflow_run_id = meta.get("mlflow_run_id")
        if not mlflow_version:
            try:
                import mlflow
                from mlops.config import MLFLOW_CHAMPION_ALIAS, MLFLOW_REGISTERED_MODEL_NAME

                db_path = Path(MODEL_DIR).parent / "mlflow.db"
                if db_path.exists():
                    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
                    client = mlflow.MlflowClient()
                    champ = client.get_model_version_by_alias(
                        MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHAMPION_ALIAS
                    )
                    if champ:
                        mlflow_version = champ.version
                        mlflow_run_id = champ.run_id
            except Exception:
                pass

        return {
            "model_type": meta.get("model_type", "unknown"),
            "feature_count": meta.get("feature_count", 0),
            "decision_threshold": meta.get("decision_threshold", 0.70),
            "training_metrics": {
                "auc_roc": metrics.get("auc_roc"),
                "auc_pr": metrics.get("auc_pr"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            },
            "training_rows": meta.get("training_rows", 0),
            "fraud_rate": meta.get("fraud_rate", 0),
            "scale_pos_weight": meta.get("scale_pos_weight", 0),
            "mlflow_run_id": mlflow_run_id,
            "mlflow_version": mlflow_version,
            "git_sha": meta.get("git_sha", os.environ.get("SENTINEL_GIT_SHA", "unknown")),
            "retrained_at": meta.get("retrained_at"),
            "last_reloaded_at": _model_loaded_at,
            "model_file_modified_at": _model_file_mtime(),
        }
    except Exception as e:
        logger.error("MLOps status error: %s", e)
        return {"error": str(e)}


# ------------------------------------------------------------------
# GET /mlops/drift
# ------------------------------------------------------------------

@router.get("/drift")
def mlops_drift():
    """Run drift check and return report."""
    try:
        from mlops.drift_detector import run_drift_check
        return run_drift_check()
    except Exception as e:
        logger.error("Drift check error: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ------------------------------------------------------------------
# GET /mlops/predictions/summary
# ------------------------------------------------------------------

@router.get("/predictions/summary")
def mlops_predictions_summary():
    """Summary of logged predictions: counts, distribution, time range."""
    try:
        from mlops.prediction_logger import read_all_prediction_logs

        logs = read_all_prediction_logs()
        if not logs:
            return {
                "total_predictions": 0,
                "message": "No predictions logged yet.",
            }

        # Risk tier distribution
        tier_counts: dict[str, int] = {}
        probs = []
        for entry in logs:
            tier = entry.get("risk_tier", "UNKNOWN")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if "fraud_probability" in entry:
                probs.append(float(entry["fraud_probability"]))

        timestamps = [e.get("timestamp", "") for e in logs if e.get("timestamp")]

        return {
            "total_predictions": len(logs),
            "risk_tier_distribution": tier_counts,
            "avg_fraud_probability": round(sum(probs) / len(probs), 4) if probs else None,
            "min_fraud_probability": round(min(probs), 4) if probs else None,
            "max_fraud_probability": round(max(probs), 4) if probs else None,
            "time_range": {
                "earliest": min(timestamps) if timestamps else None,
                "latest": max(timestamps) if timestamps else None,
            },
        }
    except Exception as e:
        logger.error("Prediction summary error: %s", e)
        return {"error": str(e)}


# ------------------------------------------------------------------
# GET /mlops/models
# ------------------------------------------------------------------

@router.get("/models")
def mlops_models():
    """List registered model versions from MLflow."""
    try:
        import mlflow
        from mlops.config import MLFLOW_REGISTERED_MODEL_NAME

        db_path = Path(MODEL_DIR).parent / "mlflow.db"
        if not db_path.exists():
            return {
                "models": [],
                "message": "MLflow database not found. Run a training with MLflow first.",
                "current_model": _get_model_metadata(),
            }

        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        client = mlflow.MlflowClient()

        try:
            versions = client.search_model_versions(
                f"name='{MLFLOW_REGISTERED_MODEL_NAME}'"
            )
        except Exception:
            return {
                "models": [],
                "message": "Model not registered in MLflow yet.",
                "current_model": _get_model_metadata(),
            }

        # Look up aliases registered on the model (e.g. {'champion': 1})
        try:
            reg_model = client.get_registered_model(MLFLOW_REGISTERED_MODEL_NAME)
            alias_map = reg_model.aliases or {}
        except Exception:
            alias_map = {}

        models = []
        for v in versions:
            run = client.get_run(v.run_id) if v.run_id else None
            metrics = run.data.metrics if run else {}

            # Map aliases pointing to this version
            v_aliases = [alias for alias, ver in alias_map.items() if str(ver) == str(v.version)]
            if not v_aliases and hasattr(v, "aliases") and v.aliases:
                v_aliases = list(v.aliases)

            canonical_auc = metrics.get("canonical_auc_pr") or metrics.get("auc_pr")

            models.append({
                "version": v.version,
                "run_id": v.run_id,
                "aliases": v_aliases,
                "creation_timestamp": v.creation_timestamp,
                "metrics": {
                    "drifted_auc_pr": metrics.get("drifted_auc_pr"),
                    "drifted_recall": metrics.get("drifted_recall"),
                    "drifted_f1": metrics.get("drifted_f1"),
                    "canonical_auc_pr": canonical_auc,
                },
            })

        return {"models": models, "total": len(models)}

    except ImportError:
        return {
            "models": [],
            "message": "MLflow not installed.",
            "current_model": _get_model_metadata(),
        }
    except Exception as e:
        logger.error("MLflow model listing error: %s", e)
        return {"error": str(e), "current_model": _get_model_metadata()}


# ------------------------------------------------------------------
# POST /mlops/reload
# ------------------------------------------------------------------

@router.post("/reload")
def mlops_reload():
    """Force-reload the model from disk."""
    global _model_loaded_at
    try:
        # Clear cached model in predictor.py
        from models import predictor
        predictor._model = None
        predictor._metadata = None
        predictor._shap_explainer = None

        # Attempt Modal volume reload
        if os.environ.get("MODAL_TASK_ID"):
            try:
                import modal
                vol = modal.Volume.from_name("sentinel-data", create_if_missing=False)
                vol.reload()
                logger.info("Modal Volume reloaded.")
            except Exception as ve:
                logger.warning("Modal Volume reload failed: %s", ve)

        # Force re-load
        predictor._load_model()
        _model_loaded_at = datetime.now(timezone.utc).isoformat()

        meta = _get_model_metadata()
        return {
            "status": "reloaded",
            "model_type": meta.get("model_type"),
            "reloaded_at": _model_loaded_at,
            "model_file_modified_at": _model_file_mtime(),
        }
    except Exception as e:
        logger.error("Model reload error: %s", e)
        return {"status": "error", "message": str(e)}


# ------------------------------------------------------------------
# POST /mlops/flush
# ------------------------------------------------------------------

@router.post("/flush")
def mlops_flush():
    """Flush prediction logs to disk and commit Modal Volume."""
    try:
        # The global logger instance (if any) needs to be flushed
        # This is best-effort; the logger may not be initialised here

        # We can't access the module-level instance directly,
        # but we can commit the volume regardless
        commit = bool(os.environ.get("MODAL_TASK_ID"))

        if commit:
            try:
                import modal
                vol = modal.Volume.from_name("sentinel-data", create_if_missing=False)
                vol.commit()
                logger.info("Modal Volume committed for prediction logs.")
            except Exception as ve:
                logger.warning("Volume commit failed: %s", ve)
                return {"status": "commit_failed", "message": str(ve)}

        return {
            "status": "flushed",
            "volume_committed": commit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Flush error: %s", e)
        return {"status": "error", "message": str(e)}

