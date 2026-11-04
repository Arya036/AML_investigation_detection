"""
mlops/retrain.py — Retraining pipeline with drift-aware evaluation.

Trains a new XGBoost model on fresh / drifted data, evaluates it against
both the new data AND a canonical held-out set from the original training,
and promotes to champion only if strict criteria are met.

Can generate a synthetic drifted dataset for demo purposes.

Usage:
    python -m mlops.retrain                        # retrain on latest processed data
    python -m mlops.retrain --generate-drift        # generate + retrain on drifted data
    python -m mlops.retrain --data-path new.csv     # retrain on a specific CSV
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, PROCESSED_DATA_PATH, XGBOOST_PARAMS
from features.engineering import engineer_features, get_feature_columns
from features.graph_features import build_account_graph_features
from mlops.config import (
    DECISION_THRESHOLD,
    EVAL_SET_DIR,
    PROJECT_ROOT,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    PROMOTION_AUC_PR_MARGIN,
)
from mlops.drift_detector import build_reference_profile, save_reference_profile

logger = logging.getLogger(__name__)

NO_REGRESSION_TOLERANCE = 0.02  # AUC-PR can drop by at most this on canonical set


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def _evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    """Compute standard metrics at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    auc_roc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else 0.0
    auc_pr = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0
    return {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "rows": int(len(y_true)),
        "fraud_rows": int(y_true.sum()),
    }


# ------------------------------------------------------------------
# Canonical eval set management
# ------------------------------------------------------------------

def save_eval_set(df: pd.DataFrame, name: str = "canonical") -> Path:
    """Save a labelled evaluation set as parquet."""
    EVAL_SET_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_SET_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Eval set '%s' saved: %d rows -> %s", name, len(df), path)
    return path


def load_eval_set(name: str = "canonical") -> pd.DataFrame | None:
    """Load a labelled evaluation set. Returns None if not found."""
    path = EVAL_SET_DIR / f"{name}.parquet"
    if not path.exists():
        alt_path = PROJECT_ROOT / "mlops" / "eval_sets" / f"{name}.parquet"
        if alt_path.exists():
            return pd.read_parquet(alt_path)
        logger.warning("Eval set '%s' not found at %s", name, path)
        return None
    return pd.read_parquet(path)


# ------------------------------------------------------------------
# Synthetic drift generator
# ------------------------------------------------------------------

def generate_drifted_dataset(
    base_df: pd.DataFrame,
    amount_shift_factor: float = 1.5,
    night_hour_boost: float = 0.30,
    fraud_rate: float | None = None,
) -> pd.DataFrame:
    """Create a controlled distribution shift for demo purposes.

    Parameters
    ----------
    base_df : pd.DataFrame
        Original processed transactions.
    amount_shift_factor : float
        Multiply amounts by this factor (e.g. 1.5 = 50% higher).
    night_hour_boost : float
        Target proportion of night-hour (0-5) transactions.
    fraud_rate : float, optional
        If set, resample to achieve this fraud rate.

    Returns
    -------
    pd.DataFrame
        Drifted copy of the dataset.
    """
    df = base_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Amount shift
    df["amount"] = df["amount"] * amount_shift_factor
    logger.info("Amount shift: x%.2f (mean %.0f -> %.0f)",
                amount_shift_factor, base_df["amount"].mean(), df["amount"].mean())

    # 2. Night hour boost — move some daytime txns to night hours
    current_night = (df["timestamp"].dt.hour < 6).mean()
    if night_hour_boost > current_night:
        n_to_shift = int((night_hour_boost - current_night) * len(df))
        daytime_mask = df["timestamp"].dt.hour >= 6
        daytime_idx = df[daytime_mask].index
        shift_idx = np.random.choice(daytime_idx, size=min(n_to_shift, len(daytime_idx)), replace=False)
        night_hours = np.random.choice([0, 1, 2, 3, 4, 5], size=len(shift_idx))
        for i, idx in enumerate(shift_idx):
            ts = df.loc[idx, "timestamp"]
            df.loc[idx, "timestamp"] = ts.replace(hour=int(night_hours[i]))
        logger.info("Night hour boost: %.1f%% -> %.1f%%",
                     current_night * 100, (df["timestamp"].dt.hour < 6).mean() * 100)

    # 3. Fraud rate adjustment
    if fraud_rate is not None and fraud_rate != base_df["is_fraud"].mean():
        n_fraud_target = int(len(df) * fraud_rate)
        n_fraud_current = int(df["is_fraud"].sum())
        if n_fraud_target > n_fraud_current:
            legit_idx = df[df["is_fraud"] == 0].index
            flip_idx = np.random.choice(legit_idx, size=min(n_fraud_target - n_fraud_current, len(legit_idx)), replace=False)
            df.loc[flip_idx, "is_fraud"] = 1
        elif n_fraud_target < n_fraud_current:
            fraud_idx = df[df["is_fraud"] == 1].index
            flip_idx = np.random.choice(fraud_idx, size=min(n_fraud_current - n_fraud_target, len(fraud_idx)), replace=False)
            df.loc[flip_idx, "is_fraud"] = 0
        logger.info("Fraud rate adjusted: %.3f%% -> %.3f%%",
                     base_df["is_fraud"].mean() * 100, df["is_fraud"].mean() * 100)

    return df


# ------------------------------------------------------------------
# Promotion decision
# ------------------------------------------------------------------

def should_promote(
    new_metrics_drifted: dict,
    old_metrics_drifted: dict,
    new_metrics_canonical: dict | None = None,
    old_metrics_canonical: dict | None = None,
) -> tuple[bool, str]:
    """Decide whether the challenger should replace the champion.

    Criteria (must pass ALL):
    1. New AUC-PR on drifted window >= old + PROMOTION_AUC_PR_MARGIN (0.01)
    2. New recall@0.70 >= old recall@0.70 (no regression)
    3. On canonical set: new AUC-PR >= old - NO_REGRESSION_TOLERANCE (0.02)
    """
    reasons = []

    # Criterion 1: AUC-PR improvement on drifted data
    new_pr = new_metrics_drifted["auc_pr"]
    old_pr = old_metrics_drifted["auc_pr"]
    margin_needed = old_pr + PROMOTION_AUC_PR_MARGIN
    if new_pr < margin_needed:
        reasons.append(
            f"FAIL Criterion 1: new AUC-PR on drifted data ({new_pr:.4f}) < "
            f"old ({old_pr:.4f}) + margin ({PROMOTION_AUC_PR_MARGIN}) = {margin_needed:.4f}"
        )

    # Criterion 2: Recall must not decrease
    new_recall = new_metrics_drifted["recall"]
    old_recall = old_metrics_drifted["recall"]
    if new_recall < old_recall:
        reasons.append(
            f"FAIL Criterion 2: new recall@{DECISION_THRESHOLD} ({new_recall:.4f}) < "
            f"old ({old_recall:.4f})"
        )

    # Criterion 3: No-regression on canonical set
    if new_metrics_canonical is not None and old_metrics_canonical is not None:
        new_canon_pr = new_metrics_canonical["auc_pr"]
        old_canon_pr = old_metrics_canonical["auc_pr"]
        floor = old_canon_pr - NO_REGRESSION_TOLERANCE
        if new_canon_pr < floor:
            reasons.append(
                f"FAIL Criterion 3: new AUC-PR on canonical set ({new_canon_pr:.4f}) < "
                f"old ({old_canon_pr:.4f}) - tolerance ({NO_REGRESSION_TOLERANCE}) = {floor:.4f}"
            )

    if reasons:
        return False, " | ".join(reasons)

    return True, (
        f"PASS: drifted AUC-PR {new_pr:.4f} >= {margin_needed:.4f}, "
        f"recall {new_recall:.4f} >= {old_recall:.4f}"
    )


# ------------------------------------------------------------------
# Main retraining pipeline
# ------------------------------------------------------------------

def retrain(
    new_data_path: str | None = None,
    drifted: bool = False,
    shift_config: dict | None = None,
) -> dict:
    """Full retraining pipeline.

    1.  Load new/drifted data
    2.  Time-split 80/20
    3.  Build graph features on train window only (leakage-safe)
    4.  Engineer features
    5.  Train new XGBoost model
    6.  Evaluate on drifted test set
    7.  Evaluate on canonical test set (no-regression)
    8.  Compare with champion
    9.  Promote if criteria met
    10. Save new reference profile
    11. Log to MLflow
    """
    feature_cols = get_feature_columns()
    data_path = new_data_path or PROCESSED_DATA_PATH

    print("=" * 60)
    print("  Sentinel AI -- Retraining Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    print("\n[1/9] Loading data...")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if drifted:
        cfg = shift_config or {}
        print("       Generating drifted dataset...")
        df = generate_drifted_dataset(
            df,
            amount_shift_factor=cfg.get("amount_shift_factor", 1.5),
            night_hour_boost=cfg.get("night_hour_boost", 0.30),
            fraud_rate=cfg.get("fraud_rate"),
        )

    print(f"       Rows: {len(df):,}  |  Fraud: {df['is_fraud'].sum():,} "
          f"({df['is_fraud'].mean()*100:.3f}%)")

    # ----------------------------------------------------------------
    # 2. Time-split
    # ----------------------------------------------------------------
    print("\n[2/9] Time-split (80/20)...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()
    print(f"       Train: {len(train_raw):,}  |  Test: {len(test_raw):,}")

    # ----------------------------------------------------------------
    # 3. Graph features (train window only)
    # ----------------------------------------------------------------
    print("\n[3/9] Building graph features (train window only)...")
    gf_train = build_account_graph_features(train_raw)
    print(f"       Graph features: {len(gf_train):,} accounts")

    # ----------------------------------------------------------------
    # 4. Feature engineering
    # ----------------------------------------------------------------
    print("\n[4/9] Engineering features...")
    train_feat = engineer_features(train_raw, graph_features=gf_train)
    test_feat = engineer_features(test_raw, graph_features=gf_train)

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat["is_fraud"].astype(int)
    X_test = test_feat[feature_cols].fillna(0)
    y_test = test_feat["is_fraud"].astype(int)

    # ----------------------------------------------------------------
    # 5. Train new model
    # ----------------------------------------------------------------
    print("\n[5/9] Training XGBoost...")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / max(pos, 1))
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"       scale_pos_weight: {spw:.1f}")

    # ----------------------------------------------------------------
    # 6. Evaluate on drifted test set
    # ----------------------------------------------------------------
    print("\n[6/9] Evaluating on drifted test set...")
    y_prob_new = model.predict_proba(X_test)[:, 1]
    new_metrics_drifted = _evaluate(y_test, y_prob_new)
    print(f"       AUC-PR: {new_metrics_drifted['auc_pr']:.4f}  "
          f"Recall@{DECISION_THRESHOLD}: {new_metrics_drifted['recall']:.4f}")

    # ----------------------------------------------------------------
    # 7. Evaluate old champion on same drifted test set
    # ----------------------------------------------------------------
    print("\n[7/9] Loading champion for comparison...")
    champion_path = os.path.join(MODEL_DIR, "xgboost_fraud.pkl")
    if os.environ.get("MODAL_TASK_ID"):
        vol_champ = Path("/data/mlops/models/xgboost_fraud.pkl")
        if vol_champ.exists():
            champion_path = str(vol_champ)
    old_metrics_drifted = None
    old_metrics_canonical = None
    new_metrics_canonical = None

    if os.path.exists(champion_path):
        champion_model = joblib.load(champion_path)
        y_prob_old = champion_model.predict_proba(X_test)[:, 1]
        old_metrics_drifted = _evaluate(y_test, y_prob_old)
        print(f"       Champion AUC-PR on drifted: {old_metrics_drifted['auc_pr']:.4f}  "
              f"Recall: {old_metrics_drifted['recall']:.4f}")

        # 7b. Canonical set comparison
        canonical_df = load_eval_set("canonical")
        if canonical_df is not None:
            print("       Evaluating both models on canonical set...")
            canon_feat = engineer_features(canonical_df, graph_features=gf_train)
            X_canon = canon_feat[feature_cols].fillna(0)
            y_canon = canon_feat["is_fraud"].astype(int)

            y_prob_new_canon = model.predict_proba(X_canon)[:, 1]
            y_prob_old_canon = champion_model.predict_proba(X_canon)[:, 1]

            new_metrics_canonical = _evaluate(y_canon, y_prob_new_canon)
            old_metrics_canonical = _evaluate(y_canon, y_prob_old_canon)
            print(f"       Canonical -- New: {new_metrics_canonical['auc_pr']:.4f}  "
                  f"Old: {old_metrics_canonical['auc_pr']:.4f}")
        else:
            print("       No canonical eval set found. Saving current test set as canonical.")
            save_eval_set(test_raw, "canonical")
    else:
        print("       No champion found. This is the first training.")
        save_eval_set(test_raw, "canonical")

    # ----------------------------------------------------------------
    # 8. Promotion decision
    # ----------------------------------------------------------------
    print("\n[8/9] Promotion decision...")
    if old_metrics_drifted is not None:
        promote, reason = should_promote(
            new_metrics_drifted, old_metrics_drifted,
            new_metrics_canonical, old_metrics_canonical,
        )
    else:
        promote, reason = True, "First model — auto-promoting as champion."

    print(f"       {'PROMOTED' if promote else 'REJECTED'}: {reason}")

    # ----------------------------------------------------------------
    # 9. Save & log to MLflow
    # ----------------------------------------------------------------
    print("\n[9/9] Saving model and logging to MLflow...")

    run_id = None
    version = None
    git_sha = os.environ.get("SENTINEL_GIT_SHA", "unknown")

    try:
        import mlflow
        from mlops.config import MLFLOW_DB_PATH, MODAL_DATA_DIR, PROJECT_ROOT, get_mlflow_db_path

        # On Modal, ensure Volume DB is initialized from container DB if not present
        if os.environ.get("MODAL_TASK_ID"):
            MODAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not MLFLOW_DB_PATH.exists():
                container_db = PROJECT_ROOT / "models" / "mlflow.db"
                if container_db.exists():
                    import shutil
                    shutil.copy2(container_db, MLFLOW_DB_PATH)

        db_path = MLFLOW_DB_PATH if os.environ.get("MODAL_TASK_ID") else get_mlflow_db_path()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Params
            mlflow.log_params({
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "scale_pos_weight": round(spw, 2),
                "drifted": str(drifted),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "git_sha": git_sha,
                "feature_count": len(feature_cols),
                "shap_method": "native_pred_contribs",
            })

            # Metrics
            for prefix, metrics in [("drifted", new_metrics_drifted),
                                     ("canonical", new_metrics_canonical or {})]:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{prefix}_{k}", v)

            # Feature importance
            feat_imp = dict(sorted(
                zip(feature_cols, model.feature_importances_.tolist()),
                key=lambda x: x[1], reverse=True,
            ))

            # Model artifact
            mlflow.xgboost.log_model(model, artifact_path="model")

            # Reference profile
            ref_profile = build_reference_profile(
                train_feat, feature_cols, prediction_col="fraud_probability"
                if "fraud_probability" in train_feat.columns else None
            )
            ref_path = save_reference_profile(ref_profile)
            mlflow.log_artifact(str(ref_path))

            # Feature schema
            schema = {
                "feature_columns": feature_cols,
                "feature_count": len(feature_cols),
                "shap_method": "native_pred_contribs",
                "feature_importance": feat_imp,
            }
            schema_path = Path(MODEL_DIR) / "feature_schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            mlflow.log_artifact(str(schema_path))

            # Register model
            model_uri = f"runs:/{run_id}/model"
            rv = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
            version = rv.version

            # Set alias
            client = mlflow.MlflowClient()
            if promote:
                client.set_registered_model_alias(
                    MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHAMPION_ALIAS, version
                )
                print(f"       Model v{version} aliased as '{MLFLOW_CHAMPION_ALIAS}'")
            else:
                client.set_registered_model_alias(
                    MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHALLENGER_ALIAS, version
                )
                print(f"       Model v{version} aliased as '{MLFLOW_CHALLENGER_ALIAS}'")

        print(f"       MLflow run: {run_id}")

    except Exception as e:
        logger.error("MLflow logging failed: %s", e)
        print(f"       MLflow logging failed: {e}")

    # Save model to disk if promoted
    if promote:
        model_path = os.path.join(MODEL_DIR, "xgboost_fraud.pkl")
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, model_path)

        metadata = {
            "model_type": "XGBClassifier",
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
            "decision_threshold": DECISION_THRESHOLD,
            "metrics": new_metrics_drifted,
            "graph_features_used": bool(gf_train),
            "graph_feature_accounts": len(gf_train),
            "feature_importance": feat_imp,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fraud_rate": round(float(y_train.mean()), 6),
            "scale_pos_weight": round(spw, 2),
            "params": params,
            "retrained_at": datetime.now(timezone.utc).isoformat(),
            "drifted": drifted,
            "mlflow_run_id": run_id,
            "mlflow_version": version,
            "git_sha": git_sha,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"       Model saved to {model_path}")

        # On Modal, also save to Volume so other containers see the new champion
        if os.environ.get("MODAL_TASK_ID"):
            from mlops.config import MODAL_DATA_DIR
            vol_models_dir = MODAL_DATA_DIR / "models"
            vol_models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, vol_models_dir / "xgboost_fraud.pkl")
            with open(vol_models_dir / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"  RETRAINING {'COMPLETE — MODEL PROMOTED' if promote else 'COMPLETE — NOT PROMOTED'}")
    print("=" * 60)

    return {
        "promoted": promote,
        "reason": reason,
        "new_metrics_drifted": new_metrics_drifted,
        "old_metrics_drifted": old_metrics_drifted,
        "new_metrics_canonical": new_metrics_canonical,
        "old_metrics_canonical": old_metrics_canonical,
        "mlflow_run_id": run_id,
        "mlflow_version": version,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel AI - Retraining Pipeline")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to new data CSV")
    parser.add_argument("--generate-drift", action="store_true",
                        help="Generate synthetic drifted data")
    parser.add_argument("--amount-shift", type=float, default=1.5,
                        help="Amount shift factor (default: 1.5)")
    parser.add_argument("--night-boost", type=float, default=0.30,
                        help="Night hour proportion (default: 0.30)")
    args = parser.parse_args()

    shift_config = {
        "amount_shift_factor": args.amount_shift,
        "night_hour_boost": args.night_boost,
    }

    retrain(
        new_data_path=args.data_path,
        drifted=args.generate_drift,
        shift_config=shift_config if args.generate_drift else None,
    )

