"""
FundFlow AI — Model Trainer
Trains XGBoost fraud detection model and reports both optimistic/random and
leakage-safe/time-split metrics.
"""
import json
import os
import sys

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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, PROCESSED_DATA_PATH, XGBOOST_PARAMS
from features.engineering import engineer_features, get_feature_columns
from features.graph_features import build_account_graph_features, load_graph_features

# MLflow (optional -- logs experiments when available)
try:
    import mlflow
    import mlflow.xgboost
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


DECISION_THRESHOLD = 0.70


def _safe_auc_roc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def _safe_auc_pr(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if int(pd.Series(y_true).sum()) == 0:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def _threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def _train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / max(pos, 1))
    params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False,
    )
    return model, params, scale_pos_weight


def _build_protocol_report(
    protocol_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    split_meta: dict,
    scale_pos_weight: float,
) -> dict:
    t50 = _threshold_metrics(y_true, y_prob, 0.50)
    t70 = _threshold_metrics(y_true, y_prob, DECISION_THRESHOLD)
    return {
        "protocol": protocol_name,
        "rows": int(len(y_true)),
        "fraud_rows": int(pd.Series(y_true).sum()),
        "fraud_rate": float(pd.Series(y_true).mean()),
        "auc_roc": _safe_auc_roc(y_true, y_prob),
        "auc_pr": _safe_auc_pr(y_true, y_prob),
        "threshold_050": t50,
        "threshold_070": t70,
        "split": split_meta,
        "scale_pos_weight": float(scale_pos_weight),
    }


def _round_report(report: dict) -> dict:
    rounded = dict(report)
    for key in ("fraud_rate", "auc_roc", "auc_pr", "scale_pos_weight"):
        rounded[key] = round(float(rounded[key]), 6)

    for key in ("threshold_050", "threshold_070"):
        block = dict(rounded[key])
        block["precision"] = round(float(block["precision"]), 6)
        block["recall"] = round(float(block["recall"]), 6)
        block["f1"] = round(float(block["f1"]), 6)
        rounded[key] = block

    return rounded


def _print_threshold_block(label: str, block: dict):
    cm = block["confusion_matrix"]
    print(f"\n       === Threshold = {label} ===")
    print(
        f"       Precision: {block['precision']:.4f}  "
        f"Recall: {block['recall']:.4f}  F1: {block['f1']:.4f}"
    )
    print(f"       TN={cm[0][0]:,}  FP={cm[0][1]:,}  FN={cm[1][0]:,}  TP={cm[1][1]:,}")


def _time_split_raw(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
    df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df_sorted) * (1.0 - test_size))
    split_idx = min(max(split_idx, 1), len(df_sorted) - 1)

    train_raw = df_sorted.iloc[:split_idx].copy()
    test_raw = df_sorted.iloc[split_idx:].copy()

    split_meta = {
        "type": "time_split",
        "test_size": float(test_size),
        "split_index": int(split_idx),
        "train_start": str(train_raw["timestamp"].min()),
        "train_end": str(train_raw["timestamp"].max()),
        "test_start": str(test_raw["timestamp"].min()),
        "test_end": str(test_raw["timestamp"].max()),
        "train_rows": int(len(train_raw)),
        "test_rows": int(len(test_raw)),
    }
    return train_raw, test_raw, split_meta


def train(data_path: str = None, model_dir: str = None, sample_size: int = None):
    """
    Full training pipeline:
    1. Load raw data
    2. Train random-split benchmark model
    3. Build leakage-safe split and train final model
    4. Evaluate both protocols
    5. Save model + metadata
    """
    data_path = data_path or PROCESSED_DATA_PATH
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("  FundFlow AI — Model Training")
    print("=" * 60)

    # ── 1. LOAD ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    if sample_size:
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        df = pd.read_csv(data_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(
        f"       Rows: {len(df):,}  |  Fraud: {df['is_fraud'].sum():,}  "
        f"({df['is_fraud'].mean()*100:.3f}%)"
    )

    feature_cols = get_feature_columns()

    # ── 2. RANDOM-SPLIT BENCHMARK (optimistic) ────────────────────────────
    print("\n[2/6] Random-split benchmark (optimistic baseline)...")
    gf_all = load_graph_features()
    if gf_all:
        print(f"       Graph features (full precompute): {len(gf_all):,} accounts")
    else:
        print("       WARNING: No precomputed graph features found for benchmark.")

    df_feat_all = engineer_features(df, graph_features=gf_all)
    X_all = df_feat_all[feature_cols].fillna(0)
    y_all = df_feat_all["is_fraud"].astype(int)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )
    print(f"       Random split Train: {len(X_train_r):,}  |  Test: {len(X_test_r):,}")

    model_random, params_random, scale_pos_weight_random = _train_xgb_model(
        X_train_r,
        y_train_r,
        X_test_r,
        y_test_r,
    )
    y_prob_random = model_random.predict_proba(X_test_r)[:, 1]
    random_report = _build_protocol_report(
        "random_stratified_split",
        y_test_r,
        y_prob_random,
        split_meta={
            "type": "random_stratified",
            "test_size": 0.2,
            "train_rows": int(len(X_train_r)),
            "test_rows": int(len(X_test_r)),
            "graph_feature_accounts": int(len(gf_all)),
        },
        scale_pos_weight=scale_pos_weight_random,
    )

    # ── 3. LEAKAGE-SAFE FEATURE BUILD ─────────────────────────────────────
    print("\n[3/6] Leakage-safe split (time) + train-only graph features...")
    train_raw, test_raw, split_meta = _time_split_raw(df, test_size=0.2)
    print(f"       Train window: {split_meta['train_start']} -> {split_meta['train_end']}")
    print(f"       Test  window: {split_meta['test_start']} -> {split_meta['test_end']}")

    gf_train_only = build_account_graph_features(train_raw)
    print(f"       Train-window graph features: {len(gf_train_only):,} accounts")

    df_train_feat = engineer_features(train_raw, graph_features=gf_train_only)
    df_test_feat = engineer_features(test_raw, graph_features=gf_train_only)

    X_train = df_train_feat[feature_cols].fillna(0)
    y_train = df_train_feat["is_fraud"].astype(int)
    X_test = df_test_feat[feature_cols].fillna(0)
    y_test = df_test_feat["is_fraud"].astype(int)
    print(f"       Leakage-safe Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"       Train fraud: {int(y_train.sum()):,}  |  Test fraud: {int(y_test.sum()):,}")

    # ── 4. TRAIN FINAL MODEL (leakage-safe) ───────────────────────────────
    print("\n[4/6] Training XGBoost on leakage-safe split...")
    model, params, scale_pos_weight = _train_xgb_model(X_train, y_train, X_test, y_test)
    print(f"       scale_pos_weight: {scale_pos_weight:.1f}")

    # ── 5. EVALUATE BOTH PROTOCOLS ─────────────────────────────────────────
    print("\n[5/6] Evaluating benchmark vs leakage-safe...")
    y_prob = model.predict_proba(X_test)[:, 1]
    leakage_report = _build_protocol_report(
        "leakage_safe_time_split_train_graph_only",
        y_test,
        y_prob,
        split_meta={
            **split_meta,
            "graph_feature_accounts": int(len(gf_train_only)),
            "graph_features_source": "train_window_only",
        },
        scale_pos_weight=scale_pos_weight,
    )

    print(
        f"\n       Random benchmark AUC-ROC: {random_report['auc_roc']:.4f}  "
        f"|  AUC-PR: {random_report['auc_pr']:.4f}"
    )
    _print_threshold_block("0.50 (Random)", random_report["threshold_050"])
    _print_threshold_block("0.70 (Random)", random_report["threshold_070"])

    print(
        f"\n       Leakage-safe AUC-ROC: {leakage_report['auc_roc']:.4f}  "
        f"|  AUC-PR: {leakage_report['auc_pr']:.4f}"
    )
    _print_threshold_block("0.50 (Leakage-safe)", leakage_report["threshold_050"])
    _print_threshold_block("0.70 (Leakage-safe)", leakage_report["threshold_070"])

    canonical_50 = leakage_report["threshold_050"]
    canonical_70 = leakage_report["threshold_070"]

    feat_imp = dict(
        sorted(
            zip(feature_cols, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # ── 6. SAVE ────────────────────────────────────────────────────────────
    print("\n[6/6] Saving model + metadata...")
    model_path = os.path.join(model_dir, "xgboost_fraud.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")
    joblib.dump(model, model_path)

    full_fraud_rate = float(pd.concat([y_train, y_test], axis=0).mean())

    metadata = {
        "model_type": "XGBClassifier",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "decision_threshold": DECISION_THRESHOLD,
        "evaluation_protocol": "leakage_safe_time_split_train_graph_only",
        "metrics": {
            "auc_roc": round(float(leakage_report["auc_roc"]), 4),
            "auc_pr": round(float(leakage_report["auc_pr"]), 4),
            "precision": round(float(canonical_70["precision"]), 4),
            "recall": round(float(canonical_70["recall"]), 4),
            "f1": round(float(canonical_70["f1"]), 4),
            "confusion_matrix": canonical_70["confusion_matrix"],
        },
        "metrics_threshold_050": {
            "precision": round(float(canonical_50["precision"]), 4),
            "recall": round(float(canonical_50["recall"]), 4),
            "f1": round(float(canonical_50["f1"]), 4),
        },
        "metrics_leakage_safe": _round_report(leakage_report),
        "metrics_random_split": _round_report(random_report),
        "graph_features_used": bool(gf_train_only),
        "graph_feature_accounts": int(len(gf_train_only)),
        "feature_importance": feat_imp,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate": round(full_fraud_rate, 6),
        "scale_pos_weight": round(float(scale_pos_weight), 2),
        "params": params,
        "benchmark_params_random_split": params_random,
        "leakage_notes": [
            "random_stratified_split is optimistic and kept only for comparison",
            "canonical metrics come from leakage_safe_time_split_train_graph_only",
            "graph features for canonical evaluation are built from train window only",
        ],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved:    {model_path}")
    print(f"  Metadata saved: {meta_path}")

    # -- MLflow logging (optional) ------------------------------------------
    if _HAS_MLFLOW:
        try:
            from pathlib import Path as _Path
            from mlops.config import (
                MLFLOW_EXPERIMENT_NAME,
                MLFLOW_REGISTERED_MODEL_NAME,
                MLFLOW_CHAMPION_ALIAS,
            )
            from mlops.drift_detector import build_reference_profile, save_reference_profile
            from mlops.retrain import save_eval_set

            mlflow.set_tracking_uri(f"sqlite:///{_Path(model_dir).parent / 'mlflow.db'}")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            git_sha = os.environ.get("SENTINEL_GIT_SHA", "unknown")

            with mlflow.start_run() as run:
                mlflow.log_params({
                    "n_estimators": params["n_estimators"],
                    "max_depth": params["max_depth"],
                    "learning_rate": params["learning_rate"],
                    "scale_pos_weight": round(scale_pos_weight, 2),
                    "training_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "git_sha": git_sha,
                    "feature_count": len(feature_cols),
                    "shap_method": "native_pred_contribs",
                })
                mlflow.log_metrics({
                    "auc_roc": round(float(leakage_report["auc_roc"]), 4),
                    "auc_pr": round(float(leakage_report["auc_pr"]), 4),
                    "precision_070": round(float(canonical_70["precision"]), 4),
                    "recall_070": round(float(canonical_70["recall"]), 4),
                    "f1_070": round(float(canonical_70["f1"]), 4),
                    "fraud_rate": round(full_fraud_rate, 6),
                })
                mlflow.xgboost.log_model(model, artifact_path="model")
                mlflow.log_artifact(meta_path)

                ref_profile = build_reference_profile(df_train_feat, feature_cols)
                ref_path = save_reference_profile(ref_profile)
                mlflow.log_artifact(str(ref_path))

                eval_path = save_eval_set(test_raw, "canonical")
                mlflow.log_artifact(str(eval_path))

                schema = {"feature_columns": feature_cols, "feature_count": len(feature_cols), "shap_method": "native_pred_contribs"}
                schema_path = os.path.join(model_dir, "feature_schema.json")
                with open(schema_path, "w") as sf:
                    json.dump(schema, sf, indent=2)
                mlflow.log_artifact(schema_path)

                model_uri = f"runs:/{run.info.run_id}/model"
                rv = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
                client = mlflow.MlflowClient()
                client.set_registered_model_alias(MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHAMPION_ALIAS, rv.version)
                print(f"  MLflow run:     {run.info.run_id}")
                print(f"  Model v{rv.version} aliased as \'{MLFLOW_CHAMPION_ALIAS}\'")

        except Exception as mlflow_err:
            print(f"  MLflow logging failed (non-fatal): {mlflow_err}")

    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return model, metadata


if __name__ == "__main__":
    train()