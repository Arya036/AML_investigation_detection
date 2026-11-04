"""
FundFlow AI — Inference Engine
Loads the trained XGBoost model and runs predictions.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR
from features.engineering import engineer_single, get_feature_columns

_model = None
_metadata = None
_model_loaded_at = None  # ISO timestamp for MLOps dashboard
_prediction_logger = None  # lazy-init prediction logger


def _load_model():
    global _model, _metadata, _model_loaded_at
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "xgboost_fraud.pkl")
        meta_path  = os.path.join(MODEL_DIR, "model_metadata.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python -m models.trainer` first."
            )
        _model = joblib.load(model_path)
        with open(meta_path) as f:
            _metadata = json.load(f)
        from datetime import datetime, timezone
        _model_loaded_at = datetime.now(timezone.utc).isoformat()
    return _model, _metadata


def predict_single(txn: dict, account_history: pd.DataFrame = None,
                   graph_features: dict = None,
                   india_extras: dict = None) -> dict:
    """
    Run fraud prediction on a single transaction in real-time.

    Args:
        txn:             Transaction dict with keys matching schema.
        account_history: Recent transactions by same sender (for rolling features).
        graph_features:  Precomputed account-level graph signals.
        india_extras:    VPA / KYC / CIBIL data per account.

    Returns:
        dict with fraud_probability, fraud_label, risk_tier, top_features.
    """
    model, metadata = _load_model()
    feature_cols = metadata['feature_columns']
    feat_imp     = metadata['feature_importance']

    features = engineer_single(
        txn, account_history,
        graph_features=graph_features or {},
        india_extras=india_extras or {},
    )
    X = np.array([[features.get(col, 0) for col in feature_cols]])

    threshold   = float(metadata.get('decision_threshold', 0.7))
    fraud_prob  = float(model.predict_proba(X)[0][1])
    fraud_label = int(fraud_prob >= threshold)


    # Risk tier
    if fraud_prob < 0.3:
        risk_tier = "LOW"
    elif fraud_prob < 0.6:
        risk_tier = "MEDIUM"
    elif fraud_prob < 0.8:
        risk_tier = "HIGH"
    else:
        risk_tier = "CRITICAL"

    # ── Real SHAP values (TreeExplainer, cached at module level) ────────────
    # Returns true log-odds contributions. Explainer loads once on first call.
    top_features = _get_shap_top_features(features, feature_cols)

    # ── MLOps prediction logging (best-effort) ────────────────────────────
    global _prediction_logger
    try:
        if _prediction_logger is None:
            from mlops.prediction_logger import PredictionLogger
            _prediction_logger = PredictionLogger()
        _prediction_logger.log(
            account_id=txn.get("sender_account", "unknown"),
            fraud_probability=fraud_prob,
            risk_tier=risk_tier,
            feature_values={col: float(features.get(col, 0)) for col in feature_cols},
            model_version=metadata.get("mlflow_version", metadata.get("model_type", "unknown")),
        )
    except Exception:
        pass  # Never let logging break predictions

    return {
        "fraud_probability": round(fraud_prob, 4),
        "fraud_label":       fraud_label,
        "decision_threshold": round(threshold, 4),
        "risk_tier":         risk_tier,
        "top_features":      top_features,
        "raw_features":      features,
    }


# ── SHAP explainer cache ──────────────────────────────────────────────────────
_shap_explainer = None


def _get_shap_top_features(features: dict, feature_cols: list) -> list:
    """
    Compute top-5 SHAP feature contributions.

    Uses XGBoost's native predict_contribs() (built-in SHAP, no torch dependency).
    Returns true log-odds SHAP contributions. Cached booster reused across calls.
    Falls back to feature×importance proxy only if XGBoost itself fails.
    """
    global _shap_explainer
    try:
        # XGBoost's native SHAP — does not require the shap package or torch.
        # model.get_booster().predict(dmatrix, pred_contribs=True) returns
        # an array of shape (n_samples, n_features + 1) where the last col
        # is the bias term. Values are log-odds contributions (true SHAP).
        import xgboost as xgb

        if _shap_explainer is None:
            model, _ = _load_model()
            _shap_explainer = model.get_booster()

        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        # pred_contribs returns shape (n, n_features + 1) — last col is bias
        contribs = _shap_explainer.predict(dmat, pred_contribs=True)
        vals = contribs[0][:-1]  # drop bias term

        pairs = [
            {
                "feature":      col,
                "shap_value":   round(float(v), 6),
                "direction":    "increases_risk" if v > 0 else "decreases_risk",
                "contribution": round(float(v), 6),  # keep key for STR compatibility
            }
            for col, v in zip(feature_cols, vals)
        ]
        pairs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return pairs[:5]

    except Exception:
        # Fallback: feature_value × tree importance (heuristic proxy)
        _, metadata = _load_model()
        feat_imp = metadata.get("feature_importance", {})
        feat_scores = {
            col: round(float(features.get(col, 0)) * float(feat_imp.get(col, 0)), 6)
            for col in feature_cols
        }
        top = sorted(feat_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        return [
            {"feature": k, "shap_value": None, "contribution": v,
             "direction": "increases_risk" if v > 0 else "decreases_risk"}
            for k, v in top
        ]



def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud prediction on a DataFrame of transactions.
    Returns df with added columns: fraud_probability, fraud_label, risk_tier.
    """
    model, metadata = _load_model()
    feature_cols    = metadata['feature_columns']
    threshold       = float(metadata.get('decision_threshold', 0.7))

    from features.engineering import engineer_features
    df_feat = engineer_features(df)
    X = df_feat[feature_cols].fillna(0)

    probs  = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)

    def tier(p):
        if p < 0.3:   return "LOW"
        if p < 0.6:   return "MEDIUM"
        if p < 0.8:   return "HIGH"
        return "CRITICAL"

    df = df.copy()
    df['fraud_probability'] = probs.round(4)
    df['fraud_label']       = labels
    df['decision_threshold'] = round(threshold, 4)
    df['risk_tier']         = [tier(p) for p in probs]
    return df


def get_model_metadata() -> dict:
    """Return model metadata (metrics, feature importance, params)."""
    _, metadata = _load_model()
    return metadata
