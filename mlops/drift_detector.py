"""
mlops/drift_detector.py — PSI-based data & prediction drift detection.

Compares live prediction feature distributions against the training-time
reference profile.  Uses Population Stability Index (PSI), the industry
standard in banking/credit-scoring (Basel II).

Label drift (comparing actual fraud rates over time) is only possible when
ground-truth labels are available.  In synthetic data we have ``is_fraud``.
In production, labels arrive late via compliance officer confirmations.
This module does NOT implement label drift — it requires a separate
feedback loop.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas  # noqa: F401 — used in type hint string

from mlops.config import (
    MIN_SAMPLES_FOR_DRIFT,
    NUM_PSI_BINS,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    REFERENCE_PROFILE_DIR,
)
from mlops.prediction_logger import read_all_prediction_logs

logger = logging.getLogger(__name__)

_EPSILON = 1e-4  # avoid log(0) in PSI


# ------------------------------------------------------------------
# Reference profile: build / save / load
# ------------------------------------------------------------------

def build_reference_profile(
    df: "pandas.DataFrame",
    feature_cols: list[str],
    prediction_col: str | None = None,
    num_bins: int = NUM_PSI_BINS,
) -> dict[str, Any]:
    """Build a reference profile from the training DataFrame.

    For each feature, compute histogram bin edges and the proportion of
    training rows in each bin.  These are later used by ``compute_psi``
    to detect drift.

    Parameters
    ----------
    df : pd.DataFrame
        Training data **after** feature engineering (must contain
        *feature_cols* as columns).
    feature_cols : list[str]
        Ordered list of feature column names (49 features).
    prediction_col : str, optional
        Column with model prediction scores.  If provided, a prediction
        distribution profile is also saved.
    num_bins : int
        Number of histogram bins per feature.

    Returns
    -------
    dict
        ``{"features": {col: {"bin_edges": [...], "proportions": [...]}},
           "prediction": {...} | None,
           "feature_columns": [...],
           "created_at": "...",
           "training_rows": N}``
    """

    profile: dict[str, Any] = {
        "features": {},
        "feature_columns": list(feature_cols),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
    }

    for col in feature_cols:
        vals = df[col].dropna().astype(float).values
        if len(vals) == 0:
            profile["features"][col] = {
                "bin_edges": [],
                "proportions": [],
            }
            continue

        # Use quantile-based bin edges for equal-frequency bins
        quantiles = np.linspace(0, 100, num_bins + 1)
        edges = np.unique(np.percentile(vals, quantiles))

        # If all values are identical, np.unique returns 1 element
        if len(edges) < 2:
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])

        counts, _ = np.histogram(vals, bins=edges)
        proportions = counts / counts.sum()

        profile["features"][col] = {
            "bin_edges": edges.tolist(),
            "proportions": proportions.tolist(),
        }

    # Optional prediction distribution
    if prediction_col and prediction_col in df.columns:
        pred_vals = df[prediction_col].dropna().astype(float).values
        edges = np.linspace(0, 1, num_bins + 1)
        counts, _ = np.histogram(pred_vals, bins=edges)
        proportions = counts / counts.sum()
        profile["prediction"] = {
            "bin_edges": edges.tolist(),
            "proportions": proportions.tolist(),
        }
    else:
        profile["prediction"] = None

    return profile


def save_reference_profile(profile: dict, path: Path | str | None = None) -> Path:
    """Save a reference profile to JSON."""
    if path is None:
        REFERENCE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = REFERENCE_PROFILE_DIR / "reference_profile.json"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logger.info("Reference profile saved to %s", path)
    return path


def load_reference_profile(path: Path | str | None = None) -> dict | None:
    """Load a reference profile from JSON. Returns None if not found."""
    if path is None:
        path = REFERENCE_PROFILE_DIR / "reference_profile.json"
    else:
        path = Path(path)

    if not path.exists():
        logger.warning("Reference profile not found at %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# PSI computation
# ------------------------------------------------------------------

def compute_psi(
    expected_proportions: list[float],
    actual_proportions: list[float],
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = SUM( (actual_i - expected_i) * ln(actual_i / expected_i) )

    Zeros are replaced with ``_EPSILON`` to avoid ``log(0)``.
    """
    expected = np.array(expected_proportions, dtype=float)
    actual = np.array(actual_proportions, dtype=float)

    # Clamp to epsilon
    expected = np.where(expected < _EPSILON, _EPSILON, expected)
    actual = np.where(actual < _EPSILON, _EPSILON, actual)

    # Re-normalise after epsilon clamping
    expected = expected / expected.sum()
    actual = actual / actual.sum()

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return round(psi, 6)


# ------------------------------------------------------------------
# Feature drift
# ------------------------------------------------------------------

def compute_feature_drift(
    reference_profile: dict,
    prediction_logs: list[dict],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Compute per-feature PSI between training and serving distributions.

    Uses the **same bin edges** from the reference profile so the
    comparison is valid.  Serving-time feature values are taken from the
    ``feature_values`` dict in each prediction log entry.
    """
    feature_psi: dict[str, float] = {}
    drifted: list[str] = []
    warning_features: list[str] = []
    ref_features = reference_profile.get("features", {})

    for col in feature_cols:
        ref = ref_features.get(col)
        if not ref or not ref.get("bin_edges") or not ref.get("proportions"):
            continue

        # Extract serving-time values for this feature
        vals = []
        for entry in prediction_logs:
            fv = entry.get("feature_values", {})
            if col in fv:
                vals.append(float(fv[col]))

        if len(vals) < 10:
            feature_psi[col] = 0.0
            continue

        bin_edges = np.array(ref["bin_edges"])
        expected_proportions = ref["proportions"]

        # Bin serving values using training bin edges
        indices = np.digitize(vals, bin_edges) - 1
        indices = np.clip(indices, 0, len(expected_proportions) - 1)

        actual_counts = np.bincount(indices, minlength=len(expected_proportions))
        actual_proportions = actual_counts / actual_counts.sum()

        psi = compute_psi(expected_proportions, actual_proportions.tolist())
        feature_psi[col] = psi

        if psi > PSI_ALERT_THRESHOLD:
            drifted.append(col)
        elif psi > PSI_WARNING_THRESHOLD:
            warning_features.append(col)

    max_psi = max(feature_psi.values()) if feature_psi else 0.0

    return {
        "feature_psi": feature_psi,
        "drifted_features": drifted,
        "warning_features": warning_features,
        "max_psi": round(max_psi, 6),
        "alert": len(drifted) > 0,
        "features_checked": len(feature_psi),
    }


# ------------------------------------------------------------------
# Prediction drift
# ------------------------------------------------------------------

def compute_prediction_drift(
    reference_profile: dict,
    prediction_logs: list[dict],
) -> dict[str, Any]:
    """Compute PSI on the prediction score distribution."""
    ref_pred = reference_profile.get("prediction")
    if not ref_pred or not ref_pred.get("bin_edges") or not ref_pred.get("proportions"):
        return {
            "prediction_psi": 0.0,
            "alert": False,
            "status": "no_prediction_reference",
        }

    scores = [
        float(e["fraud_probability"])
        for e in prediction_logs
        if "fraud_probability" in e
    ]
    if len(scores) < 10:
        return {"prediction_psi": 0.0, "alert": False, "status": "insufficient_data"}

    bin_edges = np.array(ref_pred["bin_edges"])
    expected = ref_pred["proportions"]

    indices = np.digitize(scores, bin_edges) - 1
    indices = np.clip(indices, 0, len(expected) - 1)

    actual_counts = np.bincount(indices, minlength=len(expected))
    actual_proportions = actual_counts / actual_counts.sum()

    psi = compute_psi(expected, actual_proportions.tolist())

    return {
        "prediction_psi": psi,
        "alert": psi > PSI_ALERT_THRESHOLD,
        "status": "ok",
    }


# ------------------------------------------------------------------
# Full drift report
# ------------------------------------------------------------------

def run_drift_check(
    reference_profile_path: Path | str | None = None,
    prediction_log_dir: Path | str | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Run a full drift check: feature drift + prediction drift.

    Returns a combined report dict.
    """
    # Load reference profile
    ref = load_reference_profile(reference_profile_path)
    if ref is None:
        return {
            "status": "no_reference_profile",
            "message": (
                "No reference profile found. Train with MLflow "
                "instrumentation to generate one."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Determine feature columns
    if feature_cols is None:
        feature_cols = ref.get("feature_columns", [])
    if not feature_cols:
        return {
            "status": "no_feature_columns",
            "message": "Reference profile has no feature_columns list.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Load prediction logs
    logs = read_all_prediction_logs(prediction_log_dir)
    if len(logs) < MIN_SAMPLES_FOR_DRIFT:
        return {
            "status": "insufficient_data",
            "predictions_logged": len(logs),
            "minimum_required": MIN_SAMPLES_FOR_DRIFT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Compute drift
    feat_drift = compute_feature_drift(ref, logs, feature_cols)
    pred_drift = compute_prediction_drift(ref, logs)

    alert = feat_drift["alert"] or pred_drift.get("alert", False)

    return {
        "status": "ok",
        "alert": alert,
        "feature_drift": feat_drift,
        "prediction_drift": pred_drift,
        "predictions_analysed": len(logs),
        "reference_training_rows": ref.get("training_rows", 0),
        "reference_created_at": ref.get("created_at", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

