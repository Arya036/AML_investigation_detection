"""
tests/conftest.py — Shared test fixtures for Sentinel AI.

Provides synthetic data, a temp model, and a mock DB so tests never
touch real data files (fundflow.db, ChromaDB, model pickle).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Synthetic transactions DataFrame (200 rows, 7-day span)
# ------------------------------------------------------------------

@pytest.fixture
def synthetic_transactions_df() -> pd.DataFrame:
    """200-row DataFrame mimicking processed transactions."""
    rng = np.random.RandomState(42)
    n = 200

    senders = [f"C{1000000 + i}" for i in rng.choice(20, size=n)]
    receivers = [f"C{2000000 + i}" for i in rng.choice(30, size=n)]
    branches = [
        "BR_MUMBAI_001", "BR_DELHI_001", "BR_CHENNAI_001",
        "BR_BANGALORE_001", "BR_HYDERABAD_001",
    ]

    # Timestamps spanning 7 days
    base = pd.Timestamp("2026-03-01")
    offsets = pd.to_timedelta(rng.uniform(0, 7 * 24 * 3600, size=n), unit="s")
    timestamps = base + offsets

    # Log-normal amounts (100 to ~500k)
    amounts = np.exp(rng.normal(9.0, 1.5, size=n)).clip(100, 500_000).round(2)

    # Weighted transaction types
    txn_types = rng.choice(
        ["UPI", "NEFT", "ATM", "DEPOSIT", "IMPS"],
        size=n,
        p=[0.45, 0.25, 0.15, 0.10, 0.05],
    )
    channels = rng.choice(
        ["mobile", "internet", "branch"],
        size=n,
        p=[0.45, 0.35, 0.20],
    )

    # 5% fraud rate
    is_fraud = (rng.random(n) < 0.05).astype(int)

    df = pd.DataFrame({
        "txn_id": [f"TXN{i:06d}" for i in range(n)],
        "timestamp": timestamps,
        "sender_account": senders,
        "receiver_account": receivers,
        "amount": amounts,
        "txn_type": txn_types,
        "channel": channels,
        "sender_branch": rng.choice(branches, size=n),
        "receiver_branch": rng.choice(branches, size=n),
        "is_fraud": is_fraud,
        "fraud_probability": rng.random(n).round(4),
    })
    return df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------------
# Feature column list
# ------------------------------------------------------------------

@pytest.fixture
def feature_columns() -> list[str]:
    """The 49 feature columns used by the model."""
    from features.engineering import get_feature_columns
    return get_feature_columns()


# ------------------------------------------------------------------
# Temporary trained model
# ------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir(synthetic_transactions_df, feature_columns):
    """Temp directory with a tiny trained XGBoost model on fixture data."""
    import xgboost as xgb
    from features.engineering import engineer_features

    df = synthetic_transactions_df
    feat = engineer_features(df)

    X = feat[feature_columns].fillna(0)
    y = feat["is_fraud"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=5, max_depth=3, use_label_encoder=False,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X, y, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        import joblib
        model_path = os.path.join(tmpdir, "xgboost_fraud.pkl")
        meta_path = os.path.join(tmpdir, "model_metadata.json")

        joblib.dump(model, model_path)

        metadata = {
            "model_type": "XGBClassifier",
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "decision_threshold": 0.70,
            "metrics": {
                "auc_roc": 0.90,
                "auc_pr": 0.50,
                "precision": 0.30,
                "recall": 0.80,
                "f1": 0.43,
            },
            "feature_importance": {c: 1.0 / len(feature_columns) for c in feature_columns},
            "training_rows": len(X),
            "fraud_rate": float(y.mean()),
            "scale_pos_weight": 19.0,
            "params": {"n_estimators": 5, "max_depth": 3},
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        yield tmpdir


# ------------------------------------------------------------------
# Mock SQLite DB
# ------------------------------------------------------------------

@pytest.fixture
def mock_db_path(synthetic_transactions_df):
    """Temp SQLite DB with a transactions table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    df = synthetic_transactions_df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df["risk_tier"] = "MEDIUM"
    df.to_sql("transactions", conn, index=False, if_exists="replace")

    # Create alerts table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT, timestamp TEXT, severity TEXT,
            accounts_involved TEXT, description TEXT
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    os.unlink(db_path)

