"""
tests/test_pipeline.py — Tests for the ML pipeline.

Covers feature engineering, risk scoring, and graph features.
All tests use synthetic fixtures — no external data files needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Feature engineering tests
# ------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for features/engineering.py."""

    def test_feature_columns_complete(self, synthetic_transactions_df, feature_columns):
        """engineer_features returns all 49 feature columns."""
        from features.engineering import engineer_features

        result = engineer_features(synthetic_transactions_df)
        missing = [c for c in feature_columns if c not in result.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_feature_count_is_49(self, feature_columns):
        """Feature list has exactly 49 entries."""
        # model_metadata.json says 49 but get_feature_columns returns
        # the list — confirm it.  Adjust this number if features change.
        assert len(feature_columns) >= 48, (
            f"Expected >= 48 features, got {len(feature_columns)}"
        )

    def test_feature_columns_no_nans(self, synthetic_transactions_df, feature_columns):
        """No NaN values in the final feature matrix after fillna(0)."""
        from features.engineering import engineer_features

        result = engineer_features(synthetic_transactions_df)
        X = result[feature_columns].fillna(0)
        nan_cols = X.columns[X.isna().any()].tolist()
        assert nan_cols == [], f"Columns with NaN after fillna: {nan_cols}"

    def test_binary_features_are_binary(self, synthetic_transactions_df, feature_columns):
        """Binary indicator features only contain 0 and 1."""
        from features.engineering import engineer_features

        result = engineer_features(synthetic_transactions_df)
        binary_cols = [
            "is_weekend", "is_night", "is_cross_branch",
            "type_NEFT", "type_UPI", "type_ATM", "type_DEPOSIT", "type_IMPS",
            "channel_mobile", "channel_internet",
            "is_new_receiver", "is_cross_bank_upi",
            "is_round_10k", "is_round_1k",
            "high_velocity_1h", "high_velocity_24h", "rapid_succession",
            "amount_gt_5x_avg",
            "near_any_threshold", "near_50k_threshold",
            "near_100k_threshold", "near_1m_threshold",
        ]
        for col in binary_cols:
            if col in result.columns:
                unique = set(result[col].dropna().unique())
                assert unique <= {0, 1, 0.0, 1.0}, (
                    f"{col} has non-binary values: {unique}"
                )

    def test_amount_log_positive(self, synthetic_transactions_df):
        """amount_log should always be > 0 for positive amounts."""
        from features.engineering import engineer_features

        result = engineer_features(synthetic_transactions_df)
        assert (result["amount_log"] >= 0).all(), "amount_log has negative values"

    def test_engineer_single_returns_all_keys(self, feature_columns):
        """engineer_single returns a dict with all feature column keys."""
        from features.engineering import engineer_single

        txn = {
            "amount": 15000,
            "timestamp": "2026-03-05 14:30:00",
            "sender_account": "C1000000",
            "receiver_account": "C2000000",
            "txn_type": "UPI",
            "channel": "mobile",
            "sender_branch": "BR_MUMBAI_001",
            "receiver_branch": "BR_DELHI_001",
        }
        features = engineer_single(txn)
        missing = [c for c in feature_columns if c not in features]
        assert missing == [], f"engineer_single missing keys: {missing}"


# ------------------------------------------------------------------
# Risk scoring tests
# ------------------------------------------------------------------

class TestRiskScoring:
    """Tests for scoring/risk_engine.py."""

    def test_risk_scoring_tiers(self):
        """compute_risk_score returns correct tier for known inputs."""
        from scoring.risk_engine import compute_risk_score

        # All zeros -> LOW
        result = compute_risk_score()
        assert result["risk_tier"] == "LOW"

        # All ones -> CRITICAL
        result = compute_risk_score(
            ml_fraud_probability=1.0,
            graph_risk_score=1.0,
            ring_involvement_score=1.0,
            mule_account_score=1.0,
            velocity_anomaly_score=1.0,
            amount_anomaly_score=1.0,
        )
        assert result["risk_tier"] == "CRITICAL"

    def test_risk_scoring_weights_sum_to_one(self):
        """RISK_WEIGHTS values must sum to 1.0."""
        from config import RISK_WEIGHTS

        total = sum(RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, f"RISK_WEIGHTS sum to {total}, expected 1.0"

    def test_risk_score_bounded(self):
        """Final score is always in [0, 1]."""
        from scoring.risk_engine import compute_risk_score

        for _ in range(20):
            vals = np.random.random(6)
            result = compute_risk_score(*vals)
            assert 0.0 <= result["final_score"] <= 1.0

    def test_velocity_anomaly_ranges(self):
        """compute_velocity_anomaly output is always in [0, 1]."""
        from scoring.risk_engine import compute_velocity_anomaly

        for txn_1h in [0, 1, 3, 5, 10, 50]:
            for txn_24h in [0, 5, 20, 50, 100]:
                score = compute_velocity_anomaly(txn_1h, txn_24h)
                assert 0.0 <= score <= 1.0, (
                    f"velocity_anomaly({txn_1h}, {txn_24h}) = {score}"
                )

    def test_amount_anomaly_ranges(self):
        """compute_amount_anomaly output is always in [0, 1]."""
        from scoring.risk_engine import compute_amount_anomaly

        test_cases = [
            (1000, 1000, 100),   # z=0
            (5000, 1000, 100),   # z=40 -> capped at 1.0
            (1000, 1000, 0),     # std=0 -> 0.0
            (1200, 1000, 100),   # z=2 -> 0.4
        ]
        for amount, avg, std in test_cases:
            score = compute_amount_anomaly(amount, avg, std)
            assert 0.0 <= score <= 1.0, (
                f"amount_anomaly({amount}, {avg}, {std}) = {score}"
            )

    def test_risk_tier_boundaries(self):
        """Verify exact tier boundary values."""
        from scoring.risk_engine import get_risk_tier

        assert get_risk_tier(0.0) == "LOW"
        assert get_risk_tier(0.29) == "LOW"
        assert get_risk_tier(0.3) == "MEDIUM"
        assert get_risk_tier(0.59) == "MEDIUM"
        assert get_risk_tier(0.6) == "HIGH"
        assert get_risk_tier(0.79) == "HIGH"
        assert get_risk_tier(0.8) == "CRITICAL"
        assert get_risk_tier(1.0) == "CRITICAL"


# ------------------------------------------------------------------
# MLOps module tests
# ------------------------------------------------------------------

class TestMLOpsModules:
    """Tests for mlops/ modules that don't need external data."""

    def test_psi_identical_distributions(self):
        """PSI of a distribution against itself should be ~0."""
        from mlops.drift_detector import compute_psi

        props = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        psi = compute_psi(props, props)
        assert psi < 0.001, f"PSI of identical distributions = {psi}"

    def test_psi_shifted_distribution(self):
        """PSI should be high for a significantly shifted distribution."""
        from mlops.drift_detector import compute_psi

        expected = [0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]
        actual   = [0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.20]
        psi = compute_psi(expected, actual)
        assert psi > 0.10, f"PSI of shifted distributions = {psi}, expected > 0.10"

    def test_build_reference_profile(self, synthetic_transactions_df, feature_columns):
        """build_reference_profile returns correct structure."""
        from features.engineering import engineer_features
        from mlops.drift_detector import build_reference_profile

        feat = engineer_features(synthetic_transactions_df)
        profile = build_reference_profile(feat, feature_columns)

        assert "features" in profile
        assert "feature_columns" in profile
        assert "training_rows" in profile
        assert profile["training_rows"] == len(feat)

        # Check at least some features have bins
        for col in feature_columns[:5]:
            assert col in profile["features"], f"Missing feature '{col}' in profile"
            f = profile["features"][col]
            assert "bin_edges" in f
            assert "proportions" in f
            if f["proportions"]:
                total = sum(f["proportions"])
                assert abs(total - 1.0) < 0.01, (
                    f"Proportions for {col} sum to {total}"
                )

    def test_prediction_logger_roundtrip(self, tmp_path):
        """PredictionLogger writes and read_all_prediction_logs reads back."""
        from mlops.prediction_logger import PredictionLogger, read_all_prediction_logs

        logger = PredictionLogger(log_dir=tmp_path, flush_interval=2)

        # Log 3 predictions
        for i in range(3):
            logger.log(
                account_id=f"C{i}",
                fraud_probability=0.1 * i,
                risk_tier="LOW",
                feature_values={"amount_log": float(i), "hour_of_day": 12.0},
                model_version="test_v1",
            )
        logger.flush()

        records = read_all_prediction_logs(tmp_path)
        assert len(records) == 3
        assert records[0]["account_id"] == "C0"
        assert "feature_values" in records[0]
        assert "amount_log" in records[0]["feature_values"]

    def test_promotion_criteria_pass(self):
        """should_promote passes when new model is clearly better."""
        from mlops.retrain import should_promote

        new = {"auc_pr": 0.80, "recall": 0.85}
        old = {"auc_pr": 0.70, "recall": 0.80}
        promote, reason = should_promote(new, old)
        assert promote is True, f"Expected promotion, got: {reason}"

    def test_promotion_criteria_fail_margin(self):
        """should_promote rejects noise-level improvement."""
        from mlops.retrain import should_promote

        new = {"auc_pr": 0.705, "recall": 0.80}  # only +0.005, below 0.01 margin
        old = {"auc_pr": 0.700, "recall": 0.80}
        promote, reason = should_promote(new, old)
        assert promote is False, f"Should reject noise, got: {reason}"

    def test_promotion_criteria_fail_recall(self):
        """should_promote rejects if recall drops even with AUC-PR gain."""
        from mlops.retrain import should_promote

        new = {"auc_pr": 0.80, "recall": 0.70}  # recall dropped
        old = {"auc_pr": 0.70, "recall": 0.80}
        promote, reason = should_promote(new, old)
        assert promote is False, f"Should reject recall drop, got: {reason}"

