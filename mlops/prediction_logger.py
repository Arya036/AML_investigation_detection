"""
mlops/prediction_logger.py -- Append-only prediction logging.

One JSONL file per container/process. Feature values logged in full
so PSI drift detection can bin them against the training-time reference profile.

Modal Note: Each container gets its own file. The dashboard/drift
detector reads ALL files from the directory and merges them.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlops.config import PREDICTION_LOG_DIR, PREDICTION_LOG_FLUSH_INTERVAL

logger = logging.getLogger(__name__)


def _build_log_filename() -> str:
    """Build a unique log filename using PID and Modal task ID.

    Format: predictions_{modal_task_id}_{pid}.jsonl
    This prevents write conflicts when multiple containers or
    processes write concurrently on a shared Modal Volume.
    """
    pid = os.getpid()
    modal_task_id = os.environ.get("MODAL_TASK_ID", "local")
    return f"predictions_{modal_task_id}_{pid}.jsonl"


class PredictionLogger:
    """Thread-safe, buffered prediction logger.

    Writes prediction records (including the full 49-feature vector)
    to an append-only JSONL file. Buffered writes are flushed every
    ``flush_interval`` predictions or on explicit ``flush()`` call.

    Parameters
    ----------
    log_dir : Path or str, optional
        Directory for JSONL files. Defaults to ``PREDICTION_LOG_DIR``.
    flush_interval : int, optional
        Number of buffered records before an automatic flush.
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        flush_interval: int = PREDICTION_LOG_FLUSH_INTERVAL,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else PREDICTION_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval

        self._filepath = self._log_dir / _build_log_filename()
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._total_logged: int = 0

        logger.info("PredictionLogger writing to %s", self._filepath)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        account_id: str,
        fraud_probability: float,
        risk_tier: str,
        feature_values: dict[str, float],
        model_version: str = "unknown",
    ) -> None:
        """Log a single prediction.

        Parameters
        ----------
        account_id : str
            The account or transaction identifier.
        fraud_probability : float
            Model output probability of fraud.
        risk_tier : str
            Assigned risk tier (e.g. ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``).
        feature_values : dict[str, float]
            Full feature vector (all 49 features) as ``{name: value}``.
        model_version : str
            Model version string from model metadata.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account_id": account_id,
            "fraud_probability": float(fraud_probability),
            "risk_tier": risk_tier,
            "feature_values": {k: float(v) for k, v in feature_values.items()},
            "model_version": model_version,
        }
        line = json.dumps(record, separators=(",", ":"))

        with self._lock:
            self._buffer.append(line)
            self._total_logged += 1
            if len(self._buffer) >= self._flush_interval:
                self._flush_unlocked()

    def flush(self, commit_volume: bool = False) -> None:
        """Flush buffered records to disk.

        Parameters
        ----------
        commit_volume : bool
            If ``True`` and running on Modal, call ``volume.commit()``
            after writing so other containers can see the data.
        """
        with self._lock:
            self._flush_unlocked()

        if commit_volume:
            self._commit_modal_volume()

    @property
    def total_logged(self) -> int:
        """Total number of predictions logged (including unflushed)."""
        with self._lock:
            return self._total_logged

    @property
    def filepath(self) -> Path:
        """Path to the JSONL file this logger is writing to."""
        return self._filepath

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_unlocked(self) -> None:
        """Write buffered lines to disk. Caller MUST hold ``self._lock``."""
        if not self._buffer:
            return
        try:
            with open(self._filepath, "a", encoding="utf-8") as fh:
                fh.write("\n".join(self._buffer))
                fh.write("\n")
            logger.debug(
                "Flushed %d prediction records to %s",
                len(self._buffer),
                self._filepath,
            )
        except OSError:
            logger.exception("Failed to flush prediction log to %s", self._filepath)
        finally:
            self._buffer.clear()

    @staticmethod
    def _commit_modal_volume() -> None:
        """Best-effort commit of the Modal Volume so other containers see writes."""
        if not os.environ.get("MODAL_TASK_ID"):
            return
        try:
            import modal  # type: ignore[import-untyped]

            vol = modal.Volume.from_name("sentinel-data", create_if_missing=False)
            vol.commit()
            logger.info("Modal Volume committed.")
        except Exception:
            logger.warning("Modal Volume commit failed (non-fatal).", exc_info=True)


# ------------------------------------------------------------------
# Module-level reader function
# ------------------------------------------------------------------


def read_all_prediction_logs(
    directory: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Read and merge ALL ``.jsonl`` prediction log files from *directory*.

    Parameters
    ----------
    directory : Path or str, optional
        Directory containing ``predictions_*.jsonl`` files.
        Defaults to ``PREDICTION_LOG_DIR``.

    Returns
    -------
    list[dict[str, Any]]
        Combined list of prediction records sorted by timestamp
        (oldest first).
    """
    log_dir = Path(directory) if directory else PREDICTION_LOG_DIR
    if not log_dir.is_dir():
        logger.warning("Prediction log directory does not exist: %s", log_dir)
        return []

    # Best-effort flush of active in-memory logger before reading files
    try:
        from models import predictor
        if getattr(predictor, "_prediction_logger", None) is not None:
            predictor._prediction_logger.flush()
    except Exception:
        pass

    records: list[dict[str, Any]] = []
    jsonl_files = list(log_dir.glob("predictions_*.jsonl"))

    if not jsonl_files:
        logger.info("No prediction log files found in %s", log_dir)
        return []

    for filepath in jsonl_files:
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed JSON at %s:%d", filepath, line_num
                        )
        except OSError:
            logger.exception("Failed to read prediction log %s", filepath)

    # Sort by timestamp (ISO format sorts lexicographically)
    records.sort(key=lambda r: r.get("timestamp", ""))
    logger.info(
        "Loaded %d prediction records from %d file(s) in %s",
        len(records),
        len(jsonl_files),
        log_dir,
    )
    return records
