"""Saving and loading trained artefacts.

Each trained task produces one joblib bundle containing the fitted pipeline,
the feature specification and the metrics of every candidate, plus a sidecar
JSON of the metrics for anything that wants to read them without unpickling a
model (the dashboard's comparison page, CI checks, the README).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..config import MODELS_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Canonical artefact names. Keeping them in one place stops the trainer and the
# dashboard from drifting apart over a filename.
ARTIFACTS = {
    "winner": "winner_model",
    "score": "first_innings_score_model",
    "chase": "chase_model",
    "pom": "player_of_match_model",
}


class ArtifactNotFound(FileNotFoundError):
    """Raised when a model is requested before it has been trained."""


def _json_default(value: Any) -> Any:
    """Make numpy scalars, datetimes and dataclasses JSON-serialisable."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def artifact_path(name: str) -> Path:
    """Return the on-disk path for a model bundle."""
    return MODELS_DIR / f"{name}.joblib"


def metrics_path(name: str) -> Path:
    """Return the on-disk path for a metrics sidecar."""
    return MODELS_DIR / f"{name}_metrics.json"


def save_artifact(name: str, payload: dict[str, Any]) -> Path:
    """Persist a trained bundle, stamping it with a training timestamp."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "saved_at": datetime.utcnow().isoformat(timespec="seconds")}
    path = artifact_path(name)
    joblib.dump(payload, path, compress=3)
    logger.info("Saved artefact %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    return path


def load_artifact(name: str) -> dict[str, Any]:
    """Load a trained bundle.

    Raises:
        ArtifactNotFound: the model has not been trained yet.
    """
    path = artifact_path(name)
    if not path.exists():
        raise ArtifactNotFound(
            f"No trained artefact at {path}. Run `python scripts/train_models.py` first."
        )
    return joblib.load(path)


def artifact_exists(name: str) -> bool:
    """True when the named artefact is on disk."""
    return artifact_path(name).exists()


def save_metrics(name: str, metrics: dict[str, Any]) -> Path:
    """Write the metrics sidecar as human-readable JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = metrics_path(name)
    path.write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def load_metrics(name: str) -> dict[str, Any] | None:
    """Read the metrics sidecar, or ``None`` when it does not exist."""
    path = metrics_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Metrics file %s is corrupt; ignoring.", path)
        return None


def list_artifacts() -> dict[str, dict[str, Any]]:
    """Summarise every trained artefact, for the Admin page."""
    summary: dict[str, dict[str, Any]] = {}
    for task, name in ARTIFACTS.items():
        path = artifact_path(name)
        metrics = load_metrics(name) or {}
        summary[task] = {
            "artifact": name,
            "exists": path.exists(),
            "path": str(path),
            "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0.0,
            "trained_at": metrics.get("trained_at"),
            "best_model": metrics.get("best_model"),
            "rows": metrics.get("train_rows"),
        }
    return summary
