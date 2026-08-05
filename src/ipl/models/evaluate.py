"""Model evaluation and selection.

Classifiers are scored on accuracy, precision, recall, F1 and ROC-AUC (plus
log-loss and Brier score, which matter because the dashboard reports
*probabilities*, not just labels -- a model can be accurate while being badly
calibrated, and a 78%-confident prediction should be right about 78% of the
time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Metric used to choose the production model for each task.
PRIMARY_CLASSIFICATION_METRIC = "roc_auc"
PRIMARY_REGRESSION_METRIC = "rmse"


@dataclass
class ClassificationMetrics:
    """Scores for one binary classifier on one dataset."""

    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    brier: float
    support: int
    positive_rate: float
    train_seconds: float = 0.0
    confusion: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.model:<22} acc={self.accuracy:.3f} prec={self.precision:.3f} "
            f"rec={self.recall:.3f} f1={self.f1:.3f} auc={self.roc_auc:.3f} "
            f"logloss={self.log_loss:.3f}"
        )


@dataclass
class RegressionMetrics:
    """Scores for one regressor on one dataset."""

    model: str
    rmse: float
    mae: float
    r2: float
    mean_error: float
    within_10: float
    within_20: float
    support: int
    train_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.model:<22} rmse={self.rmse:.2f} mae={self.mae:.2f} "
            f"r2={self.r2:.3f} within10={self.within_10:.1%}"
        )


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities, whatever the estimator exposes.

    Falls back to a decision function squashed through a logistic, so a model
    without ``predict_proba`` can still be ranked on AUC.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X), dtype=float)


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    name: str,
    threshold: float = 0.5,
    train_seconds: float = 0.0,
) -> ClassificationMetrics:
    """Score a fitted classifier on a held-out set."""
    y_true = np.asarray(y).astype(int)
    probabilities = _predict_proba(model, X)
    predictions = (probabilities >= threshold).astype(int)

    # A test fold containing only one class makes AUC and log-loss undefined.
    single_class = len(np.unique(y_true)) < 2

    return ClassificationMetrics(
        model=name,
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float("nan") if single_class else float(roc_auc_score(y_true, probabilities)),
        log_loss=(
            float("nan")
            if single_class
            else float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7)))
        ),
        brier=float(brier_score_loss(y_true, probabilities)),
        support=int(len(y_true)),
        positive_rate=float(y_true.mean()) if len(y_true) else 0.0,
        train_seconds=round(train_seconds, 2),
        confusion=confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    )


def evaluate_regressor(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    name: str,
    train_seconds: float = 0.0,
) -> RegressionMetrics:
    """Score a fitted regressor on a held-out set."""
    y_true = np.asarray(y, dtype=float)
    predictions = np.asarray(model.predict(X), dtype=float)
    errors = predictions - y_true

    return RegressionMetrics(
        model=name,
        rmse=float(np.sqrt(mean_squared_error(y_true, predictions))),
        mae=float(mean_absolute_error(y_true, predictions)),
        r2=float(r2_score(y_true, predictions)) if len(y_true) > 1 else float("nan"),
        mean_error=float(errors.mean()),
        # Share of predictions within 10 / 20 runs - far more intuitive for a
        # cricket audience than RMSE alone.
        within_10=float((np.abs(errors) <= 10).mean()),
        within_20=float((np.abs(errors) <= 20).mean()),
        support=int(len(y_true)),
        train_seconds=round(train_seconds, 2),
    )


def pick_best_model(
    metrics: list[ClassificationMetrics] | list[RegressionMetrics],
    *,
    metric: str | None = None,
    higher_is_better: bool | None = None,
) -> str | None:
    """Select the winning model by the primary metric.

    Ties (and NaN scores, which happen when a metric is undefined) are handled
    explicitly so selection is deterministic across runs.
    """
    if not metrics:
        return None

    is_classification = isinstance(metrics[0], ClassificationMetrics)
    if metric is None:
        metric = (
            PRIMARY_CLASSIFICATION_METRIC if is_classification else PRIMARY_REGRESSION_METRIC
        )
    if higher_is_better is None:
        higher_is_better = metric not in {"rmse", "mae", "log_loss", "brier", "mean_error"}

    scored = [
        (getattr(m, metric, float("nan")), m.model)
        for m in metrics
        if not np.isnan(getattr(m, metric, float("nan")))
    ]
    if not scored:
        logger.warning("No model produced a usable %s; falling back to the first.", metric)
        return metrics[0].model

    # Sort by score, then name, so equal scores always resolve the same way.
    scored.sort(key=lambda pair: (-pair[0] if higher_is_better else pair[0], pair[1]))
    best_score, best_name = scored[0]
    logger.info("Best model by %s: %s (%.4f)", metric, best_name, best_score)
    return best_name


def metrics_to_frame(
    metrics: list[ClassificationMetrics] | list[RegressionMetrics],
) -> pd.DataFrame:
    """Render a list of metric objects as a display-ready DataFrame."""
    if not metrics:
        return pd.DataFrame()
    frame = pd.DataFrame([m.as_dict() for m in metrics])
    # The confusion matrix is a nested list; keep it out of the flat table.
    return frame.drop(columns=[c for c in ("confusion",) if c in frame.columns])


def roc_curve_points(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
    """Return ROC curve coordinates for plotting on the comparison page."""
    y_true = np.asarray(y).astype(int)
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    probabilities = _predict_proba(model, X)
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def calibration_table(
    model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    """Bin predictions by confidence and compare to the observed win rate.

    A well-calibrated model's ``predicted`` and ``observed`` columns track each
    other; systematic divergence means the reported percentages are misleading
    even if the accuracy looks fine.
    """
    y_true = np.asarray(y).astype(int)
    probabilities = _predict_proba(model, X)
    if len(y_true) == 0:
        return pd.DataFrame()

    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)

    rows = []
    for bucket in range(bins):
        mask = indices == bucket
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"{edges[bucket]:.0%}-{edges[bucket + 1]:.0%}",
                "count": int(mask.sum()),
                "predicted": float(probabilities[mask].mean()),
                "observed": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(model: Any, feature_names: list[str], *, top_n: int = 25) -> pd.DataFrame:
    """Extract feature importances from whichever attribute the model exposes."""
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        # For linear models, magnitude of the coefficient is the analogue.
        importances = np.abs(np.asarray(model.coef_, dtype=float)).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    frame["importance"] = frame["importance"] / (frame["importance"].sum() or 1)
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
