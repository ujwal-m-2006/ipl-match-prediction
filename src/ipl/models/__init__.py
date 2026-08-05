"""Machine-learning layer: model zoo, training, evaluation and prediction."""

from .evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classifier,
    evaluate_regressor,
    pick_best_model,
)
from .persistence import (
    ARTIFACTS,
    load_artifact,
    load_metrics,
    save_artifact,
    save_metrics,
)
from .registry import (
    AVAILABLE_CLASSIFIERS,
    AVAILABLE_REGRESSORS,
    build_classifiers,
    build_regressors,
)

__all__ = [
    "AVAILABLE_CLASSIFIERS",
    "AVAILABLE_REGRESSORS",
    "build_classifiers",
    "build_regressors",
    "evaluate_classifier",
    "evaluate_regressor",
    "ClassificationMetrics",
    "RegressionMetrics",
    "pick_best_model",
    "save_artifact",
    "load_artifact",
    "save_metrics",
    "load_metrics",
    "ARTIFACTS",
]
