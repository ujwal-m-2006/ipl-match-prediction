"""The model zoo.

Every algorithm the project compares is declared here with sensible,
lightly-regularised defaults. Gradient-boosting libraries that ship as separate
wheels (XGBoost, LightGBM, CatBoost) are imported defensively: if a wheel is
missing for the running Python version the model is simply omitted from the
comparison rather than crashing the training run.

Hyperparameters are deliberately modest. With ~1100 training matches, deep
forests overfit badly; shallow trees and strong learning-rate damping give the
honest out-of-sample numbers that the model-comparison page reports.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Populated at import time with whichever optional libraries are installed.
OPTIONAL_LIBRARIES: dict[str, bool] = {}

try:  # pragma: no cover - depends on the environment
    from xgboost import XGBClassifier, XGBRegressor

    OPTIONAL_LIBRARIES["xgboost"] = True
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment]
    OPTIONAL_LIBRARIES["xgboost"] = False

try:  # pragma: no cover
    from lightgbm import LGBMClassifier, LGBMRegressor

    OPTIONAL_LIBRARIES["lightgbm"] = True
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment]
    OPTIONAL_LIBRARIES["lightgbm"] = False

try:  # pragma: no cover
    from catboost import CatBoostClassifier, CatBoostRegressor

    OPTIONAL_LIBRARIES["catboost"] = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = CatBoostRegressor = None  # type: ignore[assignment]
    OPTIONAL_LIBRARIES["catboost"] = False


# Display names, in the order the comparison table should present them.
AVAILABLE_CLASSIFIERS: tuple[str, ...] = (
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)

AVAILABLE_REGRESSORS: tuple[str, ...] = (
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)


def build_classifiers(random_state: int = 42, *, scale_pos_weight: float | None = None) -> dict[str, Any]:
    """Instantiate every available binary classifier.

    Args:
        random_state: Seed applied to every stochastic learner.
        scale_pos_weight: Positive-class weight for the boosted models. Set
            this when the target is imbalanced (as it is for Player of the
            Match, where ~1 in 22 rows is positive).
    """
    models: dict[str, Any] = {
        # `saga` handles the wide one-hot design matrix without convergence
        # warnings, and a higher iteration cap avoids a truncated fit.
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample" if scale_pos_weight else None,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=random_state,
        ),
    }

    if OPTIONAL_LIBRARIES.get("xgboost"):
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight or 1.0,
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )

    if OPTIONAL_LIBRARIES.get("lightgbm"):
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=24,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            # The default (20) is too high for our row counts and produces
            # degenerate single-leaf trees on the smaller targets.
            min_child_samples=10,
            is_unbalance=bool(scale_pos_weight),
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    if OPTIONAL_LIBRARIES.get("catboost"):
        models["CatBoost"] = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            auto_class_weights="Balanced" if scale_pos_weight else None,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    missing = [name for name, ok in OPTIONAL_LIBRARIES.items() if not ok]
    if missing:
        logger.warning(
            "Optional model libraries unavailable (%s); comparison will omit them.",
            ", ".join(missing),
        )
    return models


def build_regressors(random_state: int = 42) -> dict[str, Any]:
    """Instantiate every available regressor for the score model."""
    models: dict[str, Any] = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=random_state,
        ),
    }

    if OPTIONAL_LIBRARIES.get("xgboost"):
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )

    if OPTIONAL_LIBRARIES.get("lightgbm"):
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=24,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            min_child_samples=10,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    if OPTIONAL_LIBRARIES.get("catboost"):
        models["CatBoost"] = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def needs_scaling(model_name: str) -> bool:
    """True for models whose performance depends on feature scaling.

    Tree ensembles are invariant to monotone rescaling; the linear models are
    not, and will converge slowly (or not at all) on raw features.
    """
    return model_name in {"Logistic Regression", "Ridge Regression"}


def model_family(model_name: str) -> str:
    """Coarse family label used for grouping in the comparison UI."""
    if model_name in {"Logistic Regression", "Ridge Regression"}:
        return "Linear"
    if model_name == "Random Forest":
        return "Bagging"
    return "Boosting"


def describe_availability() -> dict[str, bool]:
    """Report which optional libraries were importable, for the Admin page."""
    return dict(OPTIONAL_LIBRARIES)
