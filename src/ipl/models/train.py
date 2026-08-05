"""Training orchestration for every prediction task.

Four models are trained, each comparing the full zoo from
:mod:`ipl.models.registry` and keeping the best by its primary metric:

============  ==================================  =======================
Task          Question                            Primary metric
============  ==================================  =======================
``winner``    Which side wins this fixture?       ROC-AUC
``score``     What will the first innings total?  RMSE
``chase``     Will this chase succeed from here?  ROC-AUC
``pom``       Who wins Player of the Match?       ROC-AUC (ranked)
============  ==================================  =======================

Every task uses the same **time-based** split: recent seasons are held out
entirely. Numbers reported here are therefore genuine out-of-sample estimates,
not the inflated figures a random split would produce on time-series data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import get_settings
from ..db import repository as repo
from ..features.inplay_features import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    POM_NUMERIC_FEATURES,
    build_chase_features,
    build_player_of_match_features,
)
from ..features.match_features import (
    MATCH_CATEGORICAL_FEATURES,
    MATCH_NUMERIC_FEATURES,
    SCORE_CATEGORICAL_FEATURES,
    SCORE_NUMERIC_FEATURES,
    build_match_features,
    build_score_features,
    mirror_fixtures,
)
from ..features.preprocessing import build_preprocessor, clean_feature_frame, split_by_season
from ..logging_utils import get_logger
from .evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    calibration_table,
    evaluate_classifier,
    evaluate_regressor,
    feature_importance,
    metrics_to_frame,
    pick_best_model,
)
from .persistence import ARTIFACTS, save_artifact, save_metrics
from .registry import build_classifiers, build_regressors, needs_scaling

logger = get_logger(__name__)

# Below this many training rows a comparison is noise, not evidence.
MIN_TRAINING_ROWS = 60


@dataclass
class TrainingResult:
    """Everything one training task produces."""

    task: str
    best_model: str | None
    metrics: list
    train_rows: int
    test_rows: int
    artifact: str | None = None

    def summary(self) -> str:
        return (
            f"[{self.task}] best={self.best_model} "
            f"train={self.train_rows} test={self.test_rows}"
        )


def _fit_and_score(
    models: dict[str, Any],
    preprocessor_factory,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    task: str,
    classification: bool,
) -> tuple[dict[str, Pipeline], list]:
    """Fit every candidate model and score it on the held-out set.

    Each model gets its own preprocessor instance -- linear models need scaled
    numerics, tree ensembles do not, and sharing one fitted transformer across
    estimators would silently couple them.
    """
    fitted: dict[str, Pipeline] = {}
    scores: list = []

    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("prep", preprocessor_factory(scale=needs_scaling(name))),
                ("model", estimator),
            ]
        )
        started = time.monotonic()
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:  # pragma: no cover - a bad wheel or bad data
            logger.error("[%s] %s failed to fit: %s", task, name, exc)
            continue
        elapsed = time.monotonic() - started

        evaluate = evaluate_classifier if classification else evaluate_regressor
        try:
            metrics = evaluate(pipeline, X_test, y_test, name=name, train_seconds=elapsed)
        except Exception as exc:  # pragma: no cover
            logger.error("[%s] %s failed to score: %s", task, name, exc)
            continue

        fitted[name] = pipeline
        scores.append(metrics)
        logger.info("[%s] %s", task, metrics.summary())

    return fitted, scores


def _feature_names(pipeline: Pipeline) -> list[str]:
    """Recover post-transform feature names for importance plots."""
    try:
        return list(pipeline.named_steps["prep"].get_feature_names_out())
    except Exception:  # pragma: no cover - older sklearn / exotic transformer
        return []


def _bundle(
    *,
    task: str,
    pipeline: Pipeline,
    best_name: str,
    scores: list,
    numeric: list[str],
    categorical: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the artefact payload written to disk."""
    payload = {
        "task": task,
        "best_model": best_name,
        "pipeline": pipeline,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "metrics": [m.as_dict() for m in scores],
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Task 1: match winner
# ---------------------------------------------------------------------------
def train_winner_model(
    features: pd.DataFrame, *, test_season_from: int, random_state: int
) -> TrainingResult:
    """Train the match-winner classifier.

    Target is ``1`` when ``team1`` (the nominal home side) wins. Because the
    fixture list is roughly balanced between home and away wins, no class
    weighting is applied.
    """
    frame = features[features["is_completed"] & features["target_team1_wins"].notna()].copy()
    if len(frame) < MIN_TRAINING_ROWS:
        logger.warning("Not enough completed matches (%d) to train a winner model", len(frame))
        return TrainingResult("winner", None, [], len(frame), 0)

    frame = clean_feature_frame(frame, MATCH_NUMERIC_FEATURES, MATCH_CATEGORICAL_FEATURES)
    train, test = split_by_season(frame, test_season_from=test_season_from)

    # Mirror the training fixtures so the classifier sees each match from both
    # sides. This removes the "team1 is listed first" artefact and doubles the
    # effective sample. The test set is left un-mirrored so the reported metrics
    # describe real fixtures, one row each.
    train = pd.concat([train, mirror_fixtures(train)], ignore_index=True)
    train = train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info("Winner training set after mirroring: %d rows", len(train))

    columns = MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES
    X_train, y_train = train[columns], train["target_team1_wins"].astype(int).to_numpy()
    X_test, y_test = test[columns], test["target_team1_wins"].astype(int).to_numpy()

    fitted, scores = _fit_and_score(
        build_classifiers(random_state),
        lambda scale: build_preprocessor(
            MATCH_NUMERIC_FEATURES, MATCH_CATEGORICAL_FEATURES, scale=scale
        ),
        X_train, y_train, X_test, y_test,
        task="winner", classification=True,
    )

    best = pick_best_model(scores)
    if best is None:
        return TrainingResult("winner", None, scores, len(train), len(test))

    pipeline = fitted[best]
    payload = _bundle(
        task="winner", pipeline=pipeline, best_name=best, scores=scores,
        numeric=MATCH_NUMERIC_FEATURES, categorical=MATCH_CATEGORICAL_FEATURES,
        extra={
            "importance": feature_importance(
                pipeline.named_steps["model"], _feature_names(pipeline)
            ).to_dict("records"),
            "calibration": calibration_table(pipeline, X_test, y_test).to_dict("records"),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_seasons": sorted(test["season"].unique().tolist()),
        },
    )
    name = ARTIFACTS["winner"]
    save_artifact(name, payload)
    save_metrics(name, {k: v for k, v in payload.items() if k != "pipeline"})
    return TrainingResult("winner", best, scores, len(train), len(test), name)


# ---------------------------------------------------------------------------
# Task 2: first-innings score
# ---------------------------------------------------------------------------
def train_score_model(
    features: pd.DataFrame, *, test_season_from: int, random_state: int
) -> TrainingResult:
    """Train the first-innings total regressor."""
    frame = build_score_features(features)
    if frame.empty or len(frame) < MIN_TRAINING_ROWS:
        logger.warning("Not enough innings (%d) to train a score model", len(frame))
        return TrainingResult("score", None, [], len(frame), 0)

    frame = clean_feature_frame(frame, SCORE_NUMERIC_FEATURES, SCORE_CATEGORICAL_FEATURES)
    train, test = split_by_season(frame, test_season_from=test_season_from)

    columns = SCORE_NUMERIC_FEATURES + SCORE_CATEGORICAL_FEATURES
    X_train = train[columns]
    y_train = train["target_first_innings_runs"].astype(float).to_numpy()
    X_test = test[columns]
    y_test = test["target_first_innings_runs"].astype(float).to_numpy()

    fitted, scores = _fit_and_score(
        build_regressors(random_state),
        lambda scale: build_preprocessor(
            SCORE_NUMERIC_FEATURES, SCORE_CATEGORICAL_FEATURES, scale=scale
        ),
        X_train, y_train, X_test, y_test,
        task="score", classification=False,
    )

    best = pick_best_model(scores)
    if best is None:
        return TrainingResult("score", None, scores, len(train), len(test))

    pipeline = fitted[best]
    payload = _bundle(
        task="score", pipeline=pipeline, best_name=best, scores=scores,
        numeric=SCORE_NUMERIC_FEATURES, categorical=SCORE_CATEGORICAL_FEATURES,
        extra={
            "importance": feature_importance(
                pipeline.named_steps["model"], _feature_names(pipeline)
            ).to_dict("records"),
            "train_rows": len(train),
            "test_rows": len(test),
            # Residual spread drives the prediction interval shown in the UI.
            "residual_std": float(
                np.std(y_test - pipeline.predict(X_test)) if len(y_test) else 0.0
            ),
        },
    )
    name = ARTIFACTS["score"]
    save_artifact(name, payload)
    save_metrics(name, {k: v for k, v in payload.items() if k != "pipeline"})
    return TrainingResult("score", best, scores, len(train), len(test), name)


# ---------------------------------------------------------------------------
# Task 3: chase success
# ---------------------------------------------------------------------------
def train_chase_model(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    test_season_from: int,
    random_state: int,
    stride: int = 1,
) -> TrainingResult:
    """Train the in-play chase-success classifier."""
    frame = build_chase_features(deliveries, matches, stride=stride)
    if frame.empty or len(frame) < MIN_TRAINING_ROWS:
        logger.warning("Not enough ball-by-ball data (%d rows) to train a chase model", len(frame))
        return TrainingResult("chase", None, [], len(frame), 0)

    frame = clean_feature_frame(frame, CHASE_NUMERIC_FEATURES, CHASE_CATEGORICAL_FEATURES)
    train, test = split_by_season(frame, test_season_from=test_season_from)

    columns = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
    X_train, y_train = train[columns], train["target_chase_success"].astype(int).to_numpy()
    X_test, y_test = test[columns], test["target_chase_success"].astype(int).to_numpy()

    fitted, scores = _fit_and_score(
        build_classifiers(random_state),
        lambda scale: build_preprocessor(
            CHASE_NUMERIC_FEATURES, CHASE_CATEGORICAL_FEATURES, scale=scale
        ),
        X_train, y_train, X_test, y_test,
        task="chase", classification=True,
    )

    best = pick_best_model(scores)
    if best is None:
        return TrainingResult("chase", None, scores, len(train), len(test))

    pipeline = fitted[best]
    payload = _bundle(
        task="chase", pipeline=pipeline, best_name=best, scores=scores,
        numeric=CHASE_NUMERIC_FEATURES, categorical=CHASE_CATEGORICAL_FEATURES,
        extra={
            "importance": feature_importance(
                pipeline.named_steps["model"], _feature_names(pipeline)
            ).to_dict("records"),
            "calibration": calibration_table(pipeline, X_test, y_test).to_dict("records"),
            "train_rows": len(train),
            "test_rows": len(test),
        },
    )
    name = ARTIFACTS["chase"]
    save_artifact(name, payload)
    save_metrics(name, {k: v for k, v in payload.items() if k != "pipeline"})
    return TrainingResult("chase", best, scores, len(train), len(test), name)


# ---------------------------------------------------------------------------
# Task 4: player of the match
# ---------------------------------------------------------------------------
def train_pom_model(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    test_season_from: int,
    random_state: int,
) -> TrainingResult:
    """Train the Player-of-the-Match ranker.

    Framed as binary classification over every player-match row, then used as a
    ranker: the highest-probability player in a match is the prediction. Only
    ~1 row in 22 is positive, so the boosted models are given a positive-class
    weight to stop them collapsing to the majority class.
    """
    frame = build_player_of_match_features(batting, bowling, matches)
    if frame.empty or len(frame) < MIN_TRAINING_ROWS:
        logger.warning("Not enough player-match rows (%d) for a POM model", len(frame))
        return TrainingResult("pom", None, [], len(frame), 0)

    frame = frame.merge(
        matches[["match_id", "season"]].drop_duplicates(), on="match_id", how="left"
    )
    frame = clean_feature_frame(frame, POM_NUMERIC_FEATURES, [])
    train, test = split_by_season(frame, test_season_from=test_season_from)

    y_train = train["target_is_pom"].astype(int).to_numpy()
    y_test = test["target_is_pom"].astype(int).to_numpy()
    positives = max(int(y_train.sum()), 1)
    imbalance = (len(y_train) - positives) / positives

    X_train, X_test = train[POM_NUMERIC_FEATURES], test[POM_NUMERIC_FEATURES]

    fitted, scores = _fit_and_score(
        build_classifiers(random_state, scale_pos_weight=imbalance),
        lambda scale: build_preprocessor(POM_NUMERIC_FEATURES, [], scale=scale),
        X_train, y_train, X_test, y_test,
        task="pom", classification=True,
    )

    best = pick_best_model(scores)
    if best is None:
        return TrainingResult("pom", None, scores, len(train), len(test))

    pipeline = fitted[best]
    # The metric that actually matters: how often is the top-ranked player in a
    # match the real award winner? Accuracy on the flat binary task is
    # misleading here because predicting "nobody" scores 95%.
    top1 = _top1_accuracy(pipeline, test)

    payload = _bundle(
        task="pom", pipeline=pipeline, best_name=best, scores=scores,
        numeric=POM_NUMERIC_FEATURES, categorical=[],
        extra={
            "importance": feature_importance(
                pipeline.named_steps["model"], _feature_names(pipeline)
            ).to_dict("records"),
            "train_rows": len(train),
            "test_rows": len(test),
            "top1_accuracy": top1,
            "positive_rate": float(y_train.mean()),
        },
    )
    name = ARTIFACTS["pom"]
    save_artifact(name, payload)
    save_metrics(name, {k: v for k, v in payload.items() if k != "pipeline"})
    logger.info("[pom] top-1 accuracy on held-out matches: %.1f%%", top1 * 100)
    return TrainingResult("pom", best, scores, len(train), len(test), name)


def _top1_accuracy(pipeline: Pipeline, test: pd.DataFrame) -> float:
    """Share of held-out matches whose top-ranked player won the award."""
    if test.empty:
        return float("nan")
    scored = test.copy()
    scored["probability"] = pipeline.predict_proba(test[POM_NUMERIC_FEATURES])[:, 1]
    picks = scored.loc[scored.groupby("match_id")["probability"].idxmax()]
    return float(picks["target_is_pom"].mean())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_all(
    *,
    test_season_from: int | None = None,
    random_state: int | None = None,
    include_chase: bool = True,
    include_pom: bool = True,
    chase_stride: int = 3,
) -> dict[str, TrainingResult]:
    """Load data, train every task and persist the artefacts.

    Returns a mapping of task name to :class:`TrainingResult`.
    """
    settings = get_settings()
    test_season_from = test_season_from or settings.test_season_from
    random_state = random_state if random_state is not None else settings.random_state

    logger.info("Loading data from the warehouse...")
    matches = repo.load_matches()
    innings = repo.load_innings()
    if matches.empty:
        raise RuntimeError(
            "No matches in the database. Run `python scripts/ingest.py` first."
        )

    batting = repo.load_batting()
    bowling = repo.load_bowling()

    logger.info("Building pre-match features...")
    features, _state = build_match_features(
        matches,
        innings,
        batting=batting,
        bowling=bowling,
        match_players=repo.load_match_players(),
    )

    results: dict[str, TrainingResult] = {}

    logger.info("=" * 70)
    logger.info("TASK 1/4: match winner")
    results["winner"] = train_winner_model(
        features, test_season_from=test_season_from, random_state=random_state
    )

    logger.info("=" * 70)
    logger.info("TASK 2/4: first-innings score")
    results["score"] = train_score_model(
        features, test_season_from=test_season_from, random_state=random_state
    )

    if include_chase:
        logger.info("=" * 70)
        logger.info("TASK 3/4: chase success")
        deliveries = repo.load_deliveries()
        if deliveries.empty:
            logger.warning("No ball-by-ball data; skipping the chase model.")
            results["chase"] = TrainingResult("chase", None, [], 0, 0)
        else:
            results["chase"] = train_chase_model(
                deliveries, matches,
                test_season_from=test_season_from,
                random_state=random_state,
                stride=chase_stride,
            )

    if include_pom:
        logger.info("=" * 70)
        logger.info("TASK 4/4: player of the match")
        results["pom"] = train_pom_model(
            batting, bowling, matches,
            test_season_from=test_season_from, random_state=random_state,
        )

    logger.info("=" * 70)
    for result in results.values():
        logger.info("%s", result.summary())

    return results


def comparison_frame(results: dict[str, TrainingResult]) -> pd.DataFrame:
    """Flatten every task's model scores into one comparison table."""
    frames = []
    for task, result in results.items():
        if not result.metrics:
            continue
        frame = metrics_to_frame(result.metrics)
        frame.insert(0, "task", task)
        frame["is_best"] = frame["model"] == result.best_model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
