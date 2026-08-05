"""Shared preprocessing: sklearn pipelines and time-based splitting.

All models consume the same column-transformer shape -- median-imputed and
scaled numerics, most-frequent-imputed one-hot categoricals -- so they can be
compared on equal footing and swapped without touching the training loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _one_hot_encoder() -> OneHotEncoder:
    """Build a OneHotEncoder that tolerates unseen categories at predict time.

    A franchise or venue that appears only in the test split (or only in
    production) must not raise -- it is encoded as all-zeros instead.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=1)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """Assemble the standard column transformer.

    Args:
        numeric_features: Columns treated as continuous.
        categorical_features: Columns treated as nominal.
        scale: Standardise numerics. Needed by Logistic Regression; harmless
            but unnecessary for tree ensembles.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_by_season(
    frame: pd.DataFrame,
    *,
    test_season_from: int,
    season_column: str = "season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: everything before ``test_season_from`` trains.

    A random split would leak the future into the past -- a model could learn
    from the 2026 final while being scored on a 2019 league game. Holding out
    whole recent seasons is the only split that answers the question the model
    is actually for: how well will it do on matches that have not happened yet.
    """
    if frame.empty:
        return frame, frame

    train = frame[frame[season_column] < test_season_from]
    test = frame[frame[season_column] >= test_season_from]

    if test.empty:
        # Guard against a misconfigured cut-off leaving nothing to evaluate on:
        # fall back to holding out the most recent season present.
        seasons = sorted(frame[season_column].unique())
        if len(seasons) > 1:
            fallback = seasons[-1]
            logger.warning(
                "No rows at/after season %s; holding out %s instead",
                test_season_from, fallback,
            )
            train = frame[frame[season_column] < fallback]
            test = frame[frame[season_column] >= fallback]

    logger.info(
        "Season split: train=%d rows (%s), test=%d rows (%s)",
        len(train),
        _season_span(train, season_column),
        len(test),
        _season_span(test, season_column),
    )
    return train, test


def _season_span(frame: pd.DataFrame, column: str) -> str:
    """Render a frame's season coverage as ``"2008-2024"``."""
    if frame.empty:
        return "empty"
    return f"{int(frame[column].min())}-{int(frame[column].max())}"


def clean_feature_frame(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Coerce feature columns to usable dtypes and drop unusable rows.

    Infinities (from run-rate divisions) become NaN so the imputer handles them
    rather than the model receiving a value it cannot split on.
    """
    out = frame.copy()

    for column in numeric_features:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out[numeric_features] = out[numeric_features].replace([np.inf, -np.inf], np.nan)

    for column in categorical_features:
        if column not in out.columns:
            out[column] = "Unknown"
        out[column] = out[column].astype("string").fillna("Unknown").astype(str)

    return out
