"""Cached data access for the dashboard.

Streamlit re-runs the whole script on every interaction, so every warehouse
read is wrapped in ``st.cache_data``. Without this, dragging a slider would
re-query a 270k-row deliveries table.

Caches are cleared by :func:`clear_caches`, which the Admin page calls after a
data refresh or a retrain.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from ..db import repository as repo
from ..features.match_features import FeatureState, build_match_features
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Most warehouse tables change only when the pipeline runs, so a long TTL is
# safe; the Admin page clears the cache explicitly after a refresh.
CACHE_TTL = 3600


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading matches...")
def load_matches() -> pd.DataFrame:
    return repo.load_matches()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading innings...")
def load_innings() -> pd.DataFrame:
    return repo.load_innings()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading batting cards...")
def load_batting() -> pd.DataFrame:
    return repo.load_batting()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading bowling cards...")
def load_bowling() -> pd.DataFrame:
    return repo.load_bowling()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading ball-by-ball data...")
def load_deliveries(season: int | None = None) -> pd.DataFrame:
    return repo.load_deliveries(season)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading squads...")
def load_match_players() -> pd.DataFrame:
    return repo.load_match_players()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading partnerships...")
def load_partnerships() -> pd.DataFrame:
    return repo.load_partnerships()


@st.cache_data(ttl=300)
def database_summary() -> dict[str, int]:
    """Row counts per table, or zeros when the schema does not exist yet."""
    try:
        return repo.database_summary()
    except SQLAlchemyError as exc:
        # A fresh clone opens the dashboard before `scripts/ingest.py` has ever
        # run. That must render an empty state, not a stack trace.
        logger.warning("Could not read database summary: %s", exc)
        return {}


@st.cache_data(ttl=60)
def latest_ingestion_run() -> dict | None:
    try:
        return repo.latest_ingestion_run()
    except SQLAlchemyError as exc:
        logger.warning("Could not read the ingestion log: %s", exc)
        return None


@st.cache_resource(show_spinner="Building feature state...")
def feature_state() -> tuple[pd.DataFrame, FeatureState]:
    """Rolling feature state, cached as a resource (it holds live objects)."""
    return build_match_features(
        load_matches(),
        load_innings(),
        batting=load_batting(),
        bowling=load_bowling(),
        match_players=load_match_players(),
    )


@st.cache_resource(show_spinner="Loading trained models...")
def prediction_service():
    """Return the shared prediction service, or ``None`` if untrained."""
    from ..models.predict import PredictionService

    return PredictionService()


def clear_caches() -> None:
    """Drop every cached frame and resource.

    Called after ingestion or training so the UI immediately reflects new data
    instead of serving hour-old results.
    """
    st.cache_data.clear()
    st.cache_resource.clear()
    from ..models.predict import reset_prediction_service

    reset_prediction_service()
    logger.info("Dashboard caches cleared")


def has_data() -> bool:
    """True when the warehouse contains at least one match."""
    return database_summary().get("matches", 0) > 0


def seasons() -> list[int]:
    """Descending list of seasons present in the warehouse."""
    matches = load_matches()
    if matches.empty:
        return []
    return sorted(matches["season"].dropna().unique().astype(int).tolist(), reverse=True)


def teams() -> list[str]:
    """Every franchise that has played, alphabetically.

    Includes defunct sides (Deccan Chargers, Pune Warriors, ...), which belong
    in historical analytics.
    """
    matches = load_matches()
    if matches.empty:
        return []
    names = pd.concat([matches["team1"], matches["team2"]]).dropna().unique()
    return sorted(names.tolist())


def prediction_teams() -> list[str]:
    """Franchises ordered for the prediction pickers: current sides first.

    Defunct franchises stay selectable -- someone may want to ask what a 2013
    matchup would look like -- but they should not be the default, which is
    what a plain alphabetical list would produce.
    """
    from ..constants import ACTIVE_TEAMS

    everyone = teams()
    active = [t for t in everyone if t in ACTIVE_TEAMS]
    retired = [t for t in everyone if t not in ACTIVE_TEAMS]
    return active + retired


def venues() -> list[str]:
    """Every venue, most-used first."""
    matches = load_matches()
    if matches.empty:
        return []
    return matches["venue"].value_counts().index.dropna().tolist()


def players(min_matches: int = 5) -> list[str]:
    """Players with at least ``min_matches`` appearances, alphabetically."""
    batting = load_batting()
    bowling = load_bowling()
    frames = [f for f in (batting, bowling) if not f.empty]
    if not frames:
        return []

    counts = (
        pd.concat([f[["player", "match_id"]] for f in frames])
        .drop_duplicates()
        .groupby("player")
        .size()
    )
    return sorted(counts[counts >= min_matches].index.tolist())
