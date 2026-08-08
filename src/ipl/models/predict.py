"""Inference service - the single entry point for every prediction.

Both the Streamlit dashboard and the FastAPI service call into
:class:`PredictionService`, so a prediction is computed exactly one way no
matter how it was requested.

The service holds the rolling :class:`~ipl.features.match_features.FeatureState`
built from the full match history. That state is what lets an *unplayed*
fixture be featurised identically to a training row: same code, same feature
definitions, no train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..constants import ACTIVE_TEAMS, TEAM_HOME_VENUES
from ..db import repository as repo
from ..features.inplay_features import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    POM_NUMERIC_FEATURES,
    build_player_of_match_features,
    chase_feature_row,
)
from ..features.match_features import (
    MATCH_CATEGORICAL_FEATURES,
    MATCH_NUMERIC_FEATURES,
    SCORE_CATEGORICAL_FEATURES,
    SCORE_NUMERIC_FEATURES,
    FeatureState,
    build_match_features,
    featurise_fixture,
)
from ..logging_utils import get_logger
from .persistence import ARTIFACTS, ArtifactNotFound, load_artifact

logger = get_logger(__name__)


@dataclass
class WinnerPrediction:
    """Result of a match-winner prediction."""

    team1: str
    team2: str
    venue: str
    predicted_winner: str
    team1_win_probability: float
    team2_win_probability: float
    confidence: float
    model: str
    drivers: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "team1": self.team1,
            "team2": self.team2,
            "venue": self.venue,
            "predicted_winner": self.predicted_winner,
            "team1_win_probability": round(self.team1_win_probability * 100, 2),
            "team2_win_probability": round(self.team2_win_probability * 100, 2),
            "confidence": round(self.confidence * 100, 2),
            "model": self.model,
            "drivers": self.drivers,
        }


@dataclass
class ScorePrediction:
    """Result of a first-innings score prediction."""

    batting_team: str
    bowling_team: str
    venue: str
    predicted_score: float
    lower_bound: float
    upper_bound: float
    model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "venue": self.venue,
            "predicted_score": round(self.predicted_score),
            "range_low": round(self.lower_bound),
            "range_high": round(self.upper_bound),
            "model": self.model,
        }


@dataclass
class ChasePrediction:
    """Result of an in-play chase-success prediction."""

    batting_team: str
    bowling_team: str
    target: int
    current_runs: int
    wickets_fallen: int
    balls_bowled: int
    runs_required: int
    balls_remaining: int
    required_run_rate: float
    chase_success_probability: float
    model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "target": self.target,
            "current_runs": self.current_runs,
            "wickets_fallen": self.wickets_fallen,
            "balls_bowled": self.balls_bowled,
            "runs_required": self.runs_required,
            "balls_remaining": self.balls_remaining,
            "required_run_rate": round(self.required_run_rate, 2),
            "chase_success_probability": round(self.chase_success_probability * 100, 2),
            "win_probability_batting": round(self.chase_success_probability * 100, 2),
            "win_probability_bowling": round((1 - self.chase_success_probability) * 100, 2),
            "model": self.model,
        }


class PredictionService:
    """Loads trained artefacts and serves predictions.

    Construction is relatively expensive (it replays the whole match history to
    rebuild the rolling feature state), so callers should reuse one instance --
    see :func:`get_prediction_service`.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._state: FeatureState | None = None
        self._matches: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None

    # -- lazy loading -------------------------------------------------------
    def artifact(self, task: str) -> dict[str, Any]:
        """Load (and memoise) a trained artefact bundle."""
        if task not in self._artifacts:
            self._artifacts[task] = load_artifact(ARTIFACTS[task])
        return self._artifacts[task]

    def has_model(self, task: str) -> bool:
        """True when the named model is trained and can actually be loaded.

        Unpickling a model needs the library that produced it -- the winner
        model is a CatBoost estimator, so a deployment where CatBoost failed to
        install raises ``ModuleNotFoundError`` here rather than at import time.
        Treating that as "not available" lets the rest of the app carry on
        serving the models that did load, instead of the page dying outright.
        """
        try:
            self.artifact(task)
            return True
        except (ArtifactNotFound, KeyError):
            return False
        except (ModuleNotFoundError, ImportError) as exc:
            logger.warning(
                "Model '%s' cannot be loaded because a library is missing (%s). "
                "Install it to enable this prediction.", task, exc,
            )
            return False
        except Exception as exc:  # pragma: no cover - corrupt artefact
            logger.error("Model '%s' failed to load: %s", task, exc)
            return False

    @property
    def matches(self) -> pd.DataFrame:
        if self._matches is None:
            self._matches = repo.load_matches()
        return self._matches

    @property
    def state(self) -> FeatureState:
        """Rolling feature state built from every completed match.

        Built with the same inputs the trainer uses, so the squad-strength and
        era features at serving time are computed identically to training time.
        """
        if self._state is None:
            features, state = build_match_features(
                self.matches,
                repo.load_innings(),
                batting=repo.load_batting(),
                bowling=repo.load_bowling(),
                match_players=repo.load_match_players(),
            )
            self._features = features
            self._state = state
        return self._state

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def default_venue(team: str) -> str:
        """A franchise's home ground, used when the caller omits a venue."""
        return TEAM_HOME_VENUES.get(team, "Wankhede Stadium")

    def available_teams(self) -> list[str]:
        """Teams that can be selected for a prediction."""
        if self.matches.empty:
            return list(ACTIVE_TEAMS)
        seen = set(self.matches["team1"].dropna()) | set(self.matches["team2"].dropna())
        return sorted(seen & set(ACTIVE_TEAMS)) or sorted(seen)

    def available_venues(self) -> list[str]:
        """Venues present in the warehouse, most-used first."""
        if self.matches.empty:
            return sorted(set(TEAM_HOME_VENUES.values()))
        counts = self.matches["venue"].value_counts()
        return counts.index.dropna().tolist()

    # -- predictions --------------------------------------------------------
    def predict_winner(
        self,
        *,
        team1: str,
        team2: str,
        venue: str | None = None,
        season: int | None = None,
        match_date: date | None = None,
        toss_winner: str | None = None,
        toss_decision: str | None = None,
        is_playoff: bool = False,
        is_neutral_venue: bool = False,
    ) -> WinnerPrediction:
        """Predict the winner of a fixture and the win probability for each side."""
        if team1 == team2:
            raise ValueError("A team cannot play itself.")

        bundle = self.artifact("winner")
        pipeline = bundle["pipeline"]

        venue = venue or self.default_venue(team1)
        season = season or (
            int(self.matches["season"].max()) if not self.matches.empty else date.today().year
        )

        row = featurise_fixture(
            self.state,
            team1=team1, team2=team2, venue=venue, season=season,
            match_date=match_date, toss_winner=toss_winner,
            toss_decision=toss_decision, is_playoff=is_playoff,
            is_neutral_venue=is_neutral_venue,
        )
        columns = MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES
        probability = float(pipeline.predict_proba(row[columns])[0, 1])

        winner = team1 if probability >= 0.5 else team2
        return WinnerPrediction(
            team1=team1,
            team2=team2,
            venue=venue,
            predicted_winner=winner,
            team1_win_probability=probability,
            team2_win_probability=1 - probability,
            # Distance from a coin flip, rescaled to 0-1.
            confidence=abs(probability - 0.5) * 2,
            model=bundle["best_model"],
            drivers=self._explain(row),
        )

    def _explain(self, row: pd.DataFrame) -> list[dict[str, Any]]:
        """Surface the handful of context numbers behind a prediction.

        Not a SHAP decomposition -- these are the raw feature values a cricket
        fan would want to see cited alongside a percentage.
        """
        if row.empty:
            return []
        record = row.iloc[0]
        interesting = [
            ("Recent form (last 5)", "team1_form_short", "team2_form_short", "pct"),
            ("Career win rate", "team1_career_win_rate", "team2_career_win_rate", "pct"),
            ("Head-to-head", "team1_h2h_win_rate", None, "pct"),
            ("Record at this venue", "team1_venue_win_rate", "team2_venue_win_rate", "pct"),
            ("Avg runs scored", "team1_avg_runs_scored", "team2_avg_runs_scored", "num"),
            ("Rest days", "team1_rest_days", "team2_rest_days", "num"),
        ]
        drivers = []
        for label, key1, key2, kind in interesting:
            if key1 not in record:
                continue
            value1 = float(record[key1])
            value2 = float(record[key2]) if key2 and key2 in record else None
            drivers.append(
                {
                    "label": label,
                    "team1_value": round(value1 * 100, 1) if kind == "pct" else round(value1, 1),
                    "team2_value": (
                        None if value2 is None
                        else (round(value2 * 100, 1) if kind == "pct" else round(value2, 1))
                    ),
                    "unit": "%" if kind == "pct" else "",
                }
            )
        return drivers

    def predict_first_innings_score(
        self,
        *,
        batting_team: str,
        bowling_team: str,
        venue: str | None = None,
        season: int | None = None,
        is_playoff: bool = False,
        is_neutral_venue: bool = False,
        batting_won_toss: bool = True,
    ) -> ScorePrediction:
        """Predict the first-innings total, with a one-sigma range."""
        bundle = self.artifact("score")
        pipeline = bundle["pipeline"]

        venue = venue or self.default_venue(batting_team)
        season = season or (
            int(self.matches["season"].max()) if not self.matches.empty else date.today().year
        )

        state = self.state
        batting = state.teams[batting_team]
        bowling = state.teams[bowling_team]
        batting_xi = state.squad_strength(state.last_lineup.get(batting_team))
        bowling_xi = state.squad_strength(state.last_lineup.get(bowling_team))

        row = pd.DataFrame(
            [
                {
                    "season": season,
                    "venue": venue,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "is_neutral_venue": int(is_neutral_venue),
                    "is_playoff": int(is_playoff),
                    "batting_is_home": int(
                        TEAM_HOME_VENUES.get(batting_team) == venue
                    ),
                    "batting_won_toss": int(batting_won_toss),
                    "batting_career_win_rate": batting.win_rate(),
                    "batting_form_short": batting.form(5),
                    "bowling_form_short": bowling.form(5),
                    "batting_avg_runs_scored": batting.avg_scored(),
                    "bowling_avg_runs_conceded": bowling.avg_conceded(),
                    "venue_avg_first_innings": state.venue_avg_first_innings(venue),
                    "batting_venue_avg_score": state.venue_team_score(venue, batting_team),
                    "batting_rest_days": 3.0,
                    "league_recent_avg_score": state.league_recent_score(),
                    "venue_recent_avg_score": state.venue_recent_score(venue),
                    "batting_xi_batting_average": batting_xi["batting_average"],
                    "batting_xi_strike_rate": batting_xi["strike_rate"],
                    "bowling_xi_bowling_economy": bowling_xi["bowling_economy"],
                }
            ]
        )

        columns = SCORE_NUMERIC_FEATURES + SCORE_CATEGORICAL_FEATURES
        predicted = float(pipeline.predict(row[columns])[0])
        spread = float(bundle.get("residual_std") or 20.0)

        return ScorePrediction(
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            predicted_score=predicted,
            lower_bound=max(predicted - spread, 0),
            upper_bound=predicted + spread,
            model=bundle["best_model"],
        )

    def predict_chase(
        self,
        *,
        batting_team: str,
        bowling_team: str,
        venue: str,
        target: int,
        current_runs: int,
        wickets_fallen: int,
        balls_bowled: int,
        runs_last_5_overs: int = 0,
        wickets_last_5_overs: int = 0,
    ) -> ChasePrediction:
        """Predict whether a chase will succeed from the current match state."""
        if target <= 0:
            raise ValueError("Target must be positive.")
        if not 0 <= wickets_fallen <= 10:
            raise ValueError("Wickets fallen must be between 0 and 10.")
        if not 0 <= balls_bowled <= 120:
            raise ValueError("Balls bowled must be between 0 and 120.")

        bundle = self.artifact("chase")
        pipeline = bundle["pipeline"]

        row = chase_feature_row(
            target=target,
            current_runs=current_runs,
            wickets_fallen=wickets_fallen,
            balls_bowled=balls_bowled,
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            runs_last_5_overs=runs_last_5_overs,
            wickets_last_5_overs=wickets_last_5_overs,
        )
        columns = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
        probability = float(pipeline.predict_proba(row[columns])[0, 1])

        # The model is only asked about live states. Terminal states are decided
        # by the laws of cricket, not by a classifier.
        if current_runs >= target:
            probability = 1.0
        elif wickets_fallen >= 10 or balls_bowled >= 120:
            probability = 0.0

        record = row.iloc[0]
        return ChasePrediction(
            batting_team=batting_team,
            bowling_team=bowling_team,
            target=target,
            current_runs=current_runs,
            wickets_fallen=wickets_fallen,
            balls_bowled=balls_bowled,
            runs_required=int(record["runs_required"]),
            balls_remaining=int(record["balls_remaining"]),
            required_run_rate=float(record["required_run_rate"]),
            chase_success_probability=probability,
            model=bundle["best_model"],
        )

    def predict_player_of_match(
        self, match_id: int, *, top_n: int = 5
    ) -> pd.DataFrame:
        """Rank the players in a completed match by award probability."""
        bundle = self.artifact("pom")
        pipeline = bundle["pipeline"]

        batting = repo.load_batting()
        bowling = repo.load_bowling()
        frame = build_player_of_match_features(
            batting[batting["match_id"] == match_id],
            bowling[bowling["match_id"] == match_id],
            self.matches[self.matches["match_id"] == match_id],
        )
        if frame.empty:
            return pd.DataFrame()

        for column in POM_NUMERIC_FEATURES:
            if column not in frame.columns:
                frame[column] = 0.0
        frame["probability"] = pipeline.predict_proba(frame[POM_NUMERIC_FEATURES])[:, 1]
        # Normalise within the match so the column reads as a share of the award.
        total = frame["probability"].sum() or 1.0
        frame["award_probability"] = (frame["probability"] / total * 100).round(2)

        columns = [
            "player", "team", "runs", "balls", "wickets", "runs_conceded",
            "total_impact", "award_probability",
        ]
        available = [c for c in columns if c in frame.columns]
        return (
            frame.sort_values("award_probability", ascending=False)
            .head(top_n)[available]
            .reset_index(drop=True)
        )

    def score_upcoming_fixtures(self, season: int | None = None) -> pd.DataFrame:
        """Predict every scheduled (not yet played) fixture in a season."""
        frame = self.matches
        if frame.empty:
            return pd.DataFrame()

        upcoming = frame[~frame["is_completed"]]
        if season is not None:
            upcoming = upcoming[upcoming["season"] == season]
        if upcoming.empty:
            return pd.DataFrame()

        rows = []
        for match in upcoming.itertuples(index=False):
            if not isinstance(match.team1, str) or not isinstance(match.team2, str):
                continue
            try:
                prediction = self.predict_winner(
                    team1=match.team1,
                    team2=match.team2,
                    venue=match.venue,
                    season=int(match.season),
                    match_date=(
                        match.match_date.date()
                        if isinstance(match.match_date, pd.Timestamp) else None
                    ),
                    is_playoff=bool(match.is_playoff),
                    is_neutral_venue=bool(match.is_neutral_venue),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not score fixture %s: %s", match.match_key, exc)
                continue
            rows.append(
                {
                    "match_date": match.match_date,
                    "team1": match.team1,
                    "team2": match.team2,
                    "venue": match.venue,
                    "predicted_winner": prediction.predicted_winner,
                    "team1_win_pct": round(prediction.team1_win_probability * 100, 1),
                    "team2_win_pct": round(prediction.team2_win_probability * 100, 1),
                }
            )
        return pd.DataFrame(rows)

    def model_summary(self) -> pd.DataFrame:
        """One row per trained task: the winning model and its headline metric."""
        rows = []
        for task in ARTIFACTS:
            if not self.has_model(task):
                rows.append({"task": task, "status": "not trained"})
                continue
            try:
                bundle = self.artifact(task)
            except Exception:  # pragma: no cover - has_model already screened this
                rows.append({"task": task, "status": "not trained"})
                continue
            best = bundle["best_model"]
            metrics = next(
                (m for m in bundle.get("metrics", []) if m.get("model") == best), {}
            )
            rows.append(
                {
                    "task": task,
                    "status": "ready",
                    "best_model": best,
                    "trained_at": bundle.get("trained_at"),
                    "accuracy": metrics.get("accuracy"),
                    "roc_auc": metrics.get("roc_auc"),
                    "rmse": metrics.get("rmse"),
                    "test_rows": bundle.get("test_rows"),
                }
            )
        return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Return the process-wide prediction service."""
    return PredictionService()


def reset_prediction_service() -> None:
    """Drop the cached service so the next call reloads models and data.

    Called after a data refresh or a retrain from the Admin page.
    """
    get_prediction_service.cache_clear()
