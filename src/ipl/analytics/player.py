"""Player analytics: career summaries, leaderboards and situational splits.

Cricket averages have conventions that a naive ``mean()`` gets wrong, and they
are applied consistently here:

* **Batting average** = runs / *dismissals* (not innings). A batter who is
  never out has an undefined average, reported as ``NaN`` rather than as their
  run total.
* **Strike rate** = runs per 100 balls faced.
* **Bowling average** = runs conceded / wickets; **economy** = runs per over;
  **bowling strike rate** = balls per wicket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import BALLS_PER_OVER
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Minimum volume before a rate statistic is meaningful on a leaderboard.
MIN_BALLS_FOR_STRIKE_RATE = 100
MIN_BALLS_BOWLED_FOR_ECONOMY = 120


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide, returning NaN instead of inf where the divisor is 0."""
    return np.where(denominator > 0, numerator / denominator.replace(0, np.nan), np.nan)


def batting_leaderboard(
    batting: pd.DataFrame,
    *,
    season: int | None = None,
    team: str | None = None,
    min_innings: int = 1,
) -> pd.DataFrame:
    """Aggregate batting statistics per player."""
    if batting.empty:
        return pd.DataFrame()

    frame = batting
    if season is not None:
        frame = frame[frame["season"] == season]
    if team is not None:
        frame = frame[frame["team"] == team]
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("player")
        .agg(
            innings=("match_id", "nunique"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
            dismissals=("is_out", "sum"),
            highest_score=("runs", "max"),
        )
        .reset_index()
    )

    summary["average"] = _safe_divide(summary["runs"], summary["dismissals"]).round(2)
    summary["strike_rate"] = (
        _safe_divide(summary["runs"] * 100, summary["balls"])
    ).round(2)
    summary["boundary_runs"] = summary["fours"] * 4 + summary["sixes"] * 6
    summary["boundary_pct"] = (
        _safe_divide(summary["boundary_runs"] * 100, summary["runs"])
    ).round(2)

    # Milestone counts.
    per_innings = frame.groupby(["player", "match_id"])["runs"].sum().reset_index()
    fifties = per_innings[(per_innings["runs"] >= 50) & (per_innings["runs"] < 100)]
    hundreds = per_innings[per_innings["runs"] >= 100]
    ducks = per_innings[per_innings["runs"] == 0]

    summary["fifties"] = summary["player"].map(fifties["player"].value_counts()).fillna(0).astype(int)
    summary["hundreds"] = summary["player"].map(hundreds["player"].value_counts()).fillna(0).astype(int)
    summary["ducks"] = summary["player"].map(ducks["player"].value_counts()).fillna(0).astype(int)

    summary = summary[summary["innings"] >= min_innings]
    return summary.sort_values("runs", ascending=False).reset_index(drop=True)


def bowling_leaderboard(
    bowling: pd.DataFrame,
    *,
    season: int | None = None,
    team: str | None = None,
    min_innings: int = 1,
) -> pd.DataFrame:
    """Aggregate bowling statistics per player."""
    if bowling.empty:
        return pd.DataFrame()

    frame = bowling
    if season is not None:
        frame = frame[frame["season"] == season]
    if team is not None:
        frame = frame[frame["team"] == team]
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("player")
        .agg(
            innings=("match_id", "nunique"),
            balls=("balls", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            wickets=("wickets", "sum"),
            maidens=("maidens", "sum"),
            best_wickets=("wickets", "max"),
            wides=("wides", "sum"),
            no_balls=("no_balls", "sum"),
        )
        .reset_index()
    )

    summary["overs"] = (summary["balls"] / BALLS_PER_OVER).round(1)
    summary["average"] = _safe_divide(summary["runs_conceded"], summary["wickets"]).round(2)
    summary["economy"] = (
        _safe_divide(summary["runs_conceded"] * BALLS_PER_OVER, summary["balls"])
    ).round(2)
    summary["strike_rate"] = _safe_divide(summary["balls"], summary["wickets"]).round(2)

    hauls = frame.groupby(["player", "match_id"])["wickets"].sum().reset_index()
    four_fers = hauls[hauls["wickets"] == 4]
    five_fers = hauls[hauls["wickets"] >= 5]
    summary["four_wicket_hauls"] = (
        summary["player"].map(four_fers["player"].value_counts()).fillna(0).astype(int)
    )
    summary["five_wicket_hauls"] = (
        summary["player"].map(five_fers["player"].value_counts()).fillna(0).astype(int)
    )

    summary = summary[summary["innings"] >= min_innings]
    return summary.sort_values("wickets", ascending=False).reset_index(drop=True)


def player_career_summary(
    player: str,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict:
    """Complete career profile for one player: batting, bowling and awards."""
    bat = batting[batting["player"] == player] if not batting.empty else pd.DataFrame()
    bowl = bowling[bowling["player"] == player] if not bowling.empty else pd.DataFrame()

    profile: dict = {
        "player": player,
        "teams": [],
        "seasons": [],
        "matches": 0,
        "player_of_match_awards": 0,
        "batting": {},
        "bowling": {},
    }

    match_ids: set = set()
    if not bat.empty:
        match_ids |= set(bat["match_id"])
        profile["teams"] = sorted(bat["team"].dropna().unique().tolist())
        profile["seasons"] = sorted(bat["season"].dropna().unique().tolist())
        stats = batting_leaderboard(bat)
        if not stats.empty:
            profile["batting"] = stats.iloc[0].to_dict()

    if not bowl.empty:
        match_ids |= set(bowl["match_id"])
        profile["teams"] = sorted(set(profile["teams"]) | set(bowl["team"].dropna().unique()))
        profile["seasons"] = sorted(set(profile["seasons"]) | set(bowl["season"].dropna().unique()))
        stats = bowling_leaderboard(bowl)
        if not stats.empty:
            profile["bowling"] = stats.iloc[0].to_dict()

    profile["matches"] = len(match_ids)
    if not matches.empty:
        profile["player_of_match_awards"] = int(
            (matches["player_of_match"] == player).sum()
        )

    return profile


def player_season_trend(
    player: str, batting: pd.DataFrame, bowling: pd.DataFrame
) -> pd.DataFrame:
    """Season-by-season batting and bowling output for one player."""
    rows: list[pd.DataFrame] = []

    if not batting.empty:
        bat = batting[batting["player"] == player]
        if not bat.empty:
            agg = (
                bat.groupby("season")
                .agg(
                    innings=("match_id", "nunique"),
                    runs=("runs", "sum"),
                    balls=("balls", "sum"),
                    dismissals=("is_out", "sum"),
                )
                .reset_index()
            )
            agg["batting_average"] = _safe_divide(agg["runs"], agg["dismissals"]).round(2)
            agg["strike_rate"] = _safe_divide(agg["runs"] * 100, agg["balls"]).round(2)
            rows.append(agg[["season", "innings", "runs", "batting_average", "strike_rate"]])

    if not bowling.empty:
        bowl = bowling[bowling["player"] == player]
        if not bowl.empty:
            agg = (
                bowl.groupby("season")
                .agg(
                    wickets=("wickets", "sum"),
                    runs_conceded=("runs_conceded", "sum"),
                    balls_bowled=("balls", "sum"),
                )
                .reset_index()
            )
            agg["economy"] = _safe_divide(
                agg["runs_conceded"] * BALLS_PER_OVER, agg["balls_bowled"]
            ).round(2)
            rows.append(agg[["season", "wickets", "economy"]])

    if not rows:
        return pd.DataFrame()

    result = rows[0]
    for extra in rows[1:]:
        result = result.merge(extra, on="season", how="outer")
    return result.sort_values("season").reset_index(drop=True)


def player_venue_split(
    player: str, batting: pd.DataFrame, *, min_innings: int = 2
) -> pd.DataFrame:
    """Batting output broken down by ground."""
    if batting.empty:
        return pd.DataFrame()
    frame = batting[batting["player"] == player]
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("venue")
        .agg(
            innings=("match_id", "nunique"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            dismissals=("is_out", "sum"),
            highest_score=("runs", "max"),
        )
        .reset_index()
    )
    summary["average"] = _safe_divide(summary["runs"], summary["dismissals"]).round(2)
    summary["strike_rate"] = _safe_divide(summary["runs"] * 100, summary["balls"]).round(2)
    summary = summary[summary["innings"] >= min_innings]
    return summary.sort_values("runs", ascending=False).reset_index(drop=True)


def player_vs_opposition(player: str, batting: pd.DataFrame) -> pd.DataFrame:
    """Batting output broken down by opposition franchise."""
    if batting.empty:
        return pd.DataFrame()
    frame = batting[batting["player"] == player]
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("opposition")
        .agg(
            innings=("match_id", "nunique"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            dismissals=("is_out", "sum"),
        )
        .reset_index()
    )
    summary["average"] = _safe_divide(summary["runs"], summary["dismissals"]).round(2)
    summary["strike_rate"] = _safe_divide(summary["runs"] * 100, summary["balls"]).round(2)
    return summary.sort_values("runs", ascending=False).reset_index(drop=True)


def compare_players(
    players: list[str],
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side comparison table for two or more players."""
    rows = []
    for player in players:
        profile = player_career_summary(player, batting, bowling, matches)
        bat = profile.get("batting") or {}
        bowl = profile.get("bowling") or {}
        rows.append(
            {
                "player": player,
                "matches": profile["matches"],
                "runs": bat.get("runs", 0),
                "batting_average": bat.get("average", np.nan),
                "strike_rate": bat.get("strike_rate", np.nan),
                "fifties": bat.get("fifties", 0),
                "hundreds": bat.get("hundreds", 0),
                "wickets": bowl.get("wickets", 0),
                "bowling_average": bowl.get("average", np.nan),
                "economy": bowl.get("economy", np.nan),
                "player_of_match_awards": profile["player_of_match_awards"],
            }
        )
    return pd.DataFrame(rows)
