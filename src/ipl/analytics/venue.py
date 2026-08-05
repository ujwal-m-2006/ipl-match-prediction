"""Venue analytics: scoring behaviour, chase success and phase profiles.

These metrics feed both the dashboard's Venue Statistics page and the venue
features used by the winner and score models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DEATH_OVERS_FROM, POWERPLAY_OVERS
from ..logging_utils import get_logger

logger = get_logger(__name__)


def venue_summary(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    *,
    min_matches: int = 1,
) -> pd.DataFrame:
    """Per-ground scoring and result profile.

    Columns: matches hosted, average and highest first-innings score, the share
    of matches won by the chasing side, and the average winning margin.
    """
    if matches.empty:
        return pd.DataFrame()

    played = matches[matches["is_completed"] & matches["winner"].notna()]
    if played.empty:
        return pd.DataFrame()

    summary = (
        played.groupby("venue")
        .agg(
            matches=("match_id", "count"),
            seasons=("season", "nunique"),
            city=("city", "first"),
        )
        .reset_index()
    )

    # Chasing side = whoever batted second.
    chase_frame = played[played["first_batting_team"].notna()].copy()
    chase_frame["chase_won"] = (
        chase_frame["winner"] != chase_frame["first_batting_team"]
    ).astype(int)
    chase = (
        chase_frame.groupby("venue")
        .agg(decided=("match_id", "count"), chase_wins=("chase_won", "sum"))
        .reset_index()
    )
    chase["chase_win_pct"] = (chase["chase_wins"] / chase["decided"] * 100).round(2)
    summary = summary.merge(
        chase[["venue", "chase_win_pct", "chase_wins", "decided"]], on="venue", how="left"
    )
    summary["bat_first_win_pct"] = (100 - summary["chase_win_pct"]).round(2)

    if not innings.empty:
        first = innings[innings["innings_no"] == 1]
        scoring = (
            first.groupby("venue")
            .agg(
                avg_first_innings=("runs", "mean"),
                highest_first_innings=("runs", "max"),
                lowest_first_innings=("runs", "min"),
                avg_run_rate=("run_rate", "mean"),
            )
            .round(2)
            .reset_index()
        )
        summary = summary.merge(scoring, on="venue", how="left")

        second = innings[innings["innings_no"] == 2]
        chase_scoring = (
            second.groupby("venue")["runs"].mean().round(2).rename("avg_second_innings")
        )
        summary = summary.merge(chase_scoring, on="venue", how="left")

    summary["avg_win_margin_runs"] = summary["venue"].map(
        played.groupby("venue")["win_margin_runs"].mean().round(1)
    )
    summary["avg_win_margin_wickets"] = summary["venue"].map(
        played.groupby("venue")["win_margin_wickets"].mean().round(1)
    )

    summary = summary[summary["matches"] >= min_matches]
    return summary.sort_values("matches", ascending=False).reset_index(drop=True)


def venue_team_performance(
    matches: pd.DataFrame, venue: str, *, min_matches: int = 1
) -> pd.DataFrame:
    """Every franchise's record at one ground."""
    if matches.empty:
        return pd.DataFrame()

    played = matches[
        (matches["venue"] == venue) & matches["is_completed"] & matches["winner"].notna()
    ]
    if played.empty:
        return pd.DataFrame()

    rows = []
    for column in ("team1", "team2"):
        rows.append(
            pd.DataFrame(
                {
                    "team": played[column],
                    "won": (played["winner"] == played[column]).astype(int),
                }
            )
        )
    long = pd.concat(rows, ignore_index=True).dropna(subset=["team"])

    summary = (
        long.groupby("team")
        .agg(matches=("won", "count"), wins=("won", "sum"))
        .reset_index()
    )
    summary["losses"] = summary["matches"] - summary["wins"]
    summary["win_pct"] = (summary["wins"] / summary["matches"] * 100).round(2)
    summary = summary[summary["matches"] >= min_matches]
    return summary.sort_values(["win_pct", "matches"], ascending=False).reset_index(drop=True)


def venue_phase_profile(deliveries: pd.DataFrame, venue: str | None = None) -> pd.DataFrame:
    """Average runs and wickets per phase (powerplay / middle / death).

    Requires ball-by-ball data; returns an empty frame when deliveries were not
    ingested.
    """
    if deliveries.empty:
        return pd.DataFrame()

    frame = deliveries
    if venue is not None:
        # `load_deliveries` does not carry the venue, so callers filter by the
        # match IDs played at the ground before calling this.
        logger.debug("venue_phase_profile called with venue=%s (pre-filter expected)", venue)

    phase = np.select(
        [frame["over_no"] <= POWERPLAY_OVERS, frame["over_no"] < DEATH_OVERS_FROM],
        ["Powerplay (1-6)", "Middle (7-15)"],
        default="Death (16-20)",
    )
    working = frame.assign(phase=phase)

    per_innings = (
        working.groupby(["match_id", "innings_no", "phase"])
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            balls=("is_legal", "sum"),
        )
        .reset_index()
    )

    summary = (
        per_innings.groupby("phase")
        .agg(
            avg_runs=("runs", "mean"),
            avg_wickets=("wickets", "mean"),
            avg_balls=("balls", "mean"),
            innings=("runs", "count"),
        )
        .round(2)
        .reset_index()
    )
    summary["run_rate"] = (summary["avg_runs"] * 6 / summary["avg_balls"]).round(2)

    order = {"Powerplay (1-6)": 0, "Middle (7-15)": 1, "Death (16-20)": 2}
    summary["_order"] = summary["phase"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def scoring_trend_by_season(innings: pd.DataFrame) -> pd.DataFrame:
    """League-wide scoring trend: how totals have moved season to season."""
    if innings.empty:
        return pd.DataFrame()

    first = innings[innings["innings_no"] == 1]
    summary = (
        first.groupby("season")
        .agg(
            innings=("runs", "count"),
            avg_score=("runs", "mean"),
            highest=("runs", "max"),
            avg_run_rate=("run_rate", "mean"),
            avg_sixes=("sixes", "mean"),
            avg_fours=("fours", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return summary.sort_values("season").reset_index(drop=True)


def venue_records(matches: pd.DataFrame, innings: pd.DataFrame, venue: str) -> dict:
    """Headline records for a single ground: highest, lowest, biggest wins."""
    played = matches[(matches["venue"] == venue) & matches["is_completed"]]
    if played.empty:
        return {}

    venue_innings = innings[innings["match_id"].isin(played["match_id"])] if not innings.empty else pd.DataFrame()

    records: dict = {
        "venue": venue,
        "matches": int(len(played)),
        "first_match": played["match_date"].min(),
        "last_match": played["match_date"].max(),
    }

    if not venue_innings.empty:
        highest = venue_innings.loc[venue_innings["runs"].idxmax()]
        lowest = venue_innings.loc[venue_innings["runs"].idxmin()]
        records["highest_total"] = (
            f"{int(highest['runs'])}/{int(highest['wickets'])} by {highest['batting_team']}"
        )
        records["lowest_total"] = (
            f"{int(lowest['runs'])}/{int(lowest['wickets'])} by {lowest['batting_team']}"
        )
        records["average_total"] = round(float(venue_innings["runs"].mean()), 1)

    biggest_runs = played["win_margin_runs"].max()
    if pd.notna(biggest_runs):
        row = played.loc[played["win_margin_runs"].idxmax()]
        records["biggest_win_runs"] = f"{row['winner']} by {int(biggest_runs)} runs"

    biggest_wickets = played["win_margin_wickets"].max()
    if pd.notna(biggest_wickets):
        row = played.loc[played["win_margin_wickets"].idxmax()]
        records["biggest_win_wickets"] = f"{row['winner']} by {int(biggest_wickets)} wickets"

    return records
