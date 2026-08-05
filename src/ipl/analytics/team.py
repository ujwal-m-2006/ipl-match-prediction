"""Team-level analytics: records, form, head-to-head and toss impact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _completed(matches: pd.DataFrame) -> pd.DataFrame:
    """Restrict to matches with a decided result."""
    if matches.empty:
        return matches
    return matches[matches["is_completed"] & matches["winner"].notna()]


def _long_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per match into two rows -- one per participating team.

    Nearly every team metric is easier to express over this shape than over the
    team1/team2 layout, so it is built once and reused.
    """
    frame = _completed(matches)
    if frame.empty:
        return pd.DataFrame()

    sides = []
    for own, opponent in (("team1", "team2"), ("team2", "team1")):
        side = pd.DataFrame(
            {
                "match_id": frame["match_id"],
                "season": frame["season"],
                "match_date": frame["match_date"],
                "venue": frame["venue"],
                "team": frame[own],
                "opponent": frame[opponent],
                "won": (frame["winner"] == frame[own]).astype(int),
                "is_home": (frame["home_team"] == frame[own]).astype(int),
                "is_playoff": frame["is_playoff"].astype(bool),
                "batted_first": (frame["first_batting_team"] == frame[own]).astype(int),
                "won_toss": (frame["toss_winner"] == frame[own]).astype(int),
                "toss_decision": frame["toss_decision"],
                "is_no_result": frame["is_no_result"].astype(bool),
            }
        )
        sides.append(side)

    combined = pd.concat(sides, ignore_index=True)
    return combined[combined["team"].notna()]


def team_summary(matches: pd.DataFrame, *, min_matches: int = 1) -> pd.DataFrame:
    """All-time record for every franchise.

    Returns columns: matches, wins, losses, win_pct, titles, home/away win
    rates and batting-first vs chasing win rates.
    """
    long = _long_form(matches)
    if long.empty:
        return pd.DataFrame()

    grouped = long.groupby("team")
    summary = grouped.agg(
        matches=("match_id", "count"),
        wins=("won", "sum"),
        home_matches=("is_home", "sum"),
    ).reset_index()
    summary["losses"] = summary["matches"] - summary["wins"]
    summary["win_pct"] = (summary["wins"] / summary["matches"] * 100).round(2)

    # Home / away split.
    home = long[long["is_home"] == 1].groupby("team")["won"].agg(["sum", "count"])
    away = long[long["is_home"] == 0].groupby("team")["won"].agg(["sum", "count"])
    summary = summary.merge(
        (home["sum"] / home["count"] * 100).round(2).rename("home_win_pct"),
        left_on="team", right_index=True, how="left",
    ).merge(
        (away["sum"] / away["count"] * 100).round(2).rename("away_win_pct"),
        left_on="team", right_index=True, how="left",
    )

    # Batting first vs chasing.
    first = long[long["batted_first"] == 1].groupby("team")["won"].agg(["sum", "count"])
    chasing = long[long["batted_first"] == 0].groupby("team")["won"].agg(["sum", "count"])
    summary = summary.merge(
        (first["sum"] / first["count"] * 100).round(2).rename("bat_first_win_pct"),
        left_on="team", right_index=True, how="left",
    ).merge(
        (chasing["sum"] / chasing["count"] * 100).round(2).rename("chasing_win_pct"),
        left_on="team", right_index=True, how="left",
    )

    summary["titles"] = summary["team"].map(_title_counts(matches)).fillna(0).astype(int)
    summary["seasons"] = summary["team"].map(
        long.groupby("team")["season"].nunique()
    ).fillna(0).astype(int)

    summary = summary[summary["matches"] >= min_matches]
    return summary.sort_values("win_pct", ascending=False).reset_index(drop=True)


def _title_counts(matches: pd.DataFrame) -> dict[str, int]:
    """Count championships: the winner of each season's Final."""
    frame = _completed(matches)
    if frame.empty:
        return {}
    finals = frame[frame["stage"].fillna("").str.lower() == "final"]
    return finals["winner"].value_counts().to_dict()


def team_season_summary(matches: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Per-season record, optionally filtered to one franchise."""
    long = _long_form(matches)
    if long.empty:
        return pd.DataFrame()
    if team:
        long = long[long["team"] == team]

    summary = (
        long.groupby(["team", "season"])
        .agg(matches=("match_id", "count"), wins=("won", "sum"))
        .reset_index()
    )
    summary["losses"] = summary["matches"] - summary["wins"]
    summary["win_pct"] = (summary["wins"] / summary["matches"] * 100).round(2)
    return summary.sort_values(["team", "season"]).reset_index(drop=True)


def team_form_timeline(matches: pd.DataFrame, team: str, window: int = 5) -> pd.DataFrame:
    """Rolling win-rate timeline for one franchise, most recent last."""
    long = _long_form(matches)
    if long.empty:
        return pd.DataFrame()

    frame = long[long["team"] == team].sort_values("match_date").reset_index(drop=True)
    if frame.empty:
        return frame

    frame["rolling_win_pct"] = (
        frame["won"].rolling(window=window, min_periods=1).mean() * 100
    ).round(2)
    frame["cumulative_win_pct"] = (
        frame["won"].expanding().mean() * 100
    ).round(2)
    frame["match_number"] = range(1, len(frame) + 1)
    return frame[
        ["match_number", "match_date", "season", "opponent", "venue", "won",
         "rolling_win_pct", "cumulative_win_pct"]
    ]


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Complete head-to-head record between two franchises.

    Returns a dict with overall counts, a venue breakdown, a season breakdown
    and the list of meetings, ready for direct display.
    """
    frame = _completed(matches)
    if frame.empty:
        return _empty_h2h(team_a, team_b)

    pair = frame[
        ((frame["team1"] == team_a) & (frame["team2"] == team_b))
        | ((frame["team1"] == team_b) & (frame["team2"] == team_a))
    ].sort_values("match_date")

    if pair.empty:
        return _empty_h2h(team_a, team_b)

    a_wins = int((pair["winner"] == team_a).sum())
    b_wins = int((pair["winner"] == team_b).sum())
    no_result = int(pair["is_no_result"].astype(bool).sum())

    venue_split = (
        pair.assign(a_won=(pair["winner"] == team_a).astype(int))
        .groupby("venue")
        .agg(matches=("match_id", "count"), a_wins=("a_won", "sum"))
        .reset_index()
    )
    venue_split["b_wins"] = venue_split["matches"] - venue_split["a_wins"]

    season_split = (
        pair.assign(a_won=(pair["winner"] == team_a).astype(int))
        .groupby("season")
        .agg(matches=("match_id", "count"), a_wins=("a_won", "sum"))
        .reset_index()
    )
    season_split["b_wins"] = season_split["matches"] - season_split["a_wins"]

    # Longest current streak, read backwards from the most recent meeting.
    streak_team, streak_len = _current_streak(pair)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "matches": int(len(pair)),
        "team_a_wins": a_wins,
        "team_b_wins": b_wins,
        "no_result": no_result,
        "team_a_win_pct": round(a_wins / len(pair) * 100, 2) if len(pair) else 0.0,
        "team_b_win_pct": round(b_wins / len(pair) * 100, 2) if len(pair) else 0.0,
        "current_streak_team": streak_team,
        "current_streak": streak_len,
        "highest_total": _highest_total(pair),
        "by_venue": venue_split.sort_values("matches", ascending=False),
        "by_season": season_split,
        "matches_list": pair[
            ["match_date", "season", "venue", "team1", "team2", "winner",
             "result_summary", "player_of_match"]
        ].sort_values("match_date", ascending=False),
    }


def _empty_h2h(team_a: str, team_b: str) -> dict:
    """Shape-compatible empty result so callers need no special-casing."""
    return {
        "team_a": team_a, "team_b": team_b, "matches": 0,
        "team_a_wins": 0, "team_b_wins": 0, "no_result": 0,
        "team_a_win_pct": 0.0, "team_b_win_pct": 0.0,
        "current_streak_team": None, "current_streak": 0, "highest_total": None,
        "by_venue": pd.DataFrame(), "by_season": pd.DataFrame(),
        "matches_list": pd.DataFrame(),
    }


def _current_streak(pair: pd.DataFrame) -> tuple[str | None, int]:
    """Return the team on the current winning streak and its length."""
    winners = pair.sort_values("match_date")["winner"].dropna().tolist()
    if not winners:
        return None, 0
    latest = winners[-1]
    streak = 0
    for winner in reversed(winners):
        if winner != latest:
            break
        streak += 1
    return latest, streak


def _highest_total(pair: pd.DataFrame) -> str | None:
    """Best-effort headline for the highest score in the fixture's history."""
    summaries = pair["result_summary"].dropna()
    return summaries.iloc[-1] if not summaries.empty else None


def toss_impact(matches: pd.DataFrame, *, by: str | None = None) -> pd.DataFrame:
    """Quantify how much winning the toss is worth.

    Args:
        by: Optionally group by ``"season"``, ``"venue"`` or ``"toss_decision"``.
    """
    frame = _completed(matches)
    frame = frame[frame["toss_winner"].notna()]
    if frame.empty:
        return pd.DataFrame()

    frame = frame.assign(
        toss_winner_won=(frame["toss_winner"] == frame["winner"]).astype(int)
    )

    if by is None:
        total = len(frame)
        wins = int(frame["toss_winner_won"].sum())
        return pd.DataFrame(
            [
                {
                    "group": "All matches",
                    "matches": total,
                    "toss_winner_wins": wins,
                    # Must match the grouped branch's column name below -- callers
                    # read the same key from either shape.
                    "toss_winner_advantage_pct": round(wins / total * 100, 2),
                }
            ]
        )

    grouped = (
        frame.groupby(by)
        .agg(matches=("match_id", "count"), toss_winner_wins=("toss_winner_won", "sum"))
        .reset_index()
        .rename(columns={by: "group"})
    )
    grouped["toss_winner_advantage_pct"] = (
        grouped["toss_winner_wins"] / grouped["matches"] * 100
    ).round(2)
    return grouped.sort_values("matches", ascending=False).reset_index(drop=True)


def batting_first_advantage(matches: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """Compare batting-first and chasing outcomes by season."""
    frame = _completed(matches)
    frame = frame[frame["first_batting_team"].notna()]
    if frame.empty:
        return pd.DataFrame()

    frame = frame.assign(
        first_batting_won=(frame["winner"] == frame["first_batting_team"]).astype(int)
    )
    summary = (
        frame.groupby("season")
        .agg(matches=("match_id", "count"), bat_first_wins=("first_batting_won", "sum"))
        .reset_index()
    )
    summary["chase_wins"] = summary["matches"] - summary["bat_first_wins"]
    summary["bat_first_win_pct"] = (
        summary["bat_first_wins"] / summary["matches"] * 100
    ).round(2)
    summary["chase_win_pct"] = (100 - summary["bat_first_win_pct"]).round(2)

    if not innings.empty:
        first_scores = (
            innings[innings["innings_no"] == 1]
            .groupby("season")["runs"]
            .mean()
            .round(1)
            .rename("avg_first_innings_score")
        )
        summary = summary.merge(first_scores, left_on="season", right_index=True, how="left")

    return summary.sort_values("season").reset_index(drop=True)
