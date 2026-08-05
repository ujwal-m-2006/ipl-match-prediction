"""In-play (second innings) features for the chase-success model.

Every legal ball of every completed run chase becomes one training row: the
match state at that instant, labelled with whether the chase ultimately
succeeded. That gives ~130k rows from ~1100 matches and lets the dashboard
answer "what are the chasing side's odds right now?" at any point in an innings.

Only the second innings is used -- a "chase" needs a target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import BALLS_PER_OVER, STANDARD_OVERS
from ..logging_utils import get_logger

logger = get_logger(__name__)

CHASE_NUMERIC_FEATURES: list[str] = [
    "target",
    "current_runs",
    "wickets_fallen",
    "wickets_in_hand",
    "balls_bowled",
    "balls_remaining",
    "runs_required",
    "current_run_rate",
    "required_run_rate",
    "run_rate_pressure",
    "over_no",
    "runs_last_5_overs",
    "wickets_last_5_overs",
    "chase_progress",
    "is_powerplay",
    "is_death",
]

CHASE_CATEGORICAL_FEATURES: list[str] = ["batting_team", "bowling_team", "venue"]

TOTAL_BALLS = STANDARD_OVERS * BALLS_PER_OVER
MAX_WICKETS = 10

# Sentinel used when no balls remain: an infinite required rate is not
# representable, so the rate is capped at a large finite value instead.
MAX_REQUIRED_RATE = 99.0


def _safe_rate(runs: float, balls: float) -> float:
    """Runs per over, guarding against division by zero."""
    return float(runs * BALLS_PER_OVER / balls) if balls > 0 else 0.0


def build_chase_features(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    stride: int = 1,
) -> pd.DataFrame:
    """Build one row per second-innings ball, labelled with the chase outcome.

    Args:
        deliveries: Output of :func:`ipl.db.repository.load_deliveries`.
        matches: Output of :func:`ipl.db.repository.load_matches`.
        stride: Emit every ``stride``-th ball. ``1`` uses every ball; a larger
            value thins the training set for faster experimentation.

    Returns:
        A DataFrame with :data:`CHASE_NUMERIC_FEATURES`,
        :data:`CHASE_CATEGORICAL_FEATURES` and a ``target_chase_success`` label.
    """
    if deliveries.empty or matches.empty:
        return pd.DataFrame()

    # Only completed, non-abandoned matches carry a usable label.
    playable = matches[
        matches["is_completed"]
        & matches["winner"].notna()
        & ~matches["is_no_result"].astype(bool)
    ]
    context = playable.set_index("match_id")[
        ["winner", "venue", "season", "match_date", "is_playoff"]
    ].to_dict("index")

    second = deliveries[deliveries["innings_no"] == 2].copy()
    if second.empty:
        return pd.DataFrame()

    # First-innings totals give each chase its target.
    first_totals = (
        deliveries[deliveries["innings_no"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .to_dict()
    )

    second = second.sort_values(["match_id", "ball_seq"], kind="stable")
    rows: list[dict] = []

    for match_id, chunk in second.groupby("match_id", sort=False):
        info = context.get(match_id)
        first_innings_runs = first_totals.get(match_id)
        if info is None or first_innings_runs is None:
            continue

        batting_team = chunk["batting_team"].iloc[0]
        bowling_team = chunk["bowling_team"].iloc[0]
        if not isinstance(batting_team, str):
            continue

        target = int(first_innings_runs) + 1
        chase_succeeded = int(info["winner"] == batting_team)

        # Vectorised running state for the whole innings.
        total_runs = chunk["total_runs"].to_numpy()
        is_legal = chunk["is_legal"].to_numpy().astype(bool)
        is_wicket = chunk["is_wicket"].to_numpy().astype(bool)
        over_numbers = chunk["over_no"].to_numpy()

        cumulative_runs = np.cumsum(total_runs)
        cumulative_wickets = np.cumsum(is_wicket)
        balls_bowled = np.cumsum(is_legal)

        for index in range(0, len(chunk), stride):
            # State *after* this ball, which is what a live scoreboard shows.
            runs = int(cumulative_runs[index])
            wickets = int(cumulative_wickets[index])
            balls = int(balls_bowled[index])
            over_no = int(over_numbers[index])

            if wickets >= MAX_WICKETS:
                break

            balls_remaining = max(TOTAL_BALLS - balls, 0)
            runs_required = max(target - runs, 0)

            # Stop once the chase is mathematically decided: rows after that
            # carry no predictive information, only label leakage.
            if runs_required == 0:
                break

            current_rate = _safe_rate(runs, balls)
            required_rate = (
                _safe_rate(runs_required, balls_remaining)
                if balls_remaining > 0
                else MAX_REQUIRED_RATE
            )

            window_start = max(0, index - 30)  # ~5 overs of deliveries
            runs_last_5 = int(cumulative_runs[index] - cumulative_runs[window_start]) if index else 0
            wickets_last_5 = (
                int(cumulative_wickets[index] - cumulative_wickets[window_start]) if index else 0
            )

            rows.append(
                {
                    "match_id": match_id,
                    "season": info["season"],
                    "match_date": info["match_date"],
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "venue": info["venue"] or "Unknown",
                    "target": target,
                    "current_runs": runs,
                    "wickets_fallen": wickets,
                    "wickets_in_hand": MAX_WICKETS - wickets,
                    "balls_bowled": balls,
                    "balls_remaining": balls_remaining,
                    "runs_required": runs_required,
                    "current_run_rate": round(current_rate, 3),
                    "required_run_rate": round(min(required_rate, MAX_REQUIRED_RATE), 3),
                    "run_rate_pressure": round(
                        min(required_rate, MAX_REQUIRED_RATE) - current_rate, 3
                    ),
                    "over_no": over_no,
                    "runs_last_5_overs": runs_last_5,
                    "wickets_last_5_overs": wickets_last_5,
                    "chase_progress": round(runs / target, 4) if target else 0.0,
                    "is_powerplay": int(over_no <= 6),
                    "is_death": int(over_no >= 16),
                    "target_chase_success": chase_succeeded,
                }
            )

    result = pd.DataFrame(rows)
    logger.info("Built chase features: %d ball-states from %d chases",
                len(result), result["match_id"].nunique() if not result.empty else 0)
    return result


def chase_feature_row(
    *,
    target: int,
    current_runs: int,
    wickets_fallen: int,
    balls_bowled: int,
    batting_team: str,
    bowling_team: str,
    venue: str,
    runs_last_5_overs: int = 0,
    wickets_last_5_overs: int = 0,
) -> pd.DataFrame:
    """Build a single chase-state row for live inference.

    Mirrors the training-row construction in :func:`build_chase_features` so
    the served features match what the model was fitted on.
    """
    balls_remaining = max(TOTAL_BALLS - balls_bowled, 0)
    runs_required = max(target - current_runs, 0)
    current_rate = _safe_rate(current_runs, balls_bowled)
    required_rate = (
        _safe_rate(runs_required, balls_remaining) if balls_remaining > 0 else MAX_REQUIRED_RATE
    )
    over_no = balls_bowled // BALLS_PER_OVER + 1

    return pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "target": target,
                "current_runs": current_runs,
                "wickets_fallen": wickets_fallen,
                "wickets_in_hand": MAX_WICKETS - wickets_fallen,
                "balls_bowled": balls_bowled,
                "balls_remaining": balls_remaining,
                "runs_required": runs_required,
                "current_run_rate": round(current_rate, 3),
                "required_run_rate": round(min(required_rate, MAX_REQUIRED_RATE), 3),
                "run_rate_pressure": round(
                    min(required_rate, MAX_REQUIRED_RATE) - current_rate, 3
                ),
                "over_no": int(over_no),
                "runs_last_5_overs": runs_last_5_overs,
                "wickets_last_5_overs": wickets_last_5_overs,
                "chase_progress": round(current_runs / target, 4) if target else 0.0,
                "is_powerplay": int(over_no <= 6),
                "is_death": int(over_no >= 16),
            }
        ]
    )


def build_player_of_match_features(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-player, per-match rows labelled with whether they won the award.

    Player of the Match is framed as a *ranking* problem rather than a
    multi-class one: a binary classifier scores every player who appeared, and
    the highest-scoring player in a match is the prediction. That keeps the
    label space fixed as the player pool changes from season to season.
    """
    if matches.empty or (batting.empty and bowling.empty):
        return pd.DataFrame()

    labelled = matches[matches["is_completed"] & matches["player_of_match"].notna()]
    if labelled.empty:
        return pd.DataFrame()

    awards = labelled.set_index("match_id")["player_of_match"].to_dict()
    winners = labelled.set_index("match_id")["winner"].to_dict()
    valid_ids = set(awards)

    bat = batting[batting["match_id"].isin(valid_ids)]
    bowl = bowling[bowling["match_id"].isin(valid_ids)]

    bat_agg = (
        bat.groupby(["match_id", "player", "team"], dropna=False)
        .agg(
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
        )
        .reset_index()
    )
    bowl_agg = (
        bowl.groupby(["match_id", "player", "team"], dropna=False)
        .agg(
            wickets=("wickets", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            balls_bowled=("balls", "sum"),
            maidens=("maidens", "sum"),
        )
        .reset_index()
    )

    frame = bat_agg.merge(bowl_agg, on=["match_id", "player", "team"], how="outer")
    if frame.empty:
        return pd.DataFrame()

    numeric = ["runs", "balls", "fours", "sixes", "wickets", "runs_conceded",
               "balls_bowled", "maidens"]
    frame[numeric] = frame[numeric].fillna(0)

    frame["strike_rate"] = np.where(
        frame["balls"] > 0, frame["runs"] * 100 / frame["balls"], 0.0
    )
    frame["economy"] = np.where(
        frame["balls_bowled"] > 0,
        frame["runs_conceded"] * BALLS_PER_OVER / frame["balls_bowled"],
        0.0,
    )
    frame["boundary_runs"] = frame["fours"] * 4 + frame["sixes"] * 6

    # A simple impact score: runs above a par strike rate plus wicket value
    # minus runs leaked. This is a strong single predictor and gives the tree
    # models an informative split point immediately.
    frame["batting_impact"] = frame["runs"] + (frame["strike_rate"] - 130) * frame["balls"] / 100
    frame["bowling_impact"] = frame["wickets"] * 20 - (frame["economy"] - 8) * (
        frame["balls_bowled"] / BALLS_PER_OVER
    )
    frame["total_impact"] = frame["batting_impact"] + frame["bowling_impact"]

    frame["team_won"] = frame.apply(
        lambda r: int(winners.get(r["match_id"]) == r["team"]), axis=1
    )
    frame["target_is_pom"] = frame.apply(
        lambda r: int(awards.get(r["match_id"]) == r["player"]), axis=1
    )

    # Rank features let the model compare players *within* a match, which is
    # what the award actually depends on.
    for column in ("total_impact", "runs", "wickets"):
        frame[f"{column}_rank"] = frame.groupby("match_id")[column].rank(
            ascending=False, method="min"
        )
        frame[f"{column}_share"] = frame[column] / frame.groupby("match_id")[column].transform(
            lambda s: s.abs().sum() or 1
        )

    logger.info(
        "Built Player-of-the-Match features: %d player-match rows across %d matches",
        len(frame), frame["match_id"].nunique(),
    )
    return frame


POM_NUMERIC_FEATURES: list[str] = [
    "runs", "balls", "fours", "sixes", "strike_rate", "boundary_runs",
    "wickets", "runs_conceded", "balls_bowled", "maidens", "economy",
    "batting_impact", "bowling_impact", "total_impact", "team_won",
    "total_impact_rank", "total_impact_share",
    "runs_rank", "runs_share", "wickets_rank", "wickets_share",
]
