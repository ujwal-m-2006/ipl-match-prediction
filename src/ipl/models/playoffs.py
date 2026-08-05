"""Playoff qualification probabilities via Monte Carlo simulation.

There is no closed form for "what are Chennai's chances of making the top
four?" -- it depends on every remaining fixture in the league, including games
they are not playing in. So the season is simulated many times:

1. Take the current standings from completed league matches.
2. For each remaining fixture, draw a winner using the trained winner model's
   probability (not a coin flip -- a strong side beating a weak one should be
   likely, not 50/50).
3. Award points, rebuild the table, and record who finished in the top four.
4. Repeat, then report the share of simulations in which each side qualified.

Net run rate is the IPL's tie-breaker. Simulating NRR credibly would require
simulating scores ball by ball, so ties on points are instead broken by each
team's *actual* NRR so far, which is a reasonable proxy and is stated plainly
in the dashboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..constants import PLAYOFF_SPOTS, POINTS_TIE_OR_NR, POINTS_WIN
from ..features.match_features import add_net_run_rate, compute_standings
from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SIMULATIONS = 5000


@dataclass
class PlayoffProjection:
    """Simulation output for one season."""

    season: int
    simulations: int
    matches_remaining: int
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        if self.table.empty:
            return f"Season {self.season}: nothing to simulate"
        leader = self.table.iloc[0]
        return (
            f"Season {self.season}: {self.matches_remaining} matches left, "
            f"{leader['team']} most likely to qualify "
            f"({leader['qualification_pct']:.1f}%)"
        )


def _remaining_fixtures(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    """League fixtures in a season that have not yet produced a result."""
    frame = matches[(matches["season"] == season)]
    frame = frame[~frame["is_playoff"].astype(bool)]
    return frame[~frame["is_completed"] | frame["winner"].isna()]


def _win_probability_matrix(
    service: Any, fixtures: pd.DataFrame, season: int
) -> list[tuple[str, str, float]]:
    """Pre-compute P(team1 wins) for each remaining fixture.

    Done once up front rather than inside the simulation loop -- the model call
    is far more expensive than the simulation arithmetic, and the probability
    does not change between draws.
    """
    rows: list[tuple[str, str, float]] = []
    for fixture in fixtures.itertuples(index=False):
        team1, team2 = fixture.team1, fixture.team2
        if not isinstance(team1, str) or not isinstance(team2, str):
            continue
        probability = 0.5
        if service is not None:
            try:
                prediction = service.predict_winner(
                    team1=team1,
                    team2=team2,
                    venue=fixture.venue,
                    season=season,
                    is_neutral_venue=bool(fixture.is_neutral_venue),
                )
                probability = prediction.team1_win_probability
            except Exception as exc:  # pragma: no cover
                logger.debug("Falling back to 50/50 for %s vs %s: %s", team1, team2, exc)
        rows.append((team1, team2, probability))
    return rows


def simulate_playoff_qualification(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    season: int,
    *,
    service: Any = None,
    simulations: int = DEFAULT_SIMULATIONS,
    random_state: int = 42,
    playoff_spots: int = PLAYOFF_SPOTS,
) -> PlayoffProjection:
    """Estimate each team's probability of finishing in the top four.

    Args:
        matches: Full match table.
        innings: Innings table, used for the current net run rate.
        season: Season to project.
        service: A :class:`~ipl.models.predict.PredictionService`. When
            ``None``, every remaining fixture is treated as a coin flip.
        simulations: Number of Monte Carlo draws.
        random_state: Seed, so the reported percentages are reproducible.
        playoff_spots: Qualifying positions (4 in the IPL).
    """
    standings = compute_standings(matches, season)
    if standings.empty:
        logger.warning("No completed league matches for season %s", season)
        return PlayoffProjection(season, 0, 0)

    standings = add_net_run_rate(standings, innings, matches, season)

    current_points = dict(zip(standings["team"], standings["points"]))
    current_nrr = dict(zip(standings["team"], standings.get("net_run_rate", pd.Series(dtype=float))))

    fixtures = _remaining_fixtures(matches, season)
    probabilities = _win_probability_matrix(service, fixtures, season)

    teams = sorted(set(current_points) | {t for pair in probabilities for t in pair[:2]})
    for team in teams:
        current_points.setdefault(team, 0.0)
        current_nrr.setdefault(team, 0.0)

    if not probabilities:
        # The league stage is over: qualification is already decided.
        logger.info("Season %s league stage complete; reporting final table", season)
        qualified = set(standings.head(playoff_spots)["team"])
        table = standings.copy()
        table["qualification_pct"] = [
            100.0 if team in qualified else 0.0 for team in table["team"]
        ]
        table["title_pct"] = np.nan
        return PlayoffProjection(season, 0, 0, table)

    rng = np.random.default_rng(random_state)
    team_index = {team: i for i, team in enumerate(teams)}
    base_points = np.array([current_points[t] for t in teams], dtype=float)
    # Tiny NRR-derived offset breaks points ties in the same order the real
    # table would, without letting NRR outweigh a win.
    nrr_offset = np.array([current_nrr.get(t, 0.0) or 0.0 for t in teams], dtype=float) * 1e-3

    fixture_indices = np.array(
        [(team_index[a], team_index[b]) for a, b, _ in probabilities], dtype=int
    )
    fixture_probs = np.array([p for _, _, p in probabilities], dtype=float)

    qualification_counts = np.zeros(len(teams), dtype=float)
    position_totals = np.zeros(len(teams), dtype=float)

    for _ in range(simulations):
        points = base_points.copy()
        # One vectorised draw per simulation decides every remaining fixture.
        team1_wins = rng.random(len(fixture_probs)) < fixture_probs
        np.add.at(
            points,
            np.where(team1_wins, fixture_indices[:, 0], fixture_indices[:, 1]),
            POINTS_WIN,
        )

        ranking_key = points + nrr_offset
        # argsort descending: highest points first.
        order = np.argsort(-ranking_key, kind="stable")
        qualification_counts[order[:playoff_spots]] += 1
        position_totals[order] += np.arange(1, len(teams) + 1)

    table = pd.DataFrame(
        {
            "team": teams,
            "current_points": [current_points[t] for t in teams],
            "net_run_rate": [round(current_nrr.get(t, 0.0) or 0.0, 3) for t in teams],
            "qualification_pct": (qualification_counts / simulations * 100).round(2),
            "expected_position": (position_totals / simulations).round(2),
        }
    )

    played = standings.set_index("team")["played"].to_dict()
    table["matches_played"] = table["team"].map(played).fillna(0).astype(int)
    table["matches_remaining"] = table["team"].map(
        _matches_remaining_per_team(probabilities)
    ).fillna(0).astype(int)
    table["max_possible_points"] = (
        table["current_points"] + table["matches_remaining"] * POINTS_WIN
    )

    table = table.sort_values(
        ["qualification_pct", "current_points"], ascending=False
    ).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    projection = PlayoffProjection(
        season=season,
        simulations=simulations,
        matches_remaining=len(probabilities),
        table=table,
    )
    logger.info("%s", projection.summary())
    return projection


def _matches_remaining_per_team(
    probabilities: list[tuple[str, str, float]]
) -> dict[str, int]:
    """Count each team's outstanding fixtures."""
    counts: dict[str, int] = defaultdict(int)
    for team1, team2, _ in probabilities:
        counts[team1] += 1
        counts[team2] += 1
    return dict(counts)


def what_if_qualification(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    season: int,
    *,
    service: Any = None,
    forced_results: dict[int, str] | None = None,
    simulations: int = DEFAULT_SIMULATIONS,
    random_state: int = 42,
) -> PlayoffProjection:
    """Re-run the projection with some remaining fixtures forced to a result.

    Args:
        forced_results: ``{match_id: winning_team}``. Those fixtures are treated
            as already decided; everything else is still simulated.
    """
    if not forced_results:
        return simulate_playoff_qualification(
            matches, innings, season, service=service,
            simulations=simulations, random_state=random_state,
        )

    adjusted = matches.copy()
    mask = adjusted["match_id"].isin(forced_results)
    adjusted.loc[mask, "winner"] = adjusted.loc[mask, "match_id"].map(forced_results)
    adjusted.loc[mask, "is_completed"] = True

    return simulate_playoff_qualification(
        adjusted, innings, season, service=service,
        simulations=simulations, random_state=random_state,
    )
