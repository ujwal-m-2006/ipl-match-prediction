"""Pre-match feature engineering for the winner and first-innings-score models.

Leakage policy
--------------
Every feature is computed from matches that finished **strictly before** the
match being described. The builder walks fixtures in chronological order and
maintains rolling state, emitting the feature row *before* folding the match's
own result into that state. This is what makes the time-based train/test split
in :mod:`ipl.features.preprocessing` an honest estimate of future performance.

The same :class:`FeatureState` is reused at inference time, so a prediction for
an unplayed fixture is built from exactly the same code path as the training
rows -- there is no separate, drift-prone "serving" implementation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..constants import POINTS_TIE_OR_NR, POINTS_WIN, STANDARD_OVERS
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Rolling windows (in matches) used for "recent form" features.
SHORT_FORM_WINDOW = 5
LONG_FORM_WINDOW = 10

# Neutral priors used before a team/venue has any history. Using 0.5 (rather
# than 0) keeps early-season rows from looking like guaranteed losses.
PRIOR_WIN_RATE = 0.5
PRIOR_FIRST_INNINGS_SCORE = 165.0
PRIOR_CHASE_WIN_RATE = 0.5

# Matches of history required before a rate is trusted; below this the prior is
# blended in (a simple empirical-Bayes shrinkage).
SHRINKAGE_STRENGTH = 5.0

# Cap on days-since-last-match, so the off-season gap does not dominate.
MAX_REST_DAYS = 30

# Rolling window (in innings) for the "current scoring era" features. T20 totals
# have risen sharply over 19 seasons, and a tree model cannot extrapolate a raw
# `season` value past its training range -- so the era is encoded as a level
# that carries forward naturally instead.
ERA_WINDOW = 40
VENUE_RECENT_WINDOW = 10

# Priors for a player with no history yet.
PRIOR_BATTING_AVERAGE = 22.0
PRIOR_STRIKE_RATE = 128.0
PRIOR_BOWLING_ECONOMY = 8.2
PRIOR_BOWLING_AVERAGE = 28.0

# Minimum volume before a player's own numbers outweigh the prior.
MIN_BALLS_FOR_BATTING = 60
MIN_BALLS_FOR_BOWLING = 60

MATCH_NUMERIC_FEATURES: list[str] = [
    "season",
    "team1_is_home",
    "is_neutral_venue",
    "is_playoff",
    "toss_winner_is_team1",
    "toss_decision_is_bat",
    "team1_career_win_rate",
    "team2_career_win_rate",
    "team1_form_short",
    "team2_form_short",
    "team1_form_long",
    "team2_form_long",
    "form_short_diff",
    "form_long_diff",
    "career_win_rate_diff",
    "team1_h2h_win_rate",
    "h2h_matches",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_win_rate_diff",
    "team1_avg_runs_scored",
    "team2_avg_runs_scored",
    "team1_avg_runs_conceded",
    "team2_avg_runs_conceded",
    "team1_net_run_strength",
    "team2_net_run_strength",
    "net_run_strength_diff",
    "team1_rest_days",
    "team2_rest_days",
    "rest_days_diff",
    "team1_season_points",
    "team2_season_points",
    "season_points_diff",
    "team1_season_matches",
    "team2_season_matches",
    "venue_avg_first_innings",
    "venue_chase_win_rate",
    "team1_win_streak",
    "team2_win_streak",
    # Playing XI strength, computed from each selected player's career record
    # up to (but not including) this match. Squad quality is the strongest
    # known pre-match signal in T20 and is what team-level form alone misses.
    "team1_xi_batting_average",
    "team2_xi_batting_average",
    "team1_xi_strike_rate",
    "team2_xi_strike_rate",
    "team1_xi_bowling_economy",
    "team2_xi_bowling_economy",
    "team1_xi_experience",
    "team2_xi_experience",
    "xi_batting_average_diff",
    "xi_strike_rate_diff",
    "xi_bowling_economy_diff",
    "xi_experience_diff",
    "has_xi_data",
]

MATCH_CATEGORICAL_FEATURES: list[str] = ["team1", "team2", "venue", "toss_decision"]

# Feature columns for the first-innings-score regression, which is framed from
# the batting side's perspective rather than team1/team2.
SCORE_NUMERIC_FEATURES: list[str] = [
    "season",
    "is_neutral_venue",
    "is_playoff",
    "batting_is_home",
    "batting_won_toss",
    "batting_career_win_rate",
    "batting_form_short",
    "bowling_form_short",
    "batting_avg_runs_scored",
    "bowling_avg_runs_conceded",
    "venue_avg_first_innings",
    "batting_venue_avg_score",
    "batting_rest_days",
    # Era-aware levels. Without these the model is anchored to the low-scoring
    # 2008-2014 era it saw most of and systematically under-predicts modern
    # totals -- the failure mode that produced a negative R² on held-out seasons.
    "league_recent_avg_score",
    "venue_recent_avg_score",
    "batting_xi_batting_average",
    "batting_xi_strike_rate",
    "bowling_xi_bowling_economy",
]

SCORE_CATEGORICAL_FEATURES: list[str] = ["batting_team", "bowling_team", "venue"]


def _shrink(successes: float, trials: float, prior: float) -> float:
    """Blend an observed rate towards a prior when the sample is small.

    With ``trials = 0`` this returns the prior; as ``trials`` grows it converges
    on the observed rate. Prevents a team that has played one match at a venue
    from showing a 100% record there.
    """
    return (successes + prior * SHRINKAGE_STRENGTH) / (trials + SHRINKAGE_STRENGTH)


@dataclass
class _TeamHistory:
    """Rolling record for a single franchise."""

    played: int = 0
    won: int = 0
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=LONG_FORM_WINDOW))
    runs_scored: deque[float] = field(default_factory=lambda: deque(maxlen=LONG_FORM_WINDOW))
    runs_conceded: deque[float] = field(default_factory=lambda: deque(maxlen=LONG_FORM_WINDOW))
    last_match_date: date | None = None
    win_streak: int = 0

    def win_rate(self) -> float:
        return _shrink(self.won, self.played, PRIOR_WIN_RATE)

    def form(self, window: int) -> float:
        recent = list(self.recent)[-window:]
        if not recent:
            return PRIOR_WIN_RATE
        return float(np.mean(recent))

    def avg_scored(self) -> float:
        return float(np.mean(self.runs_scored)) if self.runs_scored else PRIOR_FIRST_INNINGS_SCORE

    def avg_conceded(self) -> float:
        return (
            float(np.mean(self.runs_conceded))
            if self.runs_conceded
            else PRIOR_FIRST_INNINGS_SCORE
        )


@dataclass
class _PlayerHistory:
    """Career-to-date totals for one player, used for squad-strength features."""

    matches: int = 0
    runs: int = 0
    balls_faced: int = 0
    dismissals: int = 0
    runs_conceded: int = 0
    balls_bowled: int = 0
    wickets: int = 0

    def batting_average(self) -> float | None:
        """Runs per dismissal, or ``None`` below a usable sample."""
        if self.balls_faced < MIN_BALLS_FOR_BATTING:
            return None
        # A batter yet to be dismissed has an undefined average; charge them one
        # notional dismissal so the number stays finite and comparable.
        return self.runs / max(self.dismissals, 1)

    def strike_rate(self) -> float | None:
        if self.balls_faced < MIN_BALLS_FOR_BATTING:
            return None
        return self.runs * 100 / self.balls_faced

    def economy(self) -> float | None:
        if self.balls_bowled < MIN_BALLS_FOR_BOWLING:
            return None
        return self.runs_conceded * 6 / self.balls_bowled

    def bowling_average(self) -> float | None:
        if self.balls_bowled < MIN_BALLS_FOR_BOWLING or self.wickets == 0:
            return None
        return self.runs_conceded / self.wickets


@dataclass
class FeatureState:
    """All rolling state needed to describe the next fixture.

    Shared between training (fed match by match) and inference (fed the full
    history once, then queried for an unplayed fixture).
    """

    teams: dict[str, _TeamHistory] = field(default_factory=lambda: defaultdict(_TeamHistory))
    players: dict[str, _PlayerHistory] = field(
        default_factory=lambda: defaultdict(_PlayerHistory)
    )
    # League-wide and per-venue recent scoring levels, which track the era.
    recent_league_scores: deque[float] = field(
        default_factory=lambda: deque(maxlen=ERA_WINDOW)
    )
    venue_recent_scores: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=VENUE_RECENT_WINDOW))
    )
    # match_id -> {team: [player names]} for the Playing XI.
    lineups: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    # Each team's most recent known XI, used to featurise an unplayed fixture.
    last_lineup: dict[str, list[str]] = field(default_factory=dict)
    # (team_a, team_b) with team_a < team_b -> [team_a wins, matches played]
    head_to_head: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    # (venue, team) -> [wins, played]
    venue_team: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    venue_first_innings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    # venue -> [successful chases, completed matches with two innings]
    venue_chase: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # (season, team) -> [points, matches]
    season_points: dict[tuple[int, str], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0])
    )
    # (venue, team) -> list of innings totals scored there
    venue_team_scores: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # -- reads ---------------------------------------------------------
    def h2h_key(self, a: str, b: str) -> tuple[tuple[str, str], bool]:
        """Return the canonical (sorted) H2H key and whether ``a`` is first."""
        return ((a, b), True) if a <= b else ((b, a), False)

    def rest_days(self, team: str, match_date: date | None) -> float:
        """Days since the team's previous fixture, capped at :data:`MAX_REST_DAYS`.

        A missing date (``None`` or ``NaT``, as when scoring a hypothetical
        fixture) yields the cap rather than raising.
        """
        history = self.teams[team]
        if match_date is None or history.last_match_date is None or pd.isna(match_date):
            return float(MAX_REST_DAYS)
        delta = (match_date - history.last_match_date).days
        return float(min(max(delta, 0), MAX_REST_DAYS))

    def venue_avg_first_innings(self, venue: str | None) -> float:
        scores = self.venue_first_innings.get(venue or "", [])
        return float(np.mean(scores)) if scores else PRIOR_FIRST_INNINGS_SCORE

    def venue_chase_rate(self, venue: str | None) -> float:
        won, played = self.venue_chase.get(venue or "", [0, 0])
        return _shrink(won, played, PRIOR_CHASE_WIN_RATE)

    def venue_team_rate(self, venue: str | None, team: str) -> float:
        won, played = self.venue_team.get((venue or "", team), [0, 0])
        return _shrink(won, played, PRIOR_WIN_RATE)

    def venue_team_score(self, venue: str | None, team: str) -> float:
        scores = self.venue_team_scores.get((venue or "", team), [])
        return float(np.mean(scores)) if scores else self.venue_avg_first_innings(venue)

    def league_recent_score(self) -> float:
        """Mean of the most recent first-innings totals, league-wide.

        This is the "current scoring era" level. Unlike a raw season number it
        carries forward sensibly into unseen seasons, which is what lets the
        score model track the long-term rise in T20 totals.
        """
        if not self.recent_league_scores:
            return PRIOR_FIRST_INNINGS_SCORE
        return float(np.mean(self.recent_league_scores))

    def venue_recent_score(self, venue: str | None) -> float:
        """Mean of the last few first-innings totals at this ground."""
        scores = self.venue_recent_scores.get(venue or "")
        if not scores:
            return self.league_recent_score()
        return float(np.mean(scores))

    def squad_strength(self, players: list[str] | None) -> dict[str, float | None]:
        """Aggregate a Playing XI's career-to-date quality.

        Returns ``None`` values when the lineup is unknown, so the imputer -- not
        a fabricated default -- decides what to substitute.
        """
        if not players:
            return {
                "batting_average": None,
                "strike_rate": None,
                "bowling_economy": None,
                "experience": None,
            }

        histories = [self.players[name] for name in players if name in self.players]
        if not histories:
            return {
                "batting_average": None,
                "strike_rate": None,
                "bowling_economy": None,
                "experience": None,
            }

        averages = [h.batting_average() for h in histories]
        strike_rates = [h.strike_rate() for h in histories]
        economies = [h.economy() for h in histories]

        # Batting strength is driven by the top order, so the strongest seven
        # averages represent the side better than a mean over all eleven
        # (which tail-enders would drag down).
        top_averages = sorted((a for a in averages if a is not None), reverse=True)[:7]
        top_strike_rates = sorted((s for s in strike_rates if s is not None), reverse=True)[:7]
        # Bowling is the opposite: the five cheapest bowlers do the work.
        best_economies = sorted(e for e in economies if e is not None)[:5]

        return {
            "batting_average": (
                float(np.mean(top_averages)) if top_averages else PRIOR_BATTING_AVERAGE
            ),
            "strike_rate": (
                float(np.mean(top_strike_rates)) if top_strike_rates else PRIOR_STRIKE_RATE
            ),
            "bowling_economy": (
                float(np.mean(best_economies)) if best_economies else PRIOR_BOWLING_ECONOMY
            ),
            "experience": float(np.mean([h.matches for h in histories])),
        }

    def lineup_for(self, match_id: int | None, team: str) -> list[str] | None:
        """Playing XI for a team in a match, falling back to their last known XI."""
        if match_id is not None:
            lineup = self.lineups.get(int(match_id), {}).get(team)
            if lineup:
                return lineup
        return self.last_lineup.get(team)

    def update_players(
        self, batting_rows: list[dict], bowling_rows: list[dict]
    ) -> None:
        """Fold one match's scorecards into the career player histories."""
        seen: set[str] = set()

        for row in batting_rows:
            name = row.get("player")
            if not name:
                continue
            history = self.players[name]
            history.runs += int(row.get("runs") or 0)
            history.balls_faced += int(row.get("balls") or 0)
            history.dismissals += int(bool(row.get("is_out")))
            seen.add(name)

        for row in bowling_rows:
            name = row.get("player")
            if not name:
                continue
            history = self.players[name]
            history.runs_conceded += int(row.get("runs_conceded") or 0)
            history.balls_bowled += int(row.get("balls") or 0)
            history.wickets += int(row.get("wickets") or 0)
            seen.add(name)

        for name in seen:
            self.players[name].matches += 1

    # -- writes --------------------------------------------------------
    def update(self, row: pd.Series, first_innings: float | None, second_innings: float | None) -> None:
        """Fold a completed match into the rolling state.

        No-result matches update scheduling state (rest days) and season points
        but not win/loss form, since neither side actually won.
        """
        team1, team2 = row["team1"], row["team2"]
        venue = row.get("venue") or ""
        season = int(row["season"])
        match_date = row.get("match_date")
        if isinstance(match_date, pd.Timestamp):
            match_date = match_date.date()

        for team in (team1, team2):
            if isinstance(team, str):
                self.teams[team].last_match_date = match_date

        no_result = bool(row.get("is_no_result")) or not isinstance(row.get("winner"), str)

        # --- season points table (drives the playoff simulator) ---
        if not no_result:
            winner = row["winner"]
            loser = team2 if winner == team1 else team1
            self.season_points[(season, winner)][0] += POINTS_WIN
            self.season_points[(season, winner)][1] += 1
            self.season_points[(season, loser)][1] += 1
        else:
            for team in (team1, team2):
                if isinstance(team, str):
                    self.season_points[(season, team)][0] += POINTS_TIE_OR_NR
                    self.season_points[(season, team)][1] += 1

        # --- run-scoring strength ---
        batting_first = row.get("first_batting_team")
        if first_innings is not None and isinstance(batting_first, str):
            chasing = team2 if batting_first == team1 else team1
            self.teams[batting_first].runs_scored.append(first_innings)
            self.venue_first_innings[venue].append(first_innings)
            self.venue_team_scores[(venue, batting_first)].append(first_innings)
            # Era trackers: bounded windows, so they follow the current level
            # rather than being anchored by 19 seasons of history.
            self.recent_league_scores.append(first_innings)
            self.venue_recent_scores[venue].append(first_innings)
            if isinstance(chasing, str):
                self.teams[chasing].runs_conceded.append(first_innings)
        if second_innings is not None and isinstance(batting_first, str):
            chasing = team2 if batting_first == team1 else team1
            if isinstance(chasing, str):
                self.teams[chasing].runs_scored.append(second_innings)
                self.venue_team_scores[(venue, chasing)].append(second_innings)
            self.teams[batting_first].runs_conceded.append(second_innings)

        if no_result:
            return

        winner = row["winner"]

        # --- win/loss form ---
        for team in (team1, team2):
            if not isinstance(team, str):
                continue
            won = int(team == winner)
            history = self.teams[team]
            history.played += 1
            history.won += won
            history.recent.append(won)
            history.win_streak = history.win_streak + 1 if won else 0

        # --- head to head ---
        if isinstance(team1, str) and isinstance(team2, str):
            key, first_is_team1 = self.h2h_key(team1, team2)
            entry = self.head_to_head[key]
            entry[1] += 1
            first_team = key[0]
            if winner == first_team:
                entry[0] += 1

        # --- venue record ---
        for team in (team1, team2):
            if not isinstance(team, str):
                continue
            entry = self.venue_team[(venue, team)]
            entry[1] += 1
            if team == winner:
                entry[0] += 1

        # --- chase outcome at this venue ---
        if isinstance(batting_first, str) and second_innings is not None:
            chasing = team2 if batting_first == team1 else team1
            entry = self.venue_chase[venue]
            entry[1] += 1
            if winner == chasing:
                entry[0] += 1


def _fixture_features(state: FeatureState, row: pd.Series) -> dict[str, Any]:
    """Build the feature dict for one fixture from the current state."""
    team1, team2 = row["team1"], row["team2"]
    venue = row.get("venue")
    match_date = row.get("match_date")
    if isinstance(match_date, pd.Timestamp):
        match_date = match_date.date()
    season = int(row["season"])

    h1, h2 = state.teams[team1], state.teams[team2]

    key, _ = state.h2h_key(team1, team2)
    h2h_wins, h2h_played = state.head_to_head.get(key, [0, 0])
    # `h2h_wins` counts wins for key[0]; flip when team1 sorts second.
    team1_h2h_wins = h2h_wins if key[0] == team1 else h2h_played - h2h_wins
    team1_h2h_rate = _shrink(team1_h2h_wins, h2h_played, PRIOR_WIN_RATE)

    team1_venue = state.venue_team_rate(venue, team1)
    team2_venue = state.venue_team_rate(venue, team2)

    team1_rest = state.rest_days(team1, match_date)
    team2_rest = state.rest_days(team2, match_date)

    team1_points, team1_season_matches = state.season_points.get((season, team1), [0.0, 0])
    team2_points, team2_season_matches = state.season_points.get((season, team2), [0.0, 0])

    toss_winner = row.get("toss_winner")
    toss_decision = row.get("toss_decision")

    team1_strength = h1.avg_scored() - h1.avg_conceded()
    team2_strength = h2.avg_scored() - h2.avg_conceded()

    match_id = row.get("match_id")
    xi1 = state.squad_strength(state.lineup_for(match_id, team1))
    xi2 = state.squad_strength(state.lineup_for(match_id, team2))
    has_xi = int(xi1["batting_average"] is not None and xi2["batting_average"] is not None)

    def diff(key: str) -> float | None:
        left, right = xi1[key], xi2[key]
        return None if left is None or right is None else left - right

    return {
        "season": season,
        "team1": team1,
        "team2": team2,
        "venue": venue or "Unknown",
        "toss_decision": toss_decision or "unknown",
        "team1_is_home": int(row.get("home_team") == team1),
        "is_neutral_venue": int(bool(row.get("is_neutral_venue"))),
        "is_playoff": int(bool(row.get("is_playoff"))),
        "toss_winner_is_team1": int(toss_winner == team1) if isinstance(toss_winner, str) else 0,
        "toss_decision_is_bat": int(toss_decision == "bat"),
        "team1_career_win_rate": h1.win_rate(),
        "team2_career_win_rate": h2.win_rate(),
        "team1_form_short": h1.form(SHORT_FORM_WINDOW),
        "team2_form_short": h2.form(SHORT_FORM_WINDOW),
        "team1_form_long": h1.form(LONG_FORM_WINDOW),
        "team2_form_long": h2.form(LONG_FORM_WINDOW),
        "form_short_diff": h1.form(SHORT_FORM_WINDOW) - h2.form(SHORT_FORM_WINDOW),
        "form_long_diff": h1.form(LONG_FORM_WINDOW) - h2.form(LONG_FORM_WINDOW),
        "career_win_rate_diff": h1.win_rate() - h2.win_rate(),
        "team1_h2h_win_rate": team1_h2h_rate,
        "h2h_matches": float(h2h_played),
        "team1_venue_win_rate": team1_venue,
        "team2_venue_win_rate": team2_venue,
        "venue_win_rate_diff": team1_venue - team2_venue,
        "team1_avg_runs_scored": h1.avg_scored(),
        "team2_avg_runs_scored": h2.avg_scored(),
        "team1_avg_runs_conceded": h1.avg_conceded(),
        "team2_avg_runs_conceded": h2.avg_conceded(),
        "team1_net_run_strength": team1_strength,
        "team2_net_run_strength": team2_strength,
        "net_run_strength_diff": team1_strength - team2_strength,
        "team1_rest_days": team1_rest,
        "team2_rest_days": team2_rest,
        "rest_days_diff": team1_rest - team2_rest,
        "team1_season_points": float(team1_points),
        "team2_season_points": float(team2_points),
        "season_points_diff": float(team1_points - team2_points),
        "team1_season_matches": float(team1_season_matches),
        "team2_season_matches": float(team2_season_matches),
        "venue_avg_first_innings": state.venue_avg_first_innings(venue),
        "venue_chase_win_rate": state.venue_chase_rate(venue),
        "team1_win_streak": float(h1.win_streak),
        "team2_win_streak": float(h2.win_streak),
        "team1_xi_batting_average": xi1["batting_average"],
        "team2_xi_batting_average": xi2["batting_average"],
        "team1_xi_strike_rate": xi1["strike_rate"],
        "team2_xi_strike_rate": xi2["strike_rate"],
        "team1_xi_bowling_economy": xi1["bowling_economy"],
        "team2_xi_bowling_economy": xi2["bowling_economy"],
        "team1_xi_experience": xi1["experience"],
        "team2_xi_experience": xi2["experience"],
        "xi_batting_average_diff": diff("batting_average"),
        "xi_strike_rate_diff": diff("strike_rate"),
        "xi_bowling_economy_diff": diff("bowling_economy"),
        "xi_experience_diff": diff("experience"),
        "has_xi_data": has_xi,
        # Era levels, carried onto the row so the score model can reuse them.
        "league_recent_avg_score": state.league_recent_score(),
        "venue_recent_avg_score": state.venue_recent_score(venue),
    }


def _innings_totals(innings: pd.DataFrame) -> dict[int, dict[int, float]]:
    """Index innings runs as ``{match_id: {innings_no: runs}}``."""
    lookup: dict[int, dict[int, float]] = defaultdict(dict)
    if innings.empty:
        return lookup
    for match_id, innings_no, runs in innings[["match_id", "innings_no", "runs"]].itertuples(
        index=False
    ):
        if pd.notna(runs):
            lookup[int(match_id)][int(innings_no)] = float(runs)
    return lookup


def _index_lineups(match_players: pd.DataFrame | None) -> dict[int, dict[str, list[str]]]:
    """Index Playing XIs as ``{match_id: {team: [player, ...]}}``."""
    lineups: dict[int, dict[str, list[str]]] = {}
    if match_players is None or match_players.empty:
        return lineups

    xi = match_players[match_players["is_playing_xi"]]
    for (match_id, team), group in xi.groupby(["match_id", "team"], sort=False):
        lineups.setdefault(int(match_id), {})[team] = group["player"].dropna().tolist()
    return lineups


def _index_cards(frame: pd.DataFrame | None, columns: list[str]) -> dict[int, list[dict]]:
    """Index scorecard rows by match ID, keeping only the columns we need."""
    if frame is None or frame.empty:
        return {}
    available = [c for c in columns if c in frame.columns]
    grouped: dict[int, list[dict]] = {}
    for match_id, group in frame[available + ["match_id"]].groupby("match_id", sort=False):
        grouped[int(match_id)] = group[available].to_dict("records")
    return grouped


def build_match_features(
    matches: pd.DataFrame,
    innings: pd.DataFrame | None = None,
    *,
    batting: pd.DataFrame | None = None,
    bowling: pd.DataFrame | None = None,
    match_players: pd.DataFrame | None = None,
    include_incomplete: bool = False,
) -> tuple[pd.DataFrame, FeatureState]:
    """Build the pre-match feature table for every fixture.

    Args:
        matches: Output of :func:`ipl.db.repository.load_matches`.
        innings: Output of :func:`ipl.db.repository.load_innings`, used for
            run-strength and venue-scoring state. Optional but recommended.
        batting: Batting cards, used to build career player histories.
        bowling: Bowling cards, used to build career player histories.
        match_players: Playing XI membership. With ``batting``/``bowling`` this
            enables the squad-strength features, which is where most of the
            pre-match signal lives.
        include_incomplete: Emit rows for scheduled fixtures too (their target
            columns are ``NaN``). Used by the dashboard to score upcoming games.

    Returns:
        ``(features, state)`` -- the feature table (one row per fixture, in
        chronological order) and the final rolling state, which callers can
        reuse to featurise a hypothetical fixture.
    """
    if matches.empty:
        return pd.DataFrame(), FeatureState()

    frame = matches.copy()
    frame = frame[frame["team1"].notna() & frame["team2"].notna()]
    # Chronological order is essential: the whole no-leakage guarantee rests on
    # processing fixtures in the order they were played.
    frame = frame.sort_values(["match_date", "match_id"], kind="stable").reset_index(drop=True)

    totals = _innings_totals(innings if innings is not None else pd.DataFrame())
    batting_by_match = _index_cards(batting, ["player", "runs", "balls", "is_out"])
    bowling_by_match = _index_cards(bowling, ["player", "runs_conceded", "balls", "wickets"])

    state = FeatureState()
    state.lineups = _index_lineups(match_players)
    rows: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        completed = bool(series.get("is_completed")) and isinstance(series.get("winner"), str)
        match_id = int(series["match_id"])

        if completed or include_incomplete:
            features = _fixture_features(state, series)
            features["match_id"] = series["match_id"]
            features["match_key"] = series.get("match_key")
            features["match_date"] = series.get("match_date")
            features["is_completed"] = completed
            features["winner"] = series.get("winner")
            # Target: did the nominal home/first-listed side win?
            features["target_team1_wins"] = (
                int(series["winner"] == series["team1"]) if completed else np.nan
            )
            match_totals = totals.get(match_id, {})
            features["first_innings_runs"] = match_totals.get(1, np.nan)
            features["second_innings_runs"] = match_totals.get(2, np.nan)
            features["first_batting_team"] = series.get("first_batting_team")
            rows.append(features)

        if completed:
            match_totals = totals.get(match_id, {})
            state.update(series, match_totals.get(1), match_totals.get(2))
            # Player histories are folded in AFTER the feature row is emitted,
            # so a match never contributes to its own squad-strength features.
            state.update_players(
                batting_by_match.get(match_id, []), bowling_by_match.get(match_id, [])
            )
            for team, lineup in state.lineups.get(match_id, {}).items():
                state.last_lineup[team] = lineup

    result = pd.DataFrame(rows)
    logger.info(
        "Built match features: %d rows x %d columns (%d completed)",
        len(result), result.shape[1] if not result.empty else 0,
        int(result["is_completed"].sum()) if not result.empty else 0,
    )
    return result, state


def build_score_features(features: pd.DataFrame) -> pd.DataFrame:
    """Reframe the match features from the first-innings batting side's view.

    The winner model is symmetric in team1/team2, but the score model needs to
    know which side is actually batting, so the relevant columns are swapped
    when the second-listed team bats first.
    """
    if features.empty:
        return pd.DataFrame()

    frame = features[features["first_innings_runs"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    bats_first_is_team1 = frame["first_batting_team"] == frame["team1"]

    def pick(when_team1: str, when_team2: str) -> pd.Series:
        return frame[when_team1].where(bats_first_is_team1, frame[when_team2])

    out = pd.DataFrame(
        {
            "season": frame["season"],
            "venue": frame["venue"],
            "batting_team": frame["first_batting_team"],
            "bowling_team": frame["team2"].where(bats_first_is_team1, frame["team1"]),
            "is_neutral_venue": frame["is_neutral_venue"],
            "is_playoff": frame["is_playoff"],
            "batting_is_home": frame["team1_is_home"].where(bats_first_is_team1, 0),
            "batting_won_toss": (
                frame["toss_winner_is_team1"]
                .where(bats_first_is_team1, 1 - frame["toss_winner_is_team1"])
            ),
            "batting_career_win_rate": pick("team1_career_win_rate", "team2_career_win_rate"),
            "batting_form_short": pick("team1_form_short", "team2_form_short"),
            "bowling_form_short": pick("team2_form_short", "team1_form_short"),
            "batting_avg_runs_scored": pick("team1_avg_runs_scored", "team2_avg_runs_scored"),
            "bowling_avg_runs_conceded": pick(
                "team2_avg_runs_conceded", "team1_avg_runs_conceded"
            ),
            "venue_avg_first_innings": frame["venue_avg_first_innings"],
            "batting_rest_days": pick("team1_rest_days", "team2_rest_days"),
            "league_recent_avg_score": frame["league_recent_avg_score"],
            "venue_recent_avg_score": frame["venue_recent_avg_score"],
            "batting_xi_batting_average": pick(
                "team1_xi_batting_average", "team2_xi_batting_average"
            ),
            "batting_xi_strike_rate": pick("team1_xi_strike_rate", "team2_xi_strike_rate"),
            "bowling_xi_bowling_economy": pick(
                "team2_xi_bowling_economy", "team1_xi_bowling_economy"
            ),
            "match_date": frame["match_date"],
            "match_id": frame["match_id"],
            "target_first_innings_runs": frame["first_innings_runs"],
        }
    )
    # Venue-and-team scoring history is not carried on the match feature row, so
    # approximate it with the venue average - the model still gets the venue as
    # a categorical, which captures the ground effect directly.
    out["batting_venue_avg_score"] = frame["venue_avg_first_innings"]
    return out


# Column pairs that swap when a fixture is mirrored, plus the signed
# differences that must be negated. Used by :func:`mirror_fixtures`.
_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("team1", "team2"),
    ("team1_career_win_rate", "team2_career_win_rate"),
    ("team1_form_short", "team2_form_short"),
    ("team1_form_long", "team2_form_long"),
    ("team1_venue_win_rate", "team2_venue_win_rate"),
    ("team1_avg_runs_scored", "team2_avg_runs_scored"),
    ("team1_avg_runs_conceded", "team2_avg_runs_conceded"),
    ("team1_net_run_strength", "team2_net_run_strength"),
    ("team1_rest_days", "team2_rest_days"),
    ("team1_season_points", "team2_season_points"),
    ("team1_season_matches", "team2_season_matches"),
    ("team1_win_streak", "team2_win_streak"),
    ("team1_xi_batting_average", "team2_xi_batting_average"),
    ("team1_xi_strike_rate", "team2_xi_strike_rate"),
    ("team1_xi_bowling_economy", "team2_xi_bowling_economy"),
    ("team1_xi_experience", "team2_xi_experience"),
)

_MIRROR_NEGATE: tuple[str, ...] = (
    "form_short_diff",
    "form_long_diff",
    "career_win_rate_diff",
    "venue_win_rate_diff",
    "net_run_strength_diff",
    "rest_days_diff",
    "season_points_diff",
    "xi_batting_average_diff",
    "xi_strike_rate_diff",
    "xi_bowling_economy_diff",
    "xi_experience_diff",
)


def mirror_fixtures(features: pd.DataFrame) -> pd.DataFrame:
    """Return the same fixtures with the two sides swapped.

    Concatenating this with the original doubles the training set and, more
    importantly, forces the classifier to learn a *symmetric* function of the
    two teams. Without it the model can exploit the arbitrary convention that
    ``team1`` is listed first -- which is a property of the data source, not of
    cricket, and does not transfer to the held-out seasons.
    """
    if features.empty:
        return features

    mirrored = features.copy()
    for left, right in _MIRROR_PAIRS:
        if left in mirrored.columns and right in mirrored.columns:
            mirrored[left], mirrored[right] = features[right].copy(), features[left].copy()

    # Coerce first: a column that is entirely None arrives as object dtype (the
    # squad-strength diffs when no Playing XI is known), and unary minus on
    # object raises rather than propagating null.
    for column in _MIRROR_NEGATE:
        if column in mirrored.columns:
            mirrored[column] = -pd.to_numeric(features[column], errors="coerce")

    # Home advantage and toss ownership belong to the side that had them.
    if "team1_is_home" in mirrored.columns:
        mirrored["team1_is_home"] = 0
    for column in ("toss_winner_is_team1", "team1_h2h_win_rate"):
        if column in mirrored.columns:
            mirrored[column] = 1 - pd.to_numeric(features[column], errors="coerce")

    if "target_team1_wins" in mirrored.columns:
        mirrored["target_team1_wins"] = 1 - pd.to_numeric(
            features["target_team1_wins"], errors="coerce"
        )

    mirrored["is_mirrored"] = True
    return mirrored


def featurise_fixture(
    state: FeatureState,
    *,
    team1: str,
    team2: str,
    venue: str,
    season: int,
    match_date: date | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    is_playoff: bool = False,
    is_neutral_venue: bool = False,
    home_team: str | None = None,
) -> pd.DataFrame:
    """Featurise a single hypothetical fixture for inference.

    Uses the same :func:`_fixture_features` routine as training, so serving and
    training features cannot drift apart.
    """
    series = pd.Series(
        {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "season": season,
            "match_date": pd.Timestamp(match_date) if match_date else pd.NaT,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "is_playoff": is_playoff,
            "is_neutral_venue": is_neutral_venue,
            "home_team": home_team if home_team is not None else team1,
        }
    )
    return pd.DataFrame([_fixture_features(state, series)])


def compute_standings(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build the league table for a season: points, wins, losses and NRR.

    Net run rate follows the IPL's definition -- (runs scored / overs faced)
    minus (runs conceded / overs bowled) across the season, with an all-out side
    charged the full quota of overs.
    """
    frame = matches[(matches["season"] == season) & matches["is_completed"]]
    frame = frame[~frame["is_playoff"].astype(bool)]

    # Run aggregates deliberately live in `add_net_run_rate`, which reads them
    # from the innings table. Carrying same-named placeholder columns here would
    # collide on that merge and silently produce `runs_for_x` / `runs_for_y`.
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"played": 0, "won": 0, "lost": 0, "no_result": 0, "points": 0.0}
    )

    for row in frame.itertuples(index=False):
        team1, team2 = row.team1, row.team2
        if not isinstance(team1, str) or not isinstance(team2, str):
            continue
        for team in (team1, team2):
            stats[team]["played"] += 1

        if bool(getattr(row, "is_no_result", False)) or not isinstance(row.winner, str):
            for team in (team1, team2):
                stats[team]["no_result"] += 1
                stats[team]["points"] += POINTS_TIE_OR_NR
            continue

        loser = team2 if row.winner == team1 else team1
        stats[row.winner]["won"] += 1
        stats[row.winner]["points"] += POINTS_WIN
        stats[loser]["lost"] += 1

    table = pd.DataFrame(
        [{"team": team, **values} for team, values in stats.items()]
    )
    if table.empty:
        return table

    table = table.sort_values(["points", "won"], ascending=[False, False]).reset_index(drop=True)
    table.insert(0, "position", range(1, len(table) + 1))
    return table


def add_net_run_rate(standings: pd.DataFrame, innings: pd.DataFrame, matches: pd.DataFrame,
                     season: int) -> pd.DataFrame:
    """Attach net run rate to a standings table built by :func:`compute_standings`."""
    if standings.empty or innings.empty:
        return standings

    season_ids = set(
        matches[(matches["season"] == season) & matches["is_completed"]
                & ~matches["is_playoff"].astype(bool)]["match_id"]
    )
    frame = innings[innings["match_id"].isin(season_ids)].copy()
    if frame.empty:
        return standings

    # An all-out side is charged the full quota, per the IPL's NRR rules.
    full_quota_balls = STANDARD_OVERS * 6
    frame["charged_balls"] = np.where(
        frame["wickets"].fillna(0) >= 10, full_quota_balls, frame["balls"].fillna(full_quota_balls)
    )

    scored = frame.groupby("batting_team").agg(
        runs_for=("runs", "sum"), balls_for=("charged_balls", "sum")
    )
    conceded = frame.groupby("bowling_team").agg(
        runs_against=("runs", "sum"), balls_against=("charged_balls", "sum")
    )

    merged = standings.merge(
        scored, left_on="team", right_index=True, how="left"
    ).merge(conceded, left_on="team", right_index=True, how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_for = merged["runs_for"] / (merged["balls_for"] / 6)
        rate_against = merged["runs_against"] / (merged["balls_against"] / 6)
    merged["net_run_rate"] = (rate_for - rate_against).round(3)

    ordered = merged.sort_values(
        ["points", "net_run_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    ordered["position"] = range(1, len(ordered) + 1)
    return ordered
