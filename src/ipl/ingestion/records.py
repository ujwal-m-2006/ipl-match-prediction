"""Source-agnostic intermediate representation.

Both the official IPL client and the Cricsheet client emit these dataclasses.
The loader in :mod:`ipl.ingestion.pipeline` then resolves names to dimension
keys and writes them, so it never needs to know which source a match came from.

Everything here uses **canonical names** (not database IDs) -- ID resolution is
the loader's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(slots=True)
class BattingRecord:
    """A batter's scorecard line."""

    innings_no: int
    team: str | None
    player: str
    batting_position: int | None = None
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    dot_balls: int | None = None
    strike_rate: float | None = None
    is_out: bool = False
    dismissal_kind: str | None = None
    dismissal_text: str | None = None
    bowler: str | None = None
    fielder: str | None = None
    wicket_number: int | None = None


@dataclass(slots=True)
class BowlingRecord:
    """A bowler's scorecard line."""

    innings_no: int
    team: str | None
    player: str
    bowling_order: int | None = None
    overs: float = 0.0
    balls: int | None = None
    maidens: int = 0
    runs_conceded: int = 0
    wickets: int = 0
    wides: int = 0
    no_balls: int = 0
    dot_balls: int | None = None
    economy: float | None = None


@dataclass(slots=True)
class FallOfWicketRecord:
    """Score and over at which a wicket fell."""

    innings_no: int
    wicket_no: int
    player: str | None = None
    team: str | None = None
    fall_score: int | None = None
    fall_overs: float | None = None


@dataclass(slots=True)
class PartnershipRecord:
    """A batting partnership."""

    innings_no: int
    wicket_no: int
    team: str | None = None
    striker: str | None = None
    non_striker: str | None = None
    runs: int | None = None
    balls: int | None = None
    striker_runs: int | None = None
    striker_balls: int | None = None
    non_striker_runs: int | None = None
    non_striker_balls: int | None = None
    extras: int | None = None
    start_over: float | None = None
    end_over: float | None = None
    is_unbroken: bool = False


@dataclass(slots=True)
class DeliveryRecord:
    """One ball. ``ball_seq`` is 1-based and counts illegal deliveries too."""

    innings_no: int
    over_no: int
    ball_no: int
    ball_seq: int
    batting_team: str | None = None
    bowling_team: str | None = None
    batter: str | None = None
    non_striker: str | None = None
    bowler: str | None = None
    batter_runs: int = 0
    extra_runs: int = 0
    total_runs: int = 0
    is_wide: bool = False
    is_no_ball: bool = False
    is_bye: bool = False
    is_leg_bye: bool = False
    is_legal: bool = True
    is_four: bool = False
    is_six: bool = False
    is_wicket: bool = False
    wicket_type: str | None = None
    dismissed_player: str | None = None
    cumulative_runs: int | None = None
    cumulative_wickets: int | None = None

    # Runs charged to the bowler's analysis: everything off the bat, plus wide
    # and no-ball penalties, but NOT byes or leg-byes. Kept separate from
    # ``total_runs`` because the two differ on any bye. Derived at parse time
    # (where the extras breakdown is available) and not persisted -- the
    # loader maps only the database-backed fields.
    bowler_charged_runs: int | None = None


@dataclass(slots=True)
class InningsRecord:
    """Per-innings totals. Phase splits are filled in from deliveries."""

    innings_no: int
    batting_team: str | None = None
    bowling_team: str | None = None
    runs: int | None = None
    wickets: int | None = None
    overs: float | None = None
    balls: int | None = None
    run_rate: float | None = None
    extras: int | None = None
    byes: int | None = None
    leg_byes: int | None = None
    wides: int | None = None
    no_balls: int | None = None
    penalty: int | None = None
    powerplay_runs: int | None = None
    powerplay_wickets: int | None = None
    middle_runs: int | None = None
    middle_wickets: int | None = None
    death_runs: int | None = None
    death_wickets: int | None = None
    fours: int | None = None
    sixes: int | None = None
    dot_balls: int | None = None
    target: int | None = None
    is_declared: bool = False


@dataclass(slots=True)
class SquadRecord:
    """A player's inclusion in a match squad / Playing XI."""

    team: str
    player: str
    is_playing_xi: bool = True
    is_captain: bool = False
    is_wicketkeeper: bool = False
    is_overseas: bool | None = None
    is_impact_sub: bool = False
    playing_order: int | None = None
    role: str | None = None
    batting_style: str | None = None
    bowling_style: str | None = None
    source_player_id: str | None = None
    image_url: str | None = None


@dataclass(slots=True)
class MatchRecord:
    """A complete match: header fields plus every child collection."""

    # --- Identity ---
    match_key: str
    source: str
    season: int
    source_match_id: str | None = None
    competition_id: int | None = None

    # --- Scheduling ---
    match_date: date | None = None
    start_datetime: datetime | None = None
    match_number: str | None = None
    stage: str | None = None
    is_playoff: bool = False
    is_day_night: bool | None = None
    overs_per_innings: int | None = 20

    # --- Location ---
    venue: str | None = None
    city: str | None = None
    is_neutral_venue: bool = False

    # --- Participants ---
    team1: str | None = None
    team2: str | None = None
    home_team: str | None = None
    away_team: str | None = None

    # --- Toss ---
    toss_winner: str | None = None
    toss_decision: str | None = None

    # --- Innings order ---
    first_batting_team: str | None = None
    second_batting_team: str | None = None

    # --- Outcome ---
    is_completed: bool = False
    winner: str | None = None
    result_type: str | None = None
    win_margin_runs: int | None = None
    win_margin_wickets: int | None = None
    is_tie: bool = False
    is_no_result: bool = False
    is_super_over: bool = False
    is_dls_applied: bool = False
    target_runs: int | None = None
    player_of_match: str | None = None
    result_summary: str | None = None

    # --- Officials ---
    umpire1: str | None = None
    umpire2: str | None = None
    third_umpire: str | None = None
    match_referee: str | None = None

    # --- Children ---
    innings: list[InningsRecord] = field(default_factory=list)
    batting: list[BattingRecord] = field(default_factory=list)
    bowling: list[BowlingRecord] = field(default_factory=list)
    fall_of_wickets: list[FallOfWicketRecord] = field(default_factory=list)
    partnerships: list[PartnershipRecord] = field(default_factory=list)
    deliveries: list[DeliveryRecord] = field(default_factory=list)
    squads: list[SquadRecord] = field(default_factory=list)

    def teams(self) -> tuple[str | None, str | None]:
        return self.team1, self.team2

    def has_scorecard(self) -> bool:
        """True when detailed (non-header) data was collected for this match."""
        return bool(self.batting or self.bowling or self.deliveries)

    def summary(self) -> str:  # pragma: no cover - logging aid
        return (
            f"{self.season} {self.team1} vs {self.team2} @ {self.venue} "
            f"[{self.match_key}]"
        )


def as_dict(record: Any) -> dict[str, Any]:
    """Shallow dataclass -> dict, skipping ``None`` values."""
    from dataclasses import asdict

    return {k: v for k, v in asdict(record).items() if v is not None}
