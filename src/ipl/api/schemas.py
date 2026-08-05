"""Pydantic request and response models.

These double as the API's documentation: FastAPI renders them into the OpenAPI
schema served at ``/docs``, so the field descriptions and examples here are what
a consumer actually reads.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Service liveness and readiness."""

    status: Literal["ok", "degraded"] = Field(description="Overall service status.")
    version: str = Field(description="Package version.")
    database: str = Field(description="Active database dialect.")
    matches: int = Field(description="Completed matches stored.")
    seasons: list[int] = Field(default_factory=list, description="Seasons available.")
    models_ready: dict[str, bool] = Field(
        default_factory=dict, description="Which prediction models are trained."
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str = Field(description="Human-readable explanation of the failure.")


# ---------------------------------------------------------------------------
# Winner prediction
# ---------------------------------------------------------------------------
class WinnerRequest(BaseModel):
    """Inputs for a match-winner prediction."""

    team1: str = Field(description="Home / first-listed team.", examples=["Chennai Super Kings"])
    team2: str = Field(description="Away / second-listed team.", examples=["Mumbai Indians"])
    venue: str | None = Field(
        default=None,
        description="Ground name. Defaults to team1's home venue.",
        examples=["MA Chidambaram Stadium"],
    )
    season: int | None = Field(
        default=None, ge=2008, le=2100, description="Season year. Defaults to the latest stored."
    )
    match_date: date | None = Field(default=None, description="Fixture date, for rest-day features.")
    toss_winner: str | None = Field(default=None, description="Toss winner, if already known.")
    toss_decision: Literal["bat", "field"] | None = Field(
        default=None, description="What the toss winner elected to do."
    )
    is_playoff: bool = Field(default=False, description="Whether this is a playoff fixture.")
    is_neutral_venue: bool = Field(
        default=False, description="True when neither side is playing at home."
    )

    @field_validator("team2")
    @classmethod
    def _teams_must_differ(cls, value: str, info) -> str:  # noqa: ANN001
        if info.data.get("team1") == value:
            raise ValueError("team1 and team2 must be different")
        return value


class DriverItem(BaseModel):
    """One contextual factor behind a prediction."""

    label: str
    team1_value: float | None = None
    team2_value: float | None = None
    unit: str = ""


class WinnerResponse(BaseModel):
    """Match-winner prediction."""

    team1: str
    team2: str
    venue: str
    predicted_winner: str
    team1_win_probability: float = Field(description="Percentage, 0-100.")
    team2_win_probability: float = Field(description="Percentage, 0-100.")
    confidence: float = Field(description="Distance from a coin flip, 0-100.")
    model: str = Field(description="Algorithm that produced this prediction.")
    drivers: list[DriverItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Score prediction
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    """Inputs for a first-innings score prediction."""

    batting_team: str = Field(examples=["Royal Challengers Bengaluru"])
    bowling_team: str = Field(examples=["Punjab Kings"])
    venue: str | None = Field(default=None, examples=["M Chinnaswamy Stadium"])
    season: int | None = Field(default=None, ge=2008, le=2100)
    is_playoff: bool = False
    is_neutral_venue: bool = False
    batting_won_toss: bool = Field(
        default=True, description="Whether the batting side won the toss and chose to bat."
    )

    @field_validator("bowling_team")
    @classmethod
    def _teams_must_differ(cls, value: str, info) -> str:  # noqa: ANN001
        if info.data.get("batting_team") == value:
            raise ValueError("batting_team and bowling_team must be different")
        return value


class ScoreResponse(BaseModel):
    """First-innings score prediction with a one-sigma range."""

    batting_team: str
    bowling_team: str
    venue: str
    predicted_score: int
    range_low: int
    range_high: int
    model: str


# ---------------------------------------------------------------------------
# Chase prediction
# ---------------------------------------------------------------------------
class ChaseRequest(BaseModel):
    """Current second-innings match state."""

    batting_team: str = Field(description="The chasing side.", examples=["Mumbai Indians"])
    bowling_team: str = Field(description="The defending side.", examples=["Gujarat Titans"])
    venue: str = Field(examples=["Wankhede Stadium"])
    target: int = Field(gt=0, le=400, description="Runs needed to win (first innings + 1).")
    current_runs: int = Field(ge=0, le=400, description="Runs scored so far in the chase.")
    wickets_fallen: int = Field(ge=0, le=10)
    balls_bowled: int = Field(ge=0, le=120, description="Legal balls bowled in the chase.")
    runs_last_5_overs: int = Field(default=0, ge=0, le=200)
    wickets_last_5_overs: int = Field(default=0, ge=0, le=10)


class ChaseResponse(BaseModel):
    """In-play chase-success prediction."""

    batting_team: str
    bowling_team: str
    target: int
    current_runs: int
    wickets_fallen: int
    balls_bowled: int
    runs_required: int
    balls_remaining: int
    required_run_rate: float
    chase_success_probability: float = Field(description="Percentage, 0-100.")
    win_probability_batting: float
    win_probability_bowling: float
    model: str


# ---------------------------------------------------------------------------
# Player of the Match
# ---------------------------------------------------------------------------
class PlayerOfMatchItem(BaseModel):
    """One player's award probability."""

    player: str
    team: str | None = None
    runs: float = 0
    wickets: float = 0
    total_impact: float = 0
    award_probability: float = Field(description="Percentage across this match's players.")


class PlayerOfMatchResponse(BaseModel):
    """Ranked Player-of-the-Match candidates for a match."""

    match_id: int
    predicted: str | None = None
    actual: str | None = None
    candidates: list[PlayerOfMatchItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Playoffs
# ---------------------------------------------------------------------------
class PlayoffTeamProjection(BaseModel):
    """One franchise's projected playoff chances."""

    team: str
    current_points: float
    net_run_rate: float
    matches_played: int
    matches_remaining: int
    max_possible_points: float
    qualification_pct: float
    expected_position: float


class PlayoffResponse(BaseModel):
    """Monte Carlo playoff projection for a season."""

    season: int
    simulations: int
    matches_remaining: int
    projections: list[PlayoffTeamProjection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
class TeamItem(BaseModel):
    """A franchise and its all-time record."""

    team: str
    short_code: str | None = None
    matches: int = 0
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    titles: int = 0


class VenueItem(BaseModel):
    """A ground and its scoring profile."""

    venue: str
    city: str | None = None
    matches: int = 0
    avg_first_innings: float | None = None
    chase_win_pct: float | None = None


class MatchItem(BaseModel):
    """A fixture, completed or scheduled."""

    match_id: int
    season: int
    match_date: date | None = None
    stage: str | None = None
    venue: str | None = None
    team1: str | None = None
    team2: str | None = None
    toss_winner: str | None = None
    toss_decision: str | None = None
    winner: str | None = None
    result_summary: str | None = None
    player_of_match: str | None = None
    is_completed: bool = False


class HeadToHeadResponse(BaseModel):
    """Aggregate record between two franchises."""

    team_a: str
    team_b: str
    matches: int
    team_a_wins: int
    team_b_wins: int
    no_result: int
    team_a_win_pct: float
    team_b_win_pct: float
    current_streak_team: str | None = None
    current_streak: int = 0


class ModelMetricItem(BaseModel):
    """One algorithm's held-out scores for one task."""

    task: str
    model: str
    is_best: bool = False
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None


class ModelComparisonResponse(BaseModel):
    """Full model-comparison table."""

    tasks: dict[str, str] = Field(
        default_factory=dict, description="Task name -> selected model."
    )
    metrics: list[ModelMetricItem] = Field(default_factory=list)


class IngestionRunResponse(BaseModel):
    """Summary of the most recent pipeline run."""

    status: str
    trigger: str | None = None
    started_at: Any | None = None
    finished_at: Any | None = None
    matches_inserted: int = 0
    matches_updated: int = 0
    matches_skipped: int = 0
    duration_seconds: float | None = None
    message: str | None = None
