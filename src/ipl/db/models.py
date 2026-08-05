"""SQLAlchemy ORM models - the canonical IPL data warehouse schema.

Design notes
------------
* Every table uses a surrogate integer primary key plus a natural unique key.
  Source match IDs are *not* globally unique (the official feed numbers matches
  per-installation while Cricsheet uses its own IDs), so :attr:`Match.match_key`
  namespaces them as ``"<source>:<season>:<source_id>"``.
* Names are normalised to canonical forms before insert (see
  ``ipl.ingestion.normalize``), so ``Team.name`` is safe to join on.
* Column widths are explicit because MySQL cannot index an unbounded ``TEXT``.
* Deletes cascade from :class:`Match` so a match can be re-ingested cleanly by
  deleting the parent row and re-inserting.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# Cascade applied to every child collection hanging off a Match.
_CASCADE = "all, delete-orphan"


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------
class Team(Base):
    """An IPL franchise, identified by its canonical (current) name."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    short_code: Mapped[str | None] = mapped_column(String(10))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    primary_color: Mapped[str | None] = mapped_column(String(16))
    logo_url: Mapped[str | None] = mapped_column(String(400))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Team {self.short_code or self.name}>"


class Venue(Base):
    """A cricket ground. Canonicalised name; city stored separately."""

    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    city: Mapped[str | None] = mapped_column(String(80))
    country: Mapped[str | None] = mapped_column(String(60), default="India")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Venue {self.name}>"


class Player(Base):
    """A cricketer.

    Players are keyed by canonical display name. ``source_player_id`` keeps the
    official feed's opaque GUID when available so future feed pulls can match
    on ID rather than on a name that might be spelled differently.
    """

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    source_player_id: Mapped[str | None] = mapped_column(String(80), index=True)
    short_name: Mapped[str | None] = mapped_column(String(120))
    batting_style: Mapped[str | None] = mapped_column(String(60))
    bowling_style: Mapped[str | None] = mapped_column(String(80))
    role: Mapped[str | None] = mapped_column(String(40))
    is_overseas: Mapped[bool | None] = mapped_column(Boolean)
    image_url: Mapped[str | None] = mapped_column(String(400))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Player {self.name}>"


# ---------------------------------------------------------------------------
# Fact tables
# ---------------------------------------------------------------------------
class Match(Base):
    """One IPL fixture, completed or scheduled.

    The row carries both the *scheduled* framing (teams, toss, venue) and the
    *outcome* (winner, margin, player of the match). Scheduled future matches
    have all outcome columns ``NULL`` and ``is_completed = False``, which is
    exactly the filter the training pipeline uses.
    """

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Identity & provenance ---
    match_key: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    source_match_id: Mapped[str | None] = mapped_column(String(40), index=True)
    competition_id: Mapped[int | None] = mapped_column(Integer)

    # --- Scheduling ---
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    match_date: Mapped[date | None] = mapped_column(Date, index=True)
    start_datetime: Mapped[datetime | None] = mapped_column(DateTime)
    match_number: Mapped[str | None] = mapped_column(String(40))
    stage: Mapped[str | None] = mapped_column(String(40))
    is_playoff: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_day_night: Mapped[bool | None] = mapped_column(Boolean)
    overs_per_innings: Mapped[int | None] = mapped_column(Integer, default=20)

    # --- Location ---
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"), index=True)
    city: Mapped[str | None] = mapped_column(String(80))
    is_neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # --- Participants ---
    team1_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    team2_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))

    # --- Toss ---
    toss_winner_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    toss_decision: Mapped[str | None] = mapped_column(String(10))  # 'bat' | 'field'

    # --- Innings order ---
    first_batting_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    second_batting_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))

    # --- Outcome ---
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    winner_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    result_type: Mapped[str | None] = mapped_column(String(20))  # runs|wickets|tie|no result
    win_margin_runs: Mapped[int | None] = mapped_column(Integer)
    win_margin_wickets: Mapped[int | None] = mapped_column(Integer)
    is_tie: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_no_result: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_super_over: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_dls_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    target_runs: Mapped[int | None] = mapped_column(Integer)
    player_of_match_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    result_summary: Mapped[str | None] = mapped_column(String(300))

    # --- Officials ---
    umpire1: Mapped[str | None] = mapped_column(String(80))
    umpire2: Mapped[str | None] = mapped_column(String(80))
    third_umpire: Mapped[str | None] = mapped_column(String(80))
    match_referee: Mapped[str | None] = mapped_column(String(80))

    ingested_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # --- Relationships (loaded on demand; the analytics layer uses SQL directly) ---
    venue = relationship("Venue", foreign_keys=[venue_id], lazy="joined")
    team1 = relationship("Team", foreign_keys=[team1_id], lazy="joined")
    team2 = relationship("Team", foreign_keys=[team2_id], lazy="joined")
    winner = relationship("Team", foreign_keys=[winner_id], lazy="joined")
    toss_winner = relationship("Team", foreign_keys=[toss_winner_id])
    player_of_match = relationship("Player", foreign_keys=[player_of_match_id])

    innings = relationship("Innings", back_populates="match", cascade=_CASCADE)
    batting_cards = relationship("BattingCard", back_populates="match", cascade=_CASCADE)
    bowling_cards = relationship("BowlingCard", back_populates="match", cascade=_CASCADE)
    fall_of_wickets = relationship("FallOfWicket", back_populates="match", cascade=_CASCADE)
    partnerships = relationship("Partnership", back_populates="match", cascade=_CASCADE)
    deliveries = relationship("Delivery", back_populates="match", cascade=_CASCADE)
    match_players = relationship("MatchPlayer", back_populates="match", cascade=_CASCADE)

    __table_args__ = (
        Index("ix_matches_season_date", "season", "match_date"),
        Index("ix_matches_teams", "team1_id", "team2_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Match {self.match_key} season={self.season}>"


class Innings(Base):
    """Per-innings totals, including phase splits used as model features."""

    __tablename__ = "innings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)

    batting_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    bowling_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)

    runs: Mapped[int | None] = mapped_column(Integer)
    wickets: Mapped[int | None] = mapped_column(Integer)
    overs: Mapped[float | None] = mapped_column(Float)   # cricket notation, e.g. 19.4
    balls: Mapped[int | None] = mapped_column(Integer)   # legal deliveries bowled
    run_rate: Mapped[float | None] = mapped_column(Float)

    extras: Mapped[int | None] = mapped_column(Integer)
    byes: Mapped[int | None] = mapped_column(Integer)
    leg_byes: Mapped[int | None] = mapped_column(Integer)
    wides: Mapped[int | None] = mapped_column(Integer)
    no_balls: Mapped[int | None] = mapped_column(Integer)
    penalty: Mapped[int | None] = mapped_column(Integer)

    # Phase splits - derived from deliveries when ball-by-ball data is present.
    powerplay_runs: Mapped[int | None] = mapped_column(Integer)
    powerplay_wickets: Mapped[int | None] = mapped_column(Integer)
    middle_runs: Mapped[int | None] = mapped_column(Integer)
    middle_wickets: Mapped[int | None] = mapped_column(Integer)
    death_runs: Mapped[int | None] = mapped_column(Integer)
    death_wickets: Mapped[int | None] = mapped_column(Integer)

    fours: Mapped[int | None] = mapped_column(Integer)
    sixes: Mapped[int | None] = mapped_column(Integer)
    dot_balls: Mapped[int | None] = mapped_column(Integer)

    target: Mapped[int | None] = mapped_column(Integer)
    is_declared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    match = relationship("Match", back_populates="innings")

    __table_args__ = (UniqueConstraint("match_id", "innings_no", name="uq_innings_match_no"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Innings m={self.match_id} #{self.innings_no} {self.runs}/{self.wickets}>"


class BattingCard(Base):
    """One batter's line in a scorecard."""

    __tablename__ = "batting_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)

    batting_position: Mapped[int | None] = mapped_column(Integer)
    runs: Mapped[int] = mapped_column(Integer, default=0)
    balls: Mapped[int] = mapped_column(Integer, default=0)
    fours: Mapped[int] = mapped_column(Integer, default=0)
    sixes: Mapped[int] = mapped_column(Integer, default=0)
    dot_balls: Mapped[int | None] = mapped_column(Integer)
    strike_rate: Mapped[float | None] = mapped_column(Float)

    is_out: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dismissal_kind: Mapped[str | None] = mapped_column(String(40))
    dismissal_text: Mapped[str | None] = mapped_column(String(200))
    bowler_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    fielder_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    wicket_number: Mapped[int | None] = mapped_column(Integer)

    match = relationship("Match", back_populates="batting_cards")
    player = relationship("Player", foreign_keys=[player_id])

    __table_args__ = (
        UniqueConstraint("match_id", "innings_no", "player_id", name="uq_batting_card"),
    )


class BowlingCard(Base):
    """One bowler's line in a scorecard."""

    __tablename__ = "bowling_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)

    bowling_order: Mapped[int | None] = mapped_column(Integer)
    overs: Mapped[float] = mapped_column(Float, default=0.0)
    balls: Mapped[int | None] = mapped_column(Integer)
    maidens: Mapped[int] = mapped_column(Integer, default=0)
    runs_conceded: Mapped[int] = mapped_column(Integer, default=0)
    wickets: Mapped[int] = mapped_column(Integer, default=0)
    wides: Mapped[int] = mapped_column(Integer, default=0)
    no_balls: Mapped[int] = mapped_column(Integer, default=0)
    dot_balls: Mapped[int | None] = mapped_column(Integer)
    economy: Mapped[float | None] = mapped_column(Float)

    match = relationship("Match", back_populates="bowling_cards")
    player = relationship("Player", foreign_keys=[player_id])

    __table_args__ = (
        UniqueConstraint("match_id", "innings_no", "player_id", name="uq_bowling_card"),
    )


class FallOfWicket(Base):
    """Score and over at which each wicket fell."""

    __tablename__ = "fall_of_wickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)
    wicket_no: Mapped[int] = mapped_column(Integer, nullable=False)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    fall_score: Mapped[int | None] = mapped_column(Integer)
    fall_overs: Mapped[float | None] = mapped_column(Float)

    match = relationship("Match", back_populates="fall_of_wickets")

    __table_args__ = (
        UniqueConstraint("match_id", "innings_no", "wicket_no", name="uq_fow"),
    )


class Partnership(Base):
    """A batting partnership, in the order it occurred."""

    __tablename__ = "partnerships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)
    wicket_no: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))

    striker_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    non_striker_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    runs: Mapped[int | None] = mapped_column(Integer)
    balls: Mapped[int | None] = mapped_column(Integer)
    striker_runs: Mapped[int | None] = mapped_column(Integer)
    striker_balls: Mapped[int | None] = mapped_column(Integer)
    non_striker_runs: Mapped[int | None] = mapped_column(Integer)
    non_striker_balls: Mapped[int | None] = mapped_column(Integer)
    extras: Mapped[int | None] = mapped_column(Integer)
    start_over: Mapped[float | None] = mapped_column(Float)
    end_over: Mapped[float | None] = mapped_column(Float)
    is_unbroken: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    match = relationship("Match", back_populates="partnerships")

    __table_args__ = (
        UniqueConstraint("match_id", "innings_no", "wicket_no", name="uq_partnership"),
    )


class Delivery(Base):
    """A single legal or illegal delivery - the ball-by-ball grain.

    This is by far the largest table (~270k rows for 2008-2026). It powers the
    in-play chase model, phase analytics and player strike/economy profiles.
    """

    __tablename__ = "deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    innings_no: Mapped[int] = mapped_column(Integer, nullable=False)
    over_no: Mapped[int] = mapped_column(Integer, nullable=False)     # 1-based
    ball_no: Mapped[int] = mapped_column(Integer, nullable=False)     # within the over
    ball_seq: Mapped[int] = mapped_column(Integer, nullable=False)    # 1-based within innings

    batting_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    bowling_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    batter_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    non_striker_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    bowler_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)

    batter_runs: Mapped[int] = mapped_column(Integer, default=0)
    extra_runs: Mapped[int] = mapped_column(Integer, default=0)
    total_runs: Mapped[int] = mapped_column(Integer, default=0)

    is_wide: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_no_ball: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_bye: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_leg_bye: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_legal: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_four: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_six: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_wicket: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    wicket_type: Mapped[str | None] = mapped_column(String(40))
    dismissed_player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))

    # Running innings state after this ball - precomputed so the in-play chase
    # model does not need a window function at inference time.
    cumulative_runs: Mapped[int | None] = mapped_column(Integer)
    cumulative_wickets: Mapped[int | None] = mapped_column(Integer)

    match = relationship("Match", back_populates="deliveries")

    __table_args__ = (
        UniqueConstraint("match_id", "innings_no", "ball_seq", name="uq_delivery"),
        Index("ix_deliveries_match_innings", "match_id", "innings_no"),
    )


class MatchPlayer(Base):
    """Squad / Playing XI membership for one match."""

    __tablename__ = "match_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)

    is_playing_xi: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_captain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_wicketkeeper: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_overseas: Mapped[bool | None] = mapped_column(Boolean)
    is_impact_sub: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    playing_order: Mapped[int | None] = mapped_column(Integer)
    role: Mapped[str | None] = mapped_column(String(40))

    match = relationship("Match", back_populates="match_players")
    player = relationship("Player", foreign_keys=[player_id])

    __table_args__ = (
        UniqueConstraint("match_id", "team_id", "player_id", name="uq_match_player"),
    )


# ---------------------------------------------------------------------------
# Operational tables
# ---------------------------------------------------------------------------
class IngestionRun(Base):
    """Audit trail for the data pipeline.

    The dashboard's Admin page reads the latest row to show when data was last
    refreshed and whether the run succeeded.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    trigger: Mapped[str | None] = mapped_column(String(30))  # cli | dashboard | schedule
    sources: Mapped[str | None] = mapped_column(String(120))
    seasons: Mapped[str | None] = mapped_column(String(200))

    matches_seen: Mapped[int] = mapped_column(Integer, default=0)
    matches_inserted: Mapped[int] = mapped_column(Integer, default=0)
    matches_updated: Mapped[int] = mapped_column(Integer, default=0)
    matches_skipped: Mapped[int] = mapped_column(Integer, default=0)
    deliveries_inserted: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<IngestionRun {self.id} {self.status}>"
