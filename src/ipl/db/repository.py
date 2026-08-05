"""Repository layer - dimension caching, idempotent upserts and read queries.

Two responsibilities:

1. **Write path** (used by ``ipl.ingestion``): resolve canonical dimension rows
   (team/venue/player) with an in-session cache, and upsert a fully-parsed
   match together with all of its child rows in one transaction.
2. **Read path** (used by analytics, features and the dashboard): return wide,
   already-joined :class:`pandas.DataFrame` objects so downstream code never
   writes SQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from ..constants import TEAM_CODES, VENUE_CITIES, team_color
from ..logging_utils import get_logger
from .base import get_engine, session_scope
from .models import (
    BattingCard,
    BowlingCard,
    Delivery,
    FallOfWicket,
    IngestionRun,
    Innings,
    Match,
    MatchPlayer,
    Partnership,
    Player,
    Team,
    Venue,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dimension resolution
# ---------------------------------------------------------------------------
class DimensionCache:
    """Name -> primary key cache for teams, venues and players.

    Ingesting ~1100 matches touches the player dimension hundreds of thousands
    of times. Without this cache each touch is a SELECT; with it, the whole
    dimension is read once and grown in memory.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._teams: dict[str, int] = {}
        self._venues: dict[str, int] = {}
        self._players: dict[str, int] = {}
        self.warm()

    def warm(self) -> None:
        """Load existing dimension rows into memory."""
        self._teams = {n: i for n, i in self.session.execute(select(Team.name, Team.id))}
        self._venues = {n: i for n, i in self.session.execute(select(Venue.name, Venue.id))}
        self._players = {n: i for n, i in self.session.execute(select(Player.name, Player.id))}
        logger.debug(
            "Dimension cache warm: %d teams, %d venues, %d players",
            len(self._teams), len(self._venues), len(self._players),
        )

    # -- teams --------------------------------------------------------------
    def team_id(self, name: str | None, *, logo_url: str | None = None) -> int | None:
        """Return the PK for a canonical team name, inserting if unseen."""
        if not name:
            return None
        if name in self._teams:
            return self._teams[name]

        from ..constants import ACTIVE_TEAMS

        team = Team(
            name=name,
            short_code=TEAM_CODES.get(name),
            is_active=name in ACTIVE_TEAMS,
            primary_color=team_color(name),
            logo_url=logo_url,
        )
        self.session.add(team)
        self.session.flush()
        self._teams[name] = team.id
        return team.id

    # -- venues -------------------------------------------------------------
    def venue_id(self, name: str | None, *, city: str | None = None) -> int | None:
        """Return the PK for a canonical venue name, inserting if unseen."""
        if not name:
            return None
        if name in self._venues:
            return self._venues[name]

        venue = Venue(name=name, city=city or VENUE_CITIES.get(name), country="India")
        self.session.add(venue)
        self.session.flush()
        self._venues[name] = venue.id
        return venue.id

    # -- players ------------------------------------------------------------
    def player_id(self, name: str | None, **attrs: Any) -> int | None:
        """Return the PK for a canonical player name, inserting if unseen.

        Extra keyword arguments (``source_player_id``, ``batting_style``, ...)
        are applied on insert and used to *fill gaps* on an existing row, but
        never to overwrite a value that is already populated.
        """
        if not name:
            return None
        if name in self._players:
            pid = self._players[name]
            if attrs:
                self._enrich_player(pid, attrs)
            return pid

        clean = {k: v for k, v in attrs.items() if v not in (None, "")}
        player = Player(name=name, **clean)
        self.session.add(player)
        self.session.flush()
        self._players[name] = player.id
        return player.id

    def _enrich_player(self, player_id: int, attrs: dict[str, Any]) -> None:
        """Backfill NULL attributes on an existing player row."""
        interesting = {k: v for k, v in attrs.items() if v not in (None, "")}
        if not interesting:
            return
        player = self.session.get(Player, player_id)
        if player is None:
            return
        for key, value in interesting.items():
            if getattr(player, key, None) in (None, ""):
                setattr(player, key, value)


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------
# Child tables that are fully replaced whenever a match is re-ingested.
_CHILD_MODELS = (
    Innings, BattingCard, BowlingCard, FallOfWicket, Partnership, Delivery, MatchPlayer,
)


def existing_match_keys(session: Session) -> dict[str, tuple[bool, datetime | None]]:
    """Map ``match_key`` -> ``(is_completed, updated_at)`` for every stored match.

    The incremental pipeline uses this to skip matches that are already stored
    *and* already final, which is what makes a refresh run cheap.
    """
    rows = session.execute(select(Match.match_key, Match.is_completed, Match.updated_at))
    return {key: (bool(done), ts) for key, done, ts in rows}


def delete_match_children(session: Session, match_id: int) -> None:
    """Remove every child row of a match so it can be re-inserted cleanly."""
    for model in _CHILD_MODELS:
        session.execute(delete(model).where(model.match_id == match_id))


def upsert_match(session: Session, match_fields: dict[str, Any]) -> tuple[Match, bool]:
    """Insert or update a match row, keyed on ``match_key``.

    Returns:
        ``(match, created)`` where ``created`` is True on first insert.
    """
    key = match_fields["match_key"]
    match = session.scalar(select(Match).where(Match.match_key == key))
    if match is None:
        match = Match(**match_fields)
        session.add(match)
        session.flush()
        return match, True

    for field, value in match_fields.items():
        if field != "match_key":
            setattr(match, field, value)
    session.flush()
    return match, False


def bulk_insert(session: Session, model: type, rows: Iterable[dict[str, Any]]) -> int:
    """Insert child rows efficiently, returning the count written."""
    payload = list(rows)
    if not payload:
        return 0
    session.bulk_insert_mappings(model, payload)
    return len(payload)


def start_ingestion_run(trigger: str, sources: str, seasons: str) -> int:
    """Record the start of a pipeline run and return its ID."""
    with session_scope() as session:
        run = IngestionRun(
            status="running", trigger=trigger, sources=sources, seasons=seasons
        )
        session.add(run)
        session.flush()
        return run.id


def finish_ingestion_run(run_id: int, *, status: str, message: str | None = None, **counts: Any) -> None:
    """Close out a pipeline run with final counts and status."""
    with session_scope() as session:
        run = session.get(IngestionRun, run_id)
        if run is None:
            return
        run.status = status
        run.finished_at = datetime.utcnow()
        run.message = message
        if run.started_at:
            run.duration_seconds = (run.finished_at - run.started_at).total_seconds()
        for field, value in counts.items():
            if hasattr(run, field):
                setattr(run, field, value)


def latest_ingestion_run() -> dict[str, Any] | None:
    """Return the most recent pipeline run as a plain dict, or ``None``."""
    with session_scope() as session:
        run = session.scalar(select(IngestionRun).order_by(IngestionRun.id.desc()).limit(1))
        if run is None:
            return None
        return {
            "id": run.id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status,
            "trigger": run.trigger,
            "sources": run.sources,
            "seasons": run.seasons,
            "matches_seen": run.matches_seen,
            "matches_inserted": run.matches_inserted,
            "matches_updated": run.matches_updated,
            "matches_skipped": run.matches_skipped,
            "deliveries_inserted": run.deliveries_inserted,
            "duration_seconds": run.duration_seconds,
            "message": run.message,
        }


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------
# Matches joined to human-readable team/venue names. This single view backs
# nearly every analytics function and the entire feature pipeline.
_MATCHES_SQL = """
SELECT
    m.id                    AS match_id,
    m.match_key,
    m.season,
    m.match_date,
    m.start_datetime,
    m.match_number,
    m.stage,
    m.is_playoff,
    m.source,
    m.city,
    m.is_neutral_venue,
    m.is_completed,
    m.toss_decision,
    m.result_type,
    m.win_margin_runs,
    m.win_margin_wickets,
    m.is_tie,
    m.is_no_result,
    m.is_super_over,
    m.is_dls_applied,
    m.target_runs,
    m.result_summary,
    m.overs_per_innings,
    v.name                  AS venue,
    t1.name                 AS team1,
    t2.name                 AS team2,
    th.name                 AS home_team,
    ta.name                 AS away_team,
    tt.name                 AS toss_winner,
    tf.name                 AS first_batting_team,
    ts.name                 AS second_batting_team,
    tw.name                 AS winner,
    p.name                  AS player_of_match
FROM matches m
LEFT JOIN venues  v  ON v.id  = m.venue_id
LEFT JOIN teams   t1 ON t1.id = m.team1_id
LEFT JOIN teams   t2 ON t2.id = m.team2_id
LEFT JOIN teams   th ON th.id = m.home_team_id
LEFT JOIN teams   ta ON ta.id = m.away_team_id
LEFT JOIN teams   tt ON tt.id = m.toss_winner_id
LEFT JOIN teams   tf ON tf.id = m.first_batting_team_id
LEFT JOIN teams   ts ON ts.id = m.second_batting_team_id
LEFT JOIN teams   tw ON tw.id = m.winner_id
LEFT JOIN players p  ON p.id  = m.player_of_match_id
"""


def _read_sql(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a query against the shared engine and return a DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def load_matches(
    *, completed_only: bool = False, season: int | None = None
) -> pd.DataFrame:
    """Return the joined match table.

    Args:
        completed_only: Exclude scheduled/abandoned fixtures.
        season: Restrict to a single season year.
    """
    clauses, params = [], {}
    if completed_only:
        clauses.append("m.is_completed = 1" if _is_sqlite() else "m.is_completed = TRUE")
    if season is not None:
        clauses.append("m.season = :season")
        params["season"] = season

    sql = _MATCHES_SQL
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY m.match_date, m.id"

    df = _read_sql(sql, params)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        # SQLite stores booleans as 0/1 integers; normalise across backends.
        for col in (
            "is_completed", "is_playoff", "is_tie", "is_no_result",
            "is_super_over", "is_dls_applied", "is_neutral_venue",
        ):
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(bool)
    return df


def load_innings() -> pd.DataFrame:
    """Return per-innings totals joined to team names and match context."""
    sql = """
    SELECT
        i.match_id, i.innings_no, i.runs, i.wickets, i.overs, i.balls, i.run_rate,
        i.extras, i.byes, i.leg_byes, i.wides, i.no_balls,
        i.powerplay_runs, i.powerplay_wickets, i.middle_runs, i.middle_wickets,
        i.death_runs, i.death_wickets, i.fours, i.sixes, i.dot_balls, i.target,
        bt.name AS batting_team, bw.name AS bowling_team,
        m.season, m.match_date, v.name AS venue
    FROM innings i
    JOIN matches m ON m.id = i.match_id
    LEFT JOIN teams  bt ON bt.id = i.batting_team_id
    LEFT JOIN teams  bw ON bw.id = i.bowling_team_id
    LEFT JOIN venues v  ON v.id  = m.venue_id
    ORDER BY m.match_date, i.match_id, i.innings_no
    """
    df = _read_sql(sql)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def load_batting() -> pd.DataFrame:
    """Return every batting-card row joined to player, team and match context."""
    sql = """
    SELECT
        b.match_id, b.innings_no, b.batting_position, b.runs, b.balls,
        b.fours, b.sixes, b.strike_rate, b.is_out, b.dismissal_kind,
        p.name AS player, t.name AS team, m.season, m.match_date,
        v.name AS venue, opp.name AS opposition
    FROM batting_cards b
    JOIN matches m ON m.id = b.match_id
    JOIN players p ON p.id = b.player_id
    LEFT JOIN teams  t   ON t.id  = b.team_id
    LEFT JOIN venues v   ON v.id  = m.venue_id
    LEFT JOIN teams  opp ON opp.id = CASE
        WHEN b.team_id = m.team1_id THEN m.team2_id ELSE m.team1_id END
    """
    df = _read_sql(sql)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df["is_out"] = df["is_out"].fillna(0).astype(bool)
    return df


def load_bowling() -> pd.DataFrame:
    """Return every bowling-card row joined to player, team and match context."""
    sql = """
    SELECT
        b.match_id, b.innings_no, b.overs, b.balls, b.maidens, b.runs_conceded,
        b.wickets, b.wides, b.no_balls, b.economy, b.dot_balls,
        p.name AS player, t.name AS team, m.season, m.match_date,
        v.name AS venue, opp.name AS opposition
    FROM bowling_cards b
    JOIN matches m ON m.id = b.match_id
    JOIN players p ON p.id = b.player_id
    LEFT JOIN teams  t   ON t.id  = b.team_id
    LEFT JOIN venues v   ON v.id  = m.venue_id
    LEFT JOIN teams  opp ON opp.id = CASE
        WHEN b.team_id = m.team1_id THEN m.team2_id ELSE m.team1_id END
    """
    df = _read_sql(sql)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def load_deliveries(season: int | None = None) -> pd.DataFrame:
    """Return ball-by-ball data. Large -- pass ``season`` to bound the result."""
    sql = """
    SELECT
        d.match_id, d.innings_no, d.over_no, d.ball_no, d.ball_seq,
        d.batter_runs, d.extra_runs, d.total_runs, d.is_legal, d.is_wide,
        d.is_no_ball, d.is_four, d.is_six, d.is_wicket, d.wicket_type,
        d.cumulative_runs, d.cumulative_wickets,
        bat.name AS batting_team, bwl.name AS bowling_team,
        pb.name AS batter, pw.name AS bowler,
        m.season, m.match_date
    FROM deliveries d
    JOIN matches m ON m.id = d.match_id
    LEFT JOIN teams   bat ON bat.id = d.batting_team_id
    LEFT JOIN teams   bwl ON bwl.id = d.bowling_team_id
    LEFT JOIN players pb  ON pb.id  = d.batter_id
    LEFT JOIN players pw  ON pw.id  = d.bowler_id
    """
    params: dict[str, Any] = {}
    if season is not None:
        sql += " WHERE m.season = :season"
        params["season"] = season
    sql += " ORDER BY d.match_id, d.innings_no, d.ball_seq"

    df = _read_sql(sql, params)
    if not df.empty:
        for col in ("is_legal", "is_wide", "is_no_ball", "is_four", "is_six", "is_wicket"):
            df[col] = df[col].fillna(0).astype(bool)
    return df


def load_match_players() -> pd.DataFrame:
    """Return Playing XI / squad membership joined to names."""
    sql = """
    SELECT
        mp.match_id, mp.is_playing_xi, mp.is_captain, mp.is_wicketkeeper,
        mp.is_overseas, mp.playing_order, mp.role,
        p.name AS player, t.name AS team, m.season, m.match_date
    FROM match_players mp
    JOIN matches m ON m.id = mp.match_id
    JOIN players p ON p.id = mp.player_id
    JOIN teams   t ON t.id = mp.team_id
    """
    df = _read_sql(sql)
    if not df.empty:
        for col in ("is_playing_xi", "is_captain", "is_wicketkeeper"):
            df[col] = df[col].fillna(0).astype(bool)
    return df


def load_partnerships() -> pd.DataFrame:
    """Return partnership rows joined to batter names."""
    sql = """
    SELECT
        pt.match_id, pt.innings_no, pt.wicket_no, pt.runs, pt.balls,
        pt.striker_runs, pt.striker_balls, pt.non_striker_runs,
        pt.non_striker_balls, pt.extras, pt.start_over, pt.end_over,
        s.name AS striker, ns.name AS non_striker, t.name AS team,
        m.season, m.match_date
    FROM partnerships pt
    JOIN matches m ON m.id = pt.match_id
    LEFT JOIN players s  ON s.id  = pt.striker_id
    LEFT JOIN players ns ON ns.id = pt.non_striker_id
    LEFT JOIN teams   t  ON t.id  = pt.team_id
    """
    return _read_sql(sql)


def load_fall_of_wickets() -> pd.DataFrame:
    """Return fall-of-wicket rows joined to the dismissed batter."""
    sql = """
    SELECT
        f.match_id, f.innings_no, f.wicket_no, f.fall_score, f.fall_overs,
        p.name AS player, t.name AS team, m.season
    FROM fall_of_wickets f
    JOIN matches m ON m.id = f.match_id
    LEFT JOIN players p ON p.id = f.player_id
    LEFT JOIN teams   t ON t.id = f.team_id
    """
    return _read_sql(sql)


def database_summary() -> dict[str, int]:
    """Row counts per table, for the Admin page's health panel."""
    counts: dict[str, int] = {}
    with session_scope() as session:
        for model, label in (
            (Match, "matches"), (Team, "teams"), (Player, "players"), (Venue, "venues"),
            (Innings, "innings"), (BattingCard, "batting_cards"),
            (BowlingCard, "bowling_cards"), (Delivery, "deliveries"),
            (Partnership, "partnerships"), (FallOfWicket, "fall_of_wickets"),
            (MatchPlayer, "match_players"),
        ):
            counts[label] = session.scalar(select(func.count()).select_from(model)) or 0
        counts["completed_matches"] = session.scalar(
            select(func.count()).select_from(Match).where(Match.is_completed.is_(True))
        ) or 0
    return counts


def season_range() -> tuple[int, int] | None:
    """Return ``(first_season, last_season)`` present in the database."""
    with session_scope() as session:
        lo = session.scalar(select(func.min(Match.season)))
        hi = session.scalar(select(func.max(Match.season)))
    return (lo, hi) if lo is not None and hi is not None else None


def _is_sqlite() -> bool:
    """True when the active engine is SQLite (affects boolean literals)."""
    from ..config import get_settings

    return get_settings().is_sqlite
