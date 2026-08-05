"""Ingestion orchestration: fetch -> validate -> normalise -> load.

Source precedence
-----------------
The official iplt20.com feed is the **primary** source and covers every season
it publishes (currently 2019 onwards). Cricsheet is used **only** to back-fill
seasons the official feed does not expose (2008-2018). A season present in the
official registry is never overwritten by Cricsheet, which is what keeps the
official site authoritative while still giving full historical coverage.

Incremental behaviour
---------------------
A refresh run skips any match that is already stored *and* already final, so a
daily job costs a single schedule request per active season plus detail
requests for the handful of matches that finished since the last run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from sqlalchemy.orm import Session

from ..constants import (
    CRICSHEET_ONLY_SEASONS,
    IPL_COMPETITIONS,
    SOURCE_CRICSHEET,
    SOURCE_OFFICIAL,
)
from ..db.base import init_db, session_scope
from ..db.models import (
    BattingCard,
    BowlingCard,
    Delivery,
    FallOfWicket,
    Innings,
    MatchPlayer,
    Partnership,
)
from ..db.repository import (
    DimensionCache,
    bulk_insert,
    delete_match_children,
    existing_match_keys,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_match,
)
from ..logging_utils import get_logger
from .cricsheet_client import CricsheetClient
from .http_client import HttpClient
from .iplt20_client import IPLT20Client
from .records import MatchRecord
from .validation import ValidationReport, deduplicate_match, validate_match

logger = get_logger(__name__)


@dataclass
class IngestionStats:
    """Counters reported at the end of a run."""

    matches_seen: int = 0
    matches_inserted: int = 0
    matches_updated: int = 0
    matches_skipped: int = 0
    matches_rejected: int = 0
    deliveries_inserted: int = 0
    seasons: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "matches_seen": self.matches_seen,
            "matches_inserted": self.matches_inserted,
            "matches_updated": self.matches_updated,
            "matches_skipped": self.matches_skipped,
            "deliveries_inserted": self.deliveries_inserted,
        }

    def summary(self) -> str:
        return (
            f"seen={self.matches_seen} inserted={self.matches_inserted} "
            f"updated={self.matches_updated} skipped={self.matches_skipped} "
            f"rejected={self.matches_rejected} deliveries={self.deliveries_inserted}"
        )


class IngestionPipeline:
    """Coordinates the official and Cricsheet clients and writes to the database."""

    def __init__(
        self,
        *,
        official: IPLT20Client | None = None,
        cricsheet: CricsheetClient | None = None,
        ingest_deliveries: bool = True,
        enable_cricsheet: bool = True,
    ) -> None:
        http = HttpClient()
        self.official = official or IPLT20Client(http)
        self.cricsheet = cricsheet or CricsheetClient(http)
        self.ingest_deliveries = ingest_deliveries
        self.enable_cricsheet = enable_cricsheet
        self.report = ValidationReport()
        self.stats = IngestionStats()

    # ------------------------------------------------------------------
    # Season planning
    # ------------------------------------------------------------------
    @staticmethod
    def official_seasons() -> list[int]:
        """Seasons the official feed publishes."""
        return sorted(IPL_COMPETITIONS)

    @staticmethod
    def cricsheet_seasons() -> list[int]:
        """Seasons only Cricsheet covers (the official feed has no competition)."""
        return [s for s in CRICSHEET_ONLY_SEASONS if s not in IPL_COMPETITIONS]

    def plan(self, seasons: Sequence[int] | None = None) -> tuple[list[int], list[int]]:
        """Split the requested seasons into ``(official, cricsheet)`` buckets."""
        official = self.official_seasons()
        supplement = self.cricsheet_seasons() if self.enable_cricsheet else []

        if seasons is not None:
            requested = set(seasons)
            official = [s for s in official if s in requested]
            supplement = [s for s in supplement if s in requested]
            unknown = requested - set(official) - set(supplement)
            if unknown:
                logger.warning("No data source covers season(s): %s", sorted(unknown))

        return official, supplement

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        seasons: Sequence[int] | None = None,
        force_refresh: bool = False,
        skip_completed: bool = True,
        trigger: str = "cli",
        limit: int | None = None,
        commit_every: int = 25,
    ) -> IngestionStats:
        """Execute a full ingestion pass.

        Args:
            seasons: Restrict to these season years; ``None`` means all.
            force_refresh: Bypass the HTTP cache and re-fetch every feed.
            skip_completed: Skip matches already stored as final. Disable to
                force a full re-parse of history.
            trigger: Recorded on the audit row (``cli`` | ``dashboard`` | ``schedule``).
            limit: Stop after this many *new or changed* matches (for smoke tests).
            commit_every: Flush to the database every N matches. A full history
                load writes ~270k delivery rows; committing in batches keeps the
                transaction bounded and means an interrupted run leaves the
                already-loaded seasons intact instead of rolling everything back.
        """
        init_db()
        official_seasons, cricsheet_seasons = self.plan(seasons)
        self.stats.seasons = sorted(set(official_seasons) | set(cricsheet_seasons))

        sources = []
        if official_seasons:
            sources.append(SOURCE_OFFICIAL)
        if cricsheet_seasons:
            sources.append(SOURCE_CRICSHEET)

        run_id = start_ingestion_run(
            trigger=trigger,
            sources=",".join(sources) or "none",
            seasons=",".join(str(s) for s in self.stats.seasons),
        )
        started = time.monotonic()

        try:
            with session_scope() as session:
                cache = DimensionCache(session)
                known = existing_match_keys(session) if skip_completed else {}

                for record in self._iter_records(
                    official_seasons, cricsheet_seasons, known, force_refresh=force_refresh
                ):
                    processed = self.stats.matches_inserted + self.stats.matches_updated
                    if limit is not None and processed >= limit:
                        logger.info("Reached limit of %d matches; stopping", limit)
                        break

                    self._load_record(session, cache, record)

                    processed = self.stats.matches_inserted + self.stats.matches_updated
                    if commit_every and processed and processed % commit_every == 0:
                        session.commit()
                        logger.info(
                            "Committed %d matches (%d deliveries so far)",
                            processed, self.stats.deliveries_inserted,
                        )

            elapsed = time.monotonic() - started
            logger.info("Ingestion finished in %.1fs: %s", elapsed, self.stats.summary())
            logger.info("Validation: %s", self.report.summary())

            finish_ingestion_run(
                run_id,
                status="success",
                message=f"{self.stats.summary()} | validation: {self.report.summary()}",
                **self.stats.as_dict(),
            )
        except Exception as exc:
            logger.exception("Ingestion failed")
            finish_ingestion_run(
                run_id, status="failed", message=f"{type(exc).__name__}: {exc}",
                **self.stats.as_dict(),
            )
            raise

        return self.stats

    # ------------------------------------------------------------------
    # Record production
    # ------------------------------------------------------------------
    def _iter_records(
        self,
        official_seasons: Iterable[int],
        cricsheet_seasons: Sequence[int],
        known: dict,
        *,
        force_refresh: bool,
    ) -> Iterator[MatchRecord]:
        """Yield validated records from both sources, primary source first."""
        for season in official_seasons:
            logger.info("--- Official feed: IPL %s ---", season)
            for record in self.official.iter_season_matches(
                season, force_refresh=force_refresh
            ):
                self.stats.matches_seen += 1
                if self._should_skip(record, known):
                    self.stats.matches_skipped += 1
                    continue
                # Detail feeds are only worth fetching once a match has been
                # played; scheduled fixtures have no scorecard to download.
                if record.is_completed:
                    self.official.enrich_match(record, force_refresh=force_refresh)
                prepared = self._prepare(record)
                if prepared is not None:
                    yield prepared

        if cricsheet_seasons:
            logger.info("--- Cricsheet supplement: %s ---", cricsheet_seasons)
            wanted = set(cricsheet_seasons)
            for record in self.cricsheet.iter_matches(wanted, force_refresh=force_refresh):
                self.stats.matches_seen += 1
                if self._should_skip(record, known):
                    self.stats.matches_skipped += 1
                    continue
                prepared = self._prepare(record)
                if prepared is not None:
                    yield prepared

    @staticmethod
    def _should_skip(record: MatchRecord, known: dict) -> bool:
        """Skip only matches already stored *and* already final.

        A stored-but-unfinished fixture is always re-fetched, which is how a
        match that finished since the last run gets its result and scorecard.
        """
        entry = known.get(record.match_key)
        if entry is None:
            return False
        stored_completed, _ = entry
        return bool(stored_completed)

    def _prepare(self, record: MatchRecord) -> MatchRecord | None:
        """Deduplicate and validate; return ``None`` if the match is rejected."""
        # Placeholder playoff slots ("TBD vs TBD") are published before the
        # league stage decides who fills them. They are not real fixtures, so
        # they are skipped quietly rather than reported as bad data.
        if not record.team1 or not record.team2:
            self.stats.matches_skipped += 1
            logger.debug("Skipping placeholder fixture %s", record.match_key)
            return None

        deduplicate_match(record)
        if not validate_match(record, self.report):
            self.stats.matches_rejected += 1
            return None
        return record

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_record(
        self, session: Session, cache: DimensionCache, record: MatchRecord
    ) -> None:
        """Resolve names to dimension keys and write the match and its children."""
        # Register every player seen in this match up front so the squad feed's
        # richer attributes (role, styles) land on the row even if the player
        # first appears in a scorecard.
        for squad in record.squads:
            cache.player_id(
                squad.player,
                source_player_id=squad.source_player_id,
                batting_style=squad.batting_style,
                bowling_style=squad.bowling_style,
                role=squad.role,
                is_overseas=squad.is_overseas,
                image_url=squad.image_url,
            )

        match_fields = {
            "match_key": record.match_key,
            "source": record.source,
            "source_match_id": record.source_match_id,
            "competition_id": record.competition_id,
            "season": record.season,
            "match_date": record.match_date,
            "start_datetime": record.start_datetime,
            "match_number": record.match_number,
            "stage": record.stage,
            "is_playoff": record.is_playoff,
            "is_day_night": record.is_day_night,
            "overs_per_innings": record.overs_per_innings,
            "venue_id": cache.venue_id(record.venue, city=record.city),
            "city": record.city,
            "is_neutral_venue": record.is_neutral_venue,
            "team1_id": cache.team_id(record.team1),
            "team2_id": cache.team_id(record.team2),
            "home_team_id": cache.team_id(record.home_team),
            "away_team_id": cache.team_id(record.away_team),
            "toss_winner_id": cache.team_id(record.toss_winner),
            "toss_decision": record.toss_decision,
            "first_batting_team_id": cache.team_id(record.first_batting_team),
            "second_batting_team_id": cache.team_id(record.second_batting_team),
            "is_completed": record.is_completed,
            "winner_id": cache.team_id(record.winner),
            "result_type": record.result_type,
            "win_margin_runs": record.win_margin_runs,
            "win_margin_wickets": record.win_margin_wickets,
            "is_tie": record.is_tie,
            "is_no_result": record.is_no_result,
            "is_super_over": record.is_super_over,
            "is_dls_applied": record.is_dls_applied,
            "target_runs": record.target_runs,
            "player_of_match_id": cache.player_id(record.player_of_match),
            "result_summary": record.result_summary,
            "umpire1": record.umpire1,
            "umpire2": record.umpire2,
            "third_umpire": record.third_umpire,
            "match_referee": record.match_referee,
        }

        match, created = upsert_match(session, match_fields)
        if created:
            self.stats.matches_inserted += 1
        else:
            self.stats.matches_updated += 1
            # Children are replaced wholesale rather than diffed: it is simpler,
            # and a re-ingest is always a full re-parse of the same match.
            delete_match_children(session, match.id)

        self._load_children(session, cache, record, match.id)

    def _load_children(
        self, session: Session, cache: DimensionCache, record: MatchRecord, match_id: int
    ) -> None:
        """Insert every child collection for a match."""
        bulk_insert(
            session,
            Innings,
            (
                {
                    "match_id": match_id,
                    "innings_no": i.innings_no,
                    "batting_team_id": cache.team_id(i.batting_team),
                    "bowling_team_id": cache.team_id(i.bowling_team),
                    "runs": i.runs, "wickets": i.wickets, "overs": i.overs,
                    "balls": i.balls, "run_rate": i.run_rate, "extras": i.extras,
                    "byes": i.byes, "leg_byes": i.leg_byes, "wides": i.wides,
                    "no_balls": i.no_balls, "penalty": i.penalty,
                    "powerplay_runs": i.powerplay_runs,
                    "powerplay_wickets": i.powerplay_wickets,
                    "middle_runs": i.middle_runs, "middle_wickets": i.middle_wickets,
                    "death_runs": i.death_runs, "death_wickets": i.death_wickets,
                    "fours": i.fours, "sixes": i.sixes, "dot_balls": i.dot_balls,
                    "target": i.target, "is_declared": i.is_declared,
                }
                for i in record.innings
            ),
        )

        bulk_insert(
            session,
            BattingCard,
            (
                {
                    "match_id": match_id,
                    "innings_no": b.innings_no,
                    "team_id": cache.team_id(b.team),
                    "player_id": cache.player_id(b.player),
                    "batting_position": b.batting_position,
                    "runs": b.runs, "balls": b.balls, "fours": b.fours, "sixes": b.sixes,
                    "dot_balls": b.dot_balls, "strike_rate": b.strike_rate,
                    "is_out": b.is_out, "dismissal_kind": b.dismissal_kind,
                    "dismissal_text": b.dismissal_text,
                    "bowler_id": cache.player_id(b.bowler),
                    "fielder_id": cache.player_id(b.fielder),
                    "wicket_number": b.wicket_number,
                }
                for b in record.batting
            ),
        )

        bulk_insert(
            session,
            BowlingCard,
            (
                {
                    "match_id": match_id,
                    "innings_no": b.innings_no,
                    "team_id": cache.team_id(b.team),
                    "player_id": cache.player_id(b.player),
                    "bowling_order": b.bowling_order,
                    "overs": b.overs, "balls": b.balls, "maidens": b.maidens,
                    "runs_conceded": b.runs_conceded, "wickets": b.wickets,
                    "wides": b.wides, "no_balls": b.no_balls,
                    "dot_balls": b.dot_balls, "economy": b.economy,
                }
                for b in record.bowling
            ),
        )

        bulk_insert(
            session,
            FallOfWicket,
            (
                {
                    "match_id": match_id,
                    "innings_no": f.innings_no,
                    "wicket_no": f.wicket_no,
                    "player_id": cache.player_id(f.player),
                    "team_id": cache.team_id(f.team),
                    "fall_score": f.fall_score,
                    "fall_overs": f.fall_overs,
                }
                for f in record.fall_of_wickets
            ),
        )

        bulk_insert(
            session,
            Partnership,
            (
                {
                    "match_id": match_id,
                    "innings_no": p.innings_no,
                    "wicket_no": p.wicket_no,
                    "team_id": cache.team_id(p.team),
                    "striker_id": cache.player_id(p.striker),
                    "non_striker_id": cache.player_id(p.non_striker),
                    "runs": p.runs, "balls": p.balls,
                    "striker_runs": p.striker_runs, "striker_balls": p.striker_balls,
                    "non_striker_runs": p.non_striker_runs,
                    "non_striker_balls": p.non_striker_balls,
                    "extras": p.extras, "start_over": p.start_over,
                    "end_over": p.end_over, "is_unbroken": p.is_unbroken,
                }
                for p in record.partnerships
            ),
        )

        bulk_insert(
            session,
            MatchPlayer,
            (
                {
                    "match_id": match_id,
                    "team_id": cache.team_id(s.team),
                    "player_id": cache.player_id(s.player),
                    "is_playing_xi": s.is_playing_xi,
                    "is_captain": s.is_captain,
                    "is_wicketkeeper": s.is_wicketkeeper,
                    "is_overseas": s.is_overseas,
                    "is_impact_sub": s.is_impact_sub,
                    "playing_order": s.playing_order,
                    "role": s.role,
                }
                for s in record.squads
            ),
        )

        if self.ingest_deliveries and record.deliveries:
            written = bulk_insert(
                session,
                Delivery,
                (
                    {
                        "match_id": match_id,
                        "innings_no": d.innings_no,
                        "over_no": d.over_no, "ball_no": d.ball_no, "ball_seq": d.ball_seq,
                        "batting_team_id": cache.team_id(d.batting_team),
                        "bowling_team_id": cache.team_id(d.bowling_team),
                        "batter_id": cache.player_id(d.batter),
                        "non_striker_id": cache.player_id(d.non_striker),
                        "bowler_id": cache.player_id(d.bowler),
                        "batter_runs": d.batter_runs, "extra_runs": d.extra_runs,
                        "total_runs": d.total_runs,
                        "is_wide": d.is_wide, "is_no_ball": d.is_no_ball,
                        "is_bye": d.is_bye, "is_leg_bye": d.is_leg_bye,
                        "is_legal": d.is_legal, "is_four": d.is_four, "is_six": d.is_six,
                        "is_wicket": d.is_wicket, "wicket_type": d.wicket_type,
                        "dismissed_player_id": cache.player_id(d.dismissed_player),
                        "cumulative_runs": d.cumulative_runs,
                        "cumulative_wickets": d.cumulative_wickets,
                    }
                    for d in record.deliveries
                ),
            )
            self.stats.deliveries_inserted += written

    def close(self) -> None:
        self.official.close()


def run_ingestion(
    *,
    seasons: Sequence[int] | None = None,
    force_refresh: bool = False,
    skip_completed: bool = True,
    ingest_deliveries: bool | None = None,
    enable_cricsheet: bool | None = None,
    trigger: str = "cli",
    limit: int | None = None,
) -> IngestionStats:
    """Convenience wrapper used by the CLI, the API and the dashboard."""
    from ..config import get_settings

    settings = get_settings()
    pipeline = IngestionPipeline(
        ingest_deliveries=(
            settings.ingest_deliveries if ingest_deliveries is None else ingest_deliveries
        ),
        enable_cricsheet=(
            settings.enable_cricsheet if enable_cricsheet is None else enable_cricsheet
        ),
    )
    try:
        return pipeline.run(
            seasons=seasons,
            force_refresh=force_refresh,
            skip_completed=skip_completed,
            trigger=trigger,
            limit=limit,
        )
    finally:
        pipeline.close()
