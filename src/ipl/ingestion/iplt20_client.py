"""Client for the official iplt20.com data feeds.

The IPL website is a JavaScript application that renders itself from a set of
public JSONP feeds hosted on the league's S3 bucket. Those feeds -- not the
rendered HTML -- are the machine-readable form of the same data, so this client
consumes them directly instead of scraping the DOM. Feed shapes were verified
against live responses; see ``docs/DATA_SOURCES.md``.

Endpoints used
--------------
``{comp}-matchschedule.js``
    Every fixture in a competition: teams, toss, venue, result, officials.
``{match}-Innings{n}.js``
    Full innings detail: batting card, bowling card, fall of wickets,
    partnerships, extras and over-by-over (ball-by-ball) history.
``{match}-matchsummary.js``
    Result confirmation, Player of the Match, target and DLS revisions.
``{match}-squad.js``
    Both squads with captain / wicket-keeper / overseas flags.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..constants import (
    DEATH_OVERS_FROM,
    FEED_BASE_URL,
    IPL_COMPETITIONS,
    POWERPLAY_OVERS,
    SOURCE_OFFICIAL,
    TEAM_HOME_VENUES,
)
from ..logging_utils import get_logger
from .http_client import FeedNotFound, HttpClient
from .normalize import (
    balls_to_overs,
    canonical_player,
    canonical_team,
    canonical_venue,
    clean_text,
    detect_stage,
    make_match_key,
    overs_to_balls,
    parse_date,
    parse_datetime,
    parse_dismissal,
    parse_innings_summary,
    parse_player_of_match,
    parse_result,
    parse_toss,
    run_rate,
    season_from_competition_name,
    to_bool,
    to_float,
    to_int,
)
from .records import (
    BattingRecord,
    BowlingRecord,
    DeliveryRecord,
    FallOfWicketRecord,
    InningsRecord,
    MatchRecord,
    PartnershipRecord,
    SquadRecord,
)

logger = get_logger(__name__)

# Feed rows with this ball number are over-break sentinels, not deliveries.
_OVER_BREAK_BALL_NO = "99"

# `MatchStatus` values seen in the schedule feed.
_STATUS_COMPLETED = "post"

# The maximum number of innings a T20 fixture can expose (2 + super over pair).
_MAX_INNINGS = 2


class IPLT20Client:
    """Fetches and parses IPL data from the official public feeds."""

    def __init__(self, http: HttpClient | None = None, base_url: str = FEED_BASE_URL) -> None:
        self.http = http or HttpClient()
        self.base_url = base_url.rstrip("/")

    # -- URL builders -------------------------------------------------------
    def _schedule_url(self, competition_id: int) -> str:
        return f"{self.base_url}/{competition_id}-matchschedule.js"

    def _innings_url(self, match_id: Any, innings_no: int) -> str:
        return f"{self.base_url}/{match_id}-Innings{innings_no}.js"

    def _summary_url(self, match_id: Any) -> str:
        return f"{self.base_url}/{match_id}-matchsummary.js"

    def _squad_url(self, match_id: Any) -> str:
        return f"{self.base_url}/{match_id}-squad.js"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_competitions(
        self, id_range: range, *, keyword: str = "ipl"
    ) -> dict[int, int]:
        """Probe competition IDs and return ``{season: competition_id}``.

        The feed host indexes every tournament it has ever served (national
        tours, the WPL, ICC events) under a flat integer namespace, so the IPL
        seasons must be identified by name. Used by
        ``scripts/discover_competitions.py`` to pick up newly published
        seasons without a code change.
        """
        found: dict[int, int] = {}
        for competition_id in id_range:
            try:
                payload = self.http.get_jsonp(self._schedule_url(competition_id))
            except (FeedNotFound, ValueError):
                continue
            except Exception as exc:  # pragma: no cover - network flake
                logger.debug("Probe failed for competition %s: %s", competition_id, exc)
                continue

            rows = payload.get("Matchsummary") or []
            if not rows:
                continue
            name = clean_text(rows[0].get("CompetitionName")) or ""
            if keyword.lower() not in name.lower():
                continue
            season = season_from_competition_name(name)
            if season:
                found[season] = competition_id
                logger.info("Discovered %s -> competition %s", name, competition_id)
        return found

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    def fetch_schedule(self, competition_id: int, *, force_refresh: bool = False) -> list[dict]:
        """Return the raw schedule rows for a competition."""
        payload = self.http.get_jsonp(
            self._schedule_url(competition_id), force_refresh=force_refresh
        )
        rows = payload.get("Matchsummary") or []
        logger.info("Competition %s: %d fixtures in schedule", competition_id, len(rows))
        return rows

    def iter_season_matches(
        self,
        season: int,
        *,
        competition_id: int | None = None,
        force_refresh: bool = False,
    ) -> Iterator[MatchRecord]:
        """Yield a header-only :class:`MatchRecord` for every fixture in a season.

        Detail (scorecards, ball-by-ball, squads) is *not* fetched here -- the
        pipeline decides which matches need it, which is what keeps an
        incremental refresh cheap.
        """
        comp_id = competition_id or IPL_COMPETITIONS.get(season)
        if comp_id is None:
            logger.warning("No official competition ID registered for season %s", season)
            return

        for row in self.fetch_schedule(comp_id, force_refresh=force_refresh):
            try:
                record = self._parse_schedule_row(row, season=season, competition_id=comp_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed parsing fixture %s: %s", row.get("MatchID"), exc)
                continue
            if record is not None:
                yield record

    # ------------------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------------------
    def _parse_schedule_row(
        self, row: dict, *, season: int, competition_id: int
    ) -> MatchRecord | None:
        """Turn one schedule row into a header-only :class:`MatchRecord`."""
        match_id = row.get("MatchID")
        if match_id in (None, ""):
            return None

        # The feed's own competition name is more reliable than our registry
        # when a season is re-published under a new ID.
        feed_season = season_from_competition_name(row.get("CompetitionName")) or season

        venue, city = canonical_venue(row.get("GroundName"), row.get("city"))
        home = canonical_team(row.get("HomeTeamName"))
        away = canonical_team(row.get("AwayTeamName"))
        first_bat = canonical_team(row.get("FirstBattingTeamName"))
        second_bat = canonical_team(row.get("SecondBattingTeamName"))

        # `MatchName` is "<A> vs <B>" and is the only team source for fixtures
        # whose home/away fields are blank (occasionally true for playoffs).
        team1, team2 = home, away
        if not team1 or not team2:
            parsed = self._teams_from_match_name(row.get("MatchName"))
            team1 = team1 or parsed[0] or first_bat
            team2 = team2 or parsed[1] or second_bat

        status = (clean_text(row.get("MatchStatus")) or "").lower()
        result = parse_result(row.get("Comments") or row.get("Commentss"))
        toss_winner, toss_decision = parse_toss(row.get("TossDetails"), row.get("TossTeam"))
        stage, is_playoff = detect_stage(row.get("MatchOrder"), row.get("MatchName"))
        # ROUND_ID is 0 for league fixtures and a stage code for playoffs; it is
        # a useful cross-check when MatchOrder is blank in older seasons.
        if not is_playoff and to_int(row.get("ROUND_ID"), 0):
            stage, is_playoff = stage or "Playoff", True

        is_completed = status == _STATUS_COMPLETED and bool(
            result["winner"] or result["is_tie"] or result["is_no_result"]
        )

        match_key = make_match_key(SOURCE_OFFICIAL, feed_season, match_id)
        record = MatchRecord(
            match_key=match_key,
            source=SOURCE_OFFICIAL,
            season=feed_season,
            source_match_id=str(match_id),
            competition_id=competition_id,
            match_date=parse_date(row.get("MatchDate")),
            start_datetime=parse_datetime(row.get("MATCH_COMMENCE_START_DATE")),
            match_number=clean_text(row.get("MatchOrder")) or clean_text(row.get("RowNo")),
            stage=stage,
            is_playoff=is_playoff,
            # "T20 (N)" / "T20 (D/N)" denote a night fixture.
            is_day_night="(n)" in (clean_text(row.get("MatchType")) or "").lower()
            or "d/n" in (clean_text(row.get("MatchType")) or "").lower(),
            overs_per_innings=to_int(row.get("MATCH_NO_OF_OVERS"), 20),
            venue=venue,
            city=city,
            team1=team1,
            team2=team2,
            home_team=home,
            away_team=away,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            first_batting_team=first_bat,
            second_batting_team=second_bat,
            is_completed=is_completed,
            umpire1=clean_text(row.get("GroundUmpire1")),
            umpire2=clean_text(row.get("GroundUmpire2")),
            third_umpire=clean_text(row.get("ThirdUmpire")),
            **{k: v for k, v in result.items() if k != "winner"},
        )
        record.winner = result["winner"]
        record.is_neutral_venue = self._is_neutral(home, venue)

        # Innings totals are already summarised on the schedule row, so a
        # header-only ingest still yields usable scores.
        for innings_no, (team, other, summary_key) in enumerate(
            (
                (first_bat, second_bat, "FirstBattingSummary"),
                (second_bat, first_bat, "SecondBattingSummary"),
            ),
            start=1,
        ):
            parsed = parse_innings_summary(row.get(summary_key))
            # A fixture abandoned before the toss has neither a score nor a
            # named batting side; there is no innings to record.
            if parsed["runs"] is None or team is None:
                continue
            record.innings.append(
                InningsRecord(
                    innings_no=innings_no,
                    batting_team=team,
                    bowling_team=other,
                    runs=parsed["runs"],
                    wickets=parsed["wickets"],
                    overs=parsed["overs"],
                    balls=parsed["balls"],
                    run_rate=run_rate(parsed["runs"], parsed["balls"]),
                )
            )
        return record

    @staticmethod
    def _teams_from_match_name(match_name: Any) -> tuple[str | None, str | None]:
        """Split ``"Team A vs Team B"`` into canonical names."""
        text = clean_text(match_name)
        if not text:
            return None, None
        for separator in (" vs ", " Vs ", " VS ", " v "):
            if separator in text:
                left, _, right = text.partition(separator)
                return canonical_team(left), canonical_team(right)
        return None, None

    @staticmethod
    def _is_neutral(home_team: str | None, venue: str | None) -> bool:
        """True when the nominal home side is not playing at its own ground.

        IPL 2009 (South Africa), 2014 and 2021 (partly UAE) and 2020 (UAE) were
        played at neutral venues; treating those as home advantage would inject
        a systematic bias into the model.
        """
        if not home_team or not venue:
            return False
        expected = TEAM_HOME_VENUES.get(home_team)
        return bool(expected) and expected != venue

    # ------------------------------------------------------------------
    # Detail parsing
    # ------------------------------------------------------------------
    def enrich_match(self, record: MatchRecord, *, force_refresh: bool = False) -> MatchRecord:
        """Fetch and attach scorecards, ball-by-ball data and squads.

        Mutates and returns ``record``. Missing sub-feeds are tolerated: a
        match with no squad feed simply has no squad rows.
        """
        match_id = record.source_match_id
        if not match_id:
            return record

        self._apply_summary(record, match_id, force_refresh=force_refresh)
        for innings_no in range(1, _MAX_INNINGS + 1):
            self._apply_innings(record, match_id, innings_no, force_refresh=force_refresh)
        self._apply_squads(record, match_id, force_refresh=force_refresh)
        return record

    def _apply_summary(self, record: MatchRecord, match_id: str, *, force_refresh: bool) -> None:
        """Overlay the match-summary feed (MOM, target, DLS, officials)."""
        try:
            payload = self.http.get_jsonp(self._summary_url(match_id), force_refresh=force_refresh)
        except (FeedNotFound, ValueError):
            return
        rows = payload.get("MatchSummary") or []
        if not rows:
            return
        row = rows[0]

        record.player_of_match = parse_player_of_match(row.get("MOM")) or record.player_of_match
        record.target_runs = to_int(row.get("Target")) or record.target_runs
        record.match_referee = clean_text(row.get("Referee")) or record.match_referee
        record.umpire1 = clean_text(row.get("GroundUmpire1")) or record.umpire1
        record.umpire2 = clean_text(row.get("GroundUmpire2")) or record.umpire2
        record.third_umpire = clean_text(row.get("ThirdUmpire")) or record.third_umpire
        record.is_super_over = record.is_super_over or to_bool(row.get("IsSuperOver"))

        # A revised over count or target means Duckworth-Lewis-Stern was applied.
        if clean_text(row.get("RevisedOver")) or clean_text(row.get("RevisedTarget")):
            record.is_dls_applied = True

        # The summary feed confirms the result for matches whose schedule row
        # was written before the finish (e.g. a same-day refresh).
        if not record.is_completed and to_bool(row.get("IsMatchEnd")):
            outcome = parse_result(row.get("Comments"))
            if outcome["winner"] or outcome["is_tie"] or outcome["is_no_result"]:
                record.winner = outcome["winner"] or record.winner
                record.result_type = outcome["result_type"] or record.result_type
                record.win_margin_runs = outcome["win_margin_runs"]
                record.win_margin_wickets = outcome["win_margin_wickets"]
                record.is_tie = outcome["is_tie"]
                record.is_no_result = outcome["is_no_result"]
                record.result_summary = outcome["result_summary"] or record.result_summary
                record.is_completed = True

    def _apply_innings(
        self, record: MatchRecord, match_id: str, innings_no: int, *, force_refresh: bool
    ) -> None:
        """Parse one innings feed into cards, FoW, partnerships and deliveries."""
        try:
            payload = self.http.get_jsonp(
                self._innings_url(match_id, innings_no), force_refresh=force_refresh
            )
        except (FeedNotFound, ValueError):
            return

        block = payload.get(f"Innings{innings_no}")
        if not isinstance(block, dict):
            return

        batting_team, bowling_team = self._innings_teams(record, innings_no, block)

        self._parse_batting_card(record, block, innings_no, batting_team, bowling_team)
        self._parse_bowling_card(record, block, innings_no, bowling_team)
        self._parse_fall_of_wickets(record, block, innings_no, batting_team)
        self._parse_partnerships(record, block, innings_no, batting_team)
        deliveries = self._parse_over_history(record, block, innings_no, batting_team, bowling_team)
        self._merge_innings_totals(record, block, innings_no, batting_team, bowling_team, deliveries)

    @staticmethod
    def _innings_teams(
        record: MatchRecord, innings_no: int, block: dict
    ) -> tuple[str | None, str | None]:
        """Resolve which side batted in this innings.

        The Extras block names both teams explicitly, which is more reliable
        than assuming innings 1 == first batting team (super overs break that).
        """
        extras_rows = block.get("Extras") or []
        if extras_rows:
            batting = canonical_team(extras_rows[0].get("BattingTeamName"))
            bowling = canonical_team(extras_rows[0].get("BowlingTeamName"))
            if batting and bowling:
                return batting, bowling

        if innings_no == 1:
            return record.first_batting_team, record.second_batting_team
        return record.second_batting_team, record.first_batting_team

    # -- cards --------------------------------------------------------------
    def _parse_batting_card(
        self,
        record: MatchRecord,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
    ) -> None:
        for row in block.get("BattingCard") or []:
            player = canonical_player(row.get("PlayerName"))
            if not player:
                continue
            dismissal = parse_dismissal(row.get("OutDesc"))
            record.batting.append(
                BattingRecord(
                    innings_no=innings_no,
                    team=batting_team,
                    player=player,
                    batting_position=to_int(row.get("PlayingOrder")),
                    runs=to_int(row.get("Runs"), 0) or 0,
                    balls=to_int(row.get("Balls"), 0) or 0,
                    fours=to_int(row.get("Fours"), 0) or 0,
                    sixes=to_int(row.get("Sixes"), 0) or 0,
                    dot_balls=to_int(row.get("DotBalls")),
                    strike_rate=to_float(row.get("StrikeRate")),
                    is_out=dismissal["is_out"],
                    dismissal_kind=dismissal["dismissal_kind"],
                    dismissal_text=dismissal["dismissal_text"],
                    # The dedicated BowlerName field is authoritative; fall back
                    # to the name parsed out of the dismissal string.
                    bowler=canonical_player(row.get("BowlerName")) or dismissal["bowler"],
                    fielder=dismissal["fielder"],
                    wicket_number=to_int(row.get("WicketNo")),
                )
            )

    def _parse_bowling_card(
        self, record: MatchRecord, block: dict, innings_no: int, bowling_team: str | None
    ) -> None:
        for row in block.get("BowlingCard") or []:
            player = canonical_player(row.get("PlayerName"))
            if not player:
                continue
            overs = to_float(row.get("Overs"), 0.0) or 0.0
            balls = to_int(row.get("TotalLegalBallsBowled")) or overs_to_balls(overs)
            record.bowling.append(
                BowlingRecord(
                    innings_no=innings_no,
                    team=bowling_team,
                    player=player,
                    bowling_order=to_int(row.get("BowlingOrder")),
                    overs=overs,
                    balls=balls,
                    maidens=to_int(row.get("Maidens"), 0) or 0,
                    runs_conceded=to_int(row.get("Runs"), 0) or 0,
                    wickets=to_int(row.get("Wickets"), 0) or 0,
                    wides=to_int(row.get("Wides"), 0) or 0,
                    no_balls=to_int(row.get("NoBalls"), 0) or 0,
                    dot_balls=to_int(row.get("DotBalls")),
                    economy=to_float(row.get("Economy")),
                )
            )

    def _parse_fall_of_wickets(
        self, record: MatchRecord, block: dict, innings_no: int, batting_team: str | None
    ) -> None:
        for row in block.get("FallOfWickets") or []:
            wicket_no = to_int(row.get("FallWickets"))
            if wicket_no is None:
                continue
            record.fall_of_wickets.append(
                FallOfWicketRecord(
                    innings_no=innings_no,
                    wicket_no=wicket_no,
                    player=canonical_player(row.get("PlayerName")),
                    team=batting_team,
                    fall_score=to_int(row.get("FallScore")),
                    fall_overs=to_float(row.get("FallOvers")),
                )
            )

    def _parse_partnerships(
        self, record: MatchRecord, block: dict, innings_no: int, batting_team: str | None
    ) -> None:
        """Parse partnerships, marking the last one unbroken when appropriate."""
        rows = block.get("PartnershipScores") or []
        # PartnershipBreak has one row per *completed* partnership, so a final
        # partnership with no matching break row was still in progress at the end.
        completed = len(block.get("PartnershipBreak") or [])

        for index, row in enumerate(rows, start=1):
            wicket_no = to_int(row.get("RowNumber"), index) or index
            start_over = to_float(row.get("MatchMinOver"))
            end_over = to_float(row.get("MatchMaxOver"))
            striker_balls = to_int(row.get("StrikerBalls"), 0) or 0
            non_striker_balls = to_int(row.get("NonStrikerBalls"), 0) or 0
            record.partnerships.append(
                PartnershipRecord(
                    innings_no=innings_no,
                    wicket_no=wicket_no,
                    team=batting_team,
                    striker=canonical_player(row.get("Striker")),
                    non_striker=canonical_player(row.get("NonStriker")),
                    runs=to_int(row.get("PartnershipTotal")),
                    balls=striker_balls + non_striker_balls,
                    striker_runs=to_int(row.get("StrikerRuns")),
                    striker_balls=striker_balls,
                    non_striker_runs=to_int(row.get("NonStrikerRuns")),
                    non_striker_balls=non_striker_balls,
                    extras=to_int(row.get("Extras")),
                    start_over=start_over,
                    end_over=end_over,
                    is_unbroken=index > completed,
                )
            )

    # -- ball-by-ball -------------------------------------------------------
    def _parse_over_history(
        self,
        record: MatchRecord,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
    ) -> list[DeliveryRecord]:
        """Parse ``OverHistory`` into :class:`DeliveryRecord` objects.

        Two feed quirks are handled here:

        * Each over is terminated by a sentinel row with ``BallNo == "99"`` and
          every other field blank. Those are over-break markers, not balls, and
          are dropped -- leaving them in inflates the legal-ball count.
        * ``BallRuns`` is the string ``"W"`` on a wicket, so run totals are
          rebuilt from ``ActualRuns`` (off the bat) plus ``Extras``. This
          reconciles exactly with the scorecard total.
        """
        rows = block.get("OverHistory") or []
        deliveries: list[DeliveryRecord] = []
        cumulative_runs = 0
        cumulative_wickets = 0
        sequence = 0

        for row in rows:
            ball_no_raw = clean_text(row.get("BallNo"))
            if ball_no_raw == _OVER_BREAK_BALL_NO or not clean_text(row.get("ActualBallNo")):
                continue

            over_no = to_int(row.get("OverNo"))
            if over_no is None:
                continue

            batter_runs = to_int(row.get("ActualRuns"), 0) or 0
            extra_runs = to_int(row.get("Extras"), 0) or 0
            total_runs = batter_runs + extra_runs

            is_wide = to_bool(row.get("IsWide"))
            is_no_ball = to_bool(row.get("IsNoBall"))
            is_wicket = to_bool(row.get("IsWicket"))

            cumulative_runs += total_runs
            cumulative_wickets += int(is_wicket)
            sequence += 1

            deliveries.append(
                DeliveryRecord(
                    innings_no=innings_no,
                    over_no=over_no,
                    ball_no=to_int(row.get("ActualBallNo"), 0) or 0,
                    ball_seq=sequence,
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    batter=canonical_player(row.get("BatsManName")),
                    bowler=canonical_player(row.get("BowlerName")),
                    batter_runs=batter_runs,
                    extra_runs=extra_runs,
                    total_runs=total_runs,
                    is_wide=is_wide,
                    is_no_ball=is_no_ball,
                    is_bye=to_bool(row.get("IsBye")),
                    is_leg_bye=to_bool(row.get("IsLegBye")),
                    # Wides and no-balls do not count towards the over.
                    is_legal=not (is_wide or is_no_ball),
                    is_four=to_bool(row.get("IsFour")),
                    is_six=to_bool(row.get("IsSix")),
                    is_wicket=is_wicket,
                    wicket_type=clean_text(row.get("WicketType")),
                    cumulative_runs=cumulative_runs,
                    cumulative_wickets=cumulative_wickets,
                )
            )

        record.deliveries.extend(deliveries)
        return deliveries

    # -- innings totals -----------------------------------------------------
    def _merge_innings_totals(
        self,
        record: MatchRecord,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
        deliveries: list[DeliveryRecord],
    ) -> None:
        """Reconcile innings totals from the Extras block and ball-by-ball data.

        The scorecard's Extras row is treated as authoritative for the headline
        total; phase splits and boundary counts are derived from deliveries.
        """
        existing = next((i for i in record.innings if i.innings_no == innings_no), None)
        if existing is None:
            existing = InningsRecord(
                innings_no=innings_no, batting_team=batting_team, bowling_team=bowling_team
            )
            record.innings.append(existing)

        existing.batting_team = batting_team or existing.batting_team
        existing.bowling_team = bowling_team or existing.bowling_team

        extras_rows = block.get("Extras") or []
        if extras_rows:
            extras = extras_rows[0]
            parsed = parse_innings_summary(extras.get("Total"))
            existing.runs = parsed["runs"] if parsed["runs"] is not None else existing.runs
            existing.wickets = to_int(extras.get("FallWickets"), existing.wickets)
            existing.overs = to_float(extras.get("FallOvers"), existing.overs)
            existing.balls = overs_to_balls(existing.overs) or existing.balls
            existing.extras = to_int(extras.get("TotalExtras"), existing.extras)
            existing.byes = to_int(extras.get("Byes"))
            existing.leg_byes = to_int(extras.get("LegByes"))
            existing.wides = to_int(extras.get("Wides"))
            existing.no_balls = to_int(extras.get("NoBalls"))
            existing.penalty = to_int(extras.get("Penalty"))
            existing.run_rate = to_float(extras.get("CurrentRunRate")) or run_rate(
                existing.runs, existing.balls
            )

        if not deliveries:
            return

        legal_balls = sum(1 for d in deliveries if d.is_legal)
        # Prefer the derived ball count when the scorecard did not supply one.
        if existing.balls is None and legal_balls:
            existing.balls = legal_balls
            existing.overs = balls_to_overs(legal_balls)

        existing.fours = sum(1 for d in deliveries if d.is_four)
        existing.sixes = sum(1 for d in deliveries if d.is_six)
        existing.dot_balls = sum(1 for d in deliveries if d.is_legal and d.total_runs == 0)

        phases = self._phase_totals(deliveries)
        existing.powerplay_runs = phases["powerplay"][0]
        existing.powerplay_wickets = phases["powerplay"][1]
        existing.middle_runs = phases["middle"][0]
        existing.middle_wickets = phases["middle"][1]
        existing.death_runs = phases["death"][0]
        existing.death_wickets = phases["death"][1]

        # Cross-check ball-by-ball against the scorecard; a mismatch signals a
        # partial feed and is worth surfacing rather than silently trusting.
        derived_runs = sum(d.total_runs for d in deliveries)
        if existing.runs is not None and derived_runs != existing.runs:
            logger.debug(
                "Innings %s of %s: ball-by-ball total %d != scorecard %d",
                innings_no, record.match_key, derived_runs, existing.runs,
            )

    @staticmethod
    def _phase_totals(deliveries: list[DeliveryRecord]) -> dict[str, tuple[int, int]]:
        """Split runs/wickets into powerplay, middle and death phases."""
        totals = {"powerplay": [0, 0], "middle": [0, 0], "death": [0, 0]}
        for delivery in deliveries:
            if delivery.over_no <= POWERPLAY_OVERS:
                phase = "powerplay"
            elif delivery.over_no < DEATH_OVERS_FROM:
                phase = "middle"
            else:
                phase = "death"
            totals[phase][0] += delivery.total_runs
            totals[phase][1] += int(delivery.is_wicket)
        return {key: (value[0], value[1]) for key, value in totals.items()}

    # -- squads -------------------------------------------------------------
    def _apply_squads(self, record: MatchRecord, match_id: str, *, force_refresh: bool) -> None:
        """Attach both Playing XIs from the squad feed."""
        try:
            payload = self.http.get_jsonp(self._squad_url(match_id), force_refresh=force_refresh)
        except (FeedNotFound, ValueError):
            return

        for key in ("squadA", "squadB"):
            for row in payload.get(key) or []:
                player = canonical_player(row.get("PlayerName")) or canonical_player(
                    row.get("PlayerShortName")
                )
                team = canonical_team(row.get("TeamName"))
                if not player or not team:
                    continue
                order = to_int(row.get("PlayingOrder"))
                record.squads.append(
                    SquadRecord(
                        team=team,
                        player=player,
                        # The feed lists 16 players; the XI are orders 1-11 and
                        # anyone beyond that is a bench/impact option.
                        is_playing_xi=order is not None and order <= 11,
                        is_captain=to_bool(row.get("IsCaptain")),
                        is_wicketkeeper=to_bool(row.get("IsWK")),
                        is_overseas=to_bool(row.get("IsNonDomestic")),
                        is_impact_sub=order is not None and order == 12,
                        playing_order=order,
                        role=clean_text(row.get("PlayerSkill")),
                        batting_style=clean_text(row.get("BattingType")),
                        bowling_style=clean_text(row.get("BowlingProficiency")),
                        source_player_id=clean_text(row.get("PlayerID")),
                        image_url=clean_text(row.get("PlayerImage")),
                    )
                )

    def close(self) -> None:
        self.http.close()
