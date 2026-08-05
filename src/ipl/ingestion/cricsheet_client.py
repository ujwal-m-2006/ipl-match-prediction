"""Cricsheet client - back-fills the seasons the official feed does not publish.

The iplt20.com feed host only indexes competitions from IPL 2019 onwards. For
2008-2018 this module reads Cricsheet's ball-by-ball JSON archive, which is the
most complete free record of those seasons.

Cricsheet ships **no scorecards** -- only deliveries -- so batting cards,
bowling cards, fall of wickets and partnerships are all *derived* here from the
ball-by-ball stream. That derivation follows standard scoring conventions:

* Balls faced by a batter exclude wides but include no-balls.
* Runs conceded by a bowler exclude byes and leg-byes but include wides and
  no-balls.
* Only bowled / caught / lbw / stumped / caught-and-bowled / hit-wicket are
  credited to the bowler; run-outs and retirements are not.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..config import EXTERNAL_DIR
from ..constants import (
    CRICSHEET_JSON_URL,
    DEATH_OVERS_FROM,
    POWERPLAY_OVERS,
    SOURCE_CRICSHEET,
    TEAM_HOME_VENUES,
)
from ..logging_utils import get_logger
from .http_client import HttpClient
from .normalize import (
    balls_to_overs,
    canonical_player,
    canonical_team,
    canonical_venue,
    clean_text,
    detect_stage,
    make_match_key,
    parse_date,
    run_rate,
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

ARCHIVE_NAME = "ipl_json.zip"

# Dismissal kinds credited to the bowler's wicket column.
BOWLER_CREDITED_WICKETS = frozenset(
    {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
)

# Dismissals that do not put the batter "out" for batting-average purposes.
NON_DISMISSALS = frozenset({"retired hurt", "retired not out"})


@dataclass
class _BatterState:
    """Running tally for one batter while walking the deliveries."""

    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    dots: int = 0
    position: int = 0
    is_out: bool = False
    dismissal_kind: str | None = None
    bowler: str | None = None
    fielder: str | None = None
    wicket_number: int | None = None


@dataclass
class _BowlerState:
    """Running tally for one bowler while walking the deliveries."""

    balls: int = 0
    runs: int = 0
    wickets: int = 0
    wides: int = 0
    no_balls: int = 0
    dots: int = 0
    maidens: int = 0
    order: int = 0
    overs_seen: set[int] = field(default_factory=set)


class CricsheetClient:
    """Reads and parses the Cricsheet IPL ball-by-ball archive."""

    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        archive_path: Path | None = None,
        url: str = CRICSHEET_JSON_URL,
    ) -> None:
        self.http = http or HttpClient()
        self.archive_path = archive_path or (EXTERNAL_DIR / ARCHIVE_NAME)
        self.url = url

    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------
    def ensure_archive(self, *, force_refresh: bool = False) -> Path:
        """Download the archive if it is missing, then return its path."""
        if self.archive_path.exists() and not force_refresh:
            logger.debug("Using cached Cricsheet archive at %s", self.archive_path)
            return self.archive_path

        logger.info("Downloading Cricsheet IPL archive from %s", self.url)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Streamed straight to disk: the archive is a ~5 MB binary, so the
        # text-oriented HttpClient cache is bypassed deliberately here.
        response = self.http.session.get(self.url, timeout=self.http.timeout, stream=True)
        response.raise_for_status()
        with self.archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65_536):
                handle.write(chunk)
        logger.info(
            "Cricsheet archive saved (%.1f MB)", self.archive_path.stat().st_size / 1e6
        )
        return self.archive_path

    def iter_matches(
        self, seasons: set[int] | None = None, *, force_refresh: bool = False
    ) -> Iterator[MatchRecord]:
        """Yield a fully-populated :class:`MatchRecord` per archived match.

        Args:
            seasons: Restrict to these season years. ``None`` yields everything.
            force_refresh: Re-download the archive before reading.
        """
        path = self.ensure_archive(force_refresh=force_refresh)

        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".json")]
            logger.info("Cricsheet archive contains %d matches", len(names))

            for name in sorted(names):
                try:
                    payload = json.loads(archive.read(name))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping unreadable Cricsheet file %s: %s", name, exc)
                    continue

                info = payload.get("info") or {}
                match_date = parse_date((info.get("dates") or [None])[0])
                if match_date is None:
                    continue

                # Cricsheet's `season` field is ambiguous ("2007/08", "2020/21"),
                # but no IPL season has ever crossed a calendar year, so the
                # first match date's year is an unambiguous season key.
                season = match_date.year
                if seasons is not None and season not in seasons:
                    continue

                match_id = Path(name).stem
                try:
                    yield self._parse_match(payload, match_id=match_id, season=season)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed parsing Cricsheet match %s: %s", match_id, exc)

    # ------------------------------------------------------------------
    # Match parsing
    # ------------------------------------------------------------------
    def _parse_match(self, payload: dict, *, match_id: str, season: int) -> MatchRecord:
        """Convert one Cricsheet match document into a :class:`MatchRecord`."""
        info = payload.get("info") or {}

        teams = [canonical_team(t) for t in (info.get("teams") or [])]
        team1 = teams[0] if teams else None
        team2 = teams[1] if len(teams) > 1 else None

        venue, city = canonical_venue(info.get("venue"), info.get("city"))
        toss = info.get("toss") or {}
        toss_winner = canonical_team(toss.get("winner"))
        toss_decision = clean_text(toss.get("decision"))

        event = info.get("event") or {}
        match_number = clean_text(event.get("match_number")) or clean_text(event.get("stage"))
        stage, is_playoff = detect_stage(event.get("stage"), match_number)

        officials = info.get("officials") or {}
        umpires = officials.get("umpires") or []
        referees = officials.get("match_referees") or []
        tv_umpires = officials.get("tv_umpires") or []

        record = MatchRecord(
            match_key=make_match_key(SOURCE_CRICSHEET, season, match_id),
            source=SOURCE_CRICSHEET,
            season=season,
            source_match_id=match_id,
            match_date=parse_date((info.get("dates") or [None])[0]),
            match_number=match_number,
            stage=stage,
            is_playoff=is_playoff,
            overs_per_innings=to_int(info.get("overs"), 20),
            venue=venue,
            city=city,
            team1=team1,
            team2=team2,
            # Cricsheet does not label home/away. IPL convention lists the home
            # side first, which we cross-check against the venue below.
            home_team=team1,
            away_team=team2,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            umpire1=clean_text(umpires[0]) if umpires else None,
            umpire2=clean_text(umpires[1]) if len(umpires) > 1 else None,
            third_umpire=clean_text(tv_umpires[0]) if tv_umpires else None,
            match_referee=clean_text(referees[0]) if referees else None,
            player_of_match=canonical_player((info.get("player_of_match") or [None])[0]),
        )

        self._apply_outcome(record, info.get("outcome") or {})
        record.is_neutral_venue = self._is_neutral(record.home_team, venue)

        self._parse_squads(record, info)
        self._parse_innings(record, payload.get("innings") or [])
        return record

    @staticmethod
    def _apply_outcome(record: MatchRecord, outcome: dict) -> None:
        """Map Cricsheet's ``outcome`` block onto the record's result fields."""
        if not outcome:
            return

        result = clean_text(outcome.get("result"))
        method = clean_text(outcome.get("method"))
        record.is_dls_applied = bool(method and "d/l" in method.lower())

        if result and result.lower() in {"no result", "abandoned"}:
            record.is_no_result = True
            record.result_type = "no result"
            record.result_summary = f"Match abandoned ({result})"
            record.is_completed = True
            return

        if result and result.lower() == "tie":
            record.is_tie = True
            record.result_type = "tie"
            # `eliminator` names the side that won the resulting super over.
            eliminator = canonical_team(outcome.get("eliminator"))
            if eliminator:
                record.is_super_over = True
                record.winner = eliminator
                record.result_summary = f"Match tied ({eliminator} won the Super Over)"
            else:
                record.result_summary = "Match tied"
            record.is_completed = True
            return

        winner = canonical_team(outcome.get("winner"))
        if not winner:
            return

        record.winner = winner
        record.is_completed = True
        by = outcome.get("by") or {}
        if "runs" in by:
            record.result_type = "runs"
            record.win_margin_runs = to_int(by.get("runs"))
            record.result_summary = f"{winner} won by {record.win_margin_runs} runs"
        elif "wickets" in by:
            record.result_type = "wickets"
            record.win_margin_wickets = to_int(by.get("wickets"))
            record.result_summary = f"{winner} won by {record.win_margin_wickets} wickets"
        else:
            record.result_summary = f"{winner} won"

    @staticmethod
    def _is_neutral(home_team: str | None, venue: str | None) -> bool:
        """True when the nominal home side is not at its own ground."""
        if not home_team or not venue:
            return False
        expected = TEAM_HOME_VENUES.get(home_team)
        return bool(expected) and expected != venue

    @staticmethod
    def _parse_squads(record: MatchRecord, info: dict) -> None:
        """Read the per-team Playing XI listed under ``info.players``."""
        registry = (info.get("registry") or {}).get("people") or {}
        for raw_team, players in (info.get("players") or {}).items():
            team = canonical_team(raw_team)
            if not team:
                continue
            for order, raw_name in enumerate(players, start=1):
                player = canonical_player(raw_name)
                if not player:
                    continue
                record.squads.append(
                    SquadRecord(
                        team=team,
                        player=player,
                        is_playing_xi=True,
                        playing_order=order,
                        source_player_id=clean_text(registry.get(raw_name)),
                    )
                )

    # ------------------------------------------------------------------
    # Innings / deliveries
    # ------------------------------------------------------------------
    def _parse_innings(self, record: MatchRecord, innings_blocks: list[dict]) -> None:
        """Walk each innings' deliveries and derive every downstream artefact."""
        # Super-over innings appear as extra blocks; they are recorded on the
        # match flag rather than as innings 3 and 4.
        for index, block in enumerate(innings_blocks[:2], start=1):
            batting_team = canonical_team(block.get("team"))
            bowling_team = self._opponent(record, batting_team)

            deliveries = self._walk_deliveries(block, index, batting_team, bowling_team)
            record.deliveries.extend(deliveries)

            self._build_scorecards(record, block, index, batting_team, bowling_team, deliveries)
            self._build_innings_total(record, block, index, batting_team, bowling_team, deliveries)

        if len(innings_blocks) > 2:
            record.is_super_over = True

    @staticmethod
    def _opponent(record: MatchRecord, team: str | None) -> str | None:
        """Return the other side in the fixture."""
        if team is None:
            return None
        if team == record.team1:
            return record.team2
        if team == record.team2:
            return record.team1
        return None

    def _walk_deliveries(
        self,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
    ) -> list[DeliveryRecord]:
        """Flatten Cricsheet's nested overs into :class:`DeliveryRecord` rows."""
        deliveries: list[DeliveryRecord] = []
        cumulative_runs = 0
        cumulative_wickets = 0
        sequence = 0

        for over_block in block.get("overs") or []:
            # Cricsheet numbers overs from 0; the warehouse uses 1-based overs.
            over_no = (to_int(over_block.get("over"), 0) or 0) + 1
            ball_in_over = 0

            for ball in over_block.get("deliveries") or []:
                runs = ball.get("runs") or {}
                extras = ball.get("extras") or {}

                batter_runs = to_int(runs.get("batter"), 0) or 0
                extra_runs = to_int(runs.get("extras"), 0) or 0
                total_runs = to_int(runs.get("total"), batter_runs + extra_runs) or 0

                is_wide = "wides" in extras
                is_no_ball = "noballs" in extras
                is_legal = not (is_wide or is_no_ball)

                # Byes and leg-byes are debited to the team, not the bowler, so
                # they are excluded from the bowler's analysis. Wides and
                # no-balls are charged in full.
                charged = (
                    batter_runs
                    + (to_int(extras.get("wides"), 0) or 0)
                    + (to_int(extras.get("noballs"), 0) or 0)
                    + (to_int(extras.get("penalty"), 0) or 0)
                )

                ball_in_over += 1
                sequence += 1
                cumulative_runs += total_runs

                wickets = ball.get("wickets") or []
                # A single delivery can dismiss two batters only in freak cases;
                # the first wicket is the one attributed to the ball.
                primary = wickets[0] if wickets else None
                cumulative_wickets += len(wickets)

                deliveries.append(
                    DeliveryRecord(
                        innings_no=innings_no,
                        over_no=over_no,
                        ball_no=ball_in_over,
                        ball_seq=sequence,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        batter=canonical_player(ball.get("batter")),
                        non_striker=canonical_player(ball.get("non_striker")),
                        bowler=canonical_player(ball.get("bowler")),
                        batter_runs=batter_runs,
                        extra_runs=extra_runs,
                        total_runs=total_runs,
                        is_wide=is_wide,
                        is_no_ball=is_no_ball,
                        is_bye="byes" in extras,
                        is_leg_bye="legbyes" in extras,
                        is_legal=is_legal,
                        # Cricsheet does not flag boundaries, so they are
                        # inferred from runs off the bat - the standard
                        # convention (all-run fours are indistinguishable).
                        is_four=batter_runs == 4,
                        is_six=batter_runs == 6,
                        is_wicket=bool(wickets),
                        wicket_type=clean_text(primary.get("kind")) if primary else None,
                        dismissed_player=(
                            canonical_player(primary.get("player_out")) if primary else None
                        ),
                        cumulative_runs=cumulative_runs,
                        cumulative_wickets=cumulative_wickets,
                        bowler_charged_runs=charged,
                    )
                )

        return deliveries

    # -- derived scorecards -------------------------------------------------
    def _build_scorecards(
        self,
        record: MatchRecord,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
        deliveries: list[DeliveryRecord],
    ) -> None:
        """Derive batting cards, bowling cards, FoW and partnerships."""
        batters: dict[str, _BatterState] = {}
        bowlers: dict[str, _BowlerState] = {}
        over_runs: dict[tuple[str, int], int] = defaultdict(int)
        over_legal: dict[tuple[str, int], int] = defaultdict(int)

        wicket_count = 0
        partnership_runs = 0
        partnership_balls = 0
        partnership_start_over: float | None = 0.1
        current_pair: tuple[str | None, str | None] = (None, None)
        pair_contrib: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [runs, balls]
        partnership_extras = 0

        for delivery in deliveries:
            batter = delivery.batter
            bowler = delivery.bowler

            # --- batting ---
            if batter:
                state = batters.setdefault(batter, _BatterState(position=len(batters) + 1))
                state.runs += delivery.batter_runs
                # Wides are not counted as balls faced; no-balls are.
                if not delivery.is_wide:
                    state.balls += 1
                    if delivery.total_runs == 0:
                        state.dots += 1
                state.fours += int(delivery.is_four)
                state.sixes += int(delivery.is_six)

            # Ensure the non-striker is registered so an unbeaten opener who
            # never faced a ball still appears on the card.
            if delivery.non_striker and delivery.non_striker not in batters:
                batters[delivery.non_striker] = _BatterState(position=len(batters) + 1)

            # --- bowling ---
            if bowler:
                bowl = bowlers.setdefault(bowler, _BowlerState(order=len(bowlers) + 1))
                if delivery.is_legal:
                    bowl.balls += 1
                    over_legal[(bowler, delivery.over_no)] += 1

                charged = (
                    delivery.bowler_charged_runs
                    if delivery.bowler_charged_runs is not None
                    else delivery.total_runs
                )
                bowl.runs += charged
                over_runs[(bowler, delivery.over_no)] += charged

                # These columns count extra *runs* conceded, not ball counts.
                if delivery.is_wide:
                    bowl.wides += delivery.extra_runs
                if delivery.is_no_ball:
                    bowl.no_balls += delivery.extra_runs

                if delivery.is_legal and delivery.total_runs == 0:
                    bowl.dots += 1

            # --- partnership accounting ---
            if not delivery.is_wide:
                partnership_balls += 1
            partnership_runs += delivery.total_runs
            partnership_extras += delivery.extra_runs
            if batter:
                pair_contrib[batter][0] += delivery.batter_runs
                if not delivery.is_wide:
                    pair_contrib[batter][1] += 1
            current_pair = (batter, delivery.non_striker)

            # --- wicket ---
            if delivery.is_wicket:
                wicket_count += 1
                dismissed = delivery.dismissed_player or batter
                kind = (delivery.wicket_type or "").lower()

                if dismissed:
                    victim = batters.setdefault(
                        dismissed, _BatterState(position=len(batters) + 1)
                    )
                    victim.is_out = kind not in NON_DISMISSALS
                    victim.dismissal_kind = delivery.wicket_type
                    victim.wicket_number = wicket_count
                    if kind in BOWLER_CREDITED_WICKETS:
                        victim.bowler = bowler

                if bowler and kind in BOWLER_CREDITED_WICKETS:
                    bowlers[bowler].wickets += 1

                record.fall_of_wickets.append(
                    FallOfWicketRecord(
                        innings_no=innings_no,
                        wicket_no=wicket_count,
                        player=dismissed,
                        team=batting_team,
                        fall_score=delivery.cumulative_runs,
                        fall_overs=self._delivery_over_notation(delivery, deliveries),
                    )
                )

                record.partnerships.append(
                    PartnershipRecord(
                        innings_no=innings_no,
                        wicket_no=wicket_count,
                        team=batting_team,
                        striker=current_pair[0],
                        non_striker=current_pair[1],
                        runs=partnership_runs,
                        balls=partnership_balls,
                        striker_runs=pair_contrib[current_pair[0]][0] if current_pair[0] else None,
                        striker_balls=pair_contrib[current_pair[0]][1] if current_pair[0] else None,
                        non_striker_runs=(
                            pair_contrib[current_pair[1]][0] if current_pair[1] else None
                        ),
                        non_striker_balls=(
                            pair_contrib[current_pair[1]][1] if current_pair[1] else None
                        ),
                        extras=partnership_extras,
                        start_over=partnership_start_over,
                        end_over=self._delivery_over_notation(delivery, deliveries),
                        is_unbroken=False,
                    )
                )

                partnership_runs = 0
                partnership_balls = 0
                partnership_extras = 0
                pair_contrib = defaultdict(lambda: [0, 0])
                partnership_start_over = self._delivery_over_notation(delivery, deliveries)

        # An innings that ends without losing a final wicket leaves one
        # partnership unbroken - record it so the totals reconcile.
        if partnership_balls and deliveries:
            record.partnerships.append(
                PartnershipRecord(
                    innings_no=innings_no,
                    wicket_no=wicket_count + 1,
                    team=batting_team,
                    striker=current_pair[0],
                    non_striker=current_pair[1],
                    runs=partnership_runs,
                    balls=partnership_balls,
                    striker_runs=pair_contrib[current_pair[0]][0] if current_pair[0] else None,
                    striker_balls=pair_contrib[current_pair[0]][1] if current_pair[0] else None,
                    non_striker_runs=(
                        pair_contrib[current_pair[1]][0] if current_pair[1] else None
                    ),
                    non_striker_balls=(
                        pair_contrib[current_pair[1]][1] if current_pair[1] else None
                    ),
                    extras=partnership_extras,
                    start_over=partnership_start_over,
                    end_over=self._delivery_over_notation(deliveries[-1], deliveries),
                    is_unbroken=True,
                )
            )

        # A maiden is a completed over in which the bowler conceded nothing.
        for (bowler, over_no), runs in over_runs.items():
            if runs == 0 and over_legal[(bowler, over_no)] >= 6:
                bowlers[bowler].maidens += 1

        for name, state in batters.items():
            record.batting.append(
                BattingRecord(
                    innings_no=innings_no,
                    team=batting_team,
                    player=name,
                    batting_position=state.position,
                    runs=state.runs,
                    balls=state.balls,
                    fours=state.fours,
                    sixes=state.sixes,
                    dot_balls=state.dots,
                    strike_rate=(
                        round(state.runs * 100 / state.balls, 2) if state.balls else None
                    ),
                    is_out=state.is_out,
                    dismissal_kind=state.dismissal_kind or ("not out" if not state.is_out else None),
                    dismissal_text=state.dismissal_kind,
                    bowler=state.bowler,
                    fielder=state.fielder,
                    wicket_number=state.wicket_number,
                )
            )

        for name, bowl in bowlers.items():
            record.bowling.append(
                BowlingRecord(
                    innings_no=innings_no,
                    team=bowling_team,
                    player=name,
                    bowling_order=bowl.order,
                    overs=balls_to_overs(bowl.balls) or 0.0,
                    balls=bowl.balls,
                    maidens=bowl.maidens,
                    runs_conceded=bowl.runs,
                    wickets=bowl.wickets,
                    wides=bowl.wides,
                    no_balls=bowl.no_balls,
                    dot_balls=bowl.dots,
                    economy=(round(bowl.runs * 6 / bowl.balls, 2) if bowl.balls else None),
                )
            )

    @staticmethod
    def _delivery_over_notation(
        delivery: DeliveryRecord, deliveries: list[DeliveryRecord]
    ) -> float:
        """Return the over-notation position of a delivery (e.g. ``15.4``).

        Counts *legal* balls up to and including this one so that the value
        matches how a scorecard reports fall of wickets.
        """
        legal_before = sum(
            1 for d in deliveries if d.ball_seq <= delivery.ball_seq and d.is_legal
        )
        return balls_to_overs(legal_before) or 0.0

    def _build_innings_total(
        self,
        record: MatchRecord,
        block: dict,
        innings_no: int,
        batting_team: str | None,
        bowling_team: str | None,
        deliveries: list[DeliveryRecord],
    ) -> None:
        """Aggregate an innings' totals, extras breakdown and phase splits."""
        if not deliveries:
            return

        legal_balls = sum(1 for d in deliveries if d.is_legal)
        runs = sum(d.total_runs for d in deliveries)
        wickets = sum(1 for d in deliveries if d.is_wicket)

        wides = sum(d.extra_runs for d in deliveries if d.is_wide)
        no_balls = sum(d.extra_runs for d in deliveries if d.is_no_ball)
        byes = sum(d.extra_runs for d in deliveries if d.is_bye)
        leg_byes = sum(d.extra_runs for d in deliveries if d.is_leg_bye)

        phases = {"powerplay": [0, 0], "middle": [0, 0], "death": [0, 0]}
        for delivery in deliveries:
            if delivery.over_no <= POWERPLAY_OVERS:
                key = "powerplay"
            elif delivery.over_no < DEATH_OVERS_FROM:
                key = "middle"
            else:
                key = "death"
            phases[key][0] += delivery.total_runs
            phases[key][1] += int(delivery.is_wicket)

        target = (block.get("target") or {}).get("runs")

        record.innings.append(
            InningsRecord(
                innings_no=innings_no,
                batting_team=batting_team,
                bowling_team=bowling_team,
                runs=runs,
                wickets=wickets,
                overs=balls_to_overs(legal_balls),
                balls=legal_balls,
                run_rate=run_rate(runs, legal_balls),
                extras=sum(d.extra_runs for d in deliveries),
                byes=byes,
                leg_byes=leg_byes,
                wides=wides,
                no_balls=no_balls,
                powerplay_runs=phases["powerplay"][0],
                powerplay_wickets=phases["powerplay"][1],
                middle_runs=phases["middle"][0],
                middle_wickets=phases["middle"][1],
                death_runs=phases["death"][0],
                death_wickets=phases["death"][1],
                fours=sum(1 for d in deliveries if d.is_four),
                sixes=sum(1 for d in deliveries if d.is_six),
                dot_balls=sum(1 for d in deliveries if d.is_legal and d.total_runs == 0),
                target=to_int(target),
            )
        )

        if innings_no == 2 and target is not None:
            record.target_runs = record.target_runs or to_int(target)

    def close(self) -> None:
        self.http.close()
