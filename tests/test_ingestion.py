"""Tests for the ingestion clients, validation and loader.

Feed parsing is exercised against recorded payloads that mirror the real feed
structure, so these run offline. The one test that touches the network is
marked ``network`` and can be deselected with ``-m "not network"``.
"""

from __future__ import annotations

import json

import pytest

from ipl.ingestion.http_client import FeedNotFound, unwrap_jsonp
from ipl.ingestion.iplt20_client import IPLT20Client
from ipl.ingestion.records import MatchRecord
from ipl.ingestion.validation import (
    ValidationReport,
    deduplicate_match,
    validate_match,
)


# ---------------------------------------------------------------------------
# JSONP unwrapping
# ---------------------------------------------------------------------------
class TestUnwrapJsonp:
    def test_strips_the_callback(self):
        assert unwrap_jsonp('MatchSchedule({"a": 1})') == {"a": 1}

    def test_handles_nested_parentheses_in_strings(self):
        payload = 'onScoring({"note": "Kohli (c) batting"})'
        assert unwrap_jsonp(payload)["note"] == "Kohli (c) batting"

    def test_accepts_plain_json(self):
        assert unwrap_jsonp('{"a": 1}') == {"a": 1}

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            unwrap_jsonp("")

    def test_rejects_malformed(self):
        with pytest.raises(ValueError):
            unwrap_jsonp("callback({not json})")


# ---------------------------------------------------------------------------
# Official feed parsing
# ---------------------------------------------------------------------------
SCHEDULE_ROW = {
    "MatchID": 1872,
    "CompetitionID": 203,
    "CompetitionName": "Tata Ipl 2025",
    "MatchDate": "2025-06-03",
    "MATCH_COMMENCE_START_DATE": "2025-06-03 19:30:00",
    "MatchName": "Royal Challengers Bengaluru vs Punjab Kings",
    "MatchType": "T20 (N)",
    "MatchStatus": "Post",
    "MatchOrder": "Final",
    "ROUND_ID": "11",
    "MATCH_NO_OF_OVERS": "20",
    "GroundName": "Narendra Modi Stadium",
    "city": "Ahmedabad",
    "HomeTeamName": "Royal Challengers Bengaluru",
    "AwayTeamName": "Punjab Kings",
    "FirstBattingTeamName": "Royal Challengers Bengaluru",
    "SecondBattingTeamName": "Punjab Kings",
    "TossTeam": "Punjab Kings",
    "TossDetails": "Punjab Kings Won The Toss And Elected To Field",
    "Comments": "Royal Challengers Bengaluru Won by 6 Runs (Winners)",
    "FirstBattingSummary": "190/9 (20.0 Ov)",
    "SecondBattingSummary": "184/7 (20.0 Ov)",
    "GroundUmpire1": "Nitin Menon",
    "GroundUmpire2": "J Madanagopal",
    "ThirdUmpire": "Chris Gaffaney",
    "RowNo": 74,
}


class TestScheduleParsing:
    @pytest.fixture()
    def record(self) -> MatchRecord:
        client = IPLT20Client()
        return client._parse_schedule_row(SCHEDULE_ROW, season=2025, competition_id=203)

    def test_identity(self, record):
        assert record.match_key == "iplt20:2025:1872"
        assert record.source == "iplt20"
        assert record.season == 2025

    def test_teams_are_canonicalised(self, record):
        assert record.team1 == "Royal Challengers Bengaluru"
        assert record.team2 == "Punjab Kings"

    def test_venue_and_city(self, record):
        assert record.venue == "Narendra Modi Stadium"
        assert record.city == "Ahmedabad"

    def test_toss(self, record):
        assert record.toss_winner == "Punjab Kings"
        assert record.toss_decision == "field"

    def test_result(self, record):
        assert record.is_completed is True
        assert record.winner == "Royal Challengers Bengaluru"
        assert record.result_type == "runs"
        assert record.win_margin_runs == 6

    def test_stage(self, record):
        assert record.stage == "Final"
        assert record.is_playoff is True

    def test_innings_from_the_summary_strings(self, record):
        assert len(record.innings) == 2
        first, second = record.innings
        assert (first.runs, first.wickets, first.balls) == (190, 9, 120)
        assert (second.runs, second.wickets) == (184, 7)

    def test_neutral_venue_detected(self, record):
        # RCB's home ground is Chinnaswamy, so a final at Ahmedabad is neutral.
        assert record.is_neutral_venue is True

    def test_day_night(self, record):
        assert record.is_day_night is True

    def test_placeholder_fixture_yields_no_teams(self):
        client = IPLT20Client()
        row = {**SCHEDULE_ROW, "MatchID": 999, "HomeTeamName": "TBD",
               "AwayTeamName": "TBD", "MatchName": "TBD vs TBD",
               "FirstBattingTeamName": "", "SecondBattingTeamName": "",
               "Comments": "", "MatchStatus": "Pre"}
        record = client._parse_schedule_row(row, season=2025, competition_id=203)
        assert record.team1 is None or record.team2 is None


class TestOverHistoryParsing:
    """The over-break sentinel rows must not be counted as deliveries."""

    def _block(self) -> dict:
        deliveries = []
        for ball in range(1, 7):
            deliveries.append(
                {
                    "OverNo": 1, "BallNo": str(ball), "ActualBallNo": str(ball),
                    "ActualRuns": "1", "Extras": "0", "BallRuns": "1",
                    "IsWide": "0", "IsNoBall": "0", "IsBye": "0", "IsLegBye": "0",
                    "IsFour": "0", "IsSix": "0", "IsWicket": "0", "WicketType": "",
                    "BatsManName": "Batter A", "BowlerName": "Bowler A",
                }
            )
        # The sentinel the real feed appends at the end of every over.
        deliveries.append(
            {
                "OverNo": 1, "BallNo": "99", "ActualBallNo": "", "ActualRuns": "",
                "Extras": "", "BallRuns": "", "IsWide": "", "IsNoBall": "",
                "IsWicket": "", "BatsManName": "", "BowlerName": "",
            }
        )
        return {"OverHistory": deliveries, "Extras": []}

    def test_sentinel_rows_are_dropped(self):
        client = IPLT20Client()
        record = MatchRecord(match_key="k", source="iplt20", season=2025)
        deliveries = client._parse_over_history(
            record, self._block(), 1, "Team A", "Team B"
        )
        assert len(deliveries) == 6
        assert all(d.is_legal for d in deliveries)

    def test_runs_are_rebuilt_from_batter_runs_plus_extras(self):
        """`BallRuns` is the string "W" on a wicket, so it cannot be summed."""
        client = IPLT20Client()
        block = self._block()
        block["OverHistory"][2].update(
            {"BallRuns": "W", "ActualRuns": "0", "IsWicket": "1", "WicketType": "Caught"}
        )
        record = MatchRecord(match_key="k", source="iplt20", season=2025)
        deliveries = client._parse_over_history(record, block, 1, "Team A", "Team B")

        assert sum(d.total_runs for d in deliveries) == 5
        assert sum(d.is_wicket for d in deliveries) == 1

    def test_wides_are_not_legal_deliveries(self):
        client = IPLT20Client()
        block = self._block()
        block["OverHistory"][0].update(
            {"IsWide": "1", "Extras": "1", "ActualRuns": "0"}
        )
        record = MatchRecord(match_key="k", source="iplt20", season=2025)
        deliveries = client._parse_over_history(record, block, 1, "Team A", "Team B")
        assert sum(d.is_legal for d in deliveries) == 5

    def test_cumulative_state_is_monotonic(self):
        client = IPLT20Client()
        record = MatchRecord(match_key="k", source="iplt20", season=2025)
        deliveries = client._parse_over_history(
            record, self._block(), 1, "Team A", "Team B"
        )
        runs = [d.cumulative_runs for d in deliveries]
        assert runs == sorted(runs)
        assert runs[-1] == 6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _valid_record() -> MatchRecord:
    return MatchRecord(
        match_key="test:2025:1",
        source="test",
        season=2025,
        team1="Chennai Super Kings",
        team2="Mumbai Indians",
        winner="Chennai Super Kings",
        is_completed=True,
        toss_winner="Mumbai Indians",
        toss_decision="field",
    )


class TestValidation:
    def test_valid_record_passes(self):
        report = ValidationReport()
        assert validate_match(_valid_record(), report) is True
        assert not report.errors

    def test_winner_who_did_not_play_is_rejected(self):
        record = _valid_record()
        record.winner = "Rajasthan Royals"
        report = ValidationReport()
        assert validate_match(record, report) is False
        assert any(i.code == "winner_not_playing" for i in report.errors)

    def test_same_team_twice_is_rejected(self):
        record = _valid_record()
        record.team2 = record.team1
        report = ValidationReport()
        assert validate_match(record, report) is False

    def test_implausible_score_is_rejected(self):
        from ipl.ingestion.records import InningsRecord

        record = _valid_record()
        record.innings.append(InningsRecord(innings_no=1, runs=999, wickets=3))
        report = ValidationReport()
        assert validate_match(record, report) is False
        assert any(i.code == "implausible_score" for i in report.errors)

    def test_toss_winner_not_playing_is_only_a_warning(self):
        record = _valid_record()
        record.toss_winner = "Gujarat Titans"
        report = ValidationReport()
        assert validate_match(record, report) is True
        assert any(i.code == "toss_winner_not_playing" for i in report.warnings)

    def test_report_summary_is_readable(self):
        report = ValidationReport()
        validate_match(_valid_record(), report)
        assert "checked=1" in report.summary()


class TestDeduplication:
    def test_duplicate_batting_rows_collapse(self):
        from ipl.ingestion.records import BattingRecord

        record = _valid_record()
        for _ in range(3):
            record.batting.append(
                BattingRecord(innings_no=1, team="Chennai Super Kings", player="MS Dhoni")
            )
        deduplicate_match(record)
        assert len(record.batting) == 1

    def test_duplicate_deliveries_collapse(self):
        from ipl.ingestion.records import DeliveryRecord

        record = _valid_record()
        for _ in range(2):
            record.deliveries.append(
                DeliveryRecord(innings_no=1, over_no=1, ball_no=1, ball_seq=1)
            )
        deduplicate_match(record)
        assert len(record.deliveries) == 1

    def test_distinct_rows_are_kept(self):
        from ipl.ingestion.records import DeliveryRecord

        record = _valid_record()
        for seq in range(1, 4):
            record.deliveries.append(
                DeliveryRecord(innings_no=1, over_no=1, ball_no=seq, ball_seq=seq)
            )
        deduplicate_match(record)
        assert len(record.deliveries) == 3


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class TestLoader:
    def test_match_round_trips_through_the_database(self, fresh_db):
        from ipl.db.base import session_scope
        from ipl.db.repository import DimensionCache, load_matches
        from ipl.ingestion.pipeline import IngestionPipeline
        from ipl.ingestion.records import InningsRecord

        record = _valid_record()
        record.venue = "MA Chidambaram Stadium"
        record.match_date = __import__("datetime").date(2025, 4, 1)
        record.first_batting_team = "Chennai Super Kings"
        record.second_batting_team = "Mumbai Indians"
        record.innings.append(
            InningsRecord(
                innings_no=1, batting_team="Chennai Super Kings",
                bowling_team="Mumbai Indians", runs=180, wickets=4, balls=120,
            )
        )

        pipeline = IngestionPipeline(ingest_deliveries=False, enable_cricsheet=False)
        with session_scope() as session:
            cache = DimensionCache(session)
            pipeline._load_record(session, cache, record)

        stored = load_matches()
        assert len(stored) == 1
        row = stored.iloc[0]
        assert row["team1"] == "Chennai Super Kings"
        assert row["winner"] == "Chennai Super Kings"
        assert row["venue"] == "MA Chidambaram Stadium"

    def test_reingesting_updates_rather_than_duplicating(self, fresh_db):
        from ipl.db.base import session_scope
        from ipl.db.repository import DimensionCache, load_matches
        from ipl.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(ingest_deliveries=False, enable_cricsheet=False)
        for winner in ("Chennai Super Kings", "Mumbai Indians"):
            record = _valid_record()
            record.winner = winner
            with session_scope() as session:
                cache = DimensionCache(session)
                pipeline._load_record(session, cache, record)

        stored = load_matches()
        assert len(stored) == 1
        # The second load must overwrite the first, not append.
        assert stored.iloc[0]["winner"] == "Mumbai Indians"


# ---------------------------------------------------------------------------
# Live feed (opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.network
class TestLiveFeed:
    def test_official_schedule_is_reachable(self):
        """The official feed still serves the shape the client expects."""
        client = IPLT20Client()
        try:
            rows = client.fetch_schedule(203)
        except Exception as exc:  # pragma: no cover - offline CI
            pytest.skip(f"IPL feed unreachable: {exc}")

        assert rows, "schedule feed returned no fixtures"
        assert "MatchID" in rows[0]
        assert "Tata Ipl 2025" in rows[0].get("CompetitionName", "")
