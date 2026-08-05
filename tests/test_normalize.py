"""Tests for the parsing and canonicalisation layer.

These rules are where messy source strings become typed facts, so they carry
most of the correctness risk in the pipeline.
"""

from __future__ import annotations

from datetime import date

import pytest

from ipl.ingestion.normalize import (
    balls_to_overs,
    canonical_player,
    canonical_team,
    canonical_venue,
    clean_text,
    detect_stage,
    make_match_key,
    overs_to_balls,
    parse_date,
    parse_dismissal,
    parse_innings_summary,
    parse_result,
    parse_toss,
    run_rate,
    season_from_competition_name,
    to_bool,
    to_float,
    to_int,
)


class TestScalarCoercion:
    @pytest.mark.parametrize(
        "value,expected",
        [("  Virat  Kohli ", "Virat Kohli"), ("", None), (None, None), ("x", "x")],
    )
    def test_clean_text(self, value, expected):
        assert clean_text(value) == expected

    @pytest.mark.parametrize(
        "value,expected", [("3", 3), (3, 3), ("3.0", 3), ("", None), ("abc", None), (None, None)]
    )
    def test_to_int(self, value, expected):
        assert to_int(value) == expected

    def test_to_float(self):
        assert to_float("19.4") == 19.4
        assert to_float("") is None
        assert to_float("bad", default=0.0) == 0.0

    @pytest.mark.parametrize(
        "value,expected",
        [("1", True), ("0", False), ("true", True), ("", False), (1, True), (None, False)],
    )
    def test_to_bool(self, value, expected):
        assert to_bool(value) is expected


class TestOverConversion:
    """Cricket over notation is base-6 in the decimal place, not decimal."""

    @pytest.mark.parametrize(
        "overs,balls",
        [(0.0, 0), (1.0, 6), (19.4, 118), (20.0, 120), (4.3, 27), (0.5, 5)],
    )
    def test_overs_to_balls(self, overs, balls):
        assert overs_to_balls(overs) == balls

    @pytest.mark.parametrize(
        "balls,overs", [(0, 0.0), (6, 1.0), (118, 19.4), (120, 20.0), (27, 4.3)]
    )
    def test_balls_to_overs(self, balls, overs):
        assert balls_to_overs(balls) == overs

    def test_round_trip(self):
        for balls in range(0, 121):
            assert overs_to_balls(balls_to_overs(balls)) == balls

    def test_naive_multiplication_would_be_wrong(self):
        # 19.4 overs is 118 balls, not 19.4 * 6 = 116.4. This is the bug the
        # helper exists to prevent.
        assert overs_to_balls(19.4) == 118
        assert overs_to_balls(19.4) != int(19.4 * 6)

    def test_run_rate(self):
        assert run_rate(120, 120) == 6.0
        assert run_rate(190, 120) == 9.5
        assert run_rate(100, 0) is None


class TestCanonicalTeam:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Royal Challengers Bangalore", "Royal Challengers Bengaluru"),
            ("Royal Challengers Bengaluru", "Royal Challengers Bengaluru"),
            ("RCB", "Royal Challengers Bengaluru"),
            ("Delhi Daredevils", "Delhi Capitals"),
            ("Kings XI Punjab", "Punjab Kings"),
            ("Rising Pune Supergiants", "Rising Pune Supergiant"),
            ("  chennai super kings  ", "Chennai Super Kings"),
        ],
    )
    def test_aliases_fold_to_current_branding(self, raw, expected):
        assert canonical_team(raw) == expected

    @pytest.mark.parametrize("raw", ["TBD", "tba", "0", "123", "-", "", None])
    def test_placeholders_resolve_to_none(self, raw):
        assert canonical_team(raw) is None

    def test_unknown_team_passes_through(self):
        # A brand-new franchise must land in the database rather than vanish.
        assert canonical_team("Brand New Franchise") == "Brand New Franchise"


class TestCanonicalVenue:
    @pytest.mark.parametrize(
        "raw,expected_venue",
        [
            ("M Chinnaswamy Stadium", "M Chinnaswamy Stadium"),
            ("M.Chinnaswamy Stadium", "M Chinnaswamy Stadium"),
            ("Narendra Modi Stadium, Ahmedabad", "Narendra Modi Stadium"),
            ("Sardar Patel Stadium, Motera", "Narendra Modi Stadium"),
            ("Feroz Shah Kotla", "Arun Jaitley Stadium"),
            ("MA Chidambaram Stadium, Chepauk, Chennai", "MA Chidambaram Stadium"),
        ],
    )
    def test_venue_aliases(self, raw, expected_venue):
        venue, _city = canonical_venue(raw)
        assert venue == expected_venue

    def test_city_is_filled_from_the_registry(self):
        venue, city = canonical_venue("Wankhede Stadium")
        assert (venue, city) == ("Wankhede Stadium", "Mumbai")

    def test_explicit_city_wins(self):
        _venue, city = canonical_venue("Wankhede Stadium", "Bombay")
        assert city == "Bombay"


class TestCanonicalPlayer:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Rajat Patidar (c)", "Rajat Patidar"),
            ("MS Dhoni (wk)", "MS Dhoni"),
            ("Jitesh Sharma (c/wk)", "Jitesh Sharma"),
            ("  Phil  Salt  ", "Phil Salt"),
            ("Krunal Pandya (RCB)", "Krunal Pandya"),
        ],
    )
    def test_markers_are_stripped(self, raw, expected):
        assert canonical_player(raw) == expected

    def test_empty_is_none(self):
        assert canonical_player("") is None


class TestParseToss:
    def test_bat_decision(self):
        winner, decision = parse_toss(
            "Chennai Super Kings Won The Toss And Elected To Bat", "Chennai Super Kings"
        )
        assert winner == "Chennai Super Kings"
        assert decision == "bat"

    def test_field_decision(self):
        winner, decision = parse_toss("Punjab Kings Won The Toss And Elected To Field")
        assert winner == "Punjab Kings"
        assert decision == "field"

    def test_bowl_is_treated_as_field(self):
        _winner, decision = parse_toss("Mumbai Indians won the toss and elected to bowl")
        assert decision == "field"

    def test_missing_text(self):
        assert parse_toss(None) == (None, None)


class TestParseResult:
    def test_win_by_runs(self):
        result = parse_result("Royal Challengers Bengaluru Won by 6 Runs (Winners)")
        assert result["winner"] == "Royal Challengers Bengaluru"
        assert result["result_type"] == "runs"
        assert result["win_margin_runs"] == 6
        assert result["win_margin_wickets"] is None

    def test_win_by_wickets(self):
        result = parse_result("Mumbai Indians won by 4 wickets")
        assert result["winner"] == "Mumbai Indians"
        assert result["result_type"] == "wickets"
        assert result["win_margin_wickets"] == 4

    def test_abandoned(self):
        result = parse_result("Match Abandoned without a ball bowled")
        assert result["is_no_result"] is True
        assert result["result_type"] == "no result"
        assert result["winner"] is None

    def test_tie_with_super_over(self):
        result = parse_result(
            "Match Tied (Kolkata Knight Riders Won the One Over Eliminator)"
        )
        assert result["is_tie"] is True
        assert result["is_super_over"] is True
        assert result["winner"] == "Kolkata Knight Riders"

    def test_empty(self):
        result = parse_result(None)
        assert result["winner"] is None
        assert result["result_type"] is None


class TestParseInningsSummary:
    def test_standard(self):
        parsed = parse_innings_summary("190/9 (20.0 Ov)")
        assert parsed == {"runs": 190, "wickets": 9, "overs": 20.0, "balls": 120}

    def test_overs_spelling(self):
        parsed = parse_innings_summary("184/7 (19.4 Overs)")
        assert parsed["runs"] == 184
        assert parsed["balls"] == 118

    def test_unparseable(self):
        assert parse_innings_summary("yet to bat")["runs"] is None


class TestParseDismissal:
    def test_caught(self):
        parsed = parse_dismissal("c Shreyas Iyer b Kyle Jamieson")
        assert parsed["is_out"] is True
        assert parsed["dismissal_kind"] == "caught"
        assert parsed["bowler"] == "Kyle Jamieson"
        assert parsed["fielder"] == "Shreyas Iyer"

    def test_bowled(self):
        parsed = parse_dismissal("b Jasprit Bumrah")
        assert parsed["dismissal_kind"] == "bowled"
        assert parsed["bowler"] == "Jasprit Bumrah"

    def test_run_out(self):
        parsed = parse_dismissal("run out (Virat Kohli)")
        assert parsed["dismissal_kind"] == "run out"
        assert parsed["fielder"] == "Virat Kohli"
        # A run-out is not credited to any bowler.
        assert parsed["bowler"] is None

    def test_not_out(self):
        parsed = parse_dismissal("not out")
        assert parsed["is_out"] is False

    def test_stumped(self):
        parsed = parse_dismissal("st MS Dhoni b Ravindra Jadeja")
        assert parsed["dismissal_kind"] == "stumped"
        assert parsed["bowler"] == "Ravindra Jadeja"

    def test_empty_means_did_not_bat(self):
        assert parse_dismissal(None)["is_out"] is False


class TestDetectStage:
    @pytest.mark.parametrize(
        "hint,stage,playoff",
        [
            ("Final", "Final", True),
            ("Qualifier 1", "Qualifier 1", True),
            ("Qualifier 2", "Qualifier 2", True),
            ("Eliminator", "Eliminator", True),
            ("Match 42", "League", False),
        ],
    )
    def test_stage_detection(self, hint, stage, playoff):
        assert detect_stage(hint) == (stage, playoff)

    def test_eliminator_wins_over_final_substring(self):
        # Some feeds label the eliminator with text containing "final"; the more
        # specific match must win.
        assert detect_stage("Eliminator Final")[0] == "Eliminator"


class TestMisc:
    def test_parse_date_formats(self):
        assert parse_date("2025-06-03") == date(2025, 6, 3)
        assert parse_date("03 Jun 2025") == date(2025, 6, 3)
        assert parse_date("2025-06-03 19:30:00") == date(2025, 6, 3)
        assert parse_date("") is None

    def test_season_from_competition_name(self):
        assert season_from_competition_name("Tata Ipl 2025") == 2025
        assert season_from_competition_name("IPL 2019") == 2019
        assert season_from_competition_name("No year here") is None

    def test_make_match_key_namespaces_the_source(self):
        # The two sources number matches independently, so the key must carry
        # the source to stay globally unique.
        assert make_match_key("iplt20", 2025, 1872) == "iplt20:2025:1872"
        assert make_match_key("cricsheet", 2025, 1872) != make_match_key("iplt20", 2025, 1872)
