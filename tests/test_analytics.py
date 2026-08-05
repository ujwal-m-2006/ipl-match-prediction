"""Tests for the descriptive analytics layer.

Cricket statistics have conventions a naive aggregation gets wrong (an average
is runs per *dismissal*, not per innings), so those are pinned down here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ipl.analytics.player import (
    batting_leaderboard,
    bowling_leaderboard,
    compare_players,
    player_career_summary,
)
from ipl.analytics.team import (
    batting_first_advantage,
    head_to_head,
    team_form_timeline,
    team_season_summary,
    team_summary,
    toss_impact,
)
from ipl.analytics.venue import (
    scoring_trend_by_season,
    venue_records,
    venue_summary,
    venue_team_performance,
)


class TestTeamSummary:
    def test_wins_plus_losses_equal_matches(self, sample_matches):
        summary = team_summary(sample_matches)
        assert (summary["wins"] + summary["losses"] == summary["matches"]).all()

    def test_win_percentage(self, sample_matches):
        summary = team_summary(sample_matches)
        expected = (summary["wins"] / summary["matches"] * 100).round(2)
        assert np.allclose(summary["win_pct"], expected)

    def test_each_match_contributes_two_team_rows(self, sample_matches):
        summary = team_summary(sample_matches)
        assert summary["matches"].sum() == len(sample_matches) * 2

    def test_titles_count_final_wins(self, sample_matches):
        summary = team_summary(sample_matches)
        finals = sample_matches[sample_matches["stage"] == "Final"]
        assert summary["titles"].sum() == len(finals)

    def test_min_matches_filter(self, sample_matches):
        assert team_summary(sample_matches, min_matches=100).empty

    def test_empty_input(self):
        assert team_summary(pd.DataFrame()).empty


class TestSeasonSummary:
    def test_filters_to_one_team(self, sample_matches):
        summary = team_season_summary(sample_matches, "Mumbai Indians")
        assert set(summary["team"]) == {"Mumbai Indians"}

    def test_covers_every_season(self, sample_matches):
        summary = team_season_summary(sample_matches)
        assert set(summary["season"]) == set(sample_matches["season"])


class TestFormTimeline:
    def test_rows_are_chronological(self, sample_matches):
        timeline = team_form_timeline(sample_matches, "Chennai Super Kings")
        assert timeline["match_date"].is_monotonic_increasing

    def test_cumulative_win_rate_ends_at_the_career_rate(self, sample_matches):
        timeline = team_form_timeline(sample_matches, "Chennai Super Kings")
        summary = team_summary(sample_matches)
        career = summary[summary["team"] == "Chennai Super Kings"]["win_pct"].iloc[0]
        assert timeline["cumulative_win_pct"].iloc[-1] == pytest.approx(career, abs=0.01)


class TestHeadToHead:
    def test_wins_sum_to_meetings(self, sample_matches):
        record = head_to_head(sample_matches, "Chennai Super Kings", "Mumbai Indians")
        assert record["team_a_wins"] + record["team_b_wins"] == record["matches"]

    def test_is_symmetric(self, sample_matches):
        forward = head_to_head(sample_matches, "Chennai Super Kings", "Mumbai Indians")
        reverse = head_to_head(sample_matches, "Mumbai Indians", "Chennai Super Kings")
        assert forward["matches"] == reverse["matches"]
        assert forward["team_a_wins"] == reverse["team_b_wins"]

    def test_never_met_returns_empty_shape(self, sample_matches):
        record = head_to_head(sample_matches, "Chennai Super Kings", "Gujarat Titans")
        assert record["matches"] == 0
        assert record["by_venue"].empty

    def test_current_streak(self, sample_matches):
        record = head_to_head(sample_matches, "Chennai Super Kings", "Mumbai Indians")
        assert record["current_streak"] >= 1
        assert record["current_streak_team"] in {"Chennai Super Kings", "Mumbai Indians"}


class TestTossImpact:
    def test_overall_row(self, sample_matches):
        impact = toss_impact(sample_matches)
        assert len(impact) == 1
        assert 0 <= impact.iloc[0]["toss_winner_advantage_pct"] <= 100

    def test_grouped_by_season(self, sample_matches):
        impact = toss_impact(sample_matches, by="season")
        assert set(impact["group"]) == set(sample_matches["season"])

    def test_wins_never_exceed_matches(self, sample_matches):
        impact = toss_impact(sample_matches, by="season")
        assert (impact["toss_winner_wins"] <= impact["matches"]).all()

    def test_both_shapes_expose_the_same_columns(self, sample_matches):
        """Grouped and ungrouped results must be column-compatible.

        Callers (the dashboard and the EDA report) read the same keys from
        either shape, so a name that differs between the two branches is a bug
        that only surfaces at render time.
        """
        overall = toss_impact(sample_matches)
        grouped = toss_impact(sample_matches, by="season")
        assert list(overall.columns) == list(grouped.columns)
        assert "toss_winner_advantage_pct" in overall.columns

    @pytest.mark.parametrize("by", [None, "season", "venue", "toss_decision"])
    def test_every_grouping_produces_the_percentage_column(self, sample_matches, by):
        impact = toss_impact(sample_matches, by=by)
        assert "toss_winner_advantage_pct" in impact.columns
        assert impact["toss_winner_advantage_pct"].between(0, 100).all()


class TestBattingFirstAdvantage:
    def test_percentages_are_complementary(self, sample_matches, sample_innings):
        advantage = batting_first_advantage(sample_matches, sample_innings)
        total = advantage["bat_first_win_pct"] + advantage["chase_win_pct"]
        assert np.allclose(total, 100.0)


class TestBattingLeaderboard:
    def test_average_is_runs_per_dismissal(self, sample_batting):
        leaders = batting_leaderboard(sample_batting)
        for row in leaders.itertuples(index=False):
            if row.dismissals > 0:
                assert row.average == pytest.approx(row.runs / row.dismissals, abs=0.01)

    def test_never_dismissed_has_undefined_average(self, sample_batting):
        # Position 3 in the fixture is never out, so their average must be NaN
        # rather than their run total.
        never_out = sample_batting[sample_batting["batting_position"] == 3]["player"].iloc[0]
        leaders = batting_leaderboard(sample_batting)
        row = leaders[leaders["player"] == never_out]
        assert row["dismissals"].iloc[0] == 0
        assert pd.isna(row["average"].iloc[0])

    def test_strike_rate(self, sample_batting):
        leaders = batting_leaderboard(sample_batting)
        for row in leaders.itertuples(index=False):
            if row.balls > 0:
                assert row.strike_rate == pytest.approx(row.runs * 100 / row.balls, abs=0.01)

    def test_season_filter(self, sample_batting):
        leaders = batting_leaderboard(sample_batting, season=2023)
        assert not leaders.empty
        assert leaders["runs"].sum() < sample_batting["runs"].sum()

    def test_sorted_by_runs(self, sample_batting):
        leaders = batting_leaderboard(sample_batting)
        assert leaders["runs"].is_monotonic_decreasing


class TestBowlingLeaderboard:
    def test_economy(self, sample_bowling):
        leaders = bowling_leaderboard(sample_bowling)
        for row in leaders.itertuples(index=False):
            if row.balls > 0:
                assert row.economy == pytest.approx(
                    row.runs_conceded * 6 / row.balls, abs=0.01
                )

    def test_average_is_runs_per_wicket(self, sample_bowling):
        leaders = bowling_leaderboard(sample_bowling)
        for row in leaders.itertuples(index=False):
            if row.wickets > 0:
                assert row.average == pytest.approx(
                    row.runs_conceded / row.wickets, abs=0.01
                )

    def test_no_wickets_gives_undefined_average(self, sample_bowling):
        frame = sample_bowling.copy()
        frame["wickets"] = 0
        leaders = bowling_leaderboard(frame)
        assert leaders["average"].isna().all()


class TestPlayerProfile:
    def test_career_summary_has_both_disciplines(
        self, sample_batting, sample_bowling, sample_matches
    ):
        player = sample_batting["player"].iloc[0]
        profile = player_career_summary(player, sample_batting, sample_bowling, sample_matches)
        assert profile["player"] == player
        assert profile["matches"] > 0
        assert profile["batting"]

    def test_compare_players_one_row_each(
        self, sample_batting, sample_bowling, sample_matches
    ):
        players = sample_batting["player"].unique()[:3].tolist()
        comparison = compare_players(players, sample_batting, sample_bowling, sample_matches)
        assert len(comparison) == 3
        assert list(comparison["player"]) == players


class TestVenueAnalytics:
    def test_summary_counts_matches(self, sample_matches, sample_innings):
        summary = venue_summary(sample_matches, sample_innings)
        assert summary["matches"].sum() == len(sample_matches)

    def test_chase_and_bat_first_are_complementary(self, sample_matches, sample_innings):
        summary = venue_summary(sample_matches, sample_innings)
        valid = summary.dropna(subset=["chase_win_pct", "bat_first_win_pct"])
        assert np.allclose(valid["chase_win_pct"] + valid["bat_first_win_pct"], 100.0)

    def test_team_performance_at_a_venue(self, sample_matches):
        venue = sample_matches["venue"].iloc[0]
        performance = venue_team_performance(sample_matches, venue)
        assert not performance.empty
        assert (performance["wins"] + performance["losses"] == performance["matches"]).all()

    def test_records(self, sample_matches, sample_innings):
        venue = sample_matches["venue"].iloc[0]
        records = venue_records(sample_matches, sample_innings, venue)
        assert records["venue"] == venue
        assert records["matches"] > 0

    def test_scoring_trend_one_row_per_season(self, sample_innings):
        trend = scoring_trend_by_season(sample_innings)
        assert set(trend["season"]) == set(sample_innings["season"])
