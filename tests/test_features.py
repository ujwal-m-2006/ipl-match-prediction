"""Tests for feature engineering.

The critical property is **no leakage**: a feature row must be computable from
matches that finished strictly before the match it describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ipl.features.inplay_features import (
    CHASE_NUMERIC_FEATURES,
    build_chase_features,
    chase_feature_row,
)
from ipl.features.match_features import (
    MATCH_CATEGORICAL_FEATURES,
    MATCH_NUMERIC_FEATURES,
    add_net_run_rate,
    build_match_features,
    build_score_features,
    compute_standings,
    featurise_fixture,
    mirror_fixtures,
)
from ipl.features.preprocessing import clean_feature_frame, split_by_season


class TestBuildMatchFeatures:
    def test_produces_one_row_per_completed_match(self, sample_matches, sample_innings):
        features, _state = build_match_features(sample_matches, sample_innings)
        assert len(features) == len(sample_matches)

    def test_all_declared_features_are_present(self, sample_matches, sample_innings):
        features, _state = build_match_features(sample_matches, sample_innings)
        missing = set(MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES) - set(features.columns)
        assert not missing, f"missing feature columns: {sorted(missing)}"

    def test_first_match_has_no_history(self, sample_matches, sample_innings):
        """The very first fixture must fall back to priors, not to real data."""
        features, _state = build_match_features(sample_matches, sample_innings)
        first = features.iloc[0]
        assert first["h2h_matches"] == 0
        assert first["team1_career_win_rate"] == pytest.approx(0.5)
        assert first["team2_career_win_rate"] == pytest.approx(0.5)
        assert first["team1_win_streak"] == 0

    def test_no_leakage_history_only_grows_from_earlier_matches(
        self, sample_matches, sample_innings
    ):
        """H2H count at row *n* must equal the meetings that happened before it."""
        features, _state = build_match_features(sample_matches, sample_innings)
        ordered = sample_matches.sort_values("match_date").reset_index(drop=True)

        for index, row in features.iterrows():
            pair = {row["team1"], row["team2"]}
            earlier = ordered.iloc[:index]
            expected = sum(
                1 for m in earlier.itertuples(index=False) if {m.team1, m.team2} == pair
            )
            assert row["h2h_matches"] == expected, f"leak at row {index}"

    def test_target_matches_the_actual_winner(self, sample_matches, sample_innings):
        features, _state = build_match_features(sample_matches, sample_innings)
        for row in features.itertuples(index=False):
            expected = int(row.winner == row.team1)
            assert row.target_team1_wins == expected

    def test_state_is_reusable_for_inference(self, sample_matches, sample_innings):
        _features, state = build_match_features(sample_matches, sample_innings)
        row = featurise_fixture(
            state,
            team1="Chennai Super Kings",
            team2="Mumbai Indians",
            venue="MA Chidambaram Stadium",
            season=2025,
        )
        assert len(row) == 1
        # Serving must produce exactly the columns training produced.
        missing = set(MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES) - set(row.columns)
        assert not missing

    def test_squad_features_used_when_lineups_supplied(
        self, sample_matches, sample_innings, sample_batting, sample_bowling
    ):
        lineups = sample_batting[["match_id", "player", "team", "season", "match_date"]].copy()
        lineups["is_playing_xi"] = True
        lineups["is_captain"] = False
        lineups["is_wicketkeeper"] = False

        features, _state = build_match_features(
            sample_matches, sample_innings,
            batting=sample_batting, bowling=sample_bowling, match_players=lineups,
        )
        # The first match has no player history, so squad strength is unknown;
        # by the last match it must be populated.
        assert features.iloc[-1]["has_xi_data"] == 1

    def test_include_incomplete_emits_scheduled_fixtures(self, sample_matches, sample_innings):
        scheduled = sample_matches.copy()
        scheduled.loc[scheduled.index[-1], "is_completed"] = False
        scheduled.loc[scheduled.index[-1], "winner"] = None

        with_incomplete, _ = build_match_features(
            scheduled, sample_innings, include_incomplete=True
        )
        without, _ = build_match_features(scheduled, sample_innings)
        assert len(with_incomplete) == len(without) + 1

    def test_empty_input(self):
        features, state = build_match_features(pd.DataFrame())
        assert features.empty
        assert state is not None


class TestMirrorFixtures:
    def test_target_is_flipped(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        mirrored = mirror_fixtures(features)
        assert (mirrored["target_team1_wins"] == 1 - features["target_team1_wins"]).all()

    def test_teams_are_swapped(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        mirrored = mirror_fixtures(features)
        assert (mirrored["team1"] == features["team2"]).all()
        assert (mirrored["team2"] == features["team1"]).all()

    def test_signed_differences_are_negated(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        mirrored = mirror_fixtures(features)
        for column in ("form_short_diff", "career_win_rate_diff", "net_run_strength_diff"):
            assert np.allclose(mirrored[column], -features[column])

    def test_head_to_head_rate_is_complemented(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        mirrored = mirror_fixtures(features)
        assert np.allclose(
            mirrored["team1_h2h_win_rate"], 1 - features["team1_h2h_win_rate"]
        )


class TestScoreFeatures:
    def test_batting_side_is_the_first_batting_team(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        score = build_score_features(features)
        assert not score.empty
        assert (score["batting_team"] == features["first_batting_team"]).all()

    def test_target_is_the_first_innings_total(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        score = build_score_features(features)
        merged = score.merge(
            sample_innings[sample_innings["innings_no"] == 1][["match_id", "runs"]],
            on="match_id",
        )
        assert (merged["target_first_innings_runs"] == merged["runs"]).all()


class TestChaseFeatures:
    def test_builds_rows_for_second_innings_only(self, sample_deliveries, sample_matches):
        chase = build_chase_features(sample_deliveries, sample_matches)
        assert not chase.empty
        # Every row must describe a chase, so the batting side is the side that
        # batted second.
        merged = chase.merge(
            sample_matches[["match_id", "second_batting_team"]], on="match_id"
        )
        assert (merged["batting_team"] == merged["second_batting_team"]).all()

    def test_runs_required_never_negative(self, sample_deliveries, sample_matches):
        chase = build_chase_features(sample_deliveries, sample_matches)
        assert (chase["runs_required"] >= 0).all()

    def test_balls_remaining_is_consistent(self, sample_deliveries, sample_matches):
        chase = build_chase_features(sample_deliveries, sample_matches)
        assert (chase["balls_bowled"] + chase["balls_remaining"] == 120).all()

    def test_stops_once_the_chase_is_decided(self, sample_deliveries, sample_matches):
        """No rows after the target is reached - those would leak the result."""
        chase = build_chase_features(sample_deliveries, sample_matches)
        assert (chase["current_runs"] < chase["target"]).all()

    def test_stride_thins_the_output(self, sample_deliveries, sample_matches):
        full = build_chase_features(sample_deliveries, sample_matches, stride=1)
        thinned = build_chase_features(sample_deliveries, sample_matches, stride=4)
        assert len(thinned) < len(full)

    def test_serving_row_matches_training_columns(self):
        row = chase_feature_row(
            target=180, current_runs=90, wickets_fallen=3, balls_bowled=60,
            batting_team="Mumbai Indians", bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium",
        )
        missing = set(CHASE_NUMERIC_FEATURES) - set(row.columns)
        assert not missing
        assert row.iloc[0]["runs_required"] == 90
        assert row.iloc[0]["balls_remaining"] == 60
        assert row.iloc[0]["required_run_rate"] == pytest.approx(9.0)


class TestStandings:
    def test_points_are_two_per_win(self, sample_matches):
        table = compute_standings(sample_matches, 2023)
        assert not table.empty
        assert (table["points"] == table["won"] * 2).all()

    def test_played_equals_won_plus_lost(self, sample_matches):
        table = compute_standings(sample_matches, 2023)
        assert (table["played"] == table["won"] + table["lost"] + table["no_result"]).all()

    def test_playoffs_are_excluded(self, sample_matches):
        # The fixture marks match 4 as the Final; the league table must skip it.
        table = compute_standings(sample_matches, 2023)
        league_only = sample_matches[
            (sample_matches["season"] == 2023) & ~sample_matches["is_playoff"]
        ]
        assert table["played"].sum() == len(league_only) * 2

    def test_net_run_rate_is_attached(self, sample_matches, sample_innings):
        table = compute_standings(sample_matches, 2023)
        with_nrr = add_net_run_rate(table, sample_innings, sample_matches, 2023)
        assert "net_run_rate" in with_nrr.columns
        assert with_nrr["net_run_rate"].notna().any()


class TestPreprocessing:
    def test_split_is_chronological(self, sample_matches, sample_innings):
        features, _ = build_match_features(sample_matches, sample_innings)
        train, test = split_by_season(features, test_season_from=2024)
        assert train["season"].max() < 2024
        assert test["season"].min() >= 2024
        # No match may appear in both halves.
        assert not set(train["match_id"]) & set(test["match_id"])

    def test_clean_frame_replaces_infinities(self):
        frame = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": ["x", None, "z"]})
        cleaned = clean_feature_frame(frame, ["a"], ["b"])
        assert cleaned["a"].isna().sum() == 2
        assert cleaned["b"].tolist() == ["x", "Unknown", "z"]

    def test_clean_frame_adds_missing_columns(self):
        frame = pd.DataFrame({"a": [1.0]})
        cleaned = clean_feature_frame(frame, ["a", "missing_num"], ["missing_cat"])
        assert "missing_num" in cleaned.columns
        assert cleaned["missing_cat"].iloc[0] == "Unknown"
