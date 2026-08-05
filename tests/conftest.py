"""Shared pytest fixtures.

Every database-touching test runs against a throwaway SQLite file, so the suite
never reads or writes the developer's real warehouse.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_database(tmp_path_factory) -> None:
    """Point the whole session at a temporary SQLite database.

    Set before any project module is imported, because ``get_settings()`` is
    cached on first call.
    """
    db_path = tmp_path_factory.mktemp("db") / "test_ipl.db"
    os.environ["IPL_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["IPL_LOG_LEVEL"] = "WARNING"
    os.environ["IPL_USE_HTTP_CACHE"] = "false"

    from ipl.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def fresh_db():
    """Create an empty schema and drop it again afterwards."""
    from ipl.db.base import init_db

    init_db(drop_existing=True)
    yield
    init_db(drop_existing=True)


@pytest.fixture()
def sample_matches() -> pd.DataFrame:
    """A small synthetic match table shaped like ``repository.load_matches``.

    Three teams over two seasons, with deterministic results so the analytics
    assertions can be exact rather than approximate.
    """
    start = date(2023, 4, 1)
    rows = []
    fixtures = [
        # (season, day offset, team1, team2, winner, first batting, venue)
        (2023, 0, "Chennai Super Kings", "Mumbai Indians", "Chennai Super Kings",
         "Chennai Super Kings", "MA Chidambaram Stadium"),
        (2023, 3, "Mumbai Indians", "Rajasthan Royals", "Mumbai Indians",
         "Rajasthan Royals", "Wankhede Stadium"),
        (2023, 6, "Rajasthan Royals", "Chennai Super Kings", "Chennai Super Kings",
         "Rajasthan Royals", "Sawai Mansingh Stadium"),
        (2023, 9, "Chennai Super Kings", "Mumbai Indians", "Mumbai Indians",
         "Mumbai Indians", "MA Chidambaram Stadium"),
        (2024, 370, "Mumbai Indians", "Chennai Super Kings", "Mumbai Indians",
         "Chennai Super Kings", "Wankhede Stadium"),
        (2024, 373, "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals",
         "Rajasthan Royals", "Sawai Mansingh Stadium"),
        (2024, 376, "Chennai Super Kings", "Rajasthan Royals", "Chennai Super Kings",
         "Chennai Super Kings", "MA Chidambaram Stadium"),
    ]

    for index, (season, offset, team1, team2, winner, first_bat, venue) in enumerate(
        fixtures, start=1
    ):
        second_bat = team2 if first_bat == team1 else team1
        rows.append(
            {
                "match_id": index,
                "match_key": f"test:{season}:{index}",
                "season": season,
                "match_date": pd.Timestamp(start + timedelta(days=offset)),
                "start_datetime": None,
                "match_number": f"Match {index}",
                "stage": "Final" if index == 4 else "League",
                "is_playoff": index == 4,
                "source": "test",
                "city": "Test City",
                "is_neutral_venue": False,
                "is_completed": True,
                "toss_decision": "bat" if index % 2 else "field",
                "result_type": "runs",
                "win_margin_runs": 10 + index,
                "win_margin_wickets": None,
                "is_tie": False,
                "is_no_result": False,
                "is_super_over": False,
                "is_dls_applied": False,
                "target_runs": None,
                "result_summary": f"{winner} won by {10 + index} runs",
                "overs_per_innings": 20,
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "home_team": team1,
                "away_team": team2,
                "toss_winner": team1 if index % 2 else team2,
                "first_batting_team": first_bat,
                "second_batting_team": second_bat,
                "winner": winner,
                "player_of_match": f"Player {index}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_innings(sample_matches: pd.DataFrame) -> pd.DataFrame:
    """Innings totals matching ``sample_matches``."""
    rows = []
    for row in sample_matches.itertuples(index=False):
        for innings_no, team in (
            (1, row.first_batting_team),
            (2, row.second_batting_team),
        ):
            runs = 160 + row.match_id * 3 + innings_no * 5
            rows.append(
                {
                    "match_id": row.match_id,
                    "innings_no": innings_no,
                    "batting_team": team,
                    "bowling_team": (
                        row.second_batting_team if innings_no == 1 else row.first_batting_team
                    ),
                    "runs": runs,
                    "wickets": 5,
                    "overs": 20.0,
                    "balls": 120,
                    "run_rate": round(runs / 20, 2),
                    "extras": 8,
                    "byes": 0,
                    "leg_byes": 2,
                    "wides": 5,
                    "no_balls": 1,
                    "powerplay_runs": 50,
                    "powerplay_wickets": 1,
                    "middle_runs": 70,
                    "middle_wickets": 2,
                    "death_runs": runs - 120,
                    "death_wickets": 2,
                    "fours": 14,
                    "sixes": 7,
                    "dot_balls": 35,
                    "target": None,
                    "season": row.season,
                    "match_date": row.match_date,
                    "venue": row.venue,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_batting(sample_matches: pd.DataFrame) -> pd.DataFrame:
    """Batting cards matching ``sample_matches``."""
    rows = []
    for row in sample_matches.itertuples(index=False):
        for innings_no, team in ((1, row.first_batting_team), (2, row.second_batting_team)):
            opposition = (
                row.second_batting_team if innings_no == 1 else row.first_batting_team
            )
            for position in range(1, 4):
                runs = 20 * position + row.match_id
                balls = 15 * position
                rows.append(
                    {
                        "match_id": row.match_id,
                        "innings_no": innings_no,
                        "batting_position": position,
                        "runs": runs,
                        "balls": balls,
                        "fours": position,
                        "sixes": position - 1,
                        "strike_rate": round(runs * 100 / balls, 2),
                        "is_out": position != 3,
                        "dismissal_kind": "caught" if position != 3 else "not out",
                        "player": f"{team.split()[0]} Batter {position}",
                        "team": team,
                        "season": row.season,
                        "match_date": row.match_date,
                        "venue": row.venue,
                        "opposition": opposition,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_bowling(sample_matches: pd.DataFrame) -> pd.DataFrame:
    """Bowling cards matching ``sample_matches``."""
    rows = []
    for row in sample_matches.itertuples(index=False):
        for innings_no, team in ((1, row.second_batting_team), (2, row.first_batting_team)):
            opposition = (
                row.first_batting_team if innings_no == 1 else row.second_batting_team
            )
            for index in range(1, 4):
                rows.append(
                    {
                        "match_id": row.match_id,
                        "innings_no": innings_no,
                        "overs": 4.0,
                        "balls": 24,
                        "maidens": 0,
                        "runs_conceded": 25 + index * 5,
                        "wickets": 3 - index + 1,
                        "wides": 1,
                        "no_balls": 0,
                        "economy": round((25 + index * 5) / 4, 2),
                        "dot_balls": 8,
                        "player": f"{team.split()[0]} Bowler {index}",
                        "team": team,
                        "season": row.season,
                        "match_date": row.match_date,
                        "venue": row.venue,
                        "opposition": opposition,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_deliveries(sample_matches: pd.DataFrame) -> pd.DataFrame:
    """A compact ball-by-ball table: two innings of 120 legal balls each."""
    rows = []
    for row in sample_matches.itertuples(index=False):
        for innings_no, batting in (
            (1, row.first_batting_team),
            (2, row.second_batting_team),
        ):
            bowling = row.second_batting_team if innings_no == 1 else row.first_batting_team
            cumulative_runs = 0
            cumulative_wickets = 0
            for ball in range(1, 121):
                # A repeating pattern gives a realistic-ish run rate (~8/over)
                # with wickets at fixed intervals.
                runs = [1, 0, 4, 1, 2, 6][ball % 6]
                is_wicket = ball % 24 == 0 and cumulative_wickets < 9
                cumulative_runs += runs
                cumulative_wickets += int(is_wicket)
                rows.append(
                    {
                        "match_id": row.match_id,
                        "innings_no": innings_no,
                        "over_no": (ball - 1) // 6 + 1,
                        "ball_no": (ball - 1) % 6 + 1,
                        "ball_seq": ball,
                        "batter_runs": runs,
                        "extra_runs": 0,
                        "total_runs": runs,
                        "is_legal": True,
                        "is_wide": False,
                        "is_no_ball": False,
                        "is_four": runs == 4,
                        "is_six": runs == 6,
                        "is_wicket": is_wicket,
                        "wicket_type": "caught" if is_wicket else None,
                        "cumulative_runs": cumulative_runs,
                        "cumulative_wickets": cumulative_wickets,
                        "batting_team": batting,
                        "bowling_team": bowling,
                        "batter": f"{batting.split()[0]} Batter 1",
                        "bowler": f"{bowling.split()[0]} Bowler 1",
                        "season": row.season,
                        "match_date": row.match_date,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def fixtures_dir() -> Path:
    """Directory holding recorded feed payloads used by parser tests."""
    return Path(__file__).parent / "fixtures"
