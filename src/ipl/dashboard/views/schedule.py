"""Schedule & results page: fixtures, standings and match scorecards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...features.match_features import add_net_run_rate, compute_standings
from .. import data
from ..theme import bar_chart, team_palette
from ._common import page_header, require_data, show_table


def render() -> None:
    """Render the schedule and results page."""
    if not require_data():
        return

    page_header(
        "Schedule & Results",
        "Every fixture, the live league table, and full scorecards.",
    )

    matches = data.load_matches()
    innings = data.load_innings()
    seasons = data.seasons()
    if not seasons:
        st.caption("No seasons available.")
        return

    season = st.selectbox("Season", seasons, index=0, key="schedule_season")
    season_matches = matches[matches["season"] == season].sort_values("match_date")

    tab_fixtures, tab_table, tab_card = st.tabs(
        ["Fixtures & results", "League table", "Scorecard"]
    )

    with tab_fixtures:
        _fixtures(season_matches)

    with tab_table:
        _standings(matches, innings, season)

    with tab_card:
        _scorecard(season_matches)


def _fixtures(season_matches: pd.DataFrame) -> None:
    """Fixture list with a completed/upcoming filter."""
    if season_matches.empty:
        st.caption("No fixtures for this season.")
        return

    completed = season_matches[season_matches["is_completed"]]
    upcoming = season_matches[~season_matches["is_completed"]]

    left, right = st.columns([1, 3])
    with left:
        view = st.radio(
            "Show", ["All", "Completed", "Upcoming"], horizontal=False, key="fixture_view"
        )

    frame = {"All": season_matches, "Completed": completed, "Upcoming": upcoming}[view]
    if frame.empty:
        st.caption(f"No {view.lower()} fixtures.")
        return

    display = frame[
        ["match_date", "match_number", "team1", "team2", "venue", "city",
         "toss_winner", "toss_decision", "winner", "result_summary", "player_of_match"]
    ].copy()
    display["match_date"] = pd.to_datetime(display["match_date"]).dt.strftime("%d %b %Y")
    display = display.rename(
        columns={
            "match_date": "Date", "match_number": "Match", "team1": "Team 1",
            "team2": "Team 2", "venue": "Venue", "city": "City",
            "toss_winner": "Toss", "toss_decision": "Elected",
            "winner": "Winner", "result_summary": "Result",
            "player_of_match": "Player of the Match",
        }
    )
    show_table(display, height=520)

    st.caption(
        f"{len(completed)} completed · {len(upcoming)} upcoming · {len(season_matches)} total"
    )


def _standings(matches: pd.DataFrame, innings: pd.DataFrame, season: int) -> None:
    """League table with points and net run rate."""
    table = compute_standings(matches, season)
    if table.empty:
        st.caption("No completed league matches for this season yet.")
        return

    table = add_net_run_rate(table, innings, matches, season)

    display = table[
        ["position", "team", "played", "won", "lost", "no_result", "points"]
        + (["net_run_rate"] if "net_run_rate" in table.columns else [])
    ].rename(
        columns={
            "position": "#", "team": "Team", "played": "P", "won": "W",
            "lost": "L", "no_result": "NR", "points": "Pts",
            "net_run_rate": "NRR",
        }
    )
    show_table(display)
    st.caption(
        "The top four qualify for the playoffs. NRR is computed to IPL rules, "
        "with an all-out side charged the full 20 overs."
    )

    st.plotly_chart(
        bar_chart(
            table, "team", "points",
            title=f"Points — IPL {season}",
            colors=team_palette(table["team"].tolist()),
            orientation="h", height=420, text_format="d",
            x_title="Points", y_title="",
        ),
        width="stretch",
    )


def _scorecard(season_matches: pd.DataFrame) -> None:
    """Full batting and bowling scorecard for one selected match."""
    completed = season_matches[season_matches["is_completed"]]
    if completed.empty:
        st.caption("No completed matches to show a scorecard for.")
        return

    labels = {
        int(row.match_id): (
            f"{pd.to_datetime(row.match_date):%d %b} · {row.team1} vs {row.team2}"
            f" — {row.result_summary or ''}"
        )
        for row in completed.itertuples(index=False)
    }
    match_id = st.selectbox(
        "Match", list(labels), format_func=lambda mid: labels[mid], key="scorecard_match"
    )

    match = completed[completed["match_id"] == match_id].iloc[0]
    st.markdown(f"### {match['team1']} vs {match['team2']}")
    st.caption(
        f"{pd.to_datetime(match['match_date']):%d %B %Y} · {match['venue']}"
        f"{', ' + match['city'] if pd.notna(match['city']) else ''} · {match['stage'] or ''}"
    )

    columns = st.columns(3)
    columns[0].metric("Result", match["result_summary"] or "—")
    columns[1].metric(
        "Toss",
        f"{match['toss_winner']}" if pd.notna(match["toss_winner"]) else "—",
        f"elected to {match['toss_decision']}" if pd.notna(match["toss_decision"]) else None,
    )
    columns[2].metric("Player of the Match", match["player_of_match"] or "—")

    innings = data.load_innings()
    batting = data.load_batting()
    bowling = data.load_bowling()

    match_innings = innings[innings["match_id"] == match_id].sort_values("innings_no")
    for row in match_innings.itertuples(index=False):
        st.markdown(
            f"#### Innings {row.innings_no}: {row.batting_team} — "
            f"{int(row.runs)}/{int(row.wickets)} ({row.overs} overs)"
        )

        bat = batting[
            (batting["match_id"] == match_id) & (batting["innings_no"] == row.innings_no)
        ].sort_values("batting_position")
        if not bat.empty:
            display = bat[
                ["player", "runs", "balls", "fours", "sixes", "strike_rate", "dismissal_kind"]
            ].rename(
                columns={
                    "player": "Batter", "runs": "R", "balls": "B", "fours": "4s",
                    "sixes": "6s", "strike_rate": "SR", "dismissal_kind": "Dismissal",
                }
            )
            show_table(display)

        bowl = bowling[
            (bowling["match_id"] == match_id) & (bowling["innings_no"] == row.innings_no)
        ].sort_values("overs", ascending=False)
        if not bowl.empty:
            display = bowl[
                ["player", "overs", "maidens", "runs_conceded", "wickets", "economy"]
            ].rename(
                columns={
                    "player": "Bowler", "overs": "O", "maidens": "M",
                    "runs_conceded": "R", "wickets": "W", "economy": "Econ",
                }
            )
            show_table(display)

        extras = []
        for label, column in (
            ("wides", "wides"), ("no-balls", "no_balls"),
            ("byes", "byes"), ("leg-byes", "leg_byes"),
        ):
            value = getattr(row, column, None)
            if value and not pd.isna(value):
                extras.append(f"{int(value)} {label}")
        if extras:
            st.caption("Extras: " + ", ".join(extras))
