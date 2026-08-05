"""Home page: headline numbers, recent results and league-wide trends."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.team import team_summary, toss_impact
from ...analytics.venue import scoring_trend_by_season
from ...constants import IPL_WEBSITE
from .. import data
from ..theme import bar_chart, line_chart, team_palette
from ._common import metric_row, page_header, require_data, show_table


def render() -> None:
    """Render the home page."""
    if not require_data():
        return

    page_header(
        "IPL Analytics & Prediction",
        "Machine-learning powered analysis of every Indian Premier League match.",
    )

    matches = data.load_matches()
    innings = data.load_innings()
    summary = data.database_summary()

    completed = matches[matches["is_completed"]]
    seasons = data.seasons()

    # --- headline tiles ---
    metric_row(
        [
            ("Seasons", f"{len(seasons)}", f"{min(seasons)}–{max(seasons)}" if seasons else None),
            ("Matches", f"{len(completed):,}", None),
            ("Players", f"{summary.get('players', 0):,}", None),
            ("Deliveries", f"{summary.get('deliveries', 0):,}", None),
        ]
    )
    st.write("")

    # --- most recent results ---
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Latest results")
        recent = (
            completed.sort_values("match_date", ascending=False)
            .head(8)[
                ["match_date", "team1", "team2", "venue", "winner",
                 "result_summary", "player_of_match"]
            ]
            .rename(
                columns={
                    "match_date": "Date", "team1": "Team 1", "team2": "Team 2",
                    "venue": "Venue", "winner": "Winner",
                    "result_summary": "Result", "player_of_match": "Player of the Match",
                }
            )
        )
        recent["Date"] = pd.to_datetime(recent["Date"]).dt.strftime("%d %b %Y")
        show_table(recent)

    with right:
        st.subheader("Champions")
        finals = completed[completed["stage"].fillna("").str.lower() == "final"]
        if finals.empty:
            st.caption("No finals recorded.")
        else:
            titles = (
                finals["winner"].value_counts().reset_index()
            )
            titles.columns = ["Team", "Titles"]
            st.plotly_chart(
                bar_chart(
                    titles, "Team", "Titles",
                    colors=team_palette(titles["Team"].tolist()),
                    orientation="h", height=380, text_format="d",
                    x_title="Titles", y_title="",
                ),
                use_container_width=True,
            )

    st.divider()

    # --- league-wide trends ---
    st.subheader("How the league has changed")
    tab_scoring, tab_results, tab_toss = st.tabs(
        ["Scoring", "Batting first vs chasing", "Toss impact"]
    )

    with tab_scoring:
        trend = scoring_trend_by_season(innings)
        if trend.empty:
            st.caption("No innings data available.")
        else:
            st.plotly_chart(
                line_chart(
                    trend, "season", {"Average first-innings score": "avg_score"},
                    title="Average first-innings total by season",
                    x_title="Season", y_title="Runs",
                ),
                use_container_width=True,
            )
            # Boundaries are a second measure on a different scale, so they get
            # their own chart rather than a second y-axis.
            st.plotly_chart(
                line_chart(
                    trend, "season",
                    {"Fours per innings": "avg_fours", "Sixes per innings": "avg_sixes"},
                    title="Boundaries per first innings",
                    x_title="Season", y_title="Count",
                ),
                use_container_width=True,
            )
            show_table(trend, caption="Underlying figures for the charts above.")

    with tab_results:
        from ...analytics.team import batting_first_advantage

        advantage = batting_first_advantage(matches, innings)
        if advantage.empty:
            st.caption("Not enough completed matches.")
        else:
            st.plotly_chart(
                line_chart(
                    advantage, "season",
                    {"Batting first": "bat_first_win_pct", "Chasing": "chase_win_pct"},
                    title="Win rate: batting first vs chasing",
                    x_title="Season", y_title="Win %",
                ),
                use_container_width=True,
            )
            show_table(advantage)

    with tab_toss:
        overall = toss_impact(matches)
        by_season = toss_impact(matches, by="season")
        if not overall.empty:
            value = overall.iloc[0]["toss_winner_advantage_pct"]
            st.metric(
                "Toss winner's win rate, all time", f"{value:.1f}%",
                f"{value - 50:+.1f} pts vs a coin flip",
            )
        if not by_season.empty:
            st.plotly_chart(
                line_chart(
                    by_season, "group", {"Toss winner win %": "toss_winner_advantage_pct"},
                    title="Does winning the toss help?",
                    x_title="Season", y_title="Win %",
                ),
                use_container_width=True,
            )
            st.caption(
                "A value near 50% means the toss carries no advantage in that season."
            )
        show_table(toss_impact(matches, by="toss_decision"))

    st.divider()

    # --- franchise standings snapshot ---
    st.subheader("All-time franchise records")
    teams = team_summary(matches, min_matches=10)
    if not teams.empty:
        display = teams[
            ["team", "matches", "wins", "losses", "win_pct", "titles",
             "home_win_pct", "away_win_pct", "bat_first_win_pct", "chasing_win_pct"]
        ].rename(
            columns={
                "team": "Team", "matches": "Matches", "wins": "Won", "losses": "Lost",
                "win_pct": "Win %", "titles": "Titles",
                "home_win_pct": "Home win %", "away_win_pct": "Away win %",
                "bat_first_win_pct": "Bat-first win %", "chasing_win_pct": "Chasing win %",
            }
        )
        show_table(display)

    st.caption(f"Data sourced from the official IPL website: {IPL_WEBSITE}")
