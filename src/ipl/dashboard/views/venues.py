"""Venue statistics page: scoring behaviour, chase bias and team records."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.venue import (
    venue_phase_profile,
    venue_records,
    venue_summary,
    venue_team_performance,
)
from .. import data
from ..theme import CATEGORICAL, bar_chart, grouped_bar_chart, team_palette
from ._common import metric_row, page_header, require_data, show_table


def render() -> None:
    """Render the venue statistics page."""
    if not require_data():
        return

    page_header(
        "Venue Statistics",
        "How each ground plays: scoring, chase bias and franchise records.",
    )

    matches = data.load_matches()
    innings = data.load_innings()

    tab_overview, tab_detail = st.tabs(["All venues", "Venue detail"])

    with tab_overview:
        _overview(matches, innings)

    with tab_detail:
        _detail(matches, innings)


def _overview(matches: pd.DataFrame, innings: pd.DataFrame) -> None:
    """Comparison across every ground that has hosted enough matches."""
    min_matches = st.slider("Minimum matches hosted", 1, 60, 15, key="venue_min_matches")
    summary = venue_summary(matches, innings, min_matches=min_matches)

    if summary.empty:
        st.caption("No venues meet this threshold.")
        return

    if "avg_first_innings" in summary.columns:
        scoring = summary.dropna(subset=["avg_first_innings"]).sort_values(
            "avg_first_innings"
        )
        st.plotly_chart(
            bar_chart(
                scoring, "venue", "avg_first_innings",
                title="Average first-innings total by venue",
                orientation="h", height=max(420, 26 * len(scoring)),
                colors=[CATEGORICAL[0]] * len(scoring),
                text_format=".0f", x_title="Runs", y_title="",
            ),
            width="stretch",
        )

    chase = summary.dropna(subset=["chase_win_pct"]).sort_values("chase_win_pct")
    if not chase.empty:
        st.plotly_chart(
            grouped_bar_chart(
                chase, "venue",
                {"Batting first": "bat_first_win_pct", "Chasing": "chase_win_pct"},
                title="Who wins here: batting first or chasing?",
                x_title="", y_title="Win %", height=max(420, 30 * len(chase)),
            ),
            width="stretch",
        )
        st.caption(
            "A ground well above 50% for chasing suits sides bowling first "
            "(dew, or a surface that eases under lights)."
        )

    columns = [
        "venue", "city", "matches", "seasons", "avg_first_innings",
        "avg_second_innings", "highest_first_innings", "lowest_first_innings",
        "chase_win_pct", "bat_first_win_pct",
    ]
    available = [c for c in columns if c in summary.columns]
    show_table(
        summary[available].rename(
            columns={
                "venue": "Venue", "city": "City", "matches": "Matches",
                "seasons": "Seasons", "avg_first_innings": "Avg 1st inns",
                "avg_second_innings": "Avg 2nd inns",
                "highest_first_innings": "Highest", "lowest_first_innings": "Lowest",
                "chase_win_pct": "Chasing win %", "bat_first_win_pct": "Bat-first win %",
            }
        ),
        height=460,
    )


def _detail(matches: pd.DataFrame, innings: pd.DataFrame) -> None:
    """Deep dive into one ground."""
    venues = data.venues()
    if not venues:
        st.caption("No venues available.")
        return

    venue = st.selectbox("Venue", venues, key="venue_detail_select")
    records = venue_records(matches, innings, venue)
    if not records:
        st.caption("No completed matches at this ground.")
        return

    metric_row(
        [
            ("Matches hosted", f"{records['matches']}", None),
            ("Average total", f"{records.get('average_total', '—')}", None),
            ("Highest total", records.get("highest_total", "—"), None),
            ("Lowest total", records.get("lowest_total", "—"), None),
        ]
    )
    st.caption(
        f"First match: {pd.to_datetime(records['first_match']):%d %b %Y} · "
        f"Most recent: {pd.to_datetime(records['last_match']):%d %b %Y}"
    )
    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Franchise records here")
        performance = venue_team_performance(matches, venue, min_matches=2)
        if performance.empty:
            st.caption("Not enough matches.")
        else:
            st.plotly_chart(
                bar_chart(
                    performance.sort_values("win_pct"), "team", "win_pct",
                    title=f"Win % at {venue}",
                    colors=team_palette(
                        performance.sort_values("win_pct")["team"].tolist()
                    ),
                    orientation="h", height=max(360, 30 * len(performance)),
                    text_format=".1f", x_title="Win %", y_title="",
                ),
                width="stretch",
            )
            show_table(
                performance.rename(
                    columns={
                        "team": "Team", "matches": "P", "wins": "W",
                        "losses": "L", "win_pct": "Win %",
                    }
                )
            )

    with right:
        st.subheader("Scoring by phase")
        venue_match_ids = set(
            matches[(matches["venue"] == venue) & matches["is_completed"]]["match_id"]
        )
        deliveries = data.load_deliveries()
        if deliveries.empty:
            st.caption("Ball-by-ball data was not ingested, so phase splits are unavailable.")
        else:
            venue_deliveries = deliveries[deliveries["match_id"].isin(venue_match_ids)]
            phases = venue_phase_profile(venue_deliveries)
            if phases.empty:
                st.caption("No ball-by-ball data for this ground.")
            else:
                st.plotly_chart(
                    bar_chart(
                        phases, "phase", "run_rate",
                        title="Run rate by phase",
                        colors=[CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[7]],
                        text_format=".2f", x_title="", y_title="Runs per over",
                    ),
                    width="stretch",
                )
                show_table(
                    phases.rename(
                        columns={
                            "phase": "Phase", "avg_runs": "Avg runs",
                            "avg_wickets": "Avg wickets", "avg_balls": "Avg balls",
                            "innings": "Innings", "run_rate": "Run rate",
                        }
                    )
                )

    st.divider()
    st.subheader("Matches at this ground")
    played = matches[
        (matches["venue"] == venue) & matches["is_completed"]
    ].sort_values("match_date", ascending=False)
    display = played[
        ["match_date", "season", "team1", "team2", "winner", "result_summary"]
    ].copy()
    display["match_date"] = pd.to_datetime(display["match_date"]).dt.strftime("%d %b %Y")
    show_table(
        display.rename(
            columns={
                "match_date": "Date", "season": "Season", "team1": "Team 1",
                "team2": "Team 2", "winner": "Winner", "result_summary": "Result",
            }
        ),
        height=420,
    )
