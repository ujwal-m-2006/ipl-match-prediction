"""Team analytics page: single-team profile and side-by-side comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.player import batting_leaderboard, bowling_leaderboard
from ...analytics.team import team_form_timeline, team_season_summary, team_summary
from ...constants import team_color
from .. import data
from ..theme import bar_chart, grouped_bar_chart, line_chart, team_palette
from ._common import metric_row, page_header, require_data, show_table


def render() -> None:
    """Render the team analytics page."""
    if not require_data():
        return

    page_header("Team Analytics", "Franchise records, form and squad contributions.")

    teams = data.teams()
    if not teams:
        st.caption("No teams available.")
        return

    tab_profile, tab_compare = st.tabs(["Team profile", "Compare teams"])

    with tab_profile:
        _profile(teams)

    with tab_compare:
        _compare(teams)


def _profile(teams: list[str]) -> None:
    """Single-team deep dive."""
    matches = data.load_matches()
    team = st.selectbox("Team", teams, key="team_profile_select")

    summary = team_summary(matches)
    row = summary[summary["team"] == team]
    if row.empty:
        st.caption("No completed matches for this team.")
        return
    record = row.iloc[0]

    metric_row(
        [
            ("Matches", f"{int(record['matches']):,}", None),
            ("Won", f"{int(record['wins']):,}", f"{record['win_pct']:.1f}% win rate"),
            ("Titles", f"{int(record['titles'])}", None),
            ("Seasons", f"{int(record['seasons'])}", None),
        ]
    )
    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Season by season")
        seasonal = team_season_summary(matches, team)
        if seasonal.empty:
            st.caption("No seasonal data.")
        else:
            st.plotly_chart(
                line_chart(
                    seasonal, "season", {"Win %": "win_pct"},
                    title=f"{team} — win rate by season",
                    x_title="Season", y_title="Win %",
                ),
                width="stretch",
            )
            show_table(
                seasonal[["season", "matches", "wins", "losses", "win_pct"]].rename(
                    columns={
                        "season": "Season", "matches": "P", "wins": "W",
                        "losses": "L", "win_pct": "Win %",
                    }
                )
            )

    with right:
        st.subheader("Recent form")
        timeline = team_form_timeline(matches, team, window=5)
        if timeline.empty:
            st.caption("No form data.")
        else:
            recent = timeline.tail(40)
            st.plotly_chart(
                line_chart(
                    recent, "match_number",
                    {"Rolling win % (last 5)": "rolling_win_pct",
                     "Career win %": "cumulative_win_pct"},
                    title=f"{team} — form over the last 40 matches",
                    x_title="Match number", y_title="Win %",
                ),
                width="stretch",
            )
            last10 = timeline.tail(10)[
                ["match_date", "opponent", "venue", "won"]
            ].copy()
            last10["match_date"] = pd.to_datetime(last10["match_date"]).dt.strftime("%d %b %Y")
            last10["won"] = last10["won"].map({1: "Won", 0: "Lost"})
            show_table(
                last10.rename(
                    columns={
                        "match_date": "Date", "opponent": "Opponent",
                        "venue": "Venue", "won": "Result",
                    }
                ),
                caption="Most recent 10 matches.",
            )

    st.divider()

    # --- squad contributions ---
    st.subheader(f"Leading performers for {team}")
    batting = data.load_batting()
    bowling = data.load_bowling()

    left, right = st.columns(2, gap="large")
    with left:
        batters = batting_leaderboard(batting, team=team, min_innings=5).head(12)
        if batters.empty:
            st.caption("No batting data.")
        else:
            st.plotly_chart(
                bar_chart(
                    batters.sort_values("runs"), "player", "runs",
                    title="Most runs", orientation="h", height=440,
                    colors=[team_color(team)] * len(batters),
                    text_format=",d", x_title="Runs", y_title="",
                ),
                width="stretch",
            )
            show_table(
                batters[["player", "innings", "runs", "average", "strike_rate",
                         "fifties", "hundreds"]].rename(
                    columns={
                        "player": "Player", "innings": "Inns", "runs": "Runs",
                        "average": "Avg", "strike_rate": "SR",
                        "fifties": "50s", "hundreds": "100s",
                    }
                )
            )

    with right:
        bowlers = bowling_leaderboard(bowling, team=team, min_innings=5).head(12)
        if bowlers.empty:
            st.caption("No bowling data.")
        else:
            st.plotly_chart(
                bar_chart(
                    bowlers.sort_values("wickets"), "player", "wickets",
                    title="Most wickets", orientation="h", height=440,
                    colors=[team_color(team)] * len(bowlers),
                    text_format="d", x_title="Wickets", y_title="",
                ),
                width="stretch",
            )
            show_table(
                bowlers[["player", "innings", "wickets", "average", "economy",
                         "strike_rate"]].rename(
                    columns={
                        "player": "Player", "innings": "Inns", "wickets": "Wkts",
                        "average": "Avg", "economy": "Econ", "strike_rate": "SR",
                    }
                )
            )


def _compare(teams: list[str]) -> None:
    """Side-by-side comparison of two or more franchises."""
    matches = data.load_matches()

    selected = st.multiselect(
        "Teams to compare",
        teams,
        default=teams[:2] if len(teams) >= 2 else teams,
        key="team_compare_select",
        max_selections=6,
    )
    if len(selected) < 2:
        st.info("Pick at least two teams to compare.")
        return

    summary = team_summary(matches)
    comparison = summary[summary["team"].isin(selected)].copy()
    if comparison.empty:
        st.caption("No data for the selected teams.")
        return

    display = comparison[
        ["team", "matches", "wins", "losses", "win_pct", "titles",
         "home_win_pct", "away_win_pct", "bat_first_win_pct", "chasing_win_pct"]
    ].rename(
        columns={
            "team": "Team", "matches": "P", "wins": "W", "losses": "L",
            "win_pct": "Win %", "titles": "Titles",
            "home_win_pct": "Home win %", "away_win_pct": "Away win %",
            "bat_first_win_pct": "Bat-first win %", "chasing_win_pct": "Chasing win %",
        }
    )
    show_table(display)

    st.plotly_chart(
        bar_chart(
            comparison.sort_values("win_pct"), "team", "win_pct",
            title="Overall win percentage",
            colors=team_palette(comparison.sort_values("win_pct")["team"].tolist()),
            orientation="h", height=380, text_format=".1f",
            x_title="Win %", y_title="",
        ),
        width="stretch",
    )

    st.plotly_chart(
        grouped_bar_chart(
            comparison,
            "team",
            {
                "Home": "home_win_pct",
                "Away": "away_win_pct",
                "Batting first": "bat_first_win_pct",
                "Chasing": "chasing_win_pct",
            },
            title="Win rate by situation",
            x_title="", y_title="Win %",
        ),
        width="stretch",
    )

    st.subheader("Win rate by season")
    seasonal = team_season_summary(matches)
    seasonal = seasonal[seasonal["team"].isin(selected)]
    if not seasonal.empty:
        pivot = seasonal.pivot(index="season", columns="team", values="win_pct").reset_index()
        present = [t for t in selected if t in pivot.columns]
        st.plotly_chart(
            line_chart(
                pivot, "season", {t: t for t in present},
                title="Win percentage by season",
                colors=team_palette(present),
                x_title="Season", y_title="Win %",
            ),
            width="stretch",
        )
        show_table(pivot)
