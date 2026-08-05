"""Head-to-head page: the complete record between two franchises."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.team import head_to_head
from ...constants import team_color
from .. import data
from ..theme import bar_chart, grouped_bar_chart, horizontal_probability_bar, team_palette
from ._common import metric_row, page_header, require_data, show_table


def render() -> None:
    """Render the head-to-head page."""
    if not require_data():
        return

    page_header("Head to Head", "Every meeting between two franchises.")

    teams = data.teams()
    if len(teams) < 2:
        st.caption("Not enough teams.")
        return

    left, right = st.columns(2)
    with left:
        team_a = st.selectbox("Team A", teams, index=0, key="h2h_a")
    with right:
        options = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", options, index=0, key="h2h_b")

    matches = data.load_matches()
    record = head_to_head(matches, team_a, team_b)

    if record["matches"] == 0:
        st.info(f"{team_a} and {team_b} have never met in the IPL.")
        return

    metric_row(
        [
            ("Meetings", f"{record['matches']}", None),
            (f"{team_a} wins", f"{record['team_a_wins']}",
             f"{record['team_a_win_pct']:.1f}%"),
            (f"{team_b} wins", f"{record['team_b_wins']}",
             f"{record['team_b_win_pct']:.1f}%"),
            (
                "Current streak",
                f"{record['current_streak']}",
                record["current_streak_team"] or None,
            ),
        ]
    )
    st.write("")

    # The split bar is directly labelled with both team names and percentages,
    # so it reads correctly regardless of colour perception.
    st.plotly_chart(
        horizontal_probability_bar(
            team_a, team_b,
            record["team_a_wins"] / record["matches"] if record["matches"] else 0.5,
            color_left=team_color(team_a),
            color_right=team_color(team_b),
        ),
        width="stretch",
    )
    if record["no_result"]:
        st.caption(f"{record['no_result']} meeting(s) ended without a result.")

    st.divider()

    tab_season, tab_venue, tab_matches = st.tabs(
        ["By season", "By venue", "All meetings"]
    )

    with tab_season:
        by_season = record["by_season"]
        if by_season.empty:
            st.caption("No seasonal breakdown available.")
        else:
            renamed = by_season.rename(columns={"a_wins": team_a, "b_wins": team_b})
            st.plotly_chart(
                grouped_bar_chart(
                    renamed, "season", {team_a: team_a, team_b: team_b},
                    title="Wins by season",
                    colors=team_palette([team_a, team_b]),
                    x_title="Season", y_title="Wins",
                ),
                width="stretch",
            )
            show_table(
                renamed.rename(columns={"season": "Season", "matches": "Meetings"})
            )

    with tab_venue:
        by_venue = record["by_venue"]
        if by_venue.empty:
            st.caption("No venue breakdown available.")
        else:
            renamed = by_venue.rename(columns={"a_wins": team_a, "b_wins": team_b})
            st.plotly_chart(
                grouped_bar_chart(
                    renamed.head(12), "venue", {team_a: team_a, team_b: team_b},
                    title="Wins by venue",
                    colors=team_palette([team_a, team_b]),
                    x_title="", y_title="Wins", height=460,
                ),
                width="stretch",
            )
            show_table(
                renamed.rename(columns={"venue": "Venue", "matches": "Meetings"})
            )

    with tab_matches:
        meetings = record["matches_list"].copy()
        if meetings.empty:
            st.caption("No meetings recorded.")
        else:
            meetings["match_date"] = pd.to_datetime(
                meetings["match_date"]
            ).dt.strftime("%d %b %Y")
            show_table(
                meetings.rename(
                    columns={
                        "match_date": "Date", "season": "Season", "venue": "Venue",
                        "team1": "Team 1", "team2": "Team 2", "winner": "Winner",
                        "result_summary": "Result",
                        "player_of_match": "Player of the Match",
                    }
                ),
                height=520,
            )

    st.divider()

    # --- prediction shortcut ---
    st.subheader("Predict the next meeting")
    service = data.prediction_service()
    if service is None or not service.has_model("winner"):
        st.info(
            "Train the models to enable predictions: `python scripts/train_models.py`"
        )
        return

    venues = data.venues()
    venue = st.selectbox("Venue", venues, key="h2h_predict_venue")
    if st.button("Predict", key="h2h_predict_button", type="primary"):
        try:
            prediction = service.predict_winner(team1=team_a, team2=team_b, venue=venue)
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")
            return

        st.success(
            f"**{prediction.predicted_winner}** is favoured "
            f"({max(prediction.team1_win_probability, prediction.team2_win_probability):.1%})"
        )
        st.plotly_chart(
            horizontal_probability_bar(
                team_a, team_b, prediction.team1_win_probability,
                color_left=team_color(team_a), color_right=team_color(team_b),
            ),
            width="stretch",
        )
        st.caption(f"Model: {prediction.model}")
