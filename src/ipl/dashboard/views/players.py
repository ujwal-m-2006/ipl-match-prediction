"""Player analytics page: leaderboards, career profiles and comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.player import (
    batting_leaderboard,
    bowling_leaderboard,
    compare_players,
    player_career_summary,
    player_season_trend,
    player_venue_split,
    player_vs_opposition,
)
from .. import data
from ..theme import CATEGORICAL, bar_chart, grouped_bar_chart, line_chart
from ._common import (
    format_number,
    metric_row,
    page_header,
    require_data,
    season_filter,
    show_table,
)


def render() -> None:
    """Render the player analytics page."""
    if not require_data():
        return

    page_header(
        "Player Analytics",
        "Career records, season trends and head-to-head player comparison.",
    )

    tab_leaders, tab_profile, tab_compare = st.tabs(
        ["Leaderboards", "Player profile", "Compare players"]
    )

    with tab_leaders:
        _leaderboards()

    with tab_profile:
        _profile()

    with tab_compare:
        _compare()


def _leaderboards() -> None:
    """Batting and bowling leaderboards with a season filter."""
    batting = data.load_batting()
    bowling = data.load_bowling()

    left, right = st.columns([1, 1])
    with left:
        season = season_filter(data.seasons(), key="leader_season")
    with right:
        min_innings = st.slider("Minimum innings", 1, 50, 10, key="leader_min_innings")

    bat_tab, bowl_tab = st.tabs(["Batting", "Bowling"])

    with bat_tab:
        leaders = batting_leaderboard(batting, season=season, min_innings=min_innings)
        if leaders.empty:
            st.caption("No batting data for this filter.")
        else:
            top = leaders.head(15).sort_values("runs")
            st.plotly_chart(
                bar_chart(
                    top, "player", "runs",
                    title="Most runs", orientation="h", height=520,
                    colors=[CATEGORICAL[2]] * len(top),
                    text_format=",d", x_title="Runs", y_title="",
                ),
                use_container_width=True,
            )
            show_table(
                leaders.head(50)[
                    ["player", "innings", "runs", "average", "strike_rate",
                     "highest_score", "fifties", "hundreds", "fours", "sixes"]
                ].rename(
                    columns={
                        "player": "Player", "innings": "Inns", "runs": "Runs",
                        "average": "Avg", "strike_rate": "SR",
                        "highest_score": "HS", "fifties": "50s", "hundreds": "100s",
                        "fours": "4s", "sixes": "6s",
                    }
                ),
                height=460,
            )

    with bowl_tab:
        leaders = bowling_leaderboard(bowling, season=season, min_innings=min_innings)
        if leaders.empty:
            st.caption("No bowling data for this filter.")
        else:
            top = leaders.head(15).sort_values("wickets")
            st.plotly_chart(
                bar_chart(
                    top, "player", "wickets",
                    title="Most wickets", orientation="h", height=520,
                    colors=[CATEGORICAL[7]] * len(top),
                    text_format="d", x_title="Wickets", y_title="",
                ),
                use_container_width=True,
            )
            show_table(
                leaders.head(50)[
                    ["player", "innings", "overs", "wickets", "average", "economy",
                     "strike_rate", "best_wickets", "four_wicket_hauls", "five_wicket_hauls"]
                ].rename(
                    columns={
                        "player": "Player", "innings": "Inns", "overs": "Overs",
                        "wickets": "Wkts", "average": "Avg", "economy": "Econ",
                        "strike_rate": "SR", "best_wickets": "Best",
                        "four_wicket_hauls": "4W", "five_wicket_hauls": "5W",
                    }
                ),
                height=460,
            )


def _profile() -> None:
    """Career profile for one player."""
    everyone = data.players(min_matches=3)
    if not everyone:
        st.caption("No players available.")
        return

    player = st.selectbox("Player", everyone, key="player_profile_select")

    batting = data.load_batting()
    bowling = data.load_bowling()
    matches = data.load_matches()

    profile = player_career_summary(player, batting, bowling, matches)
    bat = profile.get("batting") or {}
    bowl = profile.get("bowling") or {}

    st.markdown(f"### {player}")
    teams = ", ".join(profile.get("teams") or []) or "—"
    seasons = profile.get("seasons") or []
    span = f"{int(min(seasons))}–{int(max(seasons))}" if seasons else "—"
    st.caption(f"Teams: {teams}  ·  Seasons: {span}")

    metric_row(
        [
            ("Matches", format_number(profile["matches"]), None),
            ("Runs", format_number(bat.get("runs")), f"Avg {bat.get('average', float('nan')):.1f}"
             if bat.get("average") == bat.get("average") else None),
            ("Wickets", format_number(bowl.get("wickets")),
             f"Econ {bowl.get('economy', float('nan')):.2f}"
             if bowl.get("economy") == bowl.get("economy") else None),
            ("Player of the Match", format_number(profile["player_of_match_awards"]), None),
        ]
    )
    st.write("")

    detail_left, detail_right = st.columns(2, gap="large")
    with detail_left:
        if bat:
            st.markdown("**Batting**")
            show_table(
                pd.DataFrame(
                    [
                        {"Metric": "Innings", "Value": bat.get("innings")},
                        {"Metric": "Runs", "Value": bat.get("runs")},
                        {"Metric": "Average", "Value": bat.get("average")},
                        {"Metric": "Strike rate", "Value": bat.get("strike_rate")},
                        {"Metric": "Highest score", "Value": bat.get("highest_score")},
                        {"Metric": "Fifties", "Value": bat.get("fifties")},
                        {"Metric": "Hundreds", "Value": bat.get("hundreds")},
                        {"Metric": "Fours", "Value": bat.get("fours")},
                        {"Metric": "Sixes", "Value": bat.get("sixes")},
                        {"Metric": "Boundary %", "Value": bat.get("boundary_pct")},
                    ]
                )
            )
    with detail_right:
        if bowl:
            st.markdown("**Bowling**")
            show_table(
                pd.DataFrame(
                    [
                        {"Metric": "Innings", "Value": bowl.get("innings")},
                        {"Metric": "Overs", "Value": bowl.get("overs")},
                        {"Metric": "Wickets", "Value": bowl.get("wickets")},
                        {"Metric": "Average", "Value": bowl.get("average")},
                        {"Metric": "Economy", "Value": bowl.get("economy")},
                        {"Metric": "Strike rate", "Value": bowl.get("strike_rate")},
                        {"Metric": "Best (wickets)", "Value": bowl.get("best_wickets")},
                        {"Metric": "4-wicket hauls", "Value": bowl.get("four_wicket_hauls")},
                        {"Metric": "5-wicket hauls", "Value": bowl.get("five_wicket_hauls")},
                    ]
                )
            )

    st.divider()
    st.subheader("Season trend")
    trend = player_season_trend(player, batting, bowling)
    if trend.empty:
        st.caption("No season data.")
    else:
        if "runs" in trend.columns:
            st.plotly_chart(
                line_chart(
                    trend, "season", {"Runs": "runs"},
                    title="Runs by season", x_title="Season", y_title="Runs",
                ),
                use_container_width=True,
            )
        if "wickets" in trend.columns:
            # Wickets live on a different scale from runs, so they get their own
            # chart rather than a second axis.
            st.plotly_chart(
                line_chart(
                    trend, "season", {"Wickets": "wickets"},
                    title="Wickets by season", x_title="Season", y_title="Wickets",
                    colors=[CATEGORICAL[7]],
                ),
                use_container_width=True,
            )
        show_table(trend)

    st.divider()
    split_left, split_right = st.columns(2, gap="large")
    with split_left:
        st.subheader("By venue")
        show_table(
            player_venue_split(player, batting).head(15).rename(
                columns={
                    "venue": "Venue", "innings": "Inns", "runs": "Runs",
                    "average": "Avg", "strike_rate": "SR", "highest_score": "HS",
                }
            )
        )
    with split_right:
        st.subheader("By opposition")
        show_table(
            player_vs_opposition(player, batting).head(15).rename(
                columns={
                    "opposition": "Opposition", "innings": "Inns", "runs": "Runs",
                    "average": "Avg", "strike_rate": "SR",
                }
            )
        )


def _compare() -> None:
    """Compare two or more players side by side."""
    everyone = data.players(min_matches=10)
    if len(everyone) < 2:
        st.caption("Not enough players to compare.")
        return

    selected = st.multiselect(
        "Players", everyone, default=everyone[:2], key="player_compare", max_selections=5
    )
    if len(selected) < 2:
        st.info("Pick at least two players to compare.")
        return

    batting = data.load_batting()
    bowling = data.load_bowling()
    matches = data.load_matches()

    comparison = compare_players(selected, batting, bowling, matches)
    show_table(
        comparison.rename(
            columns={
                "player": "Player", "matches": "Matches", "runs": "Runs",
                "batting_average": "Bat avg", "strike_rate": "SR",
                "fifties": "50s", "hundreds": "100s", "wickets": "Wkts",
                "bowling_average": "Bowl avg", "economy": "Econ",
                "player_of_match_awards": "PoM awards",
            }
        )
    )

    st.plotly_chart(
        grouped_bar_chart(
            comparison, "player",
            {"Runs": "runs"},
            title="Career runs", x_title="", y_title="Runs",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        grouped_bar_chart(
            comparison, "player",
            {"Wickets": "wickets"},
            title="Career wickets", x_title="", y_title="Wickets",
            colors=[CATEGORICAL[7]],
        ),
        use_container_width=True,
    )

    st.subheader("Runs by season")
    frames = []
    for player in selected:
        trend = player_season_trend(player, batting, bowling)
        if trend.empty or "runs" not in trend.columns:
            continue
        frames.append(trend[["season", "runs"]].assign(player=player))
    if frames:
        combined = pd.concat(frames)
        pivot = combined.pivot_table(
            index="season", columns="player", values="runs", aggfunc="sum"
        ).reset_index()
        present = [p for p in selected if p in pivot.columns]
        st.plotly_chart(
            line_chart(
                pivot, "season", {p: p for p in present},
                title="Runs by season", x_title="Season", y_title="Runs",
            ),
            use_container_width=True,
        )
        show_table(pivot)
