"""Predictions page: winner, score, chase, Player of the Match and playoffs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...constants import team_color
from ...models.playoffs import simulate_playoff_qualification
from .. import data
from ..theme import (
    CATEGORICAL,
    bar_chart,
    gauge,
    horizontal_probability_bar,
    line_chart,
    team_palette,
)
from ._common import metric_row, page_header, require_data, require_model, show_table


def render() -> None:
    """Render the predictions page."""
    if not require_data():
        return

    page_header(
        "Predictions",
        "Model-backed forecasts for match outcomes, scores, chases and the playoff race.",
    )

    service = data.prediction_service()
    if service is None:
        st.error("Could not initialise the prediction service.")
        return

    ready = service.model_summary()
    if not ready.empty and (ready["status"] == "not trained").all():
        st.warning(
            "No models have been trained yet. Run `python scripts/train_models.py` "
            "or use the **Admin** page."
        )
        return

    tabs = st.tabs(
        ["Match winner", "First-innings score", "Chase simulator",
         "Player of the Match", "Playoff race", "Upcoming fixtures"]
    )

    with tabs[0]:
        _winner(service)
    with tabs[1]:
        _score(service)
    with tabs[2]:
        _chase(service)
    with tabs[3]:
        _player_of_match(service)
    with tabs[4]:
        _playoffs(service)
    with tabs[5]:
        _upcoming(service)


# ---------------------------------------------------------------------------
def _winner(service) -> None:  # noqa: ANN001
    """Match-winner prediction form and result."""
    if not require_model(service, "winner", "match winner"):
        return

    teams = data.prediction_teams()
    venues = data.venues()
    seasons = data.seasons()

    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.selectbox("Team 1 (home)", teams, key="pred_team1")
    with col2:
        team2 = st.selectbox(
            "Team 2 (away)", [t for t in teams if t != team1], key="pred_team2"
        )
    with col3:
        venue = st.selectbox(
            "Venue", venues,
            index=_default_venue_index(venues, team1),
            key="pred_venue",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        toss_winner = st.selectbox(
            "Toss winner", ["Not decided", team1, team2], key="pred_toss_winner"
        )
    with col5:
        toss_decision = st.selectbox(
            "Toss decision", ["Not decided", "bat", "field"], key="pred_toss_decision"
        )
    with col6:
        is_playoff = st.checkbox("Playoff match", key="pred_playoff")

    if not st.button("Predict winner", type="primary", key="pred_winner_button"):
        return

    try:
        prediction = service.predict_winner(
            team1=team1,
            team2=team2,
            venue=venue,
            season=max(seasons) if seasons else None,
            toss_winner=None if toss_winner == "Not decided" else toss_winner,
            toss_decision=None if toss_decision == "Not decided" else toss_decision,
            is_playoff=is_playoff,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.write("")
    st.markdown(f"### Predicted winner: **{prediction.predicted_winner}**")

    st.plotly_chart(
        horizontal_probability_bar(
            team1, team2, prediction.team1_win_probability,
            color_left=team_color(team1), color_right=team_color(team2),
        ),
        width="stretch",
    )

    metric_row(
        [
            (f"{team1} win probability",
             f"{prediction.team1_win_probability:.1%}", None),
            (f"{team2} win probability",
             f"{prediction.team2_win_probability:.1%}", None),
            ("Model confidence", f"{prediction.confidence:.1%}", None),
            ("Model", prediction.model, None),
        ]
    )

    if prediction.drivers:
        st.write("")
        st.subheader("What is behind this number")
        drivers = pd.DataFrame(prediction.drivers)
        drivers = drivers.rename(
            columns={"label": "Factor", "team1_value": team1, "team2_value": team2}
        ).drop(columns=["unit"], errors="ignore")
        show_table(drivers)
        st.caption(
            "Form and win rates are percentages; runs and rest days are raw values. "
            "Head-to-head is team 1's share of previous meetings."
        )

    st.warning(
        "**Read this number with care.** Measured on held-out seasons, this model's "
        "ROC-AUC is ~0.50-0.55 — barely better than a coin toss. That is not a bug: "
        "pre-match T20 outcomes are close to genuinely unpredictable from team form "
        "and squad records alone. The **Chase simulator** (AUC ~0.90), which sees the "
        "live match state, is where the real predictive power is. See the "
        "**Model Comparison** page for the full evidence.",
        icon=":material/warning:",
    )


def _default_venue_index(venues: list[str], team: str) -> int:
    """Preselect the home ground of the chosen team where possible."""
    from ...constants import TEAM_HOME_VENUES

    home = TEAM_HOME_VENUES.get(team)
    return venues.index(home) if home in venues else 0


# ---------------------------------------------------------------------------
def _score(service) -> None:  # noqa: ANN001
    """First-innings score prediction."""
    if not require_model(service, "score", "first-innings score"):
        return

    teams = data.prediction_teams()
    venues = data.venues()

    col1, col2, col3 = st.columns(3)
    with col1:
        batting = st.selectbox("Batting first", teams, key="score_batting")
    with col2:
        bowling = st.selectbox(
            "Bowling first", [t for t in teams if t != batting], key="score_bowling"
        )
    with col3:
        venue = st.selectbox(
            "Venue", venues, index=_default_venue_index(venues, batting), key="score_venue"
        )

    if not st.button("Predict score", type="primary", key="score_button"):
        return

    try:
        prediction = service.predict_first_innings_score(
            batting_team=batting, bowling_team=bowling, venue=venue
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.write("")
    metric_row(
        [
            ("Predicted total", f"{prediction.predicted_score:.0f}", None),
            ("Likely range",
             f"{prediction.lower_bound:.0f} – {prediction.upper_bound:.0f}", None),
            ("Model", prediction.model, None),
        ]
    )
    st.caption(
        "The range is one standard deviation of the model's held-out residuals, "
        "so roughly two thirds of real totals land inside it."
    )

    # Context: how this compares to the ground's history.
    innings = data.load_innings()
    at_venue = innings[(innings["venue"] == venue) & (innings["innings_no"] == 1)]
    if not at_venue.empty:
        st.write("")
        st.subheader(f"Historical first-innings totals at {venue}")
        by_season = (
            at_venue.groupby("season")["runs"].mean().round(1).reset_index()
        )
        st.plotly_chart(
            line_chart(
                by_season, "season", {"Average first-innings total": "runs"},
                title="", x_title="Season", y_title="Runs",
            ),
            width="stretch",
        )
        metric_row(
            [
                ("Ground average", f"{at_venue['runs'].mean():.0f}", None),
                ("Highest", f"{at_venue['runs'].max():.0f}", None),
                ("Lowest", f"{at_venue['runs'].min():.0f}", None),
                ("Innings recorded", f"{len(at_venue)}", None),
            ]
        )


# ---------------------------------------------------------------------------
def _chase(service) -> None:  # noqa: ANN001
    """In-play chase-success simulator."""
    if not require_model(service, "chase", "chase success"):
        return

    teams = data.prediction_teams()
    venues = data.venues()

    col1, col2, col3 = st.columns(3)
    with col1:
        batting = st.selectbox("Chasing team", teams, key="chase_batting")
    with col2:
        bowling = st.selectbox(
            "Defending team", [t for t in teams if t != batting], key="chase_bowling"
        )
    with col3:
        venue = st.selectbox("Venue", venues, key="chase_venue")

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        target = st.number_input("Target", min_value=1, max_value=350, value=180, key="chase_target")
    with col5:
        current = st.number_input(
            "Runs scored", min_value=0, max_value=400, value=90, key="chase_runs"
        )
    with col6:
        wickets = st.number_input(
            "Wickets lost", min_value=0, max_value=10, value=3, key="chase_wickets"
        )
    with col7:
        overs = st.number_input(
            "Overs bowled", min_value=0.0, max_value=20.0, value=10.0, step=0.1,
            key="chase_overs",
        )

    # Convert cricket over notation to a ball count.
    whole = int(overs)
    balls_bowled = whole * 6 + min(int(round((overs - whole) * 10)), 5)

    if not st.button("Simulate", type="primary", key="chase_button"):
        return

    try:
        prediction = service.predict_chase(
            batting_team=batting, bowling_team=bowling, venue=venue,
            target=int(target), current_runs=int(current),
            wickets_fallen=int(wickets), balls_bowled=balls_bowled,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            gauge(
                prediction.chase_success_probability * 100,
                f"{batting} win probability",
                color=team_color(batting),
            ),
            width="stretch",
        )
    with right:
        metric_row(
            [
                ("Runs required", f"{prediction.runs_required}", None),
                ("Balls remaining", f"{prediction.balls_remaining}", None),
            ]
        )
        metric_row(
            [
                ("Required run rate", f"{prediction.required_run_rate:.2f}", None),
                ("Wickets in hand", f"{10 - prediction.wickets_fallen}", None),
            ]
        )

    st.plotly_chart(
        horizontal_probability_bar(
            batting, bowling, prediction.chase_success_probability,
            color_left=team_color(batting), color_right=team_color(bowling),
        ),
        width="stretch",
    )
    st.caption(f"Model: {prediction.model}")

    # --- sensitivity curve ---
    st.subheader("How the odds move with the score")
    rows = []
    for extra in range(0, 61, 5):
        runs = min(int(current) + extra, int(target) - 1)
        try:
            point = service.predict_chase(
                batting_team=batting, bowling_team=bowling, venue=venue,
                target=int(target), current_runs=runs,
                wickets_fallen=int(wickets), balls_bowled=balls_bowled,
            )
        except Exception:
            continue
        rows.append(
            {"runs": runs, "win_pct": point.chase_success_probability * 100}
        )
    if rows:
        curve = pd.DataFrame(rows).drop_duplicates(subset="runs")
        st.plotly_chart(
            line_chart(
                curve, "runs", {f"{batting} win %": "win_pct"},
                title=f"Win probability at {overs} overs, {wickets} down",
                x_title="Runs on the board", y_title="Win %",
                colors=[team_color(batting)],
            ),
            width="stretch",
        )


# ---------------------------------------------------------------------------
def _player_of_match(service) -> None:  # noqa: ANN001
    """Player-of-the-Match ranking for a completed fixture."""
    if not require_model(service, "pom", "Player of the Match"):
        return

    matches = data.load_matches()
    completed = matches[matches["is_completed"]].sort_values("match_date", ascending=False)
    if completed.empty:
        st.caption("No completed matches.")
        return

    seasons = data.seasons()
    season = st.selectbox("Season", seasons, key="pom_season")
    season_matches = completed[completed["season"] == season]
    if season_matches.empty:
        st.caption("No matches in this season.")
        return

    labels = {
        int(row.match_id): (
            f"{pd.to_datetime(row.match_date):%d %b} · {row.team1} vs {row.team2}"
        )
        for row in season_matches.itertuples(index=False)
    }
    match_id = st.selectbox(
        "Match", list(labels), format_func=lambda mid: labels[mid], key="pom_match"
    )

    try:
        ranking = service.predict_player_of_match(match_id, top_n=8)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    if ranking.empty:
        st.caption("No player data for this match.")
        return

    actual = season_matches[season_matches["match_id"] == match_id].iloc[0][
        "player_of_match"
    ]
    predicted = ranking.iloc[0]["player"]

    left, right = st.columns(2)
    left.metric("Model's pick", predicted)
    right.metric(
        "Actual award", actual or "—",
        "correct" if actual == predicted else ("missed" if actual else None),
    )

    st.plotly_chart(
        bar_chart(
            ranking.sort_values("award_probability"), "player", "award_probability",
            title="Award probability",
            orientation="h", height=380,
            colors=[CATEGORICAL[6]] * len(ranking),
            text_format=".1f", x_title="Probability (%)", y_title="",
        ),
        width="stretch",
    )
    show_table(
        ranking.rename(
            columns={
                "player": "Player", "team": "Team", "runs": "Runs", "balls": "Balls",
                "wickets": "Wkts", "runs_conceded": "Runs conceded",
                "total_impact": "Impact score", "award_probability": "Award %",
            }
        )
    )
    st.caption(
        "Probabilities are normalised across the players in this match, so they sum to 100%."
    )


# ---------------------------------------------------------------------------
def _playoffs(service) -> None:  # noqa: ANN001
    """Monte Carlo playoff-qualification projection."""
    matches = data.load_matches()
    innings = data.load_innings()
    seasons = data.seasons()
    if not seasons:
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        season = st.selectbox("Season", seasons, key="playoff_season")
    with col2:
        simulations = st.select_slider(
            "Simulations", options=[1000, 2500, 5000, 10000], value=5000,
            key="playoff_sims",
        )

    if not st.button("Run simulation", type="primary", key="playoff_button"):
        return

    with st.spinner(f"Simulating the rest of IPL {season}..."):
        projection = simulate_playoff_qualification(
            matches, innings, season,
            service=service if service.has_model("winner") else None,
            simulations=int(simulations),
        )

    if projection.table.empty:
        st.caption("Not enough data to simulate this season.")
        return

    if projection.matches_remaining == 0:
        st.info(
            f"The IPL {season} league stage is complete — qualification is already decided."
        )
    else:
        st.caption(
            f"{projection.matches_remaining} league matches remaining · "
            f"{projection.simulations:,} simulations"
        )

    table = projection.table
    st.plotly_chart(
        bar_chart(
            table.sort_values("qualification_pct"), "team", "qualification_pct",
            title=f"Probability of reaching the IPL {season} playoffs",
            colors=team_palette(table.sort_values("qualification_pct")["team"].tolist()),
            orientation="h", height=max(400, 34 * len(table)),
            text_format=".1f", x_title="Qualification probability (%)", y_title="",
        ),
        width="stretch",
    )

    show_table(
        table.rename(
            columns={
                "rank": "#", "team": "Team", "current_points": "Pts",
                "net_run_rate": "NRR", "matches_played": "P",
                "matches_remaining": "Left", "max_possible_points": "Max pts",
                "qualification_pct": "Qualify %", "expected_position": "Exp. finish",
            }
        )
    )
    st.caption(
        "Each remaining fixture is drawn using the winner model's probability. "
        "Points ties are broken by current net run rate, since simulating future "
        "NRR would require simulating scores ball by ball."
    )


# ---------------------------------------------------------------------------
def _upcoming(service) -> None:  # noqa: ANN001
    """Predictions for every scheduled fixture."""
    if not require_model(service, "winner", "match winner"):
        return

    seasons = data.seasons()
    season = st.selectbox("Season", seasons, key="upcoming_season")

    with st.spinner("Scoring scheduled fixtures..."):
        predictions = service.score_upcoming_fixtures(season)

    if predictions.empty:
        st.info(f"No unplayed fixtures for IPL {season}.")
        return

    display = predictions.copy()
    display["match_date"] = pd.to_datetime(display["match_date"]).dt.strftime("%d %b %Y")
    show_table(
        display.rename(
            columns={
                "match_date": "Date", "team1": "Team 1", "team2": "Team 2",
                "venue": "Venue", "predicted_winner": "Predicted winner",
                "team1_win_pct": "Team 1 win %", "team2_win_pct": "Team 2 win %",
            }
        ),
        height=480,
    )
