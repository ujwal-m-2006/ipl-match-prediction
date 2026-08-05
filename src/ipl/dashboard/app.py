"""Streamlit dashboard entry point.

Pages are declared programmatically with ``st.navigation`` so the whole app
lives inside the ``ipl`` package rather than in a loose top-level ``pages/``
directory. ``streamlit_app.py`` at the repository root is a three-line shim
that calls :func:`main`, which is what Streamlit Cloud runs.
"""

from __future__ import annotations

import streamlit as st

from ..logging_utils import setup_logging
from .theme import CUSTOM_CSS

PAGE_CONFIG = {
    "page_title": "IPL Analytics & Prediction",
    "page_icon": "🏏",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


def main() -> None:
    """Configure and run the multi-page dashboard."""
    st.set_page_config(**PAGE_CONFIG)
    setup_logging()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _ensure_schema()

    # Imported here, after set_page_config, so a slow import never delays the
    # first paint of the page chrome.
    from .views import (
        admin,
        head_to_head,
        home,
        model_comparison,
        players,
        predictions,
        schedule,
        teams,
        venues,
    )

    # Every page module exposes a function called `render`, so Streamlit would
    # derive the same URL path for all of them and refuse to build the nav.
    # `url_path` is therefore given explicitly for each page.
    navigation = st.navigation(
        {
            "Overview": [
                st.Page(
                    home.render, title="Home", icon=":material/home:",
                    url_path="home", default=True,
                ),
                st.Page(
                    schedule.render, title="Schedule & Results",
                    icon=":material/event:", url_path="schedule",
                ),
            ],
            "Analytics": [
                st.Page(
                    teams.render, title="Team Analytics",
                    icon=":material/groups:", url_path="teams",
                ),
                st.Page(
                    players.render, title="Player Analytics",
                    icon=":material/person:", url_path="players",
                ),
                st.Page(
                    head_to_head.render, title="Head to Head",
                    icon=":material/compare_arrows:", url_path="head-to-head",
                ),
                st.Page(
                    venues.render, title="Venue Statistics",
                    icon=":material/stadium:", url_path="venues",
                ),
            ],
            "Machine Learning": [
                st.Page(
                    predictions.render, title="Predictions",
                    icon=":material/insights:", url_path="predictions",
                ),
                st.Page(
                    model_comparison.render, title="Model Comparison",
                    icon=":material/analytics:", url_path="model-comparison",
                ),
            ],
            "System": [
                st.Page(
                    admin.render, title="Admin",
                    icon=":material/settings:", url_path="admin",
                ),
            ],
        }
    )

    _sidebar_footer()
    navigation.run()


@st.cache_resource(show_spinner=False)
def _ensure_schema() -> bool:
    """Create the (empty) schema on first run so a fresh clone renders.

    Cached as a resource so the ``CREATE TABLE IF NOT EXISTS`` round trip runs
    once per process rather than on every interaction.
    """
    from ..db.base import init_db

    try:
        init_db()
        return True
    except Exception as exc:  # pragma: no cover - unreachable DB
        st.error(
            f"Could not connect to the database ({exc}). Check `IPL_DATABASE_URL`."
        )
        return False


def _sidebar_footer() -> None:
    """Data provenance and freshness, shown on every page."""
    from .data import database_summary, has_data, latest_ingestion_run

    with st.sidebar:
        st.markdown("### 🏏 IPL Analytics")

        if not has_data():
            st.warning("No data yet. Open the **Admin** page to load it.")
            return

        summary = database_summary()
        st.caption(
            f"**{summary.get('completed_matches', 0):,}** matches · "
            f"**{summary.get('players', 0):,}** players · "
            f"**{summary.get('deliveries', 0):,}** deliveries"
        )

        run = latest_ingestion_run()
        if run and run.get("finished_at"):
            status = run.get("status", "unknown")
            icon = {"success": "✅", "failed": "❌"}.get(status, "⏳")
            st.caption(f"{icon} Last refresh: {run['finished_at']:%d %b %Y %H:%M}")

        st.divider()
        st.caption(
            "Primary source: [iplt20.com](https://www.iplt20.com/) official feeds. "
            "Seasons 2008-2018 supplemented from "
            "[Cricsheet](https://cricsheet.org/)."
        )


if __name__ == "__main__":  # pragma: no cover
    main()
