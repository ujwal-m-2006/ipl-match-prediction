"""Admin page: refresh data, retrain models and inspect system health.

The destructive-ish actions here (a refresh run, a retrain) are gated behind a
password so a publicly-deployed dashboard cannot be made to hammer the IPL feed
host by anyone who finds the URL.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...config import get_settings
from ...constants import CRICSHEET_JSON_URL, FEED_BASE_URL, IPL_COMPETITIONS, IPL_WEBSITE
from ...models.persistence import list_artifacts
from ...models.registry import describe_availability
from .. import data
from ._common import metric_row, page_header, show_table, stat_table

SESSION_KEY = "admin_authenticated"


def render() -> None:
    """Render the admin page."""
    page_header("Admin", "Data refresh, model retraining and system health.")

    settings = get_settings()

    tab_health, tab_refresh, tab_train, tab_config = st.tabs(
        ["Health", "Refresh data", "Retrain models", "Configuration"]
    )

    with tab_health:
        _health()
    with tab_refresh:
        _refresh(settings)
    with tab_train:
        _train(settings)
    with tab_config:
        _config(settings)


# ---------------------------------------------------------------------------
def _authenticate(settings, scope: str) -> bool:  # noqa: ANN001
    """Password-gate the write actions.

    Streamlit renders every tab's body on each run, so this function executes
    once per calling tab. Widget keys must therefore be namespaced by ``scope``
    -- two widgets sharing a key raises ``StreamlitDuplicateElementKey`` and
    takes the whole page down. Unlocking in either tab unlocks both, since the
    result is stored in session state.
    """
    if st.session_state.get(SESSION_KEY):
        return True

    if settings.admin_password == "change-me":
        st.warning(
            "The admin password is still the default. Set `IPL_ADMIN_PASSWORD` "
            "in your `.env` (or Streamlit secrets) before deploying publicly."
        )

    password = st.text_input(
        "Admin password", type="password", key=f"admin_password_input_{scope}"
    )
    if st.button("Unlock", key=f"admin_unlock_{scope}"):
        if password == settings.admin_password:
            st.session_state[SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
def _health() -> None:
    """Database counts, last pipeline run and trained artefacts."""
    if not data.has_data():
        st.warning("The database is empty. Load data from the **Refresh data** tab.")
    else:
        summary = data.database_summary()
        metric_row(
            [
                ("Matches", f"{summary.get('matches', 0):,}", None),
                ("Completed", f"{summary.get('completed_matches', 0):,}", None),
                ("Players", f"{summary.get('players', 0):,}", None),
                ("Deliveries", f"{summary.get('deliveries', 0):,}", None),
            ]
        )
        st.write("")
        counts = pd.DataFrame(
            [{"Table": k, "Rows": v} for k, v in summary.items()]
        ).sort_values("Rows", ascending=False)
        show_table(counts)

    st.divider()
    st.subheader("Last pipeline run")
    run = data.latest_ingestion_run()
    if not run:
        st.caption("No ingestion run has been recorded.")
    else:
        status = run.get("status", "unknown")
        icon = {"success": "✅", "failed": "❌", "running": "⏳"}.get(status, "•")
        st.markdown(f"**{icon} {status.title()}** · triggered by `{run.get('trigger')}`")
        metric_row(
            [
                ("Inserted", f"{run.get('matches_inserted', 0):,}", None),
                ("Updated", f"{run.get('matches_updated', 0):,}", None),
                ("Skipped", f"{run.get('matches_skipped', 0):,}", None),
                (
                    "Duration",
                    f"{run.get('duration_seconds') or 0:.0f}s",
                    None,
                ),
            ]
        )
        if run.get("message"):
            st.code(run["message"], language="text")

    st.divider()
    st.subheader("Trained models")
    artifacts = list_artifacts()
    rows = [
        {
            "Task": task,
            "Status": "ready" if info["exists"] else "not trained",
            "Best model": info.get("best_model") or "—",
            "Trained at": info.get("trained_at") or "—",
            "Size (KB)": info.get("size_kb"),
        }
        for task, info in artifacts.items()
    ]
    show_table(pd.DataFrame(rows))

    st.subheader("Model libraries")
    availability = describe_availability()
    show_table(
        pd.DataFrame(
            [
                {"Library": name, "Installed": "yes" if ok else "no"}
                for name, ok in availability.items()
            ]
        )
    )


# ---------------------------------------------------------------------------
def _refresh(settings) -> None:  # noqa: ANN001
    """Trigger an ingestion run from the browser."""
    st.markdown(
        f"""
        Collects data from the official IPL feeds ({IPL_WEBSITE}), supplemented by
        [Cricsheet]({CRICSHEET_JSON_URL}) for the 2008-2018 seasons the official
        feed does not publish.

        Requests are rate-limited to one every **{settings.request_delay:.1f}s** and
        cached on disk, so a refresh mostly re-reads local files and only fetches
        what has actually changed.
        """
    )

    if not _authenticate(settings, "refresh"):
        return

    seasons = sorted(IPL_COMPETITIONS)
    col1, col2 = st.columns(2)
    with col1:
        scope = st.radio(
            "Scope",
            ["Latest season only", "All official seasons", "Everything (incl. Cricsheet)"],
            key="admin_refresh_scope",
        )
    with col2:
        deliveries = st.checkbox(
            "Include ball-by-ball data", value=settings.ingest_deliveries,
            key="admin_refresh_deliveries",
            help="Much slower, but required by the chase model.",
        )
        full = st.checkbox(
            "Re-parse completed matches", value=False, key="admin_refresh_full",
            help="Off by default: an incremental run only fetches new results.",
        )

    if not st.button("Start refresh", type="primary", key="admin_refresh_button"):
        return

    selected: list[int] | None
    enable_cricsheet = False
    if scope == "Latest season only":
        selected = [max(seasons)]
    elif scope == "All official seasons":
        selected = seasons
    else:
        selected = None
        enable_cricsheet = True

    from ...ingestion import run_ingestion

    with st.spinner("Collecting data — this can take several minutes..."):
        try:
            stats = run_ingestion(
                seasons=selected,
                skip_completed=not full,
                ingest_deliveries=deliveries,
                enable_cricsheet=enable_cricsheet,
                trigger="dashboard",
            )
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            return

    data.clear_caches()
    st.success(f"Refresh complete — {stats.summary()}")
    st.rerun()


# ---------------------------------------------------------------------------
def _train(settings) -> None:  # noqa: ANN001
    """Trigger a retraining run from the browser."""
    st.markdown(
        """
        Retrains all four models, comparing every available algorithm and keeping
        the best of each. The chase model reads the full ball-by-ball table and is
        by far the slowest step.
        """
    )

    if not data.has_data():
        st.warning("Load data before training.")
        return

    if not _authenticate(settings, "train"):
        return

    seasons = data.seasons()
    col1, col2 = st.columns(2)
    with col1:
        test_from = st.selectbox(
            "Hold out seasons from",
            seasons,
            index=max(0, seasons.index(settings.test_season_from))
            if settings.test_season_from in seasons else 0,
            key="admin_test_season",
        )
    with col2:
        include_chase = st.checkbox(
            "Train the chase model", value=True, key="admin_include_chase"
        )
        include_pom = st.checkbox(
            "Train the Player-of-the-Match model", value=True, key="admin_include_pom"
        )

    if not st.button("Start training", type="primary", key="admin_train_button"):
        return

    from ...models.train import train_all

    with st.spinner("Training models — this can take a few minutes..."):
        try:
            results = train_all(
                test_season_from=int(test_from),
                include_chase=include_chase,
                include_pom=include_pom,
            )
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    data.clear_caches()
    st.success("Training complete.")
    show_table(
        pd.DataFrame(
            [
                {
                    "Task": task,
                    "Best model": result.best_model or "—",
                    "Train rows": result.train_rows,
                    "Test rows": result.test_rows,
                }
                for task, result in results.items()
            ]
        )
    )


# ---------------------------------------------------------------------------
def _config(settings) -> None:  # noqa: ANN001
    """Show the effective configuration (with secrets masked)."""
    st.subheader("Effective settings")
    st.caption("Change these with environment variables or a `.env` file.")

    rows = [
        ("Database dialect", settings.dialect),
        ("Database URL", _mask(settings.database_url)),
        ("Request delay (s)", settings.request_delay),
        ("Request timeout (s)", settings.request_timeout),
        ("Max retries", settings.max_retries),
        ("HTTP cache enabled", settings.use_http_cache),
        ("HTTP cache TTL (hours)", settings.http_cache_ttl_hours),
        ("Cricsheet supplement", settings.enable_cricsheet),
        ("Ingest deliveries", settings.ingest_deliveries),
        ("Test seasons from", settings.test_season_from),
        ("Random seed", settings.random_state),
        ("Log level", settings.log_level),
    ]
    # Mixed str/int/float/bool values would land as an object column, which
    # Arrow cannot serialise; stat_table renders them as text instead.
    show_table(stat_table(rows).rename(columns={"Metric": "Setting"}))

    st.divider()
    st.subheader("Data sources")
    show_table(
        pd.DataFrame(
            [
                {"Source": "IPL official website", "Endpoint": IPL_WEBSITE,
                 "Covers": "Reference"},
                {"Source": "IPL official feeds", "Endpoint": FEED_BASE_URL,
                 "Covers": f"{min(IPL_COMPETITIONS)}–{max(IPL_COMPETITIONS)}"},
                {"Source": "Cricsheet", "Endpoint": CRICSHEET_JSON_URL,
                 "Covers": "2008–2018 (supplement)"},
            ]
        )
    )

    st.subheader("Registered competitions")
    show_table(
        pd.DataFrame(
            [{"Season": s, "Competition ID": c} for s, c in sorted(IPL_COMPETITIONS.items())]
        )
    )
    st.caption(
        "Run `python scripts/discover_competitions.py` after a new season starts "
        "to find its competition ID."
    )


def _mask(url: str) -> str:
    """Hide the password in a database URL before displaying it."""
    if "://" not in url or "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    credentials, _, host = rest.rpartition("@")
    if ":" in credentials:
        user, _, _ = credentials.partition(":")
        return f"{scheme}://{user}:****@{host}"
    return url
