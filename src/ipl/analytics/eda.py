"""Exploratory data analysis: generates a standalone HTML report with figures.

Run via ``python scripts/run_eda.py``. Produces ``reports/eda_report.html``
plus PNG figures under ``reports/figures/``, both git-ignored and regenerable.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Force a non-interactive backend: the report is generated headlessly, often in
# CI, where no display is available.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from ..constants import team_color  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402
from .player import batting_leaderboard, bowling_leaderboard  # noqa: E402
from .team import batting_first_advantage, team_summary, toss_impact  # noqa: E402
from .venue import scoring_trend_by_season, venue_summary  # noqa: E402

logger = get_logger(__name__)

FIGURE_SIZE = (10, 5.5)
DPI = 110


def _save_figure(fig: plt.Figure, name: str) -> str:
    """Write a figure to ``reports/figures`` and return its relative path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure %s", path.name)
    return f"figures/{path.name}"


def _bar(data: pd.DataFrame, x: str, y: str, title: str, name: str,
         *, colors: list[str] | None = None, ylabel: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(data[x], data[y], color=colors or "#2563eb")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel or y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return _save_figure(fig, name)


def _line(data: pd.DataFrame, x: str, ys: list[str], title: str, name: str,
          *, ylabel: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for column in ys:
        ax.plot(data[x], data[column], marker="o", linewidth=2,
                label=column.replace("_", " ").title())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel or "")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if len(ys) > 1:
        ax.legend()
    return _save_figure(fig, name)


def generate_figures(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    deliveries: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Render every EDA figure and return ``{caption: relative_path}``."""
    figures: dict[str, str] = {}

    # --- team win rates ---
    teams = team_summary(matches, min_matches=10)
    if not teams.empty:
        top = teams.head(12)
        figures["Win percentage by franchise (min. 10 matches)"] = _bar(
            top, "team", "win_pct", "Win Percentage by Franchise", "team_win_pct",
            colors=[team_color(t) for t in top["team"]], ylabel="Win %",
        )

    # --- matches per season ---
    per_season = matches.groupby("season").size().reset_index(name="matches")
    if not per_season.empty:
        figures["Matches played per season"] = _bar(
            per_season, "season", "matches", "Matches per Season", "matches_per_season",
        )

    # --- scoring trend ---
    trend = scoring_trend_by_season(innings)
    if not trend.empty:
        figures["Average first-innings score by season"] = _line(
            trend, "season", ["avg_score"],
            "Average First-Innings Score by Season", "scoring_trend", ylabel="Runs",
        )
        figures["Boundaries per innings by season"] = _line(
            trend, "season", ["avg_fours", "avg_sixes"],
            "Boundaries per First Innings by Season", "boundary_trend", ylabel="Count",
        )

    # --- toss impact ---
    toss = toss_impact(matches, by="season")
    if not toss.empty:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.bar(toss["group"], toss["toss_winner_advantage_pct"], color="#7c3aed")
        # A 50% line is the "toss is worthless" null hypothesis.
        ax.axhline(50, color="#dc2626", linestyle="--", linewidth=1.5,
                   label="No advantage (50%)")
        ax.set_title("Does Winning the Toss Help?", fontsize=13, fontweight="bold")
        ax.set_ylabel("Toss winner's win %")
        ax.set_xlabel("Season")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        figures["Toss advantage by season"] = _save_figure(fig, "toss_impact")

    # --- batting first vs chasing ---
    advantage = batting_first_advantage(matches, innings)
    if not advantage.empty:
        figures["Batting first vs chasing, by season"] = _line(
            advantage, "season", ["bat_first_win_pct", "chase_win_pct"],
            "Batting First vs Chasing Win Rate", "bat_first_vs_chase", ylabel="Win %",
        )

    # --- venue scoring ---
    venues = venue_summary(matches, innings, min_matches=15)
    if not venues.empty and "avg_first_innings" in venues.columns:
        top = venues.nlargest(12, "matches").sort_values("avg_first_innings", ascending=False)
        figures["Average first-innings score by venue"] = _bar(
            top, "venue", "avg_first_innings",
            "Average First-Innings Score by Venue", "venue_scoring", ylabel="Runs",
        )

    # --- top run scorers ---
    batters = batting_leaderboard(batting, min_innings=20)
    if not batters.empty:
        figures["Leading run scorers"] = _bar(
            batters.head(12), "player", "runs",
            "Leading Run Scorers", "top_run_scorers", colors=["#059669"] * 12,
        )

    # --- top wicket takers ---
    bowlers = bowling_leaderboard(bowling, min_innings=20)
    if not bowlers.empty:
        figures["Leading wicket takers"] = _bar(
            bowlers.head(12), "player", "wickets",
            "Leading Wicket Takers", "top_wicket_takers", colors=["#dc2626"] * 12,
        )

    # --- innings score distribution ---
    first_innings = innings[innings["innings_no"] == 1]["runs"].dropna()
    if not first_innings.empty:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.hist(first_innings, bins=30, color="#2563eb", edgecolor="white")
        ax.axvline(first_innings.mean(), color="#dc2626", linestyle="--",
                   linewidth=2, label=f"Mean = {first_innings.mean():.0f}")
        ax.set_title("Distribution of First-Innings Totals", fontsize=13, fontweight="bold")
        ax.set_xlabel("Runs")
        ax.set_ylabel("Innings")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        figures["Distribution of first-innings totals"] = _save_figure(
            fig, "score_distribution"
        )

    # --- phase run rates ---
    if deliveries is not None and not deliveries.empty:
        phased = deliveries.assign(
            phase=np.select(
                [deliveries["over_no"] <= 6, deliveries["over_no"] < 16],
                ["Powerplay", "Middle"], default="Death",
            )
        )
        by_season = (
            phased.groupby(["season", "phase"])
            .agg(runs=("total_runs", "sum"), balls=("is_legal", "sum"))
            .reset_index()
        )
        by_season["run_rate"] = by_season["runs"] * 6 / by_season["balls"]
        pivot = by_season.pivot(index="season", columns="phase", values="run_rate")

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for phase in ("Powerplay", "Middle", "Death"):
            if phase in pivot.columns:
                ax.plot(pivot.index, pivot[phase], marker="o", linewidth=2, label=phase)
        ax.set_title("Run Rate by Phase Across Seasons", fontsize=13, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel("Runs per over")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        figures["Run rate by phase across seasons"] = _save_figure(fig, "phase_run_rates")

    logger.info("Generated %d EDA figures", len(figures))
    return figures


def _table_html(frame: pd.DataFrame, *, max_rows: int = 15) -> str:
    """Render a DataFrame as an HTML table, truncated for readability."""
    if frame.empty:
        return "<p><em>No data available.</em></p>"
    return frame.head(max_rows).to_html(
        index=False, classes="data", border=0, float_format=lambda v: f"{v:,.2f}"
    )


def build_report(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    deliveries: pd.DataFrame | None = None,
    *,
    output: Path | None = None,
) -> Path:
    """Generate the full EDA report and return the path to the HTML file."""
    figures = generate_figures(matches, innings, batting, bowling, deliveries)

    completed = matches[matches["is_completed"]] if not matches.empty else matches
    seasons = (
        f"{int(matches['season'].min())}-{int(matches['season'].max())}"
        if not matches.empty else "n/a"
    )

    overview = {
        "Seasons covered": seasons,
        "Total matches": f"{len(matches):,}",
        "Completed matches": f"{len(completed):,}",
        "Franchises": f"{pd.concat([matches['team1'], matches['team2']]).nunique():,}",
        "Venues": f"{matches['venue'].nunique():,}",
        "Innings recorded": f"{len(innings):,}",
        "Batting card rows": f"{len(batting):,}",
        "Bowling card rows": f"{len(bowling):,}",
        "Deliveries": f"{len(deliveries):,}" if deliveries is not None else "not ingested",
    }

    figure_blocks = "\n".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for caption, path in figures.items()
    )

    overview_rows = "\n".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in overview.items()
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPL Exploratory Data Analysis</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.6; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 2px solid currentColor; padding-bottom: 0.3rem; }}
  .meta {{ opacity: 0.7; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid rgba(128,128,128,0.3); }}
  th {{ font-weight: 600; }}
  figure {{ margin: 2rem 0; }}
  figure img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  figcaption {{ font-size: 0.9rem; opacity: 0.75; margin-top: 0.5rem; text-align: center; }}
  .overview th {{ width: 40%; }}
</style>
</head>
<body>
<h1>IPL Exploratory Data Analysis</h1>
<p class="meta">Generated {datetime.now():%d %B %Y at %H:%M} &middot;
Primary source: iplt20.com official feeds, supplemented by Cricsheet for 2008-2018.</p>

<h2>Dataset overview</h2>
<table class="overview">{overview_rows}</table>

<h2>Franchise records</h2>
{_table_html(team_summary(matches, min_matches=10))}

<h2>Toss impact</h2>
{_table_html(toss_impact(matches))}
{_table_html(toss_impact(matches, by="toss_decision"))}

<h2>Venue profile</h2>
{_table_html(venue_summary(matches, innings, min_matches=10))}

<h2>Leading run scorers</h2>
{_table_html(batting_leaderboard(batting, min_innings=20))}

<h2>Leading wicket takers</h2>
{_table_html(bowling_leaderboard(bowling, min_innings=20))}

<h2>Figures</h2>
{figure_blocks}

</body>
</html>
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = output or (REPORTS_DIR / "eda_report.html")
    path.write_text(document, encoding="utf-8")
    logger.info("EDA report written to %s", path)
    return path
