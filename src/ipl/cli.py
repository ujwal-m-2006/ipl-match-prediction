"""Command-line interface.

Installed as the console scripts ``ipl-ingest``, ``ipl-train``, ``ipl-eda`` and
``ipl-api`` (see ``pyproject.toml``). The thin wrappers in ``scripts/`` call
into the same functions so the project works either way.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import get_settings
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _season_list(value: str) -> list[int]:
    """Parse ``"2019,2021-2023"`` into ``[2019, 2021, 2022, 2023]``."""
    seasons: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(chunk))
    return sorted(set(seasons))


# ---------------------------------------------------------------------------
# ipl-ingest
# ---------------------------------------------------------------------------
def ingest_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the data-collection pipeline."""
    parser = argparse.ArgumentParser(
        prog="ipl-ingest",
        description=(
            "Collect IPL data from the official iplt20.com feeds, supplemented "
            "by Cricsheet for seasons the official feed does not publish."
        ),
    )
    parser.add_argument(
        "--seasons", type=_season_list, default=None,
        help="Seasons to ingest, e.g. '2024,2025' or '2019-2026'. Default: all.",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Bypass the HTTP cache and re-download every feed.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Re-parse matches already stored as final (default: skip them).",
    )
    parser.add_argument(
        "--no-deliveries", action="store_true",
        help="Skip ball-by-ball ingestion for a much faster run.",
    )
    parser.add_argument(
        "--no-cricsheet", action="store_true",
        help="Use only the official feed (2019 onwards).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N new/changed matches (smoke testing).",
    )
    parser.add_argument("--log-level", default=None, help="Override the log level.")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    from .ingestion import run_ingestion

    stats = run_ingestion(
        seasons=args.seasons,
        force_refresh=args.force_refresh,
        skip_completed=not args.full,
        ingest_deliveries=not args.no_deliveries,
        enable_cricsheet=not args.no_cricsheet,
        limit=args.limit,
        trigger="cli",
    )
    print(f"\nIngestion complete: {stats.summary()}")
    return 0


# ---------------------------------------------------------------------------
# ipl-train
# ---------------------------------------------------------------------------
def train_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for model training."""
    parser = argparse.ArgumentParser(
        prog="ipl-train",
        description="Train and compare every model, then persist the best of each.",
    )
    parser.add_argument(
        "--test-season-from", type=int, default=None,
        help="Hold out seasons from this year onwards (default: from settings).",
    )
    parser.add_argument("--random-state", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--skip-chase", action="store_true",
        help="Skip the ball-by-ball chase model (the slowest task).",
    )
    parser.add_argument(
        "--skip-pom", action="store_true", help="Skip the Player-of-the-Match model.",
    )
    parser.add_argument(
        "--chase-stride", type=int, default=3,
        help=(
            "Sample every Nth ball for the chase model. Consecutive balls are "
            "near-identical, so 3 keeps the signal and cuts the fit time. Use 1 "
            "for every ball."
        ),
    )
    parser.add_argument("--log-level", default=None, help="Override the log level.")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    from .models.train import comparison_frame, train_all

    results = train_all(
        test_season_from=args.test_season_from,
        random_state=args.random_state,
        include_chase=not args.skip_chase,
        include_pom=not args.skip_pom,
        chase_stride=args.chase_stride,
    )

    table = comparison_frame(results)
    if not table.empty:
        print("\n=== Model comparison ===")
        with_pandas_display(table)

    print("\n=== Selected models ===")
    for task, result in results.items():
        print(f"  {task:8} -> {result.best_model or 'not trained'}")
    return 0


def with_pandas_display(frame) -> None:  # noqa: ANN001
    """Print a DataFrame without pandas truncating the columns."""
    import pandas as pd

    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.float_format",
        lambda v: f"{v:.4f}",
    ):
        print(frame.to_string(index=False))


# ---------------------------------------------------------------------------
# ipl-eda
# ---------------------------------------------------------------------------
def eda_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the exploratory-data-analysis report."""
    parser = argparse.ArgumentParser(
        prog="ipl-eda", description="Generate the EDA report and figures."
    )
    parser.add_argument(
        "--no-deliveries", action="store_true",
        help="Skip figures that need ball-by-ball data (faster).",
    )
    parser.add_argument("--log-level", default=None, help="Override the log level.")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    from .analytics.eda import build_report
    from .db import repository as repo

    matches = repo.load_matches()
    if matches.empty:
        print("No data found. Run `python scripts/ingest.py` first.", file=sys.stderr)
        return 1

    path = build_report(
        matches,
        repo.load_innings(),
        repo.load_batting(),
        repo.load_bowling(),
        None if args.no_deliveries else repo.load_deliveries(),
    )
    print(f"EDA report written to {path}")
    return 0


# ---------------------------------------------------------------------------
# ipl-api
# ---------------------------------------------------------------------------
def api_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the FastAPI service."""
    parser = argparse.ArgumentParser(prog="ipl-api", description="Run the prediction API.")
    settings = get_settings()
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("ipl.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ---------------------------------------------------------------------------
# ipl (dispatcher)
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``python -m ipl <command>``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "ingest": ingest_main,
        "train": train_main,
        "eda": eda_main,
        "api": api_main,
    }
    if not argv or argv[0] not in commands:
        print(f"Usage: python -m ipl {{{'|'.join(commands)}}} [options]", file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
