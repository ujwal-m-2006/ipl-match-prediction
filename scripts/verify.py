#!/usr/bin/env python
"""Pre-demo verification: prove the whole system works, end to end.

Run this before every demonstration. It exercises the parts a live audience
actually touches -- every dashboard page against the *real* database, every
button, every API endpoint -- and prints one clear verdict.

    python scripts/verify.py            # full check
    python scripts/verify.py --quick    # skip the slow model/API stages

Exit code is 0 when everything passes and 1 otherwise, so it can gate a demo
or run in CI.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")

# Every page in the dashboard's navigation.
PAGES = [
    "home", "schedule", "teams", "players", "head_to_head",
    "venues", "predictions", "model_comparison", "admin",
]

# Optional model libraries: absent ones are reported, not failed.
OPTIONAL_LIBS = ["xgboost", "lightgbm", "catboost"]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Report:
    """Collects check results and renders the final verdict."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.warned: list[str] = []
        self.started = time.monotonic()

    def ok(self, label: str, detail: str = "") -> None:
        self.passed.append(label)
        suffix = f"  {DIM}{detail}{RESET}" if detail else ""
        print(f"  {GREEN}PASS{RESET}  {label}{suffix}")

    def fail(self, label: str, reason: str) -> None:
        self.failed.append((label, reason))
        print(f"  {RED}FAIL{RESET}  {label}")
        for line in reason.strip().splitlines()[-4:]:
            print(f"        {DIM}{line[:150]}{RESET}")

    def warn(self, label: str, detail: str = "") -> None:
        self.warned.append(label)
        suffix = f"  {DIM}{detail}{RESET}" if detail else ""
        print(f"  {YELLOW}WARN{RESET}  {label}{suffix}")

    def section(self, title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))

    def verdict(self) -> int:
        elapsed = time.monotonic() - self.started
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 62)
        if self.failed:
            print(f"{RED}FAILED{RESET} - {len(self.failed)} of {total} checks failed "
                  f"({elapsed:.0f}s)")
            print("\nFix these before demonstrating:")
            for label, _ in self.failed:
                print(f"  - {label}")
            return 1

        warning = f", {len(self.warned)} warning(s)" if self.warned else ""
        print(f"{GREEN}ALL {total} CHECKS PASSED{RESET}{warning} ({elapsed:.0f}s)")
        print("\nThe system is ready to demonstrate.")
        return 0


# ---------------------------------------------------------------------------
def check_environment(report: Report) -> None:
    report.section("1. Environment")

    version = sys.version_info
    if version < (3, 10):
        report.fail("Python version", f"Python {version.major}.{version.minor} is too old (need 3.10+)")
    else:
        report.ok("Python version", f"{version.major}.{version.minor}.{version.micro}")

    required = ["pandas", "numpy", "sklearn", "sqlalchemy", "streamlit",
                "plotly", "fastapi", "requests", "joblib", "matplotlib"]
    missing = [name for name in required if not _importable(name)]
    if missing:
        report.fail("Required packages", f"missing: {', '.join(missing)} - run: pip install -r requirements.txt")
    else:
        report.ok("Required packages", f"{len(required)} present")

    absent = [name for name in OPTIONAL_LIBS if not _importable(name)]
    if absent:
        report.warn("Optional model libraries", f"absent: {', '.join(absent)} (comparison will omit them)")
    else:
        report.ok("Optional model libraries", "xgboost, lightgbm, catboost")


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
def check_database(report: Report) -> bool:
    report.section("2. Database")

    try:
        from ipl.config import get_settings
        from ipl.db import repository as repo
    except Exception as exc:
        report.fail("Import project modules", traceback.format_exc())
        return False

    settings = get_settings()
    report.ok("Configuration loaded", f"dialect={settings.dialect}")

    try:
        summary = repo.database_summary()
    except Exception:
        report.fail("Connect to the database",
                    "Could not query. Run: python scripts/init_db.py")
        return False

    matches = summary.get("matches", 0)
    if matches == 0:
        report.fail("Match data present", "Database is empty. Run: python scripts/ingest.py")
        return False
    report.ok("Match data present", f"{matches:,} matches")

    span = repo.season_range()
    report.ok("Season coverage", f"{span[0]}-{span[1]} ({span[1] - span[0] + 1} seasons)")

    # Any of these being empty means an incomplete ingest.
    for table, minimum in [("innings", 100), ("batting_cards", 1000),
                           ("bowling_cards", 1000), ("players", 100)]:
        count = summary.get(table, 0)
        if count < minimum:
            report.fail(f"Table '{table}' populated", f"only {count:,} rows (expected >= {minimum:,})")
        else:
            report.ok(f"Table '{table}' populated", f"{count:,} rows")

    deliveries = summary.get("deliveries", 0)
    if deliveries == 0:
        report.warn("Ball-by-ball data", "absent - the chase model and phase analytics are unavailable")
    else:
        report.ok("Ball-by-ball data", f"{deliveries:,} deliveries")

    return True


# ---------------------------------------------------------------------------
def check_models(report: Report) -> None:
    report.section("3. Trained models")

    from ipl.models.persistence import list_artifacts

    artifacts = list_artifacts()
    for task, info in artifacts.items():
        if not info["exists"]:
            report.fail(f"Model '{task}' trained",
                        "artefact missing. Run: python scripts/train_models.py")
        else:
            report.ok(f"Model '{task}' trained",
                      f"{info.get('best_model')} ({info.get('size_kb')} KB)")


def check_predictions(report: Report) -> None:
    report.section("4. Predictions")

    from ipl.models.predict import PredictionService

    try:
        service = PredictionService()
    except Exception:
        report.fail("Initialise the prediction service", traceback.format_exc())
        return

    teams = service.available_teams()
    if len(teams) < 2:
        report.fail("Teams available for prediction", f"only {len(teams)} found")
        return

    if service.has_model("winner"):
        try:
            result = service.predict_winner(team1=teams[0], team2=teams[1])
            assert 0.0 <= result.team1_win_probability <= 1.0
            report.ok("Winner prediction",
                      f"{result.predicted_winner} "
                      f"({max(result.team1_win_probability, result.team2_win_probability):.0%})")
        except Exception:
            report.fail("Winner prediction", traceback.format_exc())

    if service.has_model("score"):
        try:
            result = service.predict_first_innings_score(
                batting_team=teams[0], bowling_team=teams[1]
            )
            assert 50 < result.predicted_score < 300, f"implausible score {result.predicted_score}"
            report.ok("First-innings score", f"{result.predicted_score:.0f} runs")
        except Exception:
            report.fail("First-innings score", traceback.format_exc())

    if service.has_model("chase"):
        try:
            result = service.predict_chase(
                batting_team=teams[0], bowling_team=teams[1],
                venue=service.default_venue(teams[0]),
                target=180, current_runs=120, wickets_fallen=3, balls_bowled=78,
            )
            assert 0.0 <= result.chase_success_probability <= 1.0
            report.ok("Chase probability", f"{result.chase_success_probability:.0%}")

            # The laws of cricket, not the model, must decide terminal states.
            won = service.predict_chase(
                batting_team=teams[0], bowling_team=teams[1],
                venue=service.default_venue(teams[0]),
                target=180, current_runs=180, wickets_fallen=3, balls_bowled=100,
            )
            assert won.chase_success_probability == 1.0, "target reached must be 100%"
            report.ok("Chase edge case", "target reached -> 100%")
        except Exception:
            report.fail("Chase probability", traceback.format_exc())

    if service.has_model("pom"):
        try:
            from ipl.db import repository as repo

            matches = repo.load_matches(completed_only=True)
            match_id = int(matches.iloc[-1]["match_id"])
            ranking = service.predict_player_of_match(match_id, top_n=3)
            assert not ranking.empty, "no candidates returned"
            report.ok("Player of the Match", f"top pick: {ranking.iloc[0]['player']}")
        except Exception:
            report.fail("Player of the Match", traceback.format_exc())


# ---------------------------------------------------------------------------
def check_dashboard(report: Report, *, with_clicks: bool = True) -> None:
    report.section("5. Dashboard pages (against the real database)")

    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        report.fail("Streamlit test harness", "streamlit is not installed")
        return

    def script(page: str) -> str:
        return (
            f"import sys\nsys.path.insert(0, {SRC!r})\n"
            f"from ipl.dashboard.views import {page}\n{page}.render()\n"
        )

    for page in PAGES:
        try:
            app = AppTest.from_string(script(page), default_timeout=300).run()
        except Exception:
            report.fail(f"Page '{page}' renders", traceback.format_exc())
            continue

        if app.exception:
            report.fail(f"Page '{page}' renders",
                        "\n".join(str(e.value) for e in app.exception))
            continue

        buttons = [b.label for b in app.button]
        report.ok(f"Page '{page}' renders",
                  f"{len(app.tabs)} tabs, {len(buttons)} buttons")

        if not with_clicks:
            continue

        # Anything behind a button never runs on a plain render, but an
        # examiner will click it.
        for index, label in enumerate(buttons):
            try:
                clicked = AppTest.from_string(script(page), default_timeout=300).run()
                clicked.button[index].click().run()
            except Exception:
                report.fail(f"Page '{page}' button '{label}'", traceback.format_exc())
                continue
            if clicked.exception:
                report.fail(f"Page '{page}' button '{label}'",
                            "\n".join(str(e.value) for e in clicked.exception))
            else:
                report.ok(f"Page '{page}' button '{label}'")


# ---------------------------------------------------------------------------
def check_api(report: Report) -> None:
    report.section("6. REST API")

    try:
        from fastapi.testclient import TestClient
    except Exception:
        report.warn("API test client", "httpx2 not installed - skipping API checks")
        return

    from ipl.api.main import app
    from ipl.db import repository as repo

    matches = repo.load_matches(completed_only=True)
    match_id = int(matches.iloc[-1]["match_id"])
    season = int(matches["season"].max())

    endpoints = [
        ("GET", "/health", None, 200),
        ("GET", "/teams", None, 200),
        ("GET", "/venues", None, 200),
        ("GET", "/matches?limit=5", None, 200),
        ("GET", "/head-to-head?team_a=Chennai Super Kings&team_b=Mumbai Indians", None, 200),
        ("GET", f"/predict/player-of-match/{match_id}", None, 200),
        ("GET", f"/predict/playoffs/{season}?simulations=200", None, 200),
        ("GET", "/models/comparison", None, 200),
        ("GET", "/openapi.json", None, 200),
        ("POST", "/predict/winner",
         {"team1": "Chennai Super Kings", "team2": "Mumbai Indians"}, 200),
        ("POST", "/predict/score",
         {"batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings"}, 200),
        ("POST", "/predict/chase",
         {"batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings",
          "venue": "Wankhede Stadium", "target": 180, "current_runs": 120,
          "wickets_fallen": 3, "balls_bowled": 78}, 200),
        # Bad input must be rejected cleanly, never with a 500.
        ("POST", "/predict/winner", {"team1": "A", "team2": "A"}, 422),
        ("GET", "/predict/playoffs/1999", None, 404),
    ]

    try:
        with TestClient(app) as client:
            for method, url, body, expected in endpoints:
                response = (
                    client.post(url, json=body) if method == "POST" else client.get(url)
                )
                label = f"{method} {url.split('?')[0]}"
                if response.status_code == expected:
                    report.ok(label, f"{response.status_code}")
                else:
                    report.fail(label,
                                f"got {response.status_code}, expected {expected}: "
                                f"{response.text[:200]}")
    except Exception:
        report.fail("API test client", traceback.format_exc())


# ---------------------------------------------------------------------------
def check_tests(report: Report) -> None:
    report.section("7. Test suite")

    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not network", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = (result.stdout or result.stderr).strip().splitlines()
    summary = tail[-1] if tail else "no output"

    if result.returncode == 0:
        report.ok("pytest", summary)
    else:
        report.fail("pytest", "\n".join(tail[-12:]))


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the whole system before a demonstration."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Skip button clicks, the API sweep and pytest.")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest.")
    args = parser.parse_args()

    print("=" * 62)
    print("  IPL Analytics - pre-demonstration verification")
    print("=" * 62)

    report = Report()

    check_environment(report)
    if not check_database(report):
        return report.verdict()

    check_models(report)
    check_predictions(report)
    check_dashboard(report, with_clicks=not args.quick)

    if not args.quick:
        check_api(report)
        if not args.no_tests:
            check_tests(report)

    return report.verdict()


if __name__ == "__main__":
    raise SystemExit(main())
