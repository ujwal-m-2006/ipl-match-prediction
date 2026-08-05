"""Smoke tests for the Streamlit dashboard.

Each page is executed through Streamlit's own ``AppTest`` harness, which runs
the script exactly as the server would but without a browser. This catches the
class of bug that only appears at render time -- an invalid widget argument, a
column that does not exist, a duplicate page path -- which unit tests on the
analytics functions cannot see.

The tests run against whatever database the session fixture points at. With an
empty database each page must still render its "no data" state rather than
raising, which is itself worth asserting.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# (module path, renderer attribute) for every page in the navigation.
PAGES = [
    "home",
    "schedule",
    "teams",
    "players",
    "head_to_head",
    "venues",
    "predictions",
    "model_comparison",
    "admin",
]

RUN_TIMEOUT = 60


def _run_page(name: str) -> AppTest:
    """Execute one dashboard page in isolation and return the result."""
    script = f"""
import sys
sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")!r})
from ipl.dashboard.views import {name}
{name}.render()
"""
    app = AppTest.from_string(script, default_timeout=RUN_TIMEOUT)
    return app.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    """No page may raise while rendering."""
    app = _run_page(page)
    assert not app.exception, (
        f"page '{page}' raised: "
        + "; ".join(str(e.value) for e in app.exception)
    )


@pytest.mark.parametrize("page", PAGES)
def test_page_produces_output(page: str) -> None:
    """A page must render something -- a title, a caption or an info box."""
    app = _run_page(page)
    produced = (
        len(app.title) + len(app.markdown) + len(app.caption)
        + len(app.info) + len(app.warning) + len(app.error) + len(app.header)
    )
    assert produced > 0, f"page '{page}' rendered nothing"


def test_navigation_paths_are_unique() -> None:
    """Every page must claim a distinct URL path.

    All nine renderers are called ``render``, so without explicit ``url_path``
    values Streamlit derives the same path for each and refuses to build the
    navigation.
    """
    import inspect

    from ipl.dashboard import app as dashboard_app

    source = inspect.getsource(dashboard_app.main)
    paths = [
        line.split('url_path="')[1].split('"')[0]
        for line in source.splitlines()
        if 'url_path="' in line
    ]
    assert len(paths) == len(PAGES), f"expected {len(PAGES)} url_path entries, found {len(paths)}"
    assert len(set(paths)) == len(paths), f"duplicate url_path values: {paths}"


def test_theme_palette_slots_are_distinct() -> None:
    """The categorical palette must not repeat a hue within its eight slots."""
    from ipl.dashboard.theme import CATEGORICAL

    assert len(CATEGORICAL) == len(set(CATEGORICAL)) == 8


def test_team_colours_cover_every_active_franchise() -> None:
    from ipl.constants import ACTIVE_TEAMS, TEAM_COLORS

    missing = [team for team in ACTIVE_TEAMS if team not in TEAM_COLORS]
    assert not missing, f"no brand colour for: {missing}"
