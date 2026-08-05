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


class TestFormatValues:
    """Regression tests for the chart data-label formatter.

    SQL aggregates return "integer" columns as floats, so an integer format
    spec must not raise -- ``format(14.0, "d")`` does, which previously took
    down the whole Schedule page over a cosmetic bar label.
    """

    def test_integer_spec_accepts_floats(self) -> None:
        import pandas as pd

        from ipl.dashboard.theme import format_values

        assert format_values(pd.Series([14.0, 2.0, 0.0]), "d") == ["14", "2", "0"]

    def test_integer_spec_rounds_rather_than_truncating(self) -> None:
        import pandas as pd

        from ipl.dashboard.theme import format_values

        assert format_values(pd.Series([2.6]), "d") == ["3"]

    def test_nulls_render_as_empty_labels(self) -> None:
        import numpy as np
        import pandas as pd

        from ipl.dashboard.theme import format_values

        assert format_values(pd.Series([1.0, np.nan, 3.0]), "d") == ["1", "", "3"]

    def test_float_spec_still_works(self) -> None:
        import pandas as pd

        from ipl.dashboard.theme import format_values

        assert format_values(pd.Series([0.8974]), ".4f") == ["0.8974"]

    def test_no_spec_returns_none(self) -> None:
        import pandas as pd

        from ipl.dashboard.theme import format_values

        assert format_values(pd.Series([1, 2]), None) is None

    def test_bar_chart_with_float_counts_does_not_raise(self) -> None:
        """The exact shape that crashed the Schedule page's points chart."""
        import pandas as pd

        from ipl.dashboard.theme import bar_chart

        frame = pd.DataFrame({"team": ["A", "B"], "points": [14.0, 0.0]})
        figure = bar_chart(frame, "team", "points", text_format="d")
        assert figure is not None


class TestStatTable:
    """The two-column metric tables must be Arrow-serialisable.

    A column mixing ints, floats and NaN becomes ``object`` dtype, which Arrow
    cannot convert -- Streamlit recovers but logs a traceback, which looks like
    a crash to anyone watching the console.
    """

    def test_values_are_all_strings(self) -> None:
        import numpy as np

        from ipl.dashboard.views._common import stat_table

        table = stat_table([("Runs", 500), ("Average", 34.567), ("Missing", np.nan)])
        assert table["Value"].map(type).eq(str).all()

    def test_missing_values_render_as_a_dash(self) -> None:
        import numpy as np

        from ipl.dashboard.views._common import stat_table

        table = stat_table([("Average", np.nan), ("Economy", None)])
        assert list(table["Value"]) == ["—", "—"]

    def test_whole_floats_lose_the_decimal(self) -> None:
        from ipl.dashboard.views._common import stat_table

        table = stat_table([("Innings", 12.0), ("Economy", 7.25)])
        assert list(table["Value"]) == ["12", "7.25"]

    def test_frame_is_arrow_serialisable(self) -> None:
        import numpy as np

        pa = pytest.importorskip("pyarrow")
        from ipl.dashboard.views._common import stat_table

        table = stat_table([("A", 1), ("B", 2.5), ("C", np.nan), ("D", "text")])
        # This is the conversion Streamlit performs internally.
        pa.Table.from_pandas(table)


def test_no_deprecated_streamlit_arguments() -> None:
    """`use_container_width` is past its removal date and warns on every call."""
    import pathlib

    offenders = [
        str(path)
        for path in pathlib.Path("src").rglob("*.py")
        if "use_container_width" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"use `width=` instead in: {offenders}"


def test_team_colours_cover_every_active_franchise() -> None:
    from ipl.constants import ACTIVE_TEAMS, TEAM_COLORS

    missing = [team for team in ACTIVE_TEAMS if team not in TEAM_COLORS]
    assert not missing, f"no brand colour for: {missing}"
