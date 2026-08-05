"""Streamlit Cloud entry point.

Streamlit Cloud runs this file. It adds ``src/`` to the import path (so the app
works from a bare clone without ``pip install -e .``) and hands off to the
dashboard package.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl.dashboard.app import main  # noqa: E402

main()
