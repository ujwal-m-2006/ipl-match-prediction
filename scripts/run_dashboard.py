#!/usr/bin/env python
"""Launch the Streamlit dashboard.

A thin convenience wrapper around:

    streamlit run streamlit_app.py
"""

import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    entry = ROOT / "streamlit_app.py"
    if not entry.exists():
        print(f"Entry point not found: {entry}", file=sys.stderr)
        return 1
    # Reuse the interpreter running this script so the venv is respected.
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)])


if __name__ == "__main__":
    raise SystemExit(main())
