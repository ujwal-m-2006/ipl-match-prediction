#!/usr/bin/env python
"""Generate the exploratory-data-analysis report into reports/eda_report.html."""

import _bootstrap  # noqa: F401

from ipl.cli import eda_main

if __name__ == "__main__":
    raise SystemExit(eda_main())
