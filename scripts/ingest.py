#!/usr/bin/env python
"""Collect IPL data into the database.

Examples
--------
    python scripts/ingest.py                       # everything, incremental
    python scripts/ingest.py --seasons 2026        # just the current season
    python scripts/ingest.py --no-deliveries       # fast header-only load
    python scripts/ingest.py --full --force-refresh  # rebuild from scratch
"""

import _bootstrap  # noqa: F401  (import for the sys.path side effect)

from ipl.cli import ingest_main

if __name__ == "__main__":
    raise SystemExit(ingest_main())
