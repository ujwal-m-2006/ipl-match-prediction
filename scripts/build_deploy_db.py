#!/usr/bin/env python
"""Build a slimmed-down database for deployment.

Streamlit Community Cloud has an ephemeral filesystem and roughly 1 GB of RAM,
so the deployed app reads a database committed to the repository rather than
ingesting on start-up. The full warehouse is ~46 MB, most of which is the
280k-row ``deliveries`` table.

That table is only needed to *train* the chase model and to draw the venue
phase chart. The trained chase model does not need it at inference time -- a
live prediction is built purely from the numbers the user types in. So the
deployment copy drops ball-by-ball data and keeps everything else, which cuts
the file dramatically while leaving every prediction working.

    python scripts/build_deploy_db.py

Writes ``data/ipl_deploy.db``.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ipl.config import DATA_DIR, get_settings
from ipl.logging_utils import setup_logging

# Tables emptied in the deployment copy, with why they can go.
DROPPABLE = {
    "deliveries": "ball-by-ball; needed only for training and venue phase charts",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deployment database.")
    parser.add_argument(
        "--output", default=str(DATA_DIR / "ipl_deploy.db"),
        help="Where to write the slimmed database.",
    )
    parser.add_argument(
        "--keep-deliveries", action="store_true",
        help="Keep ball-by-ball data (much larger, but keeps venue phase charts).",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    if not settings.is_sqlite:
        print("This script only supports SQLite sources.", file=sys.stderr)
        return 1

    source = Path(settings.database_url.replace("sqlite:///", ""))
    if not source.exists():
        print(f"Source database not found: {source}", file=sys.stderr)
        print("Run: python scripts/ingest.py", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}  ({source.stat().st_size / 1e6:.1f} MB)")
    shutil.copy2(source, output)

    connection = sqlite3.connect(output)
    try:
        if not args.keep_deliveries:
            for table, reason in DROPPABLE.items():
                count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                connection.execute(f"DELETE FROM {table}")
                print(f"  dropped {count:,} rows from '{table}' ({reason})")
        connection.commit()
        # VACUUM rebuilds the file so the freed pages are actually reclaimed;
        # without it the file stays the same size on disk.
        print("  vacuuming...")
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()

    size = output.stat().st_size / 1e6
    print(f"\nWrote {output}  ({size:.1f} MB)")

    if size > 90:
        print("\nWARNING: over 90 MB. GitHub rejects files above 100 MB; use "
              "--keep-deliveries=false or Git LFS.", file=sys.stderr)
    elif size > 45:
        print("\nNote: over 45 MB. GitHub warns above 50 MB but will accept it.")

    print("\nCommit it with:")
    print(f"  git add -f {output.as_posix()} models/artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
