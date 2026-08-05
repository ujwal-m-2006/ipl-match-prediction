#!/usr/bin/env python
"""Create (or reset) the database schema.

    python scripts/init_db.py            # create any missing tables
    python scripts/init_db.py --drop     # DESTRUCTIVE: drop and recreate
"""

import argparse

import _bootstrap  # noqa: F401

from ipl.config import get_settings
from ipl.db.base import init_db
from ipl.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the IPL database schema.")
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop every existing table first. This deletes all stored data.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt for --drop.",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    print(f"Target database: {settings.database_url}")

    if args.drop and not args.yes:
        # Dropping is irreversible, so require an explicit typed confirmation
        # rather than a bare flag.
        answer = input("This will DELETE ALL DATA. Type 'drop' to continue: ")
        if answer.strip().lower() != "drop":
            print("Aborted.")
            return 1

    init_db(drop_existing=args.drop)
    print("Schema ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
