#!/usr/bin/env python
"""Probe the official feed host for IPL competition IDs.

The feed indexes every tournament it has served under a flat integer namespace,
so a newly published IPL season shows up as a new ID. Run this after a season
starts to find it, then add the result to ``IPL_COMPETITIONS`` in
``src/ipl/constants.py``.

    python scripts/discover_competitions.py --start 1 --end 400
"""

import argparse
import json

import _bootstrap  # noqa: F401

from ipl.constants import IPL_COMPETITIONS
from ipl.ingestion.iplt20_client import IPLT20Client
from ipl.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover IPL competition IDs.")
    parser.add_argument("--start", type=int, default=1, help="First ID to probe.")
    parser.add_argument("--end", type=int, default=400, help="Last ID to probe (inclusive).")
    parser.add_argument(
        "--keyword", default="ipl",
        help="Substring that must appear in the competition name.",
    )
    args = parser.parse_args()

    setup_logging()
    print(f"Probing competition IDs {args.start}-{args.end} for '{args.keyword}'...")
    print("This is rate-limited and may take several minutes.\n")

    client = IPLT20Client()
    try:
        found = client.discover_competitions(
            range(args.start, args.end + 1), keyword=args.keyword
        )
    finally:
        client.close()

    if not found:
        print("No matching competitions found.")
        return 1

    print("\nDiscovered competitions:")
    for season in sorted(found):
        marker = " (already registered)" if IPL_COMPETITIONS.get(season) == found[season] else " <-- NEW"
        print(f"  {season}: {found[season]}{marker}")

    new = {s: c for s, c in found.items() if IPL_COMPETITIONS.get(s) != c}
    if new:
        print("\nAdd these to IPL_COMPETITIONS in src/ipl/constants.py:")
        print(json.dumps(new, indent=4))
    else:
        print("\nThe registry is already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
