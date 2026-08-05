#!/usr/bin/env python
"""Run the FastAPI prediction service.

    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --reload

Interactive docs are then at http://localhost:8000/docs
"""

import _bootstrap  # noqa: F401

from ipl.cli import api_main

if __name__ == "__main__":
    raise SystemExit(api_main())
