#!/usr/bin/env python
"""Train and compare every model, saving the best of each to models/artifacts/.

Examples
--------
    python scripts/train_models.py
    python scripts/train_models.py --test-season-from 2024
    python scripts/train_models.py --skip-chase        # skip the slow task
"""

import _bootstrap  # noqa: F401

from ipl.cli import train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
