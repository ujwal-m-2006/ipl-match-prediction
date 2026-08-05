"""Make ``src/`` importable when running scripts without installing the package.

Every script in this directory imports this module first. If the project *has*
been installed (``pip install -e .``), this is a harmless no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
