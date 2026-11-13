"""HiggsEvent — Streamlit Community Cloud Root Entrypoint.

Main web entrypoint deployed on Streamlit Community Cloud for the mid-semester evaluation.
Loads saved experiment artifacts without creating or modifying results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Launch HiggsLens Dashboard
from app.main import main

if __name__ == "__main__":
    main()
