"""Compatibility launcher. Use: streamlit run streamlit_app.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.main import main

if __name__ == "__main__":
    main()
