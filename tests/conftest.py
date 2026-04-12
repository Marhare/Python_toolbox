from pathlib import Path
import sys


# Ensure tests always import the local workspace package first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
