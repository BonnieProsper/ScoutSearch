# scout/storage/paths.py

from pathlib import Path

BASE_DIR = Path(".") / "data"
BASE_DIR.mkdir(exist_ok=True)

INDEX_FILE = BASE_DIR / "index.pkl"
STATS_FILE = BASE_DIR / "stats.pkl"
