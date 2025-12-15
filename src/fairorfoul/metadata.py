import csv
import os
from pathlib import Path

_DEFAULT_META = {
    # sample fallback metadata for testing
    1: {"match_id": 1, "home_team": "Dublin FC", "away_team": "Cork United", "home_county": "Dublin", "away_county": "Cork"},
}

def load_match_metadata(path=None):
    """Load match metadata CSV if present. Expected columns: match_id,home_team,away_team,home_county,away_county
    Returns dict keyed by int(match_id).
    """
    meta = {}
    if path is None:
        path = Path.cwd() / "data" / "match_metadata.csv"
    path = Path(path)
    if not path.exists():
        return _DEFAULT_META.copy()

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                mid = int(r.get("match_id") or r.get("id") or 0)
            except Exception:
                continue
            meta[mid] = {
                "match_id": mid,
                "home_team": r.get("home_team") or r.get("home"),
                "away_team": r.get("away_team") or r.get("away"),
                "home_county": r.get("home_county") or r.get("home_county"),
                "away_county": r.get("away_county") or r.get("away_county"),
            }
    return meta


def get_match_metadata(match_id, meta=None):
    if meta is None:
        meta = load_match_metadata()
    return meta.get(int(match_id)) or _DEFAULT_META.get(int(match_id))
