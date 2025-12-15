import pandas as pd
from .metadata import get_match_metadata, load_match_metadata


DEFAULT_COLUMNS = [
    "match_id",
    "video_path",
    "event_id",
    "frame",
    "timestamp",
    "event_type",
    "home_team",
    "away_team",
    "home_county",
    "away_county",
    "involved_players",
    "predicted_foul",
    "pred_score",
    "actual_foul",
    "prediction_error",
    "velocity",
    "area",
    "approach_angle",
    "field_x",
    "field_y",
    "notes",
]


def aggregate_and_write(events, out_csv, meta_path=None):
    meta = load_match_metadata(meta_path)
    rows = []
    for e in events:
        mid = e.get("match_id")
        mmeta = get_match_metadata(mid, meta)
        pos = e.get("centroid") or [None, None]
        field = e.get("field_pos") or {"x": None, "y": None}
        row = {
            "match_id": mid,
            "video_path": e.get("video_path"),
            "event_id": e.get("event_id"),
            "frame": e.get("frame"),
            "timestamp": e.get("timestamp"),
            "event_type": e.get("event_type"),
            "home_team": mmeta.get("home_team") if mmeta else None,
            "away_team": mmeta.get("away_team") if mmeta else None,
            "home_county": mmeta.get("home_county") if mmeta else None,
            "away_county": mmeta.get("away_county") if mmeta else None,
            "involved_players": e.get("involved_players"),
            "predicted_foul": e.get("predicted_foul"),
            "pred_score": e.get("pred_score"),
            "actual_foul": e.get("actual_foul"),
            "prediction_error": (None if e.get("actual_foul") is None else (e.get("predicted_foul") != e.get("actual_foul"))),
            "velocity": e.get("velocity"),
            "area": e.get("area"),
            "approach_angle": e.get("approach_angle"),
            "field_x": field.get("x"),
            "field_y": field.get("y"),
            "notes": e.get("notes"),
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=DEFAULT_COLUMNS)
    df.to_csv(out_csv, index=False)
    return df
