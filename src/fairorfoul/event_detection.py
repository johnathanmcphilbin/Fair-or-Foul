from typing import List, Dict
import math


def detect_tackle_events(candidates: List[Dict]) -> List[Dict]:
    """Given motion-based candidate events (from video_processing), apply heuristics
    to emit structured events with features useful for prediction.

    This is a simple heuristic implementation:
      - Treat each candidate as a possible 'tackle' if area is large and velocity is high.
      - Compute simple approach angle and mark involved_players as placeholders.
    """
    events = []
    for idx, c in enumerate(candidates):
        area = c.get("area", 0)
        vel = c.get("velocity", 0)
        # heuristic thresholds (tunable)
        if area > 5000 and vel > 20:
            # compute a fake approach angle using centroid delta when available
            approach_angle = c.get("centroid", [0, 0])[0] % 180
            ev = {
                "event_id": f"evt_{idx}",
                "match_id": c.get("match_id"),
                "frame": c.get("frame"),
                "timestamp": c.get("timestamp"),
                "event_type": "tackle",
                "bbox": c.get("bbox"),
                "centroid": c.get("centroid"),
                "area": area,
                "velocity": vel,
                "approach_angle": approach_angle,
                # placeholders for actual player ids (requires tracking)
                "involved_players": [1, 2],
            }
            events.append(ev)
    return events


def compute_position_on_field(centroid, frame_size=(1280, 720)):
    """Map pixel centroid to normalized field coordinates (0-1).
    frame_size = (width, height)
    """
    w, h = frame_size
    if not centroid:
        return {"x": None, "y": None}
    cx, cy = centroid
    return {"x": float(cx) / float(w), "y": float(cy) / float(h)}
