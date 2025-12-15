import cv2
import numpy as np
from pathlib import Path
import math


def _frame_timestamp(cap, frame_idx):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return frame_idx / float(fps)


def process_video_for_motion_events(video_path, match_id=None, frame_step=2, min_area=2000):
    """Lightweight motion-based event extractor.

    Scans the video and flags frames where large moving contours appear.
    Returns a list of candidate event dicts with basic features.
    This is a simple placeholder for a full detector+tracker pipeline.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_idx = 1
    events = []

    while True:
        # skip frames for speed
        for _ in range(frame_step):
            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 5)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large = [c for c in contours if cv2.contourArea(c) > min_area]
        if large:
            # combine bounding boxes
            x0 = min([cv2.boundingRect(c)[0] for c in large])
            y0 = min([cv2.boundingRect(c)[1] for c in large])
            x1 = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in large])
            y1 = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in large])
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            area = sum([cv2.contourArea(c) for c in large])
            ts = _frame_timestamp(cap, frame_idx)
            events.append({
                "match_id": match_id,
                "frame": frame_idx,
                "timestamp": ts,
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "centroid": [float(cx), float(cy)],
                "area": float(area),
                "video_path": str(path),
            })

        prev_gray = gray

    cap.release()
    return events


# Minimal helper to compute simple velocities between consecutive events (by centroid/time)
def compute_event_velocities(events):
    """Given a list of events sorted by timestamp, compute approximate velocity of centroid.
       Adds 'velocity' key (pixels/sec) to events where possible."""
    events = sorted(events, key=lambda e: e["timestamp"])
    for i in range(1, len(events)):
        prev = events[i - 1]
        cur = events[i]
        dt = cur["timestamp"] - prev["timestamp"]
        if dt <= 0:
            cur["velocity"] = 0.0
            continue
        dx = cur["centroid"][0] - prev["centroid"][0]
        dy = cur["centroid"][1] - prev["centroid"][1]
        dist = math.hypot(dx, dy)
        cur["velocity"] = dist / dt
    if events:
        events[0]["velocity"] = 0.0
    return events
