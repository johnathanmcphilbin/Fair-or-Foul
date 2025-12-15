#!/usr/bin/env python3
"""Runner to build an event log CSV from a folder of videos.

This is a lightweight scaffold that uses simple motion heuristics for event
candidates and a simple predictor. Replace detectors/trackers with real models
for production use.
"""
import argparse
from pathlib import Path
from fairorfoul.video_processing import process_video_for_motion_events, compute_event_velocities
from fairorfoul.event_detection import detect_tackle_events, compute_position_on_field
from fairorfoul.ai_predictor import predict_foul
from fairorfoul.dataset_writer import aggregate_and_write
from fairorfoul.metadata import load_match_metadata


def process_single_video(path, match_id=None, bias_map=None, desired_accuracy=None):
    candidates = process_video_for_motion_events(path, match_id=match_id)
    candidates = compute_event_velocities(candidates)
    events = detect_tackle_events(candidates)
    # enrich events with meta and predictions
    meta = load_match_metadata()
    enriched = []
    for e in events:
        e["match_meta"] = meta.get(e.get("match_id")) if meta else {}
        e["field_pos"] = compute_position_on_field(e.get("centroid"), frame_size=(1280, 720))
        # no human labels by default; loader can add actual_foul
        pred = predict_foul(e, bias_map=bias_map, desired_accuracy=desired_accuracy)
        e.update(pred)
        enriched.append(e)
    return enriched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", default="event_log.csv")
    p.add_argument("--bias-json", help="optional JSON file mapping county->bias")
    p.add_argument("--accuracy", type=float, help="simulate AI accuracy (0-1)")
    args = p.parse_args()

    vdir = Path(args.video_dir)
    if not vdir.exists():
        raise SystemExit("video dir not found")

    all_events = []
    for vid in sorted(vdir.glob("*.mp4")):
        print("Processing", vid)
        # try to infer match_id from filename like match_1.mp4
        mid = None
        name = vid.stem
        if "match_" in name:
            try:
                mid = int(name.split("match_")[-1].split("_")[0])
            except Exception:
                mid = None
        events = process_single_video(vid, match_id=mid, desired_accuracy=args.accuracy)
        all_events.extend(events)

    df = aggregate_and_write(all_events, args.out)
    print("Wrote", args.out, "with", len(df), "rows")


if __name__ == "__main__":
    main()
