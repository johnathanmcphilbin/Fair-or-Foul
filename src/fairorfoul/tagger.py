import cv2
import csv
import time
from pathlib import Path
from .config import CALL_TYPES


def _ts(frame_idx, fps):
    sec = frame_idx / max(fps, 1)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def tag_video(
    video_path,
    sport,
    match_id,
    referee_id,
    team_a,
    team_b,
    out_csv="data/processed/tags.csv",
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    call_types = CALL_TYPES[sport.lower()]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(out_csv).exists()
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if not exists:
        w.writerow(
            [
                "Match ID",
                "Sport",
                "Referee ID",
                "Team A Name",
                "Team B Name",
                "Call Timestamp (MM:SS)",
                "Call Type",
                "Call Against Team",
                "Player Number",
                "Score at Call",
                "Location on Pitch/Court",
            ]
        )
    sel_team = team_b
    paused = False
    last_log = time.time()
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t = _ts(frame_idx, fps)
        h, wpx, _ = frame.shape if frame is not None else (720, 1280, 3)
        overlay = frame.copy() if frame is not None else None
        if overlay is not None:
            cv2.rectangle(overlay, (0, 0), (wpx, 60), (0, 0, 0), -1)
            txt = f"{sport.upper()}  t={t}  team={sel_team}  keys: [A/B team] [1-{len(call_types)} type] [Space pause] [Q quit]"
            cv2.putText(
                overlay,
                txt,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y = 80
            for i, name in enumerate(call_types, start=1):
                cv2.putText(
                    overlay,
                    f"{i}:{name}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28
            cv2.imshow("Fair-or-Foul Tagger", overlay)
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("a"), ord("A")):
            sel_team = team_a
        if key in (ord("b"), ord("B")):
            sel_team = team_b
        if key in [ord(str(d)) for d in range(1, 10)]:
            idx = int(chr(key)) - 1
            if 0 <= idx < len(call_types):
                if time.time() - last_log > 0.15:
                    w.writerow(
                        [
                            match_id,
                            sport,
                            referee_id,
                            team_a,
                            team_b,
                            t,
                            call_types[idx],
                            sel_team,
                            "",
                            "",
                            "",
                        ]
                    )
                    f.flush()
                    last_log = time.time()
    f.close()
    cap.release()
    cv2.destroyAllWindows()
