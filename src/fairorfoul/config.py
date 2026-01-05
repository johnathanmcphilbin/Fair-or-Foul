CALL_TYPES = {
    "basketball": [
        "foul",
        "travel",
        "double_dribble",
        "charge",
        "block",
        "shooting_foul",
        "technical",
        "out_of_bounds",
    ],
    "soccer": [
        "foul",
        "yellow_card",
        "red_card",
        "offside",
        "penalty",
        "handball",
        "out_of_play",
    ],
    "martial_arts": [
        "warning",
        "point_deduction",
        "disqualification",
        "stalling",
        "illegal_move",
    ],
}

# Computer Vision Configuration
YOLO_MODEL_PATH = "models/yolov8n.pt"  # Can be yolov8s.pt, yolov8m.pt, etc.
DEEPSORT_MAX_AGE = 70
DEEPSORT_N_INIT = 3
POSE_MODEL_COMPLEXITY = 1  # 0, 1, or 2
COLLISION_IOU_THRESHOLD = 0.1
COLLISION_DISTANCE_THRESHOLD = 0.05  # Normalized distance

# Kinematic Analysis Configuration
COURT_LENGTH_M = {
    "basketball": 28.0,
    "soccer": 105.0,
    "martial_arts": 9.0  # Octagon diameter
}
COURT_WIDTH_M = {
    "basketball": 15.0,
    "soccer": 68.0,
    "martial_arts": 9.0
}
DEFAULT_FPS = 30.0
IMPACT_G_FORCE_THRESHOLD = 3.0
WHISTLE_THRESHOLD = 0.5
HIGH_G_FORCE_THRESHOLD = 2.0

# Statistical Analysis Configuration
STATISTICAL_ALPHA = 0.05
BONFERRONI_CORRECTION = True
DUBLIN_DELTA_BIAS_THRESHOLD = 0.1

# Model Configuration
MODELS_DIR = "models"
DEVICE = "auto"  # "cpu", "cuda", or "auto"
YOLO_CONF_THRESHOLD = 0.25

# Martial Arts Configuration
UFC_DATA_COLUMNS = [
    "fighter1", "fighter2", "referee", "warnings",
    "point_deductions", "disqualifications", "rounds", "result"
]
CONTROL_STUDY_ENABLED = True