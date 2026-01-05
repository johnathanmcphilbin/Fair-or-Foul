"""
Fair-or-Foul: Comprehensive Sports Referee Bias Analysis System

This package provides:
- Computer vision pipeline (YOLOv8, DeepSORT, pose estimation, collision detection)
- Kinematic analysis (homography, velocity, G-force, impact detection, whistle threshold)
- Statistical analysis (Kruskal-Wallis, Mann-Whitney U, Bonferroni, Dublin Delta, league comparisons)
- Martial arts validation (UFC data processing, correlation, control studies)
- ML model infrastructure (model management, inference pipeline)
"""

from .vision import (
    YOLOv8Detector,
    DeepSORTTracker,
    PoseEstimator,
    CollisionDetector,
    VisionPipeline
)

from .kinematics import (
    HomographyTransformer,
    VelocityCalculator,
    GForceCalculator,
    ImpactDetector,
    WhistleThresholdCalculator,
    KinematicAnalyzer
)

from .statistics import (
    StatisticalAnalyzer,
    BiasReportGenerator
)

from .martial_arts import (
    UFCDataProcessor,
    CorrelationCalculator,
    ControlStudy,
    MartialArtsValidator
)

from .models import (
    ModelManager,
    InferencePipeline,
    ModelTrainer
)

from .pipeline import CompleteAnalysisPipeline

from .soccer_rules import (
    SoccerBallTracker,
    OutOfPlayDetector,
    VelocityDropDetector,
    ShoulderToShoulderAnalyzer,
    SoccerFoulClassifier
)

from .config import (
    CALL_TYPES,
    YOLO_MODEL_PATH,
    DEFAULT_FPS,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    IMPACT_G_FORCE_THRESHOLD,
    WHISTLE_THRESHOLD
)

__version__ = "2.0.0"
__all__ = [
    # Vision
    "YOLOv8Detector",
    "DeepSORTTracker",
    "PoseEstimator",
    "CollisionDetector",
    "VisionPipeline",
    # Kinematics
    "HomographyTransformer",
    "VelocityCalculator",
    "GForceCalculator",
    "ImpactDetector",
    "WhistleThresholdCalculator",
    "KinematicAnalyzer",
    # Statistics
    "StatisticalAnalyzer",
    "BiasReportGenerator",
    # Martial Arts
    "UFCDataProcessor",
    "CorrelationCalculator",
    "ControlStudy",
    "MartialArtsValidator",
    # Models
    "ModelManager",
    "InferencePipeline",
    "ModelTrainer",
    # Pipeline
    "CompleteAnalysisPipeline",
    # Soccer Rules
    "SoccerBallTracker",
    "OutOfPlayDetector",
    "VelocityDropDetector",
    "ShoulderToShoulderAnalyzer",
    "SoccerFoulClassifier",
    # Config
    "CALL_TYPES",
    "YOLO_MODEL_PATH",
    "DEFAULT_FPS",
    "COURT_LENGTH_M",
    "COURT_WIDTH_M",
    "IMPACT_G_FORCE_THRESHOLD",
    "WHISTLE_THRESHOLD"
]

