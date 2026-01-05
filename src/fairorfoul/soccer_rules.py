"""
Soccer-Specific Rules Module for Fair-or-Foul
Implements ball tracking, out of play detection, velocity drop detection,
and shoulder-to-shoulder contact exemption

Development Notes:
- Started December 2025 after realizing generic collision detection wasn't sport-aware
- Major challenge: Ball is small, fast, and often occluded (harder than player tracking)
- Shoulder-to-shoulder logic required pose estimation integration (MediaPipe landmarks)
- Out of play detection needs pitch boundary calibration (homography integration)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from .config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD


class SoccerBallTracker:
    """
    Track soccer ball position using YOLOv8
    
    Development Notes:
    - Ball is COCO class 32 ("sports ball")
    - Much harder than player tracking: ball is small (20-30px), fast, often occluded
    - Confidence threshold set to 0.4 (lower than players) to catch fast-moving ball
    - Added temporal smoothing to handle occlusions (ball disappears behind players)
    """
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize ball tracker
        
        Args:
            model_path: Path to YOLOv8 model
            device: Device to run on
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not installed")
        
        if model_path is None:
            model_path = YOLO_MODEL_PATH
        
        self.model = YOLO(model_path)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lower confidence for ball (0.4 vs 0.25 for players)
        # Ball is smaller and moves faster, needs lower threshold
        self.conf_threshold = 0.4
        
        # Temporal smoothing: keep last N ball positions for occlusion handling
        self.ball_history = []  # List of (frame, bbox, confidence)
        self.max_history = 10  # Keep last 10 detections
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect ball in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with ball bbox and confidence, or None if not detected
        """
        results = self.model(frame, conf=self.conf_threshold, device=self.device)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                
                # COCO class 32 = "sports ball"
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter by size - balls are typically 20-50 pixels in diameter
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Reject if too large (probably not a ball) or too small (noise)
                    if 15 < bbox_width < 100 and 15 < bbox_height < 100:
                        ball_detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'class': cls
                        }
                        
                        # Add to history for temporal smoothing
                        self.ball_history.append(ball_detection)
                        if len(self.ball_history) > self.max_history:
                            self.ball_history.pop(0)
                        
                        return ball_detection
        
        # Ball not detected - try to predict from history (temporal smoothing)
        if len(self.ball_history) > 0:
            # Use last known position with reduced confidence
            last_ball = self.ball_history[-1].copy()
            last_ball['confidence'] *= 0.7  # Reduce confidence for predicted position
            last_ball['predicted'] = True
            return last_ball
        
        return None
    
    def get_ball_position(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get current ball position (center coordinates)
        
        Args:
            frame: Input frame
            
        Returns:
            (x, y) ball center position, or None if not detected
        """
        ball = self.detect_ball(frame)
        if ball:
            return ball['center']
        return None


class OutOfPlayDetector:
    """
    Detect when ball goes out of play
    
    Development Notes:
    - Soccer rule: Ball must FULLY cross the line (entire circumference)
    - Need pitch boundaries from homography calibration
    - Must distinguish: touchline (throw-in) vs goal line (goal kick/corner)
    - Challenge: Ball often occluded at boundaries (players near line)
    """
    
    def __init__(self, pitch_boundaries: Optional[Dict] = None):
        """
        Initialize out of play detector
        
        Args:
            pitch_boundaries: Dictionary with 'left', 'right', 'top', 'bottom' in world coordinates (meters)
        """
        self.pitch_boundaries = pitch_boundaries
        self.ball_radius_m = 0.11  # FIFA standard: ball radius = 11 cm
    
    def set_pitch_boundaries(self, boundaries: Dict):
        """
        Set pitch boundaries
        
        Args:
            boundaries: Dict with 'left', 'right', 'top', 'bottom' in meters
        """
        self.pitch_boundaries = boundaries
    
    def is_ball_out_of_play(
        self,
        ball_position_world: Tuple[float, float],
        last_touch_team: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if ball is out of play
        
        Args:
            ball_position_world: Ball position in world coordinates (meters)
            last_touch_team: Team that last touched ball (for restart determination)
            
        Returns:
            (is_out, restart_type): Tuple of (bool, str or None)
            restart_type: 'throw_in', 'goal_kick', 'corner_kick', or None
        """
        if self.pitch_boundaries is None:
            return False, None
        
        ball_x, ball_y = ball_position_world
        
        # Check if ENTIRE ball crossed boundary (ball radius matters)
        left_bound = self.pitch_boundaries.get('left', 0)
        right_bound = self.pitch_boundaries.get('right', 0)
        top_bound = self.pitch_boundaries.get('top', 0)
        bottom_bound = self.pitch_boundaries.get('bottom', 0)
        
        # Ball is out if center + radius is beyond boundary
        is_out_left = (ball_x - self.ball_radius_m) < left_bound
        is_out_right = (ball_x + self.ball_radius_m) > right_bound
        is_out_top = (ball_y - self.ball_radius_m) < top_bound
        is_out_bottom = (ball_y + self.ball_radius_m) > bottom_bound
        
        if not (is_out_left or is_out_right or is_out_top or is_out_bottom):
            return False, None
        
        # Determine restart type
        restart_type = None
        
        if is_out_left or is_out_right:
            # Ball crossed touchline → throw-in
            restart_type = 'throw_in'
        elif is_out_top or is_out_bottom:
            # Ball crossed goal line
            # Need to know which team's goal line and who last touched
            # Simplified: assume top = one goal, bottom = other goal
            if last_touch_team:
                # This would need team assignment logic
                # For now, return generic 'goal_kick' or 'corner_kick'
                restart_type = 'goal_kick'  # Simplified
            else:
                restart_type = 'goal_kick'
        
        return True, restart_type


class VelocityDropDetector:
    """
    Detect when play has stopped (all players slow down)
    
    Development Notes:
    - Soccer rule: Whistle blown → all players should stop moving
    - Free kick → most players stationary
    - Goalkeeper holding ball → play paused
    - Challenge: Distinguishing natural slowdown (corner kick setup) vs stoppage
    """
    
    def __init__(self, stopped_threshold: float = 1.0, stopped_percentage: float = 0.8):
        """
        Initialize velocity drop detector
        
        Args:
            stopped_threshold: Speed below which player is "stopped" (m/s)
            stopped_percentage: Percentage of players that must be stopped to trigger detection
        """
        self.stopped_threshold = stopped_threshold
        self.stopped_percentage = stopped_percentage
    
    def detect_stoppage(self, all_player_velocities: List[float]) -> Dict:
        """
        Detect when play has stopped
        
        Args:
            all_player_velocities: List of current speeds for all players (m/s)
            
        Returns:
            Dictionary with stoppage information
        """
        if len(all_player_velocities) == 0:
            return {
                'is_stopped': False,
                'stopped_count': 0,
                'total_players': 0,
                'stopped_percentage': 0.0
            }
        
        stopped_count = sum(1 for v in all_player_velocities if v < self.stopped_threshold)
        stopped_percentage = stopped_count / len(all_player_velocities)
        
        # If >80% of players stopped, play has likely paused
        is_stopped = stopped_percentage >= self.stopped_percentage
        
        return {
            'is_stopped': is_stopped,
            'stopped_count': stopped_count,
            'total_players': len(all_player_velocities),
            'stopped_percentage': stopped_percentage,
            'threshold': self.stopped_threshold
        }
    
    def detect_whistle(self, velocity_history: List[List[float]], window_size: int = 5) -> bool:
        """
        Detect whistle by sudden velocity drop across all players
        
        Args:
            velocity_history: List of velocity lists for last N frames
            window_size: Number of frames to analyze
            
        Returns:
            True if whistle likely blown
        """
        if len(velocity_history) < window_size:
            return False
        
        # Get recent frames
        recent = velocity_history[-window_size:]
        
        # Calculate average speed per frame
        avg_speeds = [np.mean(velocities) for velocities in recent]
        
        # Check for sudden drop (e.g., from 5 m/s to 1 m/s in 1-2 frames)
        if len(avg_speeds) >= 2:
            speed_before = avg_speeds[0]
            speed_after = avg_speeds[-1]
            
            # Sudden drop: >60% reduction in average speed
            if speed_before > 2.0 and speed_after < speed_before * 0.4:
                return True
        
        return False


class ShoulderToShoulderAnalyzer:
    """
    Analyze if contact is legal shoulder-to-shoulder charge
    
    Development Notes:
    - FIFA Laws of the Game, Law 12: Shoulder-to-shoulder is legal if:
      1. Both players challenging for the ball
      2. Contact is with shoulder (not elbow, hands, or from behind)
      3. No excessive force
      4. Both players have at least one foot on ground
      5. Ball is within playing distance (~1 meter)
    
    - Major challenge: Requires pose estimation (MediaPipe) to identify contact point
    - Ball proximity check requires ball tracking integration
    """
    
    def __init__(self, ball_proximity_threshold: float = 1.0, max_gforce: float = 3.0):
        """
        Initialize shoulder-to-shoulder analyzer
        
        Args:
            ball_proximity_threshold: Maximum distance to ball for "challenging" (meters)
            max_gforce: Maximum G-force for legal contact
        """
        self.ball_proximity_threshold = ball_proximity_threshold
        self.max_gforce = max_gforce
    
    def is_legal_shoulder_charge(
        self,
        collision_data: Dict,
        pose1: Optional[Dict],
        pose2: Optional[Dict],
        velocities: Dict,
        ball_position_world: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Determine if shoulder-to-shoulder contact is legal
        
        Args:
            collision_data: Collision information (bbox, G-force, etc.)
            pose1: Pose landmarks for player 1 (MediaPipe format)
            pose2: Pose landmarks for player 2
            velocities: Velocity information for both players
            ball_position_world: Ball position in world coordinates (meters)
            
        Returns:
            Dictionary with legal status and reasoning
        """
        result = {
            'is_legal': False,
            'reasons': [],
            'violations': []
        }
        
        # 1. Check if pose data available
        if pose1 is None or pose2 is None or not pose1.get('has_pose', False) or not pose2.get('has_pose', False):
            result['violations'].append('pose_data_unavailable')
            result['reasons'].append('Cannot determine contact point without pose estimation')
            return result
        
        # 2. Check contact point (shoulder vs other body parts)
        contact_point1 = self._get_contact_point(pose1, collision_data)
        contact_point2 = self._get_contact_point(pose2, collision_data)
        
        if contact_point1 != 'shoulder' or contact_point2 != 'shoulder':
            result['violations'].append('illegal_contact_point')
            result['reasons'].append(f'Contact at {contact_point1}/{contact_point2}, not shoulder-to-shoulder')
            return result
        
        result['reasons'].append('Contact is shoulder-to-shoulder')
        
        # 3. Check if contact from behind
        approach_angle = self._calculate_approach_angle(velocities)
        if approach_angle > 45:  # Coming from behind (>45 degrees)
            result['violations'].append('contact_from_behind')
            result['reasons'].append(f'Approach angle {approach_angle:.1f}° indicates contact from behind')
            return result
        
        result['reasons'].append(f'Approach angle {approach_angle:.1f}° is acceptable (front/side contact)')
        
        # 4. Check excessive force (using G-force from kinematic analysis)
        max_gforce = collision_data.get('max_gforce', 0)
        if max_gforce > self.max_gforce:
            result['violations'].append('excessive_force')
            result['reasons'].append(f'G-force {max_gforce:.2f}G exceeds threshold {self.max_gforce}G')
            return result
        
        result['reasons'].append(f'G-force {max_gforce:.2f}G is within acceptable range')
        
        # 5. Check if ball is nearby (critical for "challenging for the ball")
        if ball_position_world is None:
            result['violations'].append('ball_position_unknown')
            result['reasons'].append('Cannot verify players are challenging for ball')
            # Don't fail on this - ball might be occluded, but log it
        else:
            ball_distance = self._get_ball_distance(collision_data, ball_position_world)
            if ball_distance > self.ball_proximity_threshold:
                result['violations'].append('ball_too_far')
                result['reasons'].append(f'Ball distance {ball_distance:.2f}m exceeds threshold {self.ball_proximity_threshold}m')
                return result
            
            result['reasons'].append(f'Ball within {ball_distance:.2f}m - players challenging for ball')
        
        # 6. Check if feet are grounded (jumping into opponent is illegal)
        if not self._both_feet_grounded(pose1, pose2):
            result['violations'].append('player_in_air')
            result['reasons'].append('One or both players not grounded (jumping into opponent)')
            return result
        
        result['reasons'].append('Both players have feet on ground')
        
        # All checks passed
        result['is_legal'] = True
        return result
    
    def _get_contact_point(self, pose: Dict, collision_data: Dict) -> str:
        """
        Determine which body part is in contact
        
        Args:
            pose: Pose landmarks
            collision_data: Collision information
            
        Returns:
            Body part name: 'shoulder', 'elbow', 'hand', 'back', 'unknown'
        """
        if not pose.get('has_pose', False) or 'landmarks' not in pose:
            return 'unknown'
        
        landmarks = pose['landmarks']
        if len(landmarks) < 33:  # MediaPipe has 33 landmarks
            return 'unknown'
        
        # MediaPipe landmark indices:
        # 11: Left shoulder
        # 12: Right shoulder
        # 13: Left elbow
        # 14: Right elbow
        # 15: Left wrist
        # 16: Right wrist
        # 23: Left hip
        # 24: Right hip
        
        # Get collision center (simplified - would need actual collision point)
        collision_bbox = collision_data.get('bbox', [0, 0, 0, 0])
        collision_center = (
            (collision_bbox[0] + collision_bbox[2]) / 2,
            (collision_bbox[1] + collision_bbox[3]) / 2
        )
        
        # Get key body part positions (normalized coordinates)
        left_shoulder = landmarks[11] if len(landmarks) > 11 else None
        right_shoulder = landmarks[12] if len(landmarks) > 12 else None
        left_elbow = landmarks[13] if len(landmarks) > 13 else None
        right_elbow = landmarks[14] if len(landmarks) > 14 else None
        
        # Calculate distances (would need to convert normalized to pixel coordinates)
        # Simplified: assume collision center is closest to shoulder if in upper body region
        if collision_center[1] < collision_bbox[3] * 0.6:  # Upper 60% of bbox
            # Check if closer to shoulder than elbow
            if left_shoulder and right_shoulder:
                return 'shoulder'
            elif left_elbow or right_elbow:
                return 'elbow'
        
        return 'unknown'
    
    def _calculate_approach_angle(self, velocities: Dict) -> float:
        """
        Calculate approach angle between two players
        
        Args:
            velocities: Dict with 'player1' and 'player2' velocity vectors
            
        Returns:
            Approach angle in degrees (0 = head-on, 90 = perpendicular, >45 = from behind)
        """
        v1 = velocities.get('player1', {'vx': 0, 'vy': 0})
        v2 = velocities.get('player2', {'vx': 0, 'vy': 0})
        
        # Calculate velocity vectors
        v1_vec = np.array([v1.get('vx', 0), v1.get('vy', 0)])
        v2_vec = np.array([v2.get('vx', 0), v2.get('vy', 0)])
        
        # Calculate angle between velocity vectors
        dot_product = np.dot(v1_vec, v2_vec)
        mag1 = np.linalg.norm(v1_vec)
        mag2 = np.linalg.norm(v2_vec)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _get_ball_distance(
        self,
        collision_data: Dict,
        ball_position_world: Tuple[float, float]
    ) -> float:
        """
        Calculate distance from collision point to ball
        
        Args:
            collision_data: Collision information
            ball_position_world: Ball position in world coordinates
            
        Returns:
            Distance in meters
        """
        # Get collision center in world coordinates
        # This would need homography transformation
        # Simplified: assume collision_data has world position
        collision_world = collision_data.get('world_position', (0, 0))
        
        distance = euclidean(collision_world, ball_position_world)
        return distance
    
    def _both_feet_grounded(self, pose1: Dict, pose2: Dict) -> bool:
        """
        Check if both players have feet on ground
        
        Args:
            pose1: Pose landmarks for player 1
            pose2: Pose landmarks for player 2
            
        Returns:
            True if both players grounded
        """
        if not pose1.get('has_pose', False) or not pose2.get('has_pose', False):
            return False  # Can't determine without pose data
        
        landmarks1 = pose1.get('landmarks', [])
        landmarks2 = pose2.get('landmarks', [])
        
        if len(landmarks1) < 33 or len(landmarks2) < 33:
            return False
        
        # MediaPipe landmarks:
        # 27: Left ankle
        # 28: Right ankle
        # 29: Left heel
        # 30: Right heel
        
        # Simplified check: if ankles are below hips, player is likely grounded
        # More sophisticated: check if heels are visible and below a threshold
        left_ankle1 = landmarks1[27] if len(landmarks1) > 27 else None
        right_ankle1 = landmarks1[28] if len(landmarks1) > 28 else None
        left_hip1 = landmarks1[23] if len(landmarks1) > 23 else None
        right_hip1 = landmarks1[24] if len(landmarks1) > 24 else None
        
        left_ankle2 = landmarks2[27] if len(landmarks2) > 27 else None
        right_ankle2 = landmarks2[28] if len(landmarks2) > 28 else None
        left_hip2 = landmarks2[23] if len(landmarks2) > 23 else None
        right_hip2 = landmarks2[24] if len(landmarks2) > 24 else None
        
        # Check if ankles are below hips (simplified grounded check)
        player1_grounded = (
            (left_ankle1 and left_hip1 and left_ankle1['y'] > left_hip1['y']) or
            (right_ankle1 and right_hip1 and right_ankle1['y'] > right_hip1['y'])
        )
        
        player2_grounded = (
            (left_ankle2 and left_hip2 and left_ankle2['y'] > left_hip2['y']) or
            (right_ankle2 and right_hip2 and right_ankle2['y'] > right_hip2['y'])
        )
        
        return player1_grounded and player2_grounded


class SoccerFoulClassifier:
    """
    Classify collisions as legal or foul based on soccer rules
    
    Development Notes:
    - Integrates all soccer-specific rule checks
    - Combines ball tracking, pose estimation, and kinematic analysis
    - Returns detailed classification with reasoning
    """
    
    def __init__(
        self,
        ball_tracker: Optional[SoccerBallTracker] = None,
        shoulder_analyzer: Optional[ShoulderToShoulderAnalyzer] = None,
        out_of_play_detector: Optional[OutOfPlayDetector] = None
    ):
        """
        Initialize soccer foul classifier
        
        Args:
            ball_tracker: SoccerBallTracker instance
            shoulder_analyzer: ShoulderToShoulderAnalyzer instance
            out_of_play_detector: OutOfPlayDetector instance
        """
        self.ball_tracker = ball_tracker or SoccerBallTracker()
        self.shoulder_analyzer = shoulder_analyzer or ShoulderToShoulderAnalyzer()
        self.out_of_play_detector = out_of_play_detector or OutOfPlayDetector()
    
    def classify_foul(
        self,
        collision_data: Dict,
        kinematic_data: Dict,
        pose1: Optional[Dict] = None,
        pose2: Optional[Dict] = None,
        frame: Optional[np.ndarray] = None,
        homography: Optional = None
    ) -> Dict:
        """
        Classify collision as legal or foul based on soccer rules
        
        Args:
            collision_data: Collision information (bbox, IoU, distance)
            kinematic_data: Kinematic analysis (G-force, velocities)
            pose1: Pose landmarks for player 1
            pose2: Pose landmarks for player 2
            frame: Current frame (for ball detection)
            homography: HomographyTransformer instance (for world coordinates)
            
        Returns:
            Dictionary with foul classification and reasoning
        """
        result = {
            'foul_type': 'unknown',
            'is_foul': False,
            'is_legal': False,
            'reasoning': [],
            'confidence': 0.0
        }
        
        # 1. Check if ball is out of play (no foul possible if ball not in play)
        if frame is not None and homography is not None:
            ball = self.ball_tracker.detect_ball(frame)
            if ball:
                ball_center_pixel = ball['center']
                try:
                    ball_position_world = homography.pixel_to_world(ball_center_pixel)
                    is_out, restart_type = self.out_of_play_detector.is_ball_out_of_play(ball_position_world)
                    
                    if is_out:
                        result['foul_type'] = 'not_in_play'
                        result['is_foul'] = False
                        result['is_legal'] = True
                        result['reasoning'].append(f'Ball is out of play ({restart_type}) - no foul possible')
                        result['confidence'] = 0.9
                        return result
                except:
                    pass  # Homography not calibrated, skip out of play check
        
        # 2. Check shoulder-to-shoulder exemption
        if pose1 and pose2:
            ball_position_world = None
            if frame is not None and homography is not None:
                ball = self.ball_tracker.detect_ball(frame)
                if ball:
                    try:
                        ball_position_world = homography.pixel_to_world(ball['center'])
                    except:
                        pass
            
            velocities = {
                'player1': kinematic_data.get('player1_velocity', {}),
                'player2': kinematic_data.get('player2_velocity', {})
            }
            
            shoulder_check = self.shoulder_analyzer.is_legal_shoulder_charge(
                collision_data,
                pose1,
                pose2,
                velocities,
                ball_position_world
            )
            
            if shoulder_check['is_legal']:
                result['foul_type'] = 'legal_shoulder_charge'
                result['is_foul'] = False
                result['is_legal'] = True
                result['reasoning'] = shoulder_check['reasons']
                result['confidence'] = 0.85
                return result
        
        # 3. Check excessive force (dangerous play)
        max_gforce = kinematic_data.get('max_gforce', 0)
        if max_gforce > 4.0:
            result['foul_type'] = 'dangerous_play'
            result['is_foul'] = True
            result['reasoning'].append(f'Excessive force detected: {max_gforce:.2f}G')
            result['confidence'] = 0.8
            return result
        
        # 4. Check approach from behind
        if pose1 and pose2:
            velocities = {
                'player1': kinematic_data.get('player1_velocity', {}),
                'player2': kinematic_data.get('player2_velocity', {})
            }
            approach_angle = self.shoulder_analyzer._calculate_approach_angle(velocities)
            if approach_angle > 45:
                result['foul_type'] = 'foul_from_behind'
                result['is_foul'] = True
                result['reasoning'].append(f'Contact from behind: {approach_angle:.1f}°')
                result['confidence'] = 0.75
                return result
        
        # 5. Default: generic foul (if collision detected but doesn't meet legal criteria)
        if collision_data.get('collision', False):
            result['foul_type'] = 'foul'
            result['is_foul'] = True
            result['reasoning'].append('Collision detected - generic foul')
            result['confidence'] = 0.6
        else:
            result['foul_type'] = 'no_contact'
            result['is_foul'] = False
            result['reasoning'].append('No collision detected')
            result['confidence'] = 0.5
        
        return result

