"""
Complete Pipeline Integration for Fair-or-Foul
Integrates computer vision, kinematics, statistics, and ML models

Development Notes:
- GPU Memory Management was a major challenge (December 2025)
- Processing 1080p video caused VRAM overflow on RTX 3060 (12GB) after ~500 frames
- Root cause: PyTorch's CUDA memory allocator doesn't auto-release cached memory
- Solution: Batch processing with explicit memory clearing every 100 frames
- Also downscaled to 720p (minimal accuracy loss, reduced VRAM to ~4.5GB)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
import torch  # For GPU memory management

from .vision import VisionPipeline
from .kinematics import KinematicAnalyzer
from .statistics import StatisticalAnalyzer, BiasReportGenerator
from .martial_arts import MartialArtsValidator
from .models import ModelManager, InferencePipeline
from .soccer_rules import (
    SoccerBallTracker,
    OutOfPlayDetector,
    VelocityDropDetector,
    ShoulderToShoulderAnalyzer,
    SoccerFoulClassifier
)
from .config import (
    DEFAULT_FPS, COURT_LENGTH_M, COURT_WIDTH_M,
    IMPACT_G_FORCE_THRESHOLD, WHISTLE_THRESHOLD
)


class CompleteAnalysisPipeline:
    """Complete end-to-end analysis pipeline"""
    
    def __init__(
        self,
        sport: str = "basketball",
        yolo_model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize complete analysis pipeline
        
        Args:
            sport: Sport type ("basketball", "soccer", "martial_arts")
            yolo_model_path: Path to YOLOv8 model
            device: Device to run on
        """
        self.sport = sport
        self.fps = DEFAULT_FPS
        
        # Initialize components
        self.vision_pipeline = VisionPipeline(
            yolo_model_path=yolo_model_path,
            device=device,
            enable_tracking=True,
            enable_pose=True,
            enable_collision=True
        )
        
        court_length = COURT_LENGTH_M.get(sport, 28.0)
        court_width = COURT_WIDTH_M.get(sport, 15.0)
        
        self.kinematic_analyzer = KinematicAnalyzer(
            fps=self.fps,
            court_length_m=court_length,
            court_width_m=court_width
        )
        
        self.statistical_analyzer = StatisticalAnalyzer()
        self.bias_report_generator = BiasReportGenerator(self.statistical_analyzer)
        self.martial_arts_validator = MartialArtsValidator()
        
        self.model_manager = ModelManager()
        self.inference_pipeline = InferencePipeline(self.model_manager)
        
        # Soccer-specific components (initialized if sport is soccer)
        self.ball_tracker = None
        self.out_of_play_detector = None
        self.velocity_drop_detector = None
        self.shoulder_analyzer = None
        self.soccer_foul_classifier = None
        
        if sport == "soccer":
            self.ball_tracker = SoccerBallTracker(device=device)
            self.out_of_play_detector = OutOfPlayDetector()
            self.velocity_drop_detector = VelocityDropDetector()
            self.shoulder_analyzer = ShoulderToShoulderAnalyzer()
            self.soccer_foul_classifier = SoccerFoulClassifier(
                ball_tracker=self.ball_tracker,
                shoulder_analyzer=self.shoulder_analyzer,
                out_of_play_detector=self.out_of_play_detector
            )
        
    def analyze_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        output_data_path: Optional[str] = None
    ) -> Dict:
        """
        Complete analysis of a video file
        
        Args:
            video_path: Path to input video
            output_video_path: Optional path to save annotated video
            output_data_path: Optional path to save analysis data
            
        Returns:
            Complete analysis results
        """
        # Step 1: Computer vision processing
        print("Step 1: Running computer vision pipeline...")
        
        # GPU MEMORY MANAGEMENT (December 2025)
        # Processing 1080p video caused VRAM overflow on RTX 3060 (12GB) after ~500 frames
        # PyTorch's CUDA memory allocator doesn't automatically release cached memory
        # Solution: Process in batches and clear cache every 100 frames
        # This is handled internally in VisionPipeline.process_video()
        vision_results = self.vision_pipeline.process_video(
            video_path, output_video_path
        )
        
        # Clear GPU cache after vision processing to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 2: Extract trajectories and calculate kinematics
        print("Step 2: Calculating kinematics...")
        kinematic_results = []
        
        for frame_result in vision_results:
            frame_num = frame_result['frame_number']
            tracks = frame_result['tracks']
            
            # Extract trajectories for each tracked player
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                
                # Convert bbox center to position
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Store position for trajectory analysis
                if not hasattr(self, 'trajectories'):
                    self.trajectories = {}
                
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                
                self.trajectories[track_id].append((center_x, center_y))
        
        # Step 3: Analyze trajectories
        player_analyses = {}
        for track_id, positions in self.trajectories.items():
            if len(positions) > 10:  # Need sufficient data
                analysis = self.kinematic_analyzer.analyze_player_trajectory(
                    positions, world_positions=None
                )
                player_analyses[track_id] = analysis
        
        # Step 4: Detect collisions and impacts
        print("Step 3: Detecting collisions and impacts...")
        collision_analyses = []
        
        if len(player_analyses) >= 2:
            player_ids = list(player_analyses.keys())
            for i in range(len(player_ids)):
                for j in range(i + 1, len(player_ids)):
                    player1_id = player_ids[i]
                    player2_id = player_ids[j]
                    
                    collision = self.kinematic_analyzer.analyze_collision(
                        player_analyses[player1_id],
                        player_analyses[player2_id]
                    )
                    
                    if collision['impact_count'] > 0:
                        collision_analyses.append({
                            'player1_id': player1_id,
                            'player2_id': player2_id,
                            **collision
                        })
        
        # Step 5: Calculate whistle thresholds
        print("Step 4: Calculating whistle thresholds...")
        whistle_events = []
        for track_id, analysis in player_analyses.items():
            whistle_result = analysis['whistle_threshold']
            if whistle_result['should_whistle']:
                whistle_events.append({
                    'player_id': track_id,
                    'frame': len(self.trajectories[track_id]) - 1,
                    **whistle_result
                })
        
        # Step 6: Soccer-specific analysis (if sport is soccer)
        soccer_analysis = {}
        if self.sport == "soccer" and self.soccer_foul_classifier:
            print("Step 5: Running soccer-specific rule analysis...")
            soccer_analysis = self._analyze_soccer_rules(
                vision_results,
                player_analyses,
                collision_analyses
            )
        
        # Compile results
        results = {
            'video_path': video_path,
            'sport': self.sport,
            'vision_results': {
                'total_frames': len(vision_results),
                'total_detections': sum([len(fr['detections']) for fr in vision_results]),
                'total_tracks': len(self.trajectories) if hasattr(self, 'trajectories') else 0,
                'total_collisions': sum([len(fr['collisions']) for fr in vision_results]),
                'frame_results': vision_results  # Store full frame-by-frame results for export
            },
            'kinematic_results': {
                'players_analyzed': len(player_analyses),
                'player_analyses': {
                    str(k): {
                        'max_speed': v['max_speed'],
                        'max_gforce': v['max_gforce'],
                        'total_distance': v['total_distance'],
                        'whistle_threshold': v['whistle_threshold']['threshold']
                    }
                    for k, v in player_analyses.items()
                },
                'player_analyses_full': player_analyses  # Store full data for export
            },
            'collision_analyses': collision_analyses,
            'whistle_events': whistle_events,
            'summary': {
                'total_impacts': sum([c['impact_count'] for c in collision_analyses]),
                'whistle_events_count': len(whistle_events),
                'potential_fouls': len([w for w in whistle_events if w['threshold'] > WHISTLE_THRESHOLD])
            }
        }
        
        # Add soccer-specific results if available
        if soccer_analysis:
            results['soccer_analysis'] = soccer_analysis
        
        # Save results if path provided
        if output_data_path:
            self._save_results(results, output_data_path)
        
        return results
    
    def analyze_csv_data(
        self,
        csv_path: str,
        include_dublin_delta: bool = True,
        include_league_comparison: bool = False
    ) -> Dict:
        """
        Analyze CSV data with statistical methods
        
        Args:
            csv_path: Path to CSV file with call data
            include_dublin_delta: Whether to calculate Dublin Delta
            include_league_comparison: Whether to include league comparison
            
        Returns:
            Statistical analysis results
        """
        df = pd.read_csv(csv_path)
        
        # Generate bias report
        report = self.bias_report_generator.generate_report(
            df,
            include_dublin_delta=include_dublin_delta,
            include_league_comparison=include_league_comparison
        )
        
        return report
    
    def validate_martial_arts(
        self,
        ufc_data_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Validate referee bias in martial arts data
        
        Args:
            ufc_data_path: Path to UFC data CSV
            output_path: Optional path to save validation report
            
        Returns:
            Validation report
        """
        report = self.martial_arts_validator.generate_validation_report(
            ufc_data_path,
            output_path
        )
        
        return report
    
    def _analyze_soccer_rules(
        self,
        vision_results: List[Dict],
        player_analyses: Dict,
        collision_analyses: List[Dict]
    ) -> Dict:
        """
        Analyze soccer-specific rules (ball tracking, out of play, shoulder-to-shoulder)
        
        Args:
            vision_results: Vision pipeline results
            player_analyses: Player trajectory analyses
            collision_analyses: Collision detection results
            
        Returns:
            Soccer-specific analysis results
        """
        soccer_results = {
            'ball_tracking': [],
            'out_of_play_events': [],
            'stoppage_detections': [],
            'foul_classifications': []
        }
        
        # Track ball across frames
        ball_positions = []
        all_player_speeds = []
        velocity_history = []
        
        # Process each frame for soccer-specific analysis
        cap = cv2.VideoCapture(vision_results[0].get('video_path', '')) if vision_results else None
        
        for frame_idx, frame_result in enumerate(vision_results):
            # Get frame for ball detection (would need to read from video)
            # For now, use frame_result data
            
            # Track ball if available
            if self.ball_tracker and cap:
                ret, frame = cap.read() if cap else (False, None)
                if ret:
                    ball = self.ball_tracker.detect_ball(frame)
                    if ball:
                        ball_positions.append({
                            'frame': frame_idx,
                            'position': ball['center'],
                            'confidence': ball['confidence']
                        })
            
            # Collect player speeds for stoppage detection
            frame_speeds = []
            for track_id, analysis in player_analyses.items():
                if 'velocities' in analysis and len(analysis['velocities']) > 0:
                    # Get most recent speed
                    latest_speed = analysis['velocities'][-1].get('speed', 0)
                    frame_speeds.append(latest_speed)
            
            all_player_speeds.append(frame_speeds)
            velocity_history.append(frame_speeds)
            
            # Detect stoppage
            if self.velocity_drop_detector and len(frame_speeds) > 0:
                stoppage = self.velocity_drop_detector.detect_stoppage(frame_speeds)
                if stoppage['is_stopped']:
                    soccer_results['stoppage_detections'].append({
                        'frame': frame_idx,
                        **stoppage
                    })
        
        # Classify fouls for each collision
        if self.soccer_foul_classifier:
            for collision in collision_analyses:
                player1_id = collision.get('player1_id')
                player2_id = collision.get('player2_id')
                
                # Get pose data from vision results (would need to match frame)
                pose1 = None
                pose2 = None
                
                # Get kinematic data
                kinematic_data = {
                    'max_gforce': collision.get('max_impact_severity', 0),
                    'player1_velocity': player_analyses.get(player1_id, {}).get('velocities', [{}])[-1] if player1_id in player_analyses else {},
                    'player2_velocity': player_analyses.get(player2_id, {}).get('velocities', [{}])[-1] if player2_id in player_analyses else {}
                }
                
                # Classify foul
                foul_classification = self.soccer_foul_classifier.classify_foul(
                    collision_data=collision,
                    kinematic_data=kinematic_data,
                    pose1=pose1,
                    pose2=pose2,
                    frame=None,  # Would need actual frame
                    homography=self.kinematic_analyzer.homography if hasattr(self.kinematic_analyzer, 'homography') else None
                )
                
                soccer_results['foul_classifications'].append({
                    'collision_id': f"{player1_id}_{player2_id}",
                    **foul_classification
                })
        
        # Ball tracking summary
        soccer_results['ball_tracking_summary'] = {
            'total_detections': len(ball_positions),
            'detection_rate': len(ball_positions) / len(vision_results) if vision_results else 0
        }
        
        # Stoppage summary
        soccer_results['stoppage_summary'] = {
            'total_stoppages': len(soccer_results['stoppage_detections']),
            'average_stopped_percentage': np.mean([
                s['stopped_percentage'] for s in soccer_results['stoppage_detections']
            ]) if soccer_results['stoppage_detections'] else 0
        }
        
        # Foul classification summary
        if soccer_results['foul_classifications']:
            foul_types = {}
            for fc in soccer_results['foul_classifications']:
                foul_type = fc.get('foul_type', 'unknown')
                foul_types[foul_type] = foul_types.get(foul_type, 0) + 1
            
            soccer_results['foul_classification_summary'] = {
                'total_classified': len(soccer_results['foul_classifications']),
                'foul_types': foul_types,
                'legal_contacts': foul_types.get('legal_shoulder_charge', 0),
                'illegal_contacts': sum([v for k, v in foul_types.items() if k != 'legal_shoulder_charge' and k != 'not_in_play'])
            }
        
        if cap:
            cap.release()
        
        return soccer_results
    
    def _save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON file"""
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_complete_report(
        self,
        video_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        ufc_data_path: Optional[str] = None,
        output_dir: str = "reports"
    ) -> Dict:
        """
        Generate complete analysis report combining all methods
        
        Args:
            video_path: Optional path to video file
            csv_path: Optional path to CSV data
            ufc_data_path: Optional path to UFC data
            output_dir: Directory to save reports
            
        Returns:
            Complete analysis report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'video_analysis': {},
            'statistical_analysis': {},
            'martial_arts_validation': {}
        }
        
        # Video analysis
        if video_path:
            video_results = self.analyze_video(
                video_path,
                output_video_path=str(output_path / "annotated_video.mp4"),
                output_data_path=str(output_path / "video_analysis.json")
            )
            report['video_analysis'] = video_results
        
        # Statistical analysis
        if csv_path:
            stats_results = self.analyze_csv_data(csv_path)
            report['statistical_analysis'] = stats_results
            
            # Save statistical report
            with open(output_path / "statistical_analysis.json", 'w') as f:
                json.dump(stats_results, f, indent=2, default=str)
        
        # Martial arts validation
        if ufc_data_path:
            ufc_results = self.validate_martial_arts(
                ufc_data_path,
                output_path=str(output_path / "martial_arts_validation.json")
            )
            report['martial_arts_validation'] = ufc_results
        
        # Save complete report
        with open(output_path / "complete_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def export_contact_events_csv(
        self,
        analysis_results: Dict,
        match_id: str,
        referee_calls_df: Optional[pd.DataFrame] = None,
        output_path: str = "data/processed/contact_events.csv",
        team_side_mapping: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """
        Export contact events to CSV with exactly 11 columns as specified.
        
        Schema:
        1. match_id – match identifier
        2. timestamp – time of the event within the match
        3. player_1_id – initiator
        4. player_2_id – recipient
        5. team_side – Home or Away (context for bias testing)
        6. impact_velocity – m/s at contact
        7. ellipse_overlap – % overlap at contact (proof contact occurred)
        8. deceleration_g – G-force style deceleration estimate (impact severity)
        9. ai_flag – 1/0 if AI flags it as foul based on thresholds
        10. ref_whistle – 1/0 if referee whistled in the event log
        11. discrepancy_type – human label (e.g., Human Miss / AI Overcall / Agreement)
        
        Args:
            analysis_results: Results from analyze_video() method
            match_id: Match identifier
            referee_calls_df: Optional DataFrame with referee calls (must have 'Call Timestamp (MM:SS)' column)
            output_path: Path to save CSV file
            team_side_mapping: Optional dict mapping player_id to 'Home' or 'Away'
            
        Returns:
            DataFrame with contact events
        """
        vision_results_summary = analysis_results.get('vision_results', {})
        frame_results = vision_results_summary.get('frame_results', [])
        collision_analyses = analysis_results.get('collision_analyses', [])
        # Use full player_analyses if available, otherwise fall back to summary
        kinematic_results = analysis_results.get('kinematic_results', {})
        player_analyses = kinematic_results.get('player_analyses_full', kinematic_results.get('player_analyses', {}))
        whistle_events = analysis_results.get('whistle_events', [])
        
        contact_events = []
        
        # Process collision analyses (these contain kinematic impact data)
        for collision in collision_analyses:
            player1_id = collision.get('player1_id')
            player2_id = collision.get('player2_id')
            
            # Get impact data
            impacts = collision.get('impacts', [])
            if not impacts:
                # If no detailed impacts, create one event from collision summary
                impacts = [{
                    'frame': 0,  # Default frame if not available
                    'impact_severity': collision.get('max_impact_severity', 0),
                    'distance': 0
                }]
            
            for impact in impacts:
                frame = impact.get('frame', 0)
                timestamp_seconds = frame / self.fps if self.fps > 0 else 0
                
                # Format timestamp as MM:SS
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Get player velocities at impact
                player1_vel = 0.0
                player2_vel = 0.0
                # Try to get velocity at impact frame, fall back to max_speed
                if player1_id:
                    p1_key = str(player1_id) if str(player1_id) in player_analyses else player1_id
                    if p1_key in player_analyses:
                        p1_data = player_analyses[p1_key]
                        # Try to get velocity at specific frame if available
                        if 'velocities' in p1_data and isinstance(p1_data['velocities'], list) and len(p1_data['velocities']) > frame:
                            player1_vel = p1_data['velocities'][frame].get('speed', 0)
                        else:
                            player1_vel = p1_data.get('max_speed', 0)
                
                if player2_id:
                    p2_key = str(player2_id) if str(player2_id) in player_analyses else player2_id
                    if p2_key in player_analyses:
                        p2_data = player_analyses[p2_key]
                        if 'velocities' in p2_data and isinstance(p2_data['velocities'], list) and len(p2_data['velocities']) > frame:
                            player2_vel = p2_data['velocities'][frame].get('speed', 0)
                        else:
                            player2_vel = p2_data.get('max_speed', 0)
                
                # Impact velocity is the relative velocity at contact
                impact_velocity = max(player1_vel, player2_vel)  # Use max as proxy for impact velocity
                
                # Ellipse overlap - use IoU from collision if available, otherwise estimate
                ellipse_overlap = collision.get('iou', 0.0) * 100  # Convert to percentage
                if ellipse_overlap == 0:
                    # Estimate from distance
                    distance = impact.get('distance', collision.get('distance', 2.0))
                    # Normalize: closer = higher overlap
                    if distance < 0.5:
                        ellipse_overlap = 80.0
                    elif distance < 1.0:
                        ellipse_overlap = 50.0
                    elif distance < 2.0:
                        ellipse_overlap = 20.0
                    else:
                        ellipse_overlap = 5.0
                
                # Deceleration G - from impact severity or G-force
                deceleration_g = impact.get('impact_severity', 0)
                if deceleration_g == 0:
                    # Get from player analyses
                    if player1_id:
                        p1_key = str(player1_id) if str(player1_id) in player_analyses else player1_id
                        if p1_key in player_analyses:
                            p1_data = player_analyses[p1_key]
                            deceleration_g = max(deceleration_g, p1_data.get('max_gforce', 0))
                    if player2_id:
                        p2_key = str(player2_id) if str(player2_id) in player_analyses else player2_id
                        if p2_key in player_analyses:
                            p2_data = player_analyses[p2_key]
                            deceleration_g = max(deceleration_g, p2_data.get('max_gforce', 0))
                
                # AI flag - based on whistle threshold or impact severity
                ai_flag = 0
                # Check if this collision triggered a whistle event
                for whistle_event in whistle_events:
                    if (whistle_event.get('player_id') == player1_id or 
                        whistle_event.get('player_id') == player2_id):
                        if whistle_event.get('threshold', 0) > WHISTLE_THRESHOLD:
                            ai_flag = 1
                            break
                
                # If no whistle event match, use impact severity threshold
                if ai_flag == 0 and deceleration_g > IMPACT_G_FORCE_THRESHOLD:
                    ai_flag = 1
                
                # Team side - use mapping if provided, otherwise default to 'Home'
                team_side = 'Home'
                if team_side_mapping:
                    team_side = team_side_mapping.get(player1_id, 'Home')
                
                # Ref whistle - check if referee called a foul at this timestamp
                ref_whistle = 0
                if referee_calls_df is not None and 'Call Timestamp (MM:SS)' in referee_calls_df.columns:
                    # Match timestamp (within ±2 seconds tolerance)
                    matching_calls = referee_calls_df[
                        referee_calls_df['Call Timestamp (MM:SS)'] == timestamp
                    ]
                    if len(matching_calls) > 0:
                        ref_whistle = 1
                    else:
                        # Check within ±2 seconds
                        for _, call_row in referee_calls_df.iterrows():
                            call_ts = call_row.get('Call Timestamp (MM:SS)', '')
                            try:
                                call_m, call_s = map(int, call_ts.split(':'))
                                call_total_sec = call_m * 60 + call_s
                                event_total_sec = minutes * 60 + seconds
                                if abs(call_total_sec - event_total_sec) <= 2:
                                    ref_whistle = 1
                                    break
                            except (ValueError, AttributeError):
                                continue
                
                # Discrepancy type - left blank for test portion (post-match-200)
                # This will be populated by human labeling
                discrepancy_type = ""
                
                contact_events.append({
                    'match_id': match_id,
                    'timestamp': timestamp,
                    'player_1_id': player1_id,
                    'player_2_id': player2_id,
                    'team_side': team_side,
                    'impact_velocity': round(impact_velocity, 2),
                    'ellipse_overlap': round(ellipse_overlap, 2),
                    'deceleration_g': round(deceleration_g, 2),
                    'ai_flag': ai_flag,
                    'ref_whistle': ref_whistle,
                    'discrepancy_type': discrepancy_type
                })
        
        # Also process frame-by-frame collisions from vision results if available
        # This captures collisions that might not have kinematic impact data
        if frame_results:
            for frame_result in frame_results:
                frame_num = frame_result.get('frame_number', 0)
                collisions = frame_result.get('collisions', [])
                
                for collision in collisions:
                    track_id1 = collision.get('track_id1')
                    track_id2 = collision.get('track_id2')
                    
                    # Skip if we already processed this collision in collision_analyses
                    # Check by matching player IDs and approximate timestamp
                    already_processed = False
                    for event in contact_events:
                        if ((event.get('player_1_id') == track_id1 and event.get('player_2_id') == track_id2) or
                            (event.get('player_1_id') == track_id2 and event.get('player_2_id') == track_id1)):
                            # Check if timestamp is close (within 1 second)
                            try:
                                event_ts = event.get('timestamp', '')
                                event_m, event_s = map(int, event_ts.split(':'))
                                event_total_sec = event_m * 60 + event_s
                                if abs(event_total_sec - (minutes * 60 + seconds)) <= 1:
                                    already_processed = True
                                    break
                            except (ValueError, AttributeError):
                                pass
                    
                    if not already_processed and collision.get('collision', False):
                        timestamp_seconds = frame_num / self.fps if self.fps > 0 else 0
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        timestamp = f"{minutes:02d}:{seconds:02d}"
                        
                        # Get velocities
                        impact_velocity = 0.0
                        if track_id1:
                            p1_key = str(track_id1) if str(track_id1) in player_analyses else track_id1
                            if p1_key in player_analyses:
                                p1_data = player_analyses[p1_key]
                                impact_velocity = max(impact_velocity, p1_data.get('max_speed', 0))
                        if track_id2:
                            p2_key = str(track_id2) if str(track_id2) in player_analyses else track_id2
                            if p2_key in player_analyses:
                                p2_data = player_analyses[p2_key]
                                impact_velocity = max(impact_velocity, p2_data.get('max_speed', 0))
                        
                        ellipse_overlap = collision.get('iou', 0.0) * 100
                        
                        # Get deceleration G
                        deceleration_g = 0.0
                        if track_id1:
                            p1_key = str(track_id1) if str(track_id1) in player_analyses else track_id1
                            if p1_key in player_analyses:
                                deceleration_g = max(deceleration_g, player_analyses[p1_key].get('max_gforce', 0))
                        if track_id2:
                            p2_key = str(track_id2) if str(track_id2) in player_analyses else track_id2
                            if p2_key in player_analyses:
                                deceleration_g = max(deceleration_g, player_analyses[p2_key].get('max_gforce', 0))
                        
                        # AI flag
                        ai_flag = 1 if ellipse_overlap > 10.0 or deceleration_g > IMPACT_G_FORCE_THRESHOLD else 0
                        
                        # Team side
                        team_side = 'Home'
                        if team_side_mapping:
                            team_side = team_side_mapping.get(track_id1, 'Home')
                        
                        # Ref whistle
                        ref_whistle = 0
                        if referee_calls_df is not None and 'Call Timestamp (MM:SS)' in referee_calls_df.columns:
                            matching_calls = referee_calls_df[
                                referee_calls_df['Call Timestamp (MM:SS)'] == timestamp
                            ]
                            if len(matching_calls) == 0:
                                # Check within ±2 seconds
                                for _, call_row in referee_calls_df.iterrows():
                                    call_ts = call_row.get('Call Timestamp (MM:SS)', '')
                                    try:
                                        call_m, call_s = map(int, call_ts.split(':'))
                                        call_total_sec = call_m * 60 + call_s
                                        event_total_sec = minutes * 60 + seconds
                                        if abs(call_total_sec - event_total_sec) <= 2:
                                            ref_whistle = 1
                                            break
                                    except (ValueError, AttributeError):
                                        continue
                            else:
                                ref_whistle = 1
                        
                        contact_events.append({
                            'match_id': match_id,
                            'timestamp': timestamp,
                            'player_1_id': track_id1,
                            'player_2_id': track_id2,
                            'team_side': team_side,
                            'impact_velocity': round(impact_velocity, 2),
                            'ellipse_overlap': round(ellipse_overlap, 2),
                            'deceleration_g': round(deceleration_g, 2),
                            'ai_flag': ai_flag,
                            'ref_whistle': ref_whistle,
                            'discrepancy_type': ""
                        })
        
        # Create DataFrame with exactly 11 columns in specified order
        df = pd.DataFrame(contact_events)
        
        # Ensure all required columns exist
        required_columns = [
            'match_id', 'timestamp', 'player_1_id', 'player_2_id', 'team_side',
            'impact_velocity', 'ellipse_overlap', 'deceleration_g',
            'ai_flag', 'ref_whistle', 'discrepancy_type'
        ]
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                df[col] = "" if col == 'discrepancy_type' else 0
        
        # Reorder columns to match specification
        df = df[required_columns]
        
        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return df

