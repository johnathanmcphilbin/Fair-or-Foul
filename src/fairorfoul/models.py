"""
ML Model Infrastructure for Fair-or-Foul
Handles model weights, inference pipeline, and model management
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle
from ultralytics import YOLO
import joblib


class ModelManager:
    """Manage ML models and weights"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        
    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save model weights and metadata
        
        Args:
            model: PyTorch model
            model_name: Name for the model
            metadata: Optional metadata dictionary
        """
        model_path = self.models_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        if metadata:
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_model(
        self,
        model_class: torch.nn.Module,
        model_name: str,
        device: str = "auto"
    ) -> torch.nn.Module:
        """
        Load model weights
        
        Args:
            model_class: Model class to instantiate
            model_name: Name of the model
            device: Device to load on
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.loaded_models[model_name] = model
        return model
    
    def save_yolo_model(self, model_path: str, model_name: str):
        """
        Save YOLO model
        
        Args:
            model_path: Path to YOLO model file
            model_name: Name for the model
        """
        target_path = self.models_dir / f"{model_name}.pt"
        import shutil
        shutil.copy(model_path, target_path)
    
    def load_yolo_model(self, model_name: str) -> YOLO:
        """
        Load YOLO model
        
        Args:
            model_name: Name of the model
            
        Returns:
            YOLO model instance
        """
        model_path = self.models_dir / f"{model_name}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model {model_name} not found at {model_path}")
        
        model = YOLO(str(model_path))
        self.loaded_models[model_name] = model
        return model
    
    def save_sklearn_model(self, model, model_name: str):
        """
        Save scikit-learn model
        
        Args:
            model: scikit-learn model
            model_name: Name for the model
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
    
    def load_sklearn_model(self, model_name: str):
        """
        Load scikit-learn model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        model = joblib.load(model_path)
        self.loaded_models[model_name] = model
        return model


class InferencePipeline:
    """Inference pipeline for model predictions"""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize inference pipeline
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        
    def predict_collision(
        self,
        frame: np.ndarray,
        bbox1: List[float],
        bbox2: List[float],
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Predict collision using ML model
        
        Args:
            frame: Input frame
            bbox1: First bounding box
            bbox2: Second bounding box
            model_name: Optional model name (uses default if None)
            
        Returns:
            Prediction dictionary
        """
        # Extract features from bounding boxes
        features = self._extract_collision_features(bbox1, bbox2)
        
        # If model available, use it; otherwise use rule-based
        if model_name and model_name in self.model_manager.loaded_models:
            model = self.model_manager.loaded_models[model_name]
            # Model inference would go here
            # For now, return rule-based prediction
            return self._rule_based_collision(bbox1, bbox2)
        else:
            return self._rule_based_collision(bbox1, bbox2)
    
    def predict_foul(
        self,
        kinematic_data: Dict,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Predict foul based on kinematic data
        
        Args:
            kinematic_data: Dictionary with kinematic features
            model_name: Optional model name
            
        Returns:
            Foul prediction dictionary
        """
        # Extract features
        features = self._extract_foul_features(kinematic_data)
        
        # Model inference would go here
        # For now, return rule-based prediction
        return self._rule_based_foul_prediction(kinematic_data)
    
    def _extract_collision_features(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> np.ndarray:
        """Extract features for collision prediction"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate features
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        return np.array([distance, area1, area2, iou])
    
    def _extract_foul_features(self, kinematic_data: Dict) -> np.ndarray:
        """Extract features for foul prediction"""
        features = []
        
        # G-force features
        if 'gforces' in kinematic_data and len(kinematic_data['gforces']) > 0:
            max_gforce = max([abs(gf['g_force']) for gf in kinematic_data['gforces']])
            avg_gforce = np.mean([abs(gf['g_force']) for gf in kinematic_data['gforces']])
            features.extend([max_gforce, avg_gforce])
        else:
            features.extend([0, 0])
        
        # Velocity features
        if 'velocities' in kinematic_data and len(kinematic_data['velocities']) > 0:
            max_velocity = max([v['speed'] for v in kinematic_data['velocities']])
            avg_velocity = np.mean([v['speed'] for v in kinematic_data['velocities']])
            features.extend([max_velocity, avg_velocity])
        else:
            features.extend([0, 0])
        
        # Whistle threshold
        if 'whistle_threshold' in kinematic_data:
            features.append(kinematic_data['whistle_threshold'].get('threshold', 0))
        else:
            features.append(0)
        
        return np.array(features)
    
    def _rule_based_collision(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> Dict:
        """Rule-based collision detection"""
        features = self._extract_collision_features(bbox1, bbox2)
        distance, iou = features[0], features[3]
        
        collision = iou > 0.1 or distance < 50  # Thresholds
        
        return {
            'collision': collision,
            'confidence': min(iou, 1.0) if collision else 0.0,
            'distance': distance,
            'iou': iou
        }
    
    def _rule_based_foul_prediction(self, kinematic_data: Dict) -> Dict:
        """Rule-based foul prediction"""
        whistle_result = kinematic_data.get('whistle_threshold', {})
        threshold = whistle_result.get('threshold', 0)
        
        foul = threshold > 0.5
        
        return {
            'foul': foul,
            'confidence': threshold,
            'threshold': threshold
        }


class ModelTrainer:
    """Train models for foul detection"""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize model trainer
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
    
    def train_collision_detector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "collision_detector"
    ):
        """
        Train collision detection model
        
        Args:
            X: Feature matrix
            y: Labels (0 = no collision, 1 = collision)
            model_name: Name for the model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save model
        self.model_manager.save_sklearn_model(model, model_name)
        
        return {
            'model_name': model_name,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
    
    def train_foul_predictor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "foul_predictor"
    ):
        """
        Train foul prediction model
        
        Args:
            X: Feature matrix
            y: Labels (0 = no foul, 1 = foul)
            model_name: Name for the model
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save model
        self.model_manager.save_sklearn_model(model, model_name)
        
        return {
            'model_name': model_name,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }

