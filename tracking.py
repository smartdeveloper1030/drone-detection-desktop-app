"""
Object tracking and prediction module.
Tracks objects across frames and predicts future positions using Kalman Filter.
"""
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from detection import Detection
import logging

logger = logging.getLogger(__name__)


class KalmanFilter2D:
    """
    2D Kalman Filter for object tracking with position, velocity, and acceleration handling.
    State vector: [x, y, vx, vy]
    """
    
    def __init__(self, initial_x: float = 0.0, initial_y: float = 0.0):
        """
        Initialize Kalman Filter.
        
        Args:
            initial_x: Initial x position
            initial_y: Initial y position
        """
        # State vector: [x, y, vx, vy]
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float32)
        
        # State covariance matrix (uncertainty in state)
        self.P = np.eye(4, dtype=np.float32) * 1000.0  # Large initial uncertainty
        
        # Process noise covariance (Q) - accounts for acceleration/process uncertainty
        # Higher values = more uncertainty in motion model
        dt = 1.0  # Time step (will be updated dynamically)
        q = 0.1  # Process noise scale
        
        # Q matrix for constant velocity model with acceleration uncertainty
        # Acceleration affects velocity and position
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float32) * q
        
        # Measurement matrix (H) - we only observe position
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Measurement noise covariance (R) - uncertainty in detections
        self.R = np.eye(2, dtype=np.float32) * 10.0  # Detection noise
        
        # Identity matrix
        self.I = np.eye(4, dtype=np.float32)
    
    def predict(self, dt: float) -> Tuple[float, float]:
        """
        Predict next state.
        
        Args:
            dt: Time step in seconds
        
        Returns:
            Predicted (x, y) position
        """
        # State transition matrix (F) for constant velocity model
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Update process noise based on time step
        q = 0.1
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float32) * q
        
        # Predict state: x' = F * x
        self.state = F @ self.state
        
        # Predict covariance: P' = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Return predicted position
        return (float(self.state[0]), float(self.state[1]))
    
    def update(self, measurement_x: float, measurement_y: float):
        """
        Update filter with measurement.
        
        Args:
            measurement_x: Measured x position
            measurement_y: Measured y position
        """
        # Measurement vector
        z = np.array([measurement_x, measurement_y], dtype=np.float32)
        
        # Predicted measurement: z' = H * x
        z_pred = self.H @ self.state
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K * H) * P
        self.P = (self.I - K @ self.H) @ self.P
    
    def get_position(self) -> Tuple[float, float]:
        """Get current estimated position."""
        return (float(self.state[0]), float(self.state[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated velocity."""
        return (float(self.state[2]), float(self.state[3]))
    
    def predict_future(self, time_ahead_s: float) -> Tuple[float, float]:
        """
        Predict position at a future time.
        
        Args:
            time_ahead_s: Time ahead in seconds
        
        Returns:
            Predicted (x, y) position
        """
        # Use current state to predict future
        x, y, vx, vy = self.state
        
        # Predict with constant velocity model
        # x_future = x + vx * dt
        # y_future = y + vy * dt
        predicted_x = x + vx * time_ahead_s
        predicted_y = y + vy * time_ahead_s
        
        return (float(predicted_x), float(predicted_y))


@dataclass
class Track:
    """Track data for a single object with Kalman Filter."""
    track_id: int
    detection: Detection
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    age: int = 0  # Number of frames since last update
    hits: int = 1  # Number of successful matches
    kalman_filter: Optional[KalmanFilter2D] = field(default=None, init=False)
    last_timestamp: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize Kalman Filter after object creation."""
        if self.kalman_filter is None:
            self.kalman_filter = KalmanFilter2D(self.detection.x, self.detection.y)
            self.last_timestamp = time.time()
            # Initialize position history
            self.position_history.append((self.detection.x, self.detection.y, self.last_timestamp))
    
    def update(self, detection: Detection, timestamp: float):
        """Update track with new detection using Kalman Filter."""
        self.detection = detection
        self.position_history.append((detection.x, detection.y, timestamp))
        self.hits += 1
        self.age = 0
        
        # Keep only recent history (last 10 positions)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Calculate time step
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.0
        if dt <= 0:
            dt = 0.033  # Default to ~30 FPS if invalid
        
        # Predict with Kalman Filter
        if dt > 0:
            self.kalman_filter.predict(dt)
        
        # Update Kalman Filter with measurement
        self.kalman_filter.update(detection.x, detection.y)
        
        self.last_timestamp = timestamp
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Get velocity from Kalman Filter."""
        if self.kalman_filter:
            return self.kalman_filter.get_velocity()
        return (0.0, 0.0)
    
    def predict(self, time_ahead_ms: float) -> Optional[Tuple[float, float]]:
        """
        Predict future position using Kalman Filter.
        
        Args:
            time_ahead_ms: Time to predict ahead in milliseconds
        
        Returns:
            Predicted (x, y) position or None if prediction not possible
        """
        if self.kalman_filter is None:
            return None
        
        # Convert time to seconds
        time_ahead_s = time_ahead_ms / 1000.0
        
        # Use Kalman Filter to predict future position
        return self.kalman_filter.predict_future(time_ahead_s)
    
    def increment_age(self):
        """Increment age counter (called when track is not matched)."""
        self.age += 1


class Tracker:
    """Multi-object tracker with prediction capabilities."""
    
    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames a track can be unmatched before deletion
            min_hits: Minimum hits required to confirm a track
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.last_timestamp = time.time()
    
    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of current detections
            timestamp: Current timestamp (defaults to current time)
        
        Returns:
            List of active tracks
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.0
        self.last_timestamp = timestamp
        
        # Increment age for all tracks
        for track in self.tracks.values():
            track.increment_age()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_id, detection in matched_tracks.items():
            self.tracks[track_id].update(detection, timestamp)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            new_track = Track(track_id, detection)
            new_track.last_timestamp = timestamp
            # Position history is already initialized in __post_init__, just update timestamp
            if new_track.position_history:
                new_track.position_history[-1] = (detection.x, detection.y, timestamp)
            self.tracks[track_id] = new_track
        
        # Remove old tracks
        tracks_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return only confirmed tracks (with enough hits)
        confirmed_tracks = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
        
        return confirmed_tracks
    
    def _match_detections_to_tracks(self, detections: List[Detection]) -> Tuple[Dict[int, Detection], List[Detection]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            Tuple of (matched_tracks dict, unmatched_detections list)
        """
        if len(self.tracks) == 0:
            return {}, detections
        
        if len(detections) == 0:
            return {}, []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track in enumerate(self.tracks.values()):
            track_bbox = self._get_bbox(track.detection)
            for j, detection in enumerate(detections):
                det_bbox = self._get_bbox(detection)
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Simple greedy matching (can be improved with Hungarian algorithm)
        matched_tracks = {}
        matched_detection_indices = set()
        
        # Sort by IoU (highest first)
        matches = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy assignment
        for iou, track_idx, det_idx in matches:
            if track_idx not in matched_tracks and det_idx not in matched_detection_indices:
                track_id = track_ids[track_idx]
                matched_tracks[track_id] = detections[det_idx]
                matched_detection_indices.add(det_idx)
        
        # Find unmatched detections
        unmatched_detections = [
            detections[i] for i in range(len(detections))
            if i not in matched_detection_indices
        ]
        
        return matched_tracks, unmatched_detections
    
    def _get_bbox(self, detection: Detection) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates (x1, y1, x2, y2)."""
        x1 = detection.x - detection.width // 2
        y1 = detection.y - detection.height // 2
        x2 = detection.x + detection.width // 2
        y2 = detection.y + detection.height // 2
        return (x1, y1, x2, y2)
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_predictions(self, time_ahead_ms: float) -> List[Tuple[float, float]]:
        """
        Get predicted positions for all active tracks.
        
        Args:
            time_ahead_ms: Time to predict ahead in milliseconds
        
        Returns:
            List of predicted (x, y) positions
        """
        predictions = []
        for track in self.tracks.values():
            if track.hits >= self.min_hits:
                pred = track.predict(time_ahead_ms)
                if pred is not None:
                    predictions.append(pred)
        return predictions
    
    def get_primary_prediction(self, time_ahead_ms: float) -> Optional[Tuple[float, float]]:
        """
        Get primary predicted position (for blacklist/threat objects, or first track).
        
        Args:
            time_ahead_ms: Time to predict ahead in milliseconds
        
        Returns:
            Predicted (x, y) position or None
        """
        # Prioritize blacklist/threat tracks
        blacklist_tracks = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits and self._is_blacklist(track.detection)
        ]
        
        if blacklist_tracks:
            # Use the most recent/confident blacklist track
            track = max(blacklist_tracks, key=lambda t: (t.hits, t.detection.confidence))
            return track.predict(time_ahead_ms)
        
        # Otherwise, use first confirmed track
        confirmed_tracks = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
        
        if confirmed_tracks:
            # Use the most confident track
            track = max(confirmed_tracks, key=lambda t: t.detection.confidence)
            return track.predict(time_ahead_ms)
        
        return None
    
    def _is_blacklist(self, detection: Detection) -> bool:
        """Check if detection is blacklist (threat)."""
        from config import Config
        from detection import ColorClassifier
        
        # Drones are always blacklist
        if 'drone' in detection.class_name.lower():
            return True
        
        # Balloons: check color
        if 'balloon' in detection.class_name.lower() and detection.color_class:
            return ColorClassifier.is_blacklist(detection.color_class)
        
        return False
    
    def clear(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.next_id = 0

