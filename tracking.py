"""
Object tracking and prediction module.
Tracks objects across frames and predicts future positions.
"""
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from detection import Detection
import logging

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Track data for a single object."""
    track_id: int
    detection: Detection
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels per second
    age: int = 0  # Number of frames since last update
    hits: int = 1  # Number of successful matches
    
    def update(self, detection: Detection, timestamp: float):
        """Update track with new detection."""
        self.detection = detection
        self.position_history.append((detection.x, detection.y, timestamp))
        self.hits += 1
        self.age = 0
        
        # Keep only recent history (last 10 positions)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Calculate velocity from recent positions
        self._calculate_velocity()
    
    def _calculate_velocity(self):
        """Calculate velocity from position history."""
        if len(self.position_history) < 2:
            self.velocity = (0.0, 0.0)
            return
        
        # Use last 2 positions for velocity calculation (most recent movement)
        recent_positions = self.position_history[-2:]
        
        if len(recent_positions) < 2:
            self.velocity = (0.0, 0.0)
            return
        
        x1, y1, t1 = recent_positions[0]
        x2, y2, t2 = recent_positions[1]
        
        dt = t2 - t1
        if dt > 0:
            # Velocity in pixels per second
            self.velocity = ((x2 - x1) / dt, (y2 - y1) / dt)
        else:
            self.velocity = (0.0, 0.0)
    
    def predict(self, time_ahead_ms: float) -> Optional[Tuple[float, float]]:
        """
        Predict future position.
        
        Args:
            time_ahead_ms: Time to predict ahead in milliseconds
        
        Returns:
            Predicted (x, y) position or None if prediction not possible
        """
        if len(self.position_history) == 0:
            return None
        
        # Get current position
        current_x, current_y, _ = self.position_history[-1]
        
        # Convert time to seconds
        time_ahead_s = time_ahead_ms / 1000.0
        
        # Predict using velocity
        vx, vy = self.velocity
        
        predicted_x = current_x + vx * time_ahead_s
        predicted_y = current_y + vy * time_ahead_s
        
        return (predicted_x, predicted_y)
    
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
            self.tracks[track_id] = Track(track_id, detection)
            self.tracks[track_id].position_history.append((detection.x, detection.y, timestamp))
        
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

