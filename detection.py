"""
YOLOv8 detection module for detecting drones, balloons, and human shapes.
Includes color classification for balloons.
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from config import Config
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result data class."""
    x: int  # Center x coordinate
    y: int  # Center y coordinate
    width: int  # Bounding box width
    height: int  # Bounding box height
    confidence: float  # Detection confidence
    class_id: int  # Class ID
    class_name: str  # Class name
    color_class: Optional[str] = None  # Color classification for balloons
    distance: Optional[float] = None  # Distance (placeholder for future laser integration)
    
    def to_dict(self) -> Dict:
        """Convert detection to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'distance': self.distance,
            'color_class': self.color_class,
            'confidence': self.confidence,
            'class_name': self.class_name
        }


class ColorClassifier:
    """Classifies balloon colors for whitelist/blacklist determination."""
    
    # Color ranges in HSV
    COLOR_RANGES = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'red': ([0, 100, 100], [10, 255, 255], [170, 100, 100], [180, 255, 255]),  # Red wraps around
        'green': ([40, 50, 50], [80, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'black': ([0, 0, 0], [180, 255, 30]),
        'orange': ([10, 100, 100], [25, 255, 255]),
        'yellow': ([25, 100, 100], [35, 255, 255]),
    }
    
    @classmethod
    def classify_color(cls, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Classify the color of a balloon in the bounding box.
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            str: Color name or None
        """
        x1, y1, x2, y2 = bbox
        # Extract ROI with padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 50, 50], [0, 180, 0, 256, 0, 256])
        
        # Check each color range
        color_scores = {}
        for color_name, ranges in cls.COLOR_RANGES.items():
            if len(ranges) == 2:
                # Single range
                lower, upper = ranges
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            else:
                # Multiple ranges (e.g., red)
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            
            # Calculate percentage of pixels matching this color
            match_percentage = np.sum(mask > 0) / mask.size
            color_scores[color_name] = match_percentage
        
        # Return color with highest match percentage
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            if color_scores[best_color] > 0.1:  # At least 10% match
                return best_color
        
        return None
    
    @classmethod
    def is_whitelist(cls, color: Optional[str]) -> bool:
        """Check if color is in whitelist."""
        if color is None:
            return False
        return color.lower() in [c.lower() for c in Config.BALLOON_WHITELIST_COLORS]
    
    @classmethod
    def is_blacklist(cls, color: Optional[str]) -> bool:
        """Check if color is in blacklist."""
        if color is None:
            return False
        return color.lower() in [c.lower() for c in Config.BALLOON_BLACKLIST_COLORS]


class DetectionModule:
    """YOLOv8 detection module."""
    
    # COCO class names (YOLOv8 uses COCO dataset)
    COCO_CLASSES = {
        0: 'person',
        # Note: YOLOv8 doesn't have built-in drone/balloon classes
        # We'll need to use a custom model or map detections
    }
    
    def __init__(self):
        """Initialize the detection module."""
        self.model: Optional[YOLO] = None
        self.confidence_threshold = Config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.YOLO_IOU_THRESHOLD
        self.color_classifier = ColorClassifier()
        
    def load_model(self) -> bool:
        """
        Load the YOLOv8 model.
        Model will be downloaded automatically if not found.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            model_path = Config.YOLO_MODEL_PATH
            logger.info(f"Loading YOLO model from {model_path}")
            
            # YOLO will automatically download the model if it doesn't exist
            # For custom models, ensure the file exists
            if not model_path.startswith('yolov8') and not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Using default YOLOv8n model (will download if needed)")
                model_path = 'yolov8n.pt'
            
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            logger.info("Trying to use default YOLOv8n model...")
            try:
                self.model = YOLO('yolov8n.pt')
                logger.info("Default YOLOv8n model loaded successfully")
                return True
            except Exception as e2:
                logger.error(f"Failed to load default model: {str(e2)}")
                return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in the frame.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List[Detection]: List of detections
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate center and dimensions
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # Check if we should detect this class
                    if not self._should_detect(class_name, class_id):
                        continue
                    
                    # Classify color for balloons
                    color_class = None
                    # Try to classify color for any detected object (useful for balloons)
                    # In production, you'd filter by class_id for balloon-specific classes
                    if 'balloon' in class_name.lower():
                        color_class = self.color_classifier.classify_color(frame, (x1, y1, x2, y2))
                    
                    # Create detection object
                    detection = Detection(
                        x=center_x,
                        y=center_y,
                        width=width,
                        height=height,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        color_class=color_class,
                        distance=None  # Placeholder for future laser integration
                    )
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def _should_detect(self, class_name: str, class_id: int) -> bool:
        """
        Check if we should detect this class based on configuration.
        
        Args:
            class_name: Name of the class
            class_id: ID of the class
        
        Returns:
            bool: True if should detect, False otherwise
        """
        class_name_lower = class_name.lower()
        
        # Check for person
        if 'person' in class_name_lower or class_id == 0:
            return Config.DETECT_PERSON
        
        # Check for drone
        if 'drone' in class_name_lower:
            return Config.DETECT_DRONE
        
        # Check for balloon
        if 'balloon' in class_name_lower:
            return Config.DETECT_BALLOON
        
        # Default: detect if it's a person (COCO class 0)
        return class_id == 0 and Config.DETECT_PERSON
    
    def get_blacklist_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections to get only blacklist items.
        
        Args:
            detections: List of all detections
        
        Returns:
            List[Detection]: Blacklist detections
        """
        blacklist = []
        for det in detections:
            # Drones are always blacklist
            if 'drone' in det.class_name.lower():
                blacklist.append(det)
            # Balloons: check color
            elif 'balloon' in det.class_name.lower() and det.color_class:
                if self.color_classifier.is_blacklist(det.color_class):
                    blacklist.append(det)
        
        return blacklist

