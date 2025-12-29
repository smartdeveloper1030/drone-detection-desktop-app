"""
YOLOv8 detection module for detecting drones, balloons, and human shapes.
Includes color classification for balloons.
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import torch
from ultralytics import YOLO
from config import Config
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# PyTorch 2.6 compatibility: Patch torch.load to handle weights_only parameter
# This is needed because PyTorch 2.6 changed the default value from False to True
# and Ultralytics YOLO models require weights_only=False
try:
    # Try to add safe globals first (PyTorch 2.6+ preferred method)
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except (ImportError, AttributeError):
            pass
    
    # Patch torch.load as fallback/complementary solution
    original_load = torch.load
    def patched_load(*args, **kwargs):
        # Set weights_only=False if not explicitly provided (for PyTorch 2.6+)
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception as e:
    logger.warning(f"Could not patch torch.load for PyTorch 2.6 compatibility: {e}")


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
    track_id: Optional[int] = None  # Track ID for tracking
    velocity: Optional[Tuple[float, float]] = None  # Velocity (vx, vy) in pixels per second
    
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
    """Classifies balloon colors for whitelist/blacklist determination.
    Optimized with pixel sampling, ROI subsampling, and caching."""
    
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
    
    # Cache for recent color classifications (LRU)
    _color_cache = {}
    _cache_access_order = []
    _max_cache_size = Config.COLOR_CACHE_SIZE
    
    @classmethod
    def _get_cache_key(cls, bbox: Tuple[int, int, int, int]) -> str:
        """Generate cache key from bounding box (rounded to reduce cache misses)."""
        x1, y1, x2, y2 = bbox
        return f"{x1//10}_{y1//10}_{x2//10}_{y2//10}"
    
    @classmethod
    def _update_cache(cls, cache_key: str, color: Optional[str]):
        """Update LRU cache."""
        if cache_key in cls._color_cache:
            cls._cache_access_order.remove(cache_key)
        elif len(cls._color_cache) >= cls._max_cache_size:
            # Evict least recently used
            lru_key = cls._cache_access_order.pop(0)
            del cls._color_cache[lru_key]
        
        cls._color_cache[cache_key] = color
        cls._cache_access_order.append(cache_key)
    
    @classmethod
    def classify_color(cls, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Classify the color of a balloon in the bounding box.
        Optimized with pixel sampling, ROI subsampling, and caching.
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            str: Color name or None
        """
        # Check cache first
        cache_key = cls._get_cache_key(bbox)
        if cache_key in cls._color_cache:
            # Update access order for LRU
            if cache_key in cls._cache_access_order:
                cls._cache_access_order.remove(cache_key)
            cls._cache_access_order.append(cache_key)
            return cls._color_cache[cache_key]
        
        x1, y1, x2, y2 = bbox
        # Extract ROI with padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            cls._update_cache(cache_key, None)
            return None
        
        # Subsample large ROIs for performance
        max_sample_size = 100  # Maximum dimension for sampling
        if roi.shape[0] > max_sample_size or roi.shape[1] > max_sample_size:
            scale = max_sample_size / max(roi.shape[0], roi.shape[1])
            new_width = int(roi.shape[1] * scale)
            new_height = int(roi.shape[0] * scale)
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Use pixel sampling instead of full histogram (faster)
        # Sample pixels in a grid pattern
        sample_step = max(2, min(roi.shape[0] // 20, roi.shape[1] // 20))
        sampled_pixels = hsv[::sample_step, ::sample_step]
        
        # Check each color range using sampled pixels
        color_scores = {}
        total_pixels = sampled_pixels.shape[0] * sampled_pixels.shape[1]
        
        for color_name, ranges in cls.COLOR_RANGES.items():
            if len(ranges) == 2:
                # Single range
                lower, upper = np.array(ranges[0]), np.array(ranges[1])
                mask = np.all((sampled_pixels >= lower) & (sampled_pixels <= upper), axis=2)
            else:
                # Multiple ranges (e.g., red)
                lower1, upper1 = np.array(ranges[0]), np.array(ranges[1])
                lower2, upper2 = np.array(ranges[2]), np.array(ranges[3])
                mask1 = np.all((sampled_pixels >= lower1) & (sampled_pixels <= upper1), axis=2)
                mask2 = np.all((sampled_pixels >= lower2) & (sampled_pixels <= upper2), axis=2)
                mask = mask1 | mask2
            
            # Calculate percentage of pixels matching this color
            match_percentage = np.sum(mask) / total_pixels if total_pixels > 0 else 0
            color_scores[color_name] = match_percentage
        
        # Return color with highest match percentage
        result = None
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            if color_scores[best_color] > 0.1:  # At least 10% match
                result = best_color
        
        # Update cache
        cls._update_cache(cache_key, result)
        return result
    
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
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold dynamically.
        
        Args:
            threshold: New confidence threshold value (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold:.2f}")
        else:
            logger.warning(f"Invalid confidence threshold: {threshold}. Must be between 0.0 and 1.0")
        
    def load_model(self) -> bool:
        """
        Load the YOLOv8 model based on detect mode.
        - If detect mode is "balloon", uses models/best_balloon_nano.pt
        - If detect mode is "drone", uses models/best_drone_nano.pt
        - If detect mode is "person", uses models/yolov8n.pt (default COCO model for human detection)
        Model will be downloaded automatically if not found.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Determine model path based on detect mode
            detect_mode = Config.DETECT_MODE.lower()
            if detect_mode == "balloon":
                model_path = "models/best_balloon_nano.pt"
            elif detect_mode == "drone":
                model_path = "models/best_drone_nano.pt"
            elif detect_mode == "person":
                model_path = "models/yolov8n.pt"  # Default COCO model for human detection
            else:
                # Fallback to configured path if mode is not recognized
                model_path = Config.YOLO_MODEL_PATH
                logger.warning(f"Unknown detect mode: {detect_mode}, using configured model path: {model_path}")
            
            logger.info(f"Loading YOLO model from {model_path} (detect mode: {detect_mode})")
            
            # For custom models, ensure the file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                # Try fallback to configured path
                if model_path != Config.YOLO_MODEL_PATH and os.path.exists(Config.YOLO_MODEL_PATH):
                    logger.info(f"Using fallback model path: {Config.YOLO_MODEL_PATH}")
                    model_path = Config.YOLO_MODEL_PATH
                elif detect_mode == "person" and not os.path.exists(model_path):
                    # For person mode, try models/yolov8n.pt first, then download if needed
                    logger.info(f"Model not found at {model_path}, will download if needed")
                    # YOLO will automatically download if file doesn't exist
                elif not model_path.startswith('yolov8') and not model_path.startswith('models/yolov8'):
                    logger.error(f"Custom model file not found: {model_path}")
                    logger.info("Using default YOLOv8n model (will download if needed)")
                    model_path = 'models/yolov8n.pt'
            
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            logger.info("Trying to use default YOLOv8n model from models folder...")
            try:
                # Try models folder first, then root (will download if needed)
                try:
                    self.model = YOLO('models/yolov8n.pt')
                    logger.info("Default YOLOv8n model loaded successfully from models folder")
                except:
                    self.model = YOLO('yolov8n.pt')
                    logger.info("Default YOLOv8n model loaded successfully (will be downloaded if needed)")
                return True
            except Exception as e2:
                logger.error(f"Failed to load default model: {str(e2)}")
                return False
    
    def warmup(self, dummy_frame_size: Tuple[int, int] = None):
        """
        Warm up the model with a dummy inference to avoid slow first inference.
        
        Args:
            dummy_frame_size: Size of dummy frame for warmup (width, height). 
                              If None, uses YOLO_INPUT_SIZE from config.
        """
        if self.model is None:
            logger.warning("Cannot warmup: model not loaded")
            return
        
        try:
            # Use configured input size for warmup
            if dummy_frame_size is None:
                warmup_size = Config.YOLO_INPUT_SIZE
                dummy_frame_size = (warmup_size, warmup_size)
            
            logger.info(f"Warming up model with dummy inference (size: {dummy_frame_size[0]}x{dummy_frame_size[1]})...")
            dummy_frame = np.zeros((dummy_frame_size[1], dummy_frame_size[0], 3), dtype=np.uint8)
            _ = self.model(dummy_frame, imgsz=Config.YOLO_INPUT_SIZE, verbose=False)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
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
            # Run YOLO inference with optimized input size
            # Using imgsz parameter to reduce input resolution for faster inference
            results = self.model(
                frame,
                imgsz=Config.YOLO_INPUT_SIZE,  # Reduced input size (416 vs default 640) for better FPS
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
        Check if we should detect this class based on configuration and current mode.
        
        Args:
            class_name: Name of the class
            class_id: ID of the class
        
        Returns:
            bool: True if should detect, False otherwise
        """
        class_name_lower = class_name.lower()
        detect_mode = Config.DETECT_MODE.lower()
        
        # If mode is "person", only detect persons
        if detect_mode == "person":
            return 'person' in class_name_lower or class_id == 0
        
        # If mode is "drone", only detect drones
        if detect_mode == "drone":
            return 'drone' in class_name_lower
        
        # If mode is "balloon", only detect balloons
        if detect_mode == "balloon":
            return 'balloon' in class_name_lower
        
        # Fallback to configuration-based detection
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

