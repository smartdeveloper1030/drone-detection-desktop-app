"""
Configuration management for the drone detection system.
Loads settings from .env file and provides configuration access.
"""
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the application."""
    
    # Camera Configuration
    CAMERA_TYPE: str = os.getenv("CAMERA_TYPE", "rtsp")
    CAMERA_RTSP_URL: str = os.getenv("CAMERA_RTSP_URL", "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101")
    CAMERA_USB_INDEX: int = int(os.getenv("CAMERA_USB_INDEX", "0"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))  # Default to 720p for lower latency
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))  # Default to 720p for lower latency
    CAMERA_BUFFER_SIZE: int = int(os.getenv("CAMERA_BUFFER_SIZE", "1"))  # Buffer size (1 = drop old frames, minimal latency)
    
    # Detection Configuration
    DETECT_MODE: str = os.getenv("DETECT_MODE", "drone").lower()  # "balloon", "drone", or "person"
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "auto").lower()  # "auto", "cpu", "cuda", "cuda:0", etc.
    YOLO_INPUT_SIZE: int = int(os.getenv("YOLO_INPUT_SIZE", "320"))  # Input resolution (320 for lowest latency)
    YOLO_HALF_PRECISION: bool = os.getenv("YOLO_HALF_PRECISION", "true").lower() == "true"  # Use FP16 for GPU inference
    YOLO_USE_TENSORRT: bool = os.getenv("YOLO_USE_TENSORRT", "true").lower() == "true"  # Use TensorRT if available
    
    # Balloon Color Classification
    BALLOON_WHITELIST_COLORS: List[str] = [
        c.strip() for c in os.getenv("BALLOON_WHITELIST_COLORS", "white,red,green,blue").split(",")
    ]
    BALLOON_BLACKLIST_COLORS: List[str] = [
        c.strip() for c in os.getenv("BALLOON_BLACKLIST_COLORS", "black,orange,yellow").split(",")
    ]
    SELECTED_BALLOON_COLOR: str = os.getenv("SELECTED_BALLOON_COLOR", "red").lower()  # Selected color for balloon detection
    
    # Detection Classes
    DETECT_PERSON: bool = os.getenv("DETECT_PERSON", "true").lower() == "true"
    DETECT_DRONE: bool = os.getenv("DETECT_DRONE", "true").lower() == "true"
    DETECT_BALLOON: bool = os.getenv("DETECT_BALLOON", "true").lower() == "true"
    
    # UI Configuration
    UI_REFRESH_RATE: int = int(os.getenv("UI_REFRESH_RATE", "60"))  # Increased to 60 FPS for smoother display
    UI_SHOW_FPS: bool = os.getenv("UI_SHOW_FPS", "true").lower() == "true"
    UI_DISABLE_ANIMATIONS: bool = os.getenv("UI_DISABLE_ANIMATIONS", "true").lower() == "true"  # Disable animations for lower latency
    
    # Performance Optimization
    DETECTION_FRAME_SKIP: int = int(os.getenv("DETECTION_FRAME_SKIP", "3"))  # Run detection every N frames (reduced for lower latency)
    DETECTION_QUEUE_SIZE: int = int(os.getenv("DETECTION_QUEUE_SIZE", "1"))  # Max frames in detection queue (1 = always use latest)
    COLOR_CACHE_SIZE: int = int(os.getenv("COLOR_CACHE_SIZE", "50"))  # Color classification cache size
    DETECTION_GPU_ID: int = int(os.getenv("DETECTION_GPU_ID", "0"))  # GPU ID for detection (0 = primary GPU, 1 = secondary GPU if available)
    
    # Tracking & Prediction Configuration
    PREDICTION_HORIZON_MS: int = int(os.getenv("PREDICTION_HORIZON_MS", "500"))  # Prediction horizon in milliseconds
    
    @classmethod
    def get_camera_source(cls) -> Optional[str]:
        """Get the camera source based on configuration."""
        if cls.CAMERA_TYPE == "rtsp":
            return cls.CAMERA_RTSP_URL
        elif cls.CAMERA_TYPE == "usb":
            return cls.CAMERA_USB_INDEX
        return None

