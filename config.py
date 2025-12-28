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
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    
    # Test Mode Configuration
    TEST_OPTION: bool = os.getenv("TEST_OPTION", "false").lower() == "true"
    TEST_VIDEO_PATH: str = os.getenv("TEST_VIDEO_PATH", "test_videos/sample.mp4")
    
    # Detection Configuration
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    
    # Balloon Color Classification
    BALLOON_WHITELIST_COLORS: List[str] = [
        c.strip() for c in os.getenv("BALLOON_WHITELIST_COLORS", "white,red,green,blue").split(",")
    ]
    BALLOON_BLACKLIST_COLORS: List[str] = [
        c.strip() for c in os.getenv("BALLOON_BLACKLIST_COLORS", "black,orange,yellow").split(",")
    ]
    
    # Detection Classes
    DETECT_PERSON: bool = os.getenv("DETECT_PERSON", "true").lower() == "true"
    DETECT_DRONE: bool = os.getenv("DETECT_DRONE", "true").lower() == "true"
    DETECT_BALLOON: bool = os.getenv("DETECT_BALLOON", "true").lower() == "true"
    
    # UI Configuration
    UI_REFRESH_RATE: int = int(os.getenv("UI_REFRESH_RATE", "30"))
    UI_SHOW_FPS: bool = os.getenv("UI_SHOW_FPS", "true").lower() == "true"
    
    @classmethod
    def get_camera_source(cls) -> Optional[str]:
        """Get the camera source based on configuration."""
        if cls.TEST_OPTION:
            return cls.TEST_VIDEO_PATH
        elif cls.CAMERA_TYPE == "rtsp":
            return cls.CAMERA_RTSP_URL
        elif cls.CAMERA_TYPE == "usb":
            return cls.CAMERA_USB_INDEX
        return None

