"""
Camera module for handling video streams from RTSP, USB, or test video files.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class CameraModule:
    """Handles camera/video input from various sources."""
    
    def __init__(self):
        """Initialize the camera module."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_test_mode = Config.TEST_OPTION
        self.source = Config.get_camera_source()
        self.fps = Config.CAMERA_FPS
        self.width = Config.CAMERA_WIDTH
        self.height = Config.CAMERA_HEIGHT
        self.frame_count = 0
        self.is_running = False
        
    def connect(self) -> bool:
        """
        Connect to the camera or video source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.is_test_mode:
                # Test mode: use video file
                if isinstance(self.source, str):
                    import os
                    if not os.path.exists(self.source):
                        logger.error(f"Test video file not found: {self.source}")
                        return False
                    self.cap = cv2.VideoCapture(self.source)
                    logger.info(f"Test mode: Loading video from {self.source}")
                else:
                    logger.error("Test mode enabled but TEST_VIDEO_PATH is not a valid string")
                    return False
            elif Config.CAMERA_TYPE == "rtsp":
                # RTSP stream
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                logger.info(f"Connecting to RTSP stream: {self.source}")
            elif Config.CAMERA_TYPE == "usb":
                # USB camera
                self.cap = cv2.VideoCapture(int(self.source))
                logger.info(f"Connecting to USB camera at index {self.source}")
            else:
                logger.error(f"Unknown camera type: {Config.CAMERA_TYPE}")
                return False
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera/video source")
                return False
            
            # Set camera properties if not in test mode
            if not self.is_test_mode:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set buffer size to minimize latency (drop old frames)
            # Buffer size of 1 means we always get the latest frame, dropping old ones
            # This is critical for reducing latency and preventing frame buildup
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE)
                actual_buffer = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
                logger.info(f"Camera buffer size set to {Config.CAMERA_BUFFER_SIZE} (actual: {actual_buffer})")
            except Exception as e:
                logger.warning(f"Could not set camera buffer size: {str(e)}")
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Store actual FPS for synchronization
            self.actual_fps = actual_fps if actual_fps > 0 else self.fps
            # If FPS is 0 or invalid, use configured FPS or default to 30
            if self.actual_fps <= 0:
                self.actual_fps = self.fps if self.fps > 0 else 30.0
            
            logger.info(f"Camera connected: {actual_width}x{actual_height} @ {self.actual_fps} FPS")
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to camera: {str(e)}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera/video source.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            # In test mode, loop the video if it reaches the end
            if self.is_test_mode and not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        
        return ret, frame
    
    def get_fps(self) -> float:
        """
        Get the actual FPS of the video source.
        
        Returns:
            float: FPS value
        """
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the frame size.
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def release(self):
        """Release the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Camera released")
    
    def is_connected(self) -> bool:
        """
        Check if camera is connected and running.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()

