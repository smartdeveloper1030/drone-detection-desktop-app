"""
Ultra-low latency camera module.
"""
import cv2
import numpy as np
import os
import time
from typing import Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class CameraModule:
    """Handles camera/video input with ultra-low latency."""
    
    def __init__(self):
        """Initialize the camera module."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = Config.get_camera_source()
        
        # CAPTURE at high resolution (for detection)
        self.capture_width = Config.CAMERA_WIDTH
        self.capture_height = Config.CAMERA_HEIGHT
        
        # DISPLAY at low resolution (for UI)
        self.display_width = 640  # Low resolution for fast UI
        self.display_height = 480
        
        self.frame_count = 0
        self.is_running = False
        
    def connect(self) -> bool:
        """
        Connect to camera with ultra-low latency settings.
        """
        self.source = Config.get_camera_source()
        
        try:
            if Config.CAMERA_TYPE == "rtsp":
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;1'
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            elif Config.CAMERA_TYPE == "usb":
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(int(self.source))
            
            if not self.cap.isOpened():
                return False
            
            # CRITICAL: Set ABSOLUTE MINIMAL settings
            # 1. Buffer size = 1 (most important!)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
            
            # 2. Set resolution to capture resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            
            # 3. Try MJPG compression (faster than YUYV)
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass
            
            # 4. Set FPS to match your processing rate
            target_fps = 30  # Match your UI refresh
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # 5. Disable auto settings
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Manual exposure
            except:
                pass
            
            # 6. Flush buffer once at startup
            for _ in range(5):
                self.cap.grab()
            
            # Get actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            return False
    
    def read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Ultra-fast frame reading.
        Returns frame at camera's native resolution.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            # FASTEST METHOD: Simple read with buffer=1
            # With buffer=1, read() gives latest frame
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_count += 1
                return True, frame
            
            return False, None
            
        except Exception as e:
            # Camera might have been released
            if "release" in str(e).lower():
                self.is_running = False
            return False, None
    
    def read_frame_for_detection(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame for detection (same as read_latest_frame).
        """
        return self.read_latest_frame()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the current frame size from the camera.
        Returns the actual capture resolution.
        """
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def get_fps(self) -> float:
        """Get the current FPS from the camera."""
        if self.cap and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def release(self):
        """Release the camera resource."""
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        self.is_running = False
    
    def is_connected(self) -> bool:
        """Check if camera is connected and running."""
        return self.is_running and self.cap is not None and self.cap.isOpened()