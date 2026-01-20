"""
Camera module for handling video streams from RTSP or USB cameras.
"""
import cv2
import numpy as np
import os
from typing import Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class CameraModule:
    """Handles camera/video input from various sources."""
    
    def __init__(self):
        """Initialize the camera module."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = Config.get_camera_source()
        self.fps = Config.CAMERA_FPS
        self.width = Config.CAMERA_WIDTH
        self.height = Config.CAMERA_HEIGHT
        self.frame_count = 0
        self.is_running = False
    
    def connect(self) -> bool:
        """
        Connect to the camera source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        # Update source before connecting
        self.source = Config.get_camera_source()
        
        try:
            if Config.CAMERA_TYPE == "rtsp":
                # RTSP stream with low-latency options
                # Set environment variables for FFmpeg low-latency options
                # These reduce RTSP stream buffering significantly
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'  # Use TCP instead of UDP for reliability
                    'buffer_size;1|'  # Minimal buffer (1 frame)
                    'max_delay;500000|'  # Max delay 500ms in microseconds
                    'stimeout;2000000'  # Socket timeout 2s
                )
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                logger.info(f"Connecting to RTSP stream with low-latency options: {self.source}")
            elif Config.CAMERA_TYPE == "usb":
                # USB camera
                self.cap = cv2.VideoCapture(int(self.source))
                logger.info(f"Connecting to USB camera at index {self.source}")
            else:
                logger.error(f"Unknown camera type: {Config.CAMERA_TYPE}")
                return False
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera source")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set buffer size to 1 for minimal latency (drop old frames)
            # Buffer size of 1 means we always get the latest frame, dropping old ones
            # This is critical for reducing latency and preventing frame buildup
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Force buffer size to 1 for lowest latency
                actual_buffer = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
                logger.info(f"Camera buffer size set to {Config.CAMERA_BUFFER_SIZE} (actual: {actual_buffer})")
            except Exception as e:
                logger.warning(f"Could not set camera buffer size: {str(e)}")
            
            # For RTSP: Set additional low-latency properties
            if Config.CAMERA_TYPE == "rtsp":
                try:
                    # Try to set OpenCV's FFMPEG options for low latency
                    # These may not work on all systems, but worth trying
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                except Exception as e:
                    logger.debug(f"Could not set RTSP codec options: {str(e)}")
            
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
        Read a frame from the camera source.
        For low latency: skip old buffered frames to get the latest one.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            # For low latency: skip old frames in buffer to get the latest frame
            # This reduces latency by discarding stale frames
            if Config.CAMERA_TYPE == "rtsp" and Config.CAMERA_BUFFER_SIZE > 1:
                # Read and discard old frames, keeping only the latest
                for _ in range(Config.CAMERA_BUFFER_SIZE - 1):
                    ret_temp, _ = self.cap.read()
                    if not ret_temp:
                        break
            
            # Read the latest frame
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_count += 1
            else:
                logger.warning("Failed to read frame from camera source")
            
            return ret, frame
        except Exception as e:
            # Camera might have been released during read
            logger.warning(f"Error reading frame (camera may have been released): {str(e)}")
            return False, None
    
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

