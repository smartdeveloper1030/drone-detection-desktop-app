"""
Camera module FIXED for low latency.
"""
import cv2
import numpy as np
import os
import time  # Add this
from typing import Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class CameraModule:
    """Handles camera/video input with minimal latency."""
    
    def __init__(self):
        """Initialize the camera module."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = Config.get_camera_source()
        self.fps = Config.CAMERA_FPS
        self.width = Config.CAMERA_WIDTH
        self.height = Config.CAMERA_HEIGHT
        self.frame_count = 0
        self.is_running = False
        self.last_read_time = 0
        self.read_interval = 1.0 / 30.0  # Target 30 FPS
    
    def connect(self) -> bool:
        """
        Connect to the camera source with minimal latency settings.
        """
        self.source = Config.get_camera_source()
        
        try:
            if Config.CAMERA_TYPE == "rtsp":
                # RTSP with ultra-low latency options
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'
                    'buffer_size;102400|'  # 100KB buffer (not frames!)
                    'max_delay;100000|'    # 100ms max delay
                    'stimeout;1000000'     # 1s timeout
                )
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            elif Config.CAMERA_TYPE == "usb":
                # USB camera with DirectShow on Windows for better control
                if os.name == 'nt':  # Windows
                    self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(int(self.source))
            else:
                return False
            
            if not self.cap.isOpened():
                return False
            
            # CRITICAL: Set properties in right order
            # 1. Buffer size FIRST (some cameras need this first)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass  # Some cameras ignore this
            
            # 2. Then resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Don't set FPS for RTSP (let it use stream FPS)
            if Config.CAMERA_TYPE == "usb":
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 3. Try to disable auto settings (reduces latency)
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            except:
                pass
            
            # Flush buffer once at startup to clear any stale frames
            # This prevents decoding old frames during normal operation
            flush_count = 0
            max_flush = 100  # Enough for several seconds of backlog
            while flush_count < max_flush:
                if not self.cap.grab():
                    break
                flush_count += 1
            
            if flush_count > 0:
                logger.info(f"Flushed {flush_count} stale frames from camera buffer at startup")
            
            self.is_running = True
            logger.info(f"Camera connected: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            return False
    
    def read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame with minimal latency.
        Uses grab() to clear buffer (fast, no decode) and retrieve() to get only the latest frame.
        Buffer is flushed once at startup, so we only need to grab until we get the latest.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            # Step 1: Grab frames in buffer without decoding (FAST)
            # Since buffer was flushed at startup, we only need to grab until we get the latest
            # Typically this means grabbing 0-2 frames (if any accumulated since last read)
            grab_count = 0
            max_grabs = 10  # Safety limit (should rarely need more than 1-2)
            
            while grab_count < max_grabs:
                grabbed = self.cap.grab()  # Fast: no decode, just advances buffer
                if not grabbed:
                    # No more frames in buffer, break
                    break
                grab_count += 1
            
            # Step 2: Retrieve and decode ONLY the latest frame
            ret, frame = self.cap.retrieve()
            
            if ret:
                self.frame_count += 1
                return True, frame
            else:
                # Fallback: try normal read if retrieve failed
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                return ret, frame
                
        except Exception as e:
            logger.warning(f"Error reading latest frame (camera may have been released): {str(e)}")
            return False, None
    
    # Keep other methods the same...
    def get_fps(self) -> float:
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def get_frame_size(self) -> Tuple[int, int]:
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_running = False
    
    def is_connected(self) -> bool:
        return self.is_running and self.cap is not None and self.cap.isOpened()