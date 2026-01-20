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
            
            # Flush initial buffer
            for _ in range(5):
                self.cap.grab()
            
            self.is_running = True
            logger.info(f"Camera connected: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            return False
    
    def read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the ABSOLUTE latest frame with minimal latency.
        This is the key fix for 3-second delay.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        start_time = time.perf_counter()
        
        try:
            # METHOD A: For 3-second delay, we need AGGRESSIVE buffer clearing
            # Estimate: 3 seconds × 30 FPS = 90 frames backlogged
            
            # Step 1: Grab ALL buffered frames without decoding (FAST)
            grab_count = 0
            max_grabs = 100  # Enough for 3+ seconds at 30 FPS
            
            while grab_count < max_grabs:
                grabbed = self.cap.grab()  # Returns True if frame grabbed
                if not grabbed:
                    break
                grab_count += 1
            
            if grab_count > 10:
                logger.warning(f"Cleared {grab_count} buffered frames! That was your 3-second delay.")
            
            # Step 2: Retrieve ONLY the latest frame
            ret, frame = self.cap.retrieve()
            
            read_time = (time.perf_counter() - start_time) * 1000
            
            if ret:
                # Log if we're still slow
                if read_time > 50:  # >50ms is too slow
                    logger.warning(f"Frame read took {read_time:.0f}ms")
                
                self.frame_count += 1
                return True, frame
            else:
                # Fallback: normal read
                ret, frame = self.cap.read()
                return ret, frame
                
        except Exception as e:
            logger.error(f"Error in read_latest_frame: {e}")
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