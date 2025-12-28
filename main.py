"""
Main application entry point for Drone Detection System - Milestone 1.
"""
import sys
import logging
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import cv2
import numpy as np

from config import Config
from camera_module import CameraModule
from detection import DetectionModule
from ui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DroneDetectionApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = QApplication(sys.argv)
        self.main_window = MainWindow()
        self.camera = CameraModule()
        self.detector = DetectionModule()
        
        # Processing state
        self.is_running = False
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing Drone Detection System...")
        
        # Load YOLO model
        if not self.detector.load_model():
            logger.error("Failed to load YOLO model")
            return False
        
        # Connect camera
        if not self.camera.connect():
            logger.error("Failed to connect to camera")
            self.main_window.get_system_view().add_alert(
                "Failed to connect to camera/video source", "ERROR"
            )
            return False
        
        self.main_window.get_system_view().update_camera_status(True)
        self.main_window.get_system_view().add_alert(
            f"Camera connected - Test Mode: {Config.TEST_OPTION}", "INFO"
        )
        
        # Start frame processing
        frame_interval = int(1000 / Config.UI_REFRESH_RATE)  # Convert to milliseconds
        self.frame_timer.start(frame_interval)
        self.is_running = True
        
        logger.info("Initialization complete")
        return True
    
    def process_frame(self):
        """Process a single frame."""
        if not self.is_running:
            return
        
        try:
            # Read frame
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                self.main_window.get_system_view().update_camera_status(False)
                return
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            self.main_window.get_system_view().update_camera_status(False)
            return
        
        # Update camera status
        self.main_window.get_system_view().update_camera_status(True)
        
        try:
            # Run detection
            detections = self.detector.detect(frame)
            
            # Update detection status
            self.main_window.get_system_view().update_detection_status(len(detections) > 0)
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            detections = []
        
        # Check for blacklist detections
        blacklist_detections = self.detector.get_blacklist_detections(detections)
        if blacklist_detections:
            # Threat detected
            det = blacklist_detections[0]
            color_info = f" ({det.color_class})" if det.color_class else ""
            self.main_window.get_system_view().add_alert(
                f"THREAT DETECTED: {det.class_name}{color_info} at ({det.x}, {det.y})",
                "THREAT"
            )
        elif len(detections) > 0 and len(blacklist_detections) == 0:
            # Only whitelist detected
            det = detections[0]
            self.main_window.get_system_view().add_alert(
                f"Whitelist object detected: {det.class_name}",
                "INFO"
            )
        
        # For now, predicted point and servo crosshair are None
        # These will be implemented in Milestone 2 (Tracking & Prediction)
        predicted_point = None
        servo_crosshair = None
        
        # Update operator view
        self.main_window.get_operator_view().update_frame(
            frame,
            detections,
            predicted_point,
            servo_crosshair
        )
        
        # Update FPS
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            fps = self.fps_frame_count / elapsed
            self.main_window.statusBar().showMessage(f"Processing at {fps:.1f} FPS")
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Run the application."""
        if not self.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Show main window
        self.main_window.show()
        
        # Add startup alert
        self.main_window.get_system_view().add_alert(
            "System started successfully", "INFO"
        )
        
        # Run application
        return self.app.exec_()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")
        self.is_running = False
        self.frame_timer.stop()
        self.camera.release()
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    app = DroneDetectionApp()
    try:
        exit_code = app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_code = 0
    finally:
        app.cleanup()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

