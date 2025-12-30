"""
Main application entry point for Drone Detection System - Milestone 1.
"""
import sys
import logging
import time
import threading
from queue import Queue, Empty

# IMPORTANT: Import torch BEFORE PyQt5 to avoid DLL conflicts on Windows
# This must be done before any PyQt5 imports
import torch

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import cv2
import numpy as np

from config import Config
from camera_module import CameraModule
from detection import DetectionModule, Detection
from tracking import Tracker
from ui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionThread(threading.Thread):
    """
    Separate thread for running detection to prevent UI blocking.
    Uses non-blocking queue to drop frames if detection is too slow.
    """
    
    def __init__(self, detector, max_queue_size=2):
        """
        Initialize the detection thread.
        
        Args:
            detector: DetectionModule instance
            max_queue_size: Maximum frames in queue (drops frames if exceeded)
        """
        super().__init__(daemon=True)
        self.detector = detector
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=1)  # Only keep latest result
        self.is_running = False
        self.current_frame_id = 0
        self.processed_count = 0
        self.dropped_count = 0
        
    def run(self):
        """Main thread loop - continuously process frames."""
        self.is_running = True
        logger.info("Detection thread started")
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame, frame_id = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Put result in result queue (replace old result if queue is full)
                try:
                    self.result_queue.put_nowait((frame_id, detections))
                except:
                    # Queue full, replace with new result
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((frame_id, detections))
                    except:
                        pass
                
                self.processed_count += 1
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in detection thread: {str(e)}")
                if not self.is_running:
                    break
        
        logger.info(f"Detection thread stopped. Processed {self.processed_count} frames, dropped {self.dropped_count}")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add frame to detection queue (non-blocking).
        
        Args:
            frame: Frame to process
            
        Returns:
            bool: True if frame was added, False if queue was full (frame dropped)
        """
        self.current_frame_id += 1
        
        # If queue is full, remove oldest frame
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.dropped_count += 1
            except Empty:
                pass
        
        try:
            self.frame_queue.put_nowait((frame.copy(), self.current_frame_id))
            return True
        except:
            self.dropped_count += 1
            return False
    
    def get_latest_result(self):
        """
        Get the latest detection result (non-blocking).
        
        Returns:
            Tuple[frame_id, detections] or (None, []) if no result available
        """
        try:
            # Get the latest result (skip older ones)
            latest_result = None
            latest_id = None
            
            while True:
                try:
                    frame_id, detections = self.result_queue.get_nowait()
                    latest_result = detections
                    latest_id = frame_id
                except Empty:
                    break
            
            return latest_id, latest_result if latest_result is not None else []
        except Exception as e:
            logger.error(f"Error getting detection result: {str(e)}")
            return None, []
    
    def stop(self):
        """Stop the detection thread."""
        self.is_running = False
        logger.info("Stopping detection thread...")


class DroneDetectionApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = QApplication(sys.argv)
        self.main_window = MainWindow()
        self.camera = CameraModule()
        self.detector = DetectionModule()
        self.tracker = Tracker(max_age=5, min_hits=1, iou_threshold=0.3)
        
        # Connect mode change signal
        self.main_window.mode_changed.connect(self._on_mode_changed)
        
        # Connect color change signal
        self.main_window.color_changed.connect(self._on_color_changed)
        
        # Connect confidence threshold change signal
        self.main_window.get_system_view().confidence_threshold_changed.connect(self._on_confidence_threshold_changed)
        
        # Connect prediction horizon change signal
        self.main_window.get_system_view().prediction_horizon_changed.connect(self._on_prediction_horizon_changed)
        
        # Processing state
        self.is_running = False
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        
        # Detection threading
        self.detection_thread = None
        
        # Frame skipping
        self.frame_counter = 0
        
        # FPS tracking (separate for display and detection)
        self.display_fps_start = time.time()
        self.display_fps_count = 0
        self.detection_fps_start = time.time()
        self.detection_fps_count = 0
        
        # Latest detections (from detection thread)
        self.latest_detections = []
        self.last_detection_frame_id = -1
        
        # Prediction horizon (in milliseconds)
        self.prediction_horizon_ms = Config.PREDICTION_HORIZON_MS
        
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
        
        # Warmup model (first inference is slow)
        # Uses configured YOLO_INPUT_SIZE for warmup (optimized for performance)
        logger.info("Warming up model...")
        self.detector.warmup()  # Uses YOLO_INPUT_SIZE from config (default 416)
        
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
        
        # Start detection thread
        self.detection_thread = DetectionThread(
            self.detector,
            max_queue_size=Config.DETECTION_QUEUE_SIZE
        )
        self.detection_thread.start()
        logger.info(f"Detection thread started (frame skip: {Config.DETECTION_FRAME_SKIP}, queue size: {Config.DETECTION_QUEUE_SIZE})")
        
        # Start frame processing
        frame_interval = int(1000 / Config.UI_REFRESH_RATE)  # Convert to milliseconds
        self.frame_timer.start(frame_interval)
        self.is_running = True
        
        # Set initial mode in UI
        self.main_window.set_mode(Config.DETECT_MODE)
        
        # Set initial color if in balloon mode
        if Config.DETECT_MODE.lower() == "balloon":
            selected_color = getattr(Config, 'SELECTED_BALLOON_COLOR', 'red')
            # Update blacklist to only include selected color
            Config.BALLOON_BLACKLIST_COLORS = [selected_color]
            self.main_window.set_color(selected_color)
        
        # Initialize prediction horizon display
        self.main_window.get_system_view().update_prediction_horizon(self.prediction_horizon_ms)
        
        logger.info("Initialization complete")
        return True
    
    def _on_mode_changed(self, mode: str):
        """
        Handle detection mode change from UI.
        
        Args:
            mode: "balloon", "drone", or "person"
        """
        logger.info(f"Detection mode changed to: {mode}")
        
        # Update config
        Config.DETECT_MODE = mode
        
        # If switching to balloon mode, initialize color selection
        if mode.lower() == "balloon":
            selected_color = getattr(Config, 'SELECTED_BALLOON_COLOR', 'red')
            # Update blacklist to only include selected color
            Config.BALLOON_BLACKLIST_COLORS = [selected_color]
            # Ensure color selector is set correctly
            self.main_window.set_color(selected_color)
        
        # Stop current detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            logger.info("Stopping current detection thread...")
            self.detection_thread.stop()
            self.detection_thread.join(timeout=2.0)
            self.detection_thread = None
        
        # Clear current detections
        self.latest_detections = []
        self.last_detection_frame_id = -1
        
        # Reload model with new mode
        logger.info(f"Reloading model for {mode} mode...")
        if not self.detector.load_model():
            logger.error(f"Failed to load model for {mode} mode")
            self.main_window.get_system_view().add_alert(
                f"Failed to load model for {mode} mode", "ERROR"
            )
            return
        
        # Warmup new model
        logger.info("Warming up new model...")
        self.detector.warmup()
        
        # Clear tracker when mode changes (objects may be different)
        self.tracker.clear()
        
        # Restart detection thread
        self.detection_thread = DetectionThread(
            self.detector,
            max_queue_size=Config.DETECTION_QUEUE_SIZE
        )
        self.detection_thread.start()
        
        # Update system view
        self.main_window.get_system_view().add_alert(
            f"Detection mode changed to: {mode.capitalize()}", "INFO"
        )
        
        logger.info(f"Detection mode changed to {mode} successfully")
    
    def _on_confidence_threshold_changed(self, threshold: float):
        """
        Handle confidence threshold change from UI.
        
        Args:
            threshold: New confidence threshold value (0.0 to 1.0)
        """
        logger.info(f"Confidence threshold changed to: {threshold:.2f}")
        self.detector.set_confidence_threshold(threshold)
        # Update config for persistence
        Config.YOLO_CONFIDENCE_THRESHOLD = threshold
        self.main_window.get_system_view().add_alert(
            f"Confidence threshold updated to {threshold:.2f}", "INFO"
        )
    
    def _on_prediction_horizon_changed(self, horizon_ms: int):
        """
        Handle prediction horizon change from UI.
        
        Args:
            horizon_ms: New prediction horizon in milliseconds
        """
        logger.info(f"Prediction horizon changed to: {horizon_ms} ms")
        self.prediction_horizon_ms = horizon_ms
        # Update config for persistence
        Config.PREDICTION_HORIZON_MS = horizon_ms
        self.main_window.get_system_view().add_alert(
            f"Prediction horizon updated to {horizon_ms} ms", "INFO"
        )
    
    def _on_color_changed(self, color: str):
        """
        Handle color selection change from UI.
        
        Args:
            color: Selected color name
        """
        logger.info(f"Balloon color selection changed to: {color}")
        # Update config
        Config.SELECTED_BALLOON_COLOR = color
        # Update blacklist colors to only include selected color
        Config.BALLOON_BLACKLIST_COLORS = [color]
        self.main_window.get_system_view().add_alert(
            f"Balloon detection color set to: {color.capitalize()}", "INFO"
        )
    
    def process_frame(self):
        """Process a single frame - display immediately, detection runs asynchronously."""
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
        
        # CRITICAL: Display frame immediately with latest available detections (don't wait for detection)
        # Get latest detection results (non-blocking, may be from older frame)
        detections = self.latest_detections  # Use cached detections immediately
        
        # Update tracker with current detections and get predictions
        current_time = time.time()
        tracks = self.tracker.update(detections, current_time)
        predicted_point = self.tracker.get_primary_prediction(self.prediction_horizon_ms)
        
        # Enrich detections with track information
        # Create a mapping from detection to track by comparing detection objects
        enriched_detections = []
        for det in detections:
            # Find matching track for this detection
            matching_track = None
            for track in tracks:
                # Match by position and class (within small tolerance for position)
                if (abs(track.detection.x - det.x) < 5 and 
                    abs(track.detection.y - det.y) < 5 and
                    track.detection.class_name == det.class_name):
                    matching_track = track
                    break
            
            if matching_track:
                # Create enriched detection with track info
                enriched_det = Detection(
                    x=det.x,
                    y=det.y,
                    width=det.width,
                    height=det.height,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    color_class=det.color_class,
                    distance=det.distance,
                    track_id=matching_track.track_id,
                    velocity=matching_track.velocity
                )
                enriched_detections.append(enriched_det)
            else:
                enriched_detections.append(det)
        
        # Update prediction horizon display
        self.main_window.get_system_view().update_prediction_horizon(self.prediction_horizon_ms)
        
        # Servo crosshair (placeholder for future implementation)
        servo_crosshair = None
        
        # Update operator view IMMEDIATELY with current frame and latest detections
        # This ensures smooth display regardless of detection speed
        self.main_window.get_operator_view().update_frame(
            frame,
            enriched_detections,
            predicted_point,
            servo_crosshair,
            self.prediction_horizon_ms
        )
        
        # Now handle detection asynchronously (non-blocking)
        # Frame skipping: only run detection every N frames
        self.frame_counter += 1
        should_detect = (self.frame_counter % Config.DETECTION_FRAME_SKIP == 0)
        
        if should_detect and self.detection_thread:
            # Add frame to detection queue (non-blocking)
            self.detection_thread.add_frame(frame)
            self.detection_fps_count += 1
        
        # Get latest detection results (non-blocking) and update cache
        if self.detection_thread:
            frame_id, new_detections = self.detection_thread.get_latest_result()
            if frame_id is not None and frame_id != self.last_detection_frame_id:
                # Update cached detections for next frame display
                self.latest_detections = new_detections
                self.last_detection_frame_id = frame_id
                
                # Update detection status (non-blocking UI update)
                self.main_window.get_system_view().update_detection_status(len(new_detections) > 0)
                
                # Check for blacklist detections and add alerts (non-blocking)
                blacklist_detections = self.detector.get_blacklist_detections(new_detections)
                if blacklist_detections:
                    # Threat detected
                    det = blacklist_detections[0]
                    color_info = f" ({det.color_class})" if det.color_class else ""
                    self.main_window.get_system_view().add_alert(
                        f"THREAT DETECTED: {det.class_name}{color_info} at ({det.x}, {det.y})",
                        "THREAT"
                    )
                elif len(new_detections) > 0 and len(blacklist_detections) == 0:
                    # Only whitelist detected
                    det = new_detections[0]
                    self.main_window.get_system_view().add_alert(
                        f"Whitelist object detected: {det.class_name}",
                        "INFO"
                    )
        
        # Update FPS (separate tracking for display and detection)
        self.display_fps_count += 1
        current_time = time.time()
        
        # Display FPS (UI refresh rate)
        display_elapsed = current_time - self.display_fps_start
        if display_elapsed >= 1.0:
            display_fps = self.display_fps_count / display_elapsed
            self.display_fps_count = 0
            self.display_fps_start = current_time
            
            # Detection FPS
            detection_elapsed = current_time - self.detection_fps_start
            if detection_elapsed >= 1.0:
                detection_fps = self.detection_fps_count / detection_elapsed
                self.detection_fps_count = 0
                self.detection_fps_start = current_time
            else:
                detection_fps = 0.0
            
            # Update status bar with both FPS
            dropped_info = ""
            if self.detection_thread:
                dropped = self.detection_thread.dropped_count
                if dropped > 0:
                    dropped_info = f" (Dropped: {dropped})"
            
            status_msg = f"Display: {display_fps:.1f} FPS | Detection: {detection_fps:.1f} FPS{dropped_info}"
            self.main_window.statusBar().showMessage(status_msg)
    
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
        
        # Stop detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.stop()
            self.detection_thread.join(timeout=2.0)
        
        # Stop timer
        self.frame_timer.stop()
        
        # Release camera
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

