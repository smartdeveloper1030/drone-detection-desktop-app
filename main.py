"""
Main application entry point for Drone Detection System - Milestone 1.
"""
import sys
import logging
import time
import threading
from queue import Queue, Empty
from typing import Tuple, Optional

# IMPORTANT: Import torch BEFORE PyQt5 to avoid DLL conflicts on Windows
# This must be done before any PyQt5 imports
import torch

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import cv2
import numpy as np

from config import Config
from camera_module import CameraModule
from detection import DetectionModule, Detection
from tracking import Tracker
from ui.main_window import MainWindow
from ptu_control import PTUControl
from coordinate_converter import CoordinateConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraReadingThread(threading.Thread):
    """
    Thread for continuously reading camera frames to prevent UI blocking.
    All camera I/O operations happen in this thread.
    """
    
    def __init__(self, camera, frame_queue, max_queue_size=2):
        """
        Initialize the camera reading thread.
        
        Args:
            camera: CameraModule instance
            frame_queue: Queue to put frames into
            max_queue_size: Maximum frames in queue (drops old frames if exceeded)
        """
        super().__init__(daemon=True)
        self.camera = camera
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.is_running = False
        self.read_count = 0
        self.dropped_count = 0
        
    def run(self):
        """Main thread loop - continuously read frames."""
        self.is_running = True
        logger.info("Camera reading thread started")
        
        while self.is_running:
            try:
                # Check if camera is connected
                if not self.camera.is_connected():
                    time.sleep(0.033)  # ~30 FPS check interval
                    continue
                
                # Read frame from camera (this can block, but it's OK in this thread)
                ret, frame = self.camera.read_latest_frame()
                
                if ret and frame is not None:
                    # Clear old frames to keep only latest (low latency)
                    while self.frame_queue.qsize() >= self.max_queue_size:
                        try:
                            self.frame_queue.get_nowait()
                            self.dropped_count += 1
                        except Empty:
                            break
                    
                    # Put frame in queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait((frame.copy(), time.time()))
                        self.read_count += 1
                    except:
                        self.dropped_count += 1
                else:
                    # No frame available, small sleep to avoid busy-waiting
                    time.sleep(0.001)  # 1ms
                    
            except Exception as e:
                logger.error(f"Error in camera reading thread: {str(e)}")
                if not self.is_running:
                    break
                # Small sleep on error to prevent tight loop
                time.sleep(0.01)
        
        logger.info(f"Camera reading thread stopped. Read {self.read_count} frames, dropped {self.dropped_count}")
    
    def stop(self):
        """Stop the camera reading thread."""
        self.is_running = False
        logger.info("Stopping camera reading thread...")


class ProcessingResultSignals(QObject):
    """Signals for thread-safe UI updates from processing thread."""
    result_ready = pyqtSignal(int, list, object, list)  # frame_id, enriched_detections, predicted_point, tracks


class DetectionProcessingThread(threading.Thread):
    """
    Thread for running detection, tracking, and prediction.
    Handles the complete processing pipeline in the background.
    """
    
    def __init__(self, detector, tracker, prediction_horizon_ms, signals, max_queue_size=1, detection_frame_skip=3):
        """
        Initialize the processing thread.
        
        Args:
            detector: DetectionModule instance
            tracker: Tracker instance
            prediction_horizon_ms: Prediction horizon in milliseconds
            signals: ProcessingResultSignals instance for thread-safe UI updates
            max_queue_size: Maximum frames in queue (drops frames if exceeded)
            detection_frame_skip: Run detection every N frames (track/predict every frame)
        """
        super().__init__(daemon=True)
        self.detector = detector
        self.tracker = tracker
        self.prediction_horizon_ms = prediction_horizon_ms
        self.signals = signals
        self.detection_frame_skip = detection_frame_skip
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.current_frame_id = 0
        self.frame_counter = 0  # Internal counter for detection skipping
        self.processed_count = 0
        self.dropped_count = 0
        
    def run(self):
        """Main thread loop - continuously process frames."""
        self.is_running = True
        logger.info(f"Processing thread started (detection every {self.detection_frame_skip} frames, track/predict every frame)")
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame, frame_id, timestamp = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                self.frame_counter += 1
                should_detect = (self.frame_counter % self.detection_frame_skip == 0)
                
                # Step 1: Run detection ONLY every N frames
                if should_detect:
                    detections = self.detector.detect(frame)
                    # Filter to keep only the largest detection (by area)
                    if detections:
                        largest_detection = max(detections, key=lambda d: d.width * d.height)
                        detections = [largest_detection]
                    # Step 2: Update tracker with detections
                    tracks = self.tracker.update(detections, timestamp)
                else:
                    # Step 2: Track without new detections (predict only)
                    detections = []  # No new detections
                    tracks = self.tracker.predict_only(timestamp)
                
                # Step 3: Get prediction from tracker (ALWAYS, every frame)
                predicted_point = self.tracker.get_primary_prediction(self.prediction_horizon_ms)
                
                # Step 4: Get primary track (largest) for enriched detections
                primary_track = self.tracker.get_primary_track()
                
                # Step 5: Enrich detections with track information (only primary/largest track)
                enriched_detections = []
                
                if primary_track:
                    # Use primary track (largest object)
                    if should_detect and detections and len(detections) > 0:
                        # Frame with detections: use the detection
                        det = detections[0]
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
                            track_id=primary_track.track_id,
                            velocity=primary_track.velocity
                        )
                        enriched_detections.append(enriched_det)
                    else:
                        # Frame without detection: use predicted position from primary track
                        enriched_det = Detection(
                            x=primary_track.detection.x,  # Predicted position
                            y=primary_track.detection.y,  # Predicted position
                            width=primary_track.detection.width,
                            height=primary_track.detection.height,
                            confidence=primary_track.detection.confidence * 0.8,  # Lower confidence for predicted
                            class_id=primary_track.detection.class_id,
                            class_name=primary_track.detection.class_name,
                            color_class=primary_track.detection.color_class,
                            distance=primary_track.detection.distance,
                            track_id=primary_track.track_id,
                            velocity=primary_track.velocity
                        )
                        enriched_detections.append(enriched_det)
                
                # Step 5: Emit result signal for UI update (thread-safe)
                self.signals.result_ready.emit(frame_id, enriched_detections, predicted_point, tracks)
                
                self.processed_count += 1
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing thread: {str(e)}")
                if not self.is_running:
                    break
        
        logger.info(f"Processing thread stopped. Processed {self.processed_count} frames, dropped {self.dropped_count}")
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """
        Add frame to processing queue (non-blocking).
        For low latency: always keep only the latest frame.
        
        Args:
            frame: Frame to process
            timestamp: Current timestamp for tracking
            
        Returns:
            bool: True if frame was added, False if queue was full (frame dropped)
        """
        self.current_frame_id += 1
        
        # For low latency: always clear old frames and keep only the latest
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.dropped_count += 1
            except Empty:
                break
        
        try:
            # Copy frame to avoid issues if frame is modified elsewhere
            self.frame_queue.put_nowait((frame.copy(), self.current_frame_id, timestamp))
            return True
        except:
            self.dropped_count += 1
            return False
    
    def stop(self):
        """Stop the processing thread."""
        self.is_running = False
        logger.info("Stopping processing thread...")


class DetectionThread(threading.Thread):
    """
    Separate thread for running detection to prevent UI blocking.
    Uses non-blocking queue to drop frames if detection is too slow.
    DEPRECATED: Use DetectionProcessingThread instead.
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
        For low latency: always keep only the latest frame.
        
        Args:
            frame: Frame to process
            
        Returns:
            bool: True if frame was added, False if queue was full (frame dropped)
        """
        self.current_frame_id += 1
        
        # For low latency: always clear old frames and keep only the latest
        # This ensures we process the most recent frame, not stale ones
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.dropped_count += 1
            except Empty:
                break
        
        try:
            # Copy frame to avoid issues if frame is modified elsewhere
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
        
        # PTU control
        self.ptu = PTUControl()
        self.coordinate_converter = None  # Will be initialized after camera connection
        self.ptu_tracking_enabled = False
        
        # Connect mode change signal
        self.main_window.mode_changed.connect(self._on_mode_changed)
        
        # Connect color change signal
        self.main_window.color_changed.connect(self._on_color_changed)
        
        # Connect confidence threshold change signal
        self.main_window.get_system_view().confidence_threshold_changed.connect(self._on_confidence_threshold_changed)
        
        # Connect prediction horizon change signal
        self.main_window.get_system_view().prediction_horizon_changed.connect(self._on_prediction_horizon_changed)
        
        # Connect PTU control signals
        ptu_view = self.main_window.get_ptu_control_view()
        ptu_view.connect_requested.connect(self._on_ptu_connect)
        ptu_view.disconnect_requested.connect(self._on_ptu_disconnect)
        ptu_view.move_to_position.connect(self._on_ptu_move_to_position)
        ptu_view.move_relative.connect(self._on_ptu_move_relative)
        ptu_view.move_directional.connect(self._on_ptu_move_directional)
        ptu_view.stop_requested.connect(self._on_ptu_stop)
        ptu_view.go_to_zero.connect(self._on_ptu_go_to_zero)
        ptu_view.set_speed.connect(self._on_ptu_set_speed)
        ptu_view.set_acceleration.connect(self._on_ptu_set_acceleration)
        ptu_view.tracking_enabled_changed.connect(self.enable_ptu_tracking)
        ptu_view.send_raw_command.connect(self._on_ptu_send_raw_command)
        ptu_view.get_position_requested.connect(self._on_ptu_get_position)
        
        # Set up command history callback to show all communication in UI
        def history_callback(command, status, response):
            """Callback to update UI with command history."""
            ptu_view.add_command_history(command, status, response)
        
        self.ptu.set_history_callback(history_callback)
        
        # Processing state
        self.is_running = False
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        
        # Camera reading thread (prevents UI blocking)
        self.camera_frame_queue = Queue(maxsize=2)  # Keep only latest 2 frames
        self.camera_reading_thread = None
        
        # Processing threading (detection, tracking, prediction)
        self.processing_thread = None
        self.processing_signals = ProcessingResultSignals()
        self.processing_signals.result_ready.connect(self._on_processing_result)
        
        # Latest processing results
        self.latest_enriched_detections = []
        self.latest_predicted_point = None
        self.latest_tracks = []
        self.last_processing_frame_id = -1
        
        # Keep old detection_thread for backward compatibility during transition
        self.detection_thread = None
        
        # Frame counter (for FPS tracking only - all frames sent to processing thread)
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
        
        # Track last prediction distance for logging
        self.last_prediction_distance: Optional[Tuple[float, float]] = None  # (distance_x, distance_y)
        self.last_prediction_distance_log_time: float = 0.0
        self.prediction_distance_log_interval: float = 1.0  # Log distance every 1 second
        
        # PTU auto-tracking calibration constants
        # Based on calibration: initial (1094, 153), 1° move → (1123, 190), 5° move → (1220, 300)
        # Average: ~29 pixels/degree for azimuth (horizontal), ~37 pixels/degree for pitch (vertical)
        self.PTU_PIXELS_PER_DEGREE_AZIMUTH = 29.0  # Horizontal movement
        self.PTU_PIXELS_PER_DEGREE_PITCH = 37.0    # Vertical movement
        
        # Minimum pixel offset threshold to trigger PTU movement (avoid jitter)
        self.PTU_TRACKING_THRESHOLD_PIXELS = 3.0  # Only move if offset > 3 pixels (reduced for better responsiveness)
        
        # Throttle PTU movements to avoid excessive commands
        self.last_ptu_tracking_time: float = 0.0
        self.PTU_TRACKING_MIN_INTERVAL: float = 0.05  # Minimum 50ms between movements (increased frequency)
        
        # Smooth tracking parameters - move in incremental steps
        self.PTU_TRACKING_STEP_PERCENTAGE = 0.15  # Move 15% of calculated offset per step (very smooth movement, more steps)
        self.PTU_TRACKING_MAX_STEP_DEGREES = 0.2  # Maximum step size in degrees (smaller steps for smoother movement)
        self.PTU_TRACKING_CONVERGENCE_THRESHOLD = 2.0  # Stop tracking when offset < 2 pixels
        
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
        
        # Connect camera (allow app to start even if camera fails)
        camera_connected = self.camera.connect()
        if camera_connected:
            self.main_window.get_system_view().update_camera_status(True)
            self.main_window.get_system_view().add_alert(
                "Camera connected", "INFO"
            )
            
            # Initialize coordinate converter with camera dimensions
            frame_width, frame_height = self.camera.get_frame_size()
            if frame_width > 0 and frame_height > 0:
                self.coordinate_converter = CoordinateConverter(
                    image_width=frame_width,
                    image_height=frame_height,
                    horizontal_fov=60.0,  # Default FOV, can be configured
                    vertical_fov=45.0
                )
                logger.info(f"Coordinate converter initialized: {frame_width}x{frame_height}")
        else:
            logger.warning("Camera not connected - app will continue without video feed")
            self.main_window.get_system_view().update_camera_status(False)
            self.main_window.get_system_view().add_alert(
                "Camera is not connected", "WARNING"
            )
        
        # Update available PTU ports
        self._update_ptu_ports()
        
        # Start camera reading thread (prevents UI blocking on camera I/O)
        if camera_connected:
            self.camera_reading_thread = CameraReadingThread(
                self.camera,
                self.camera_frame_queue,
                max_queue_size=2
            )
            self.camera_reading_thread.start()
            logger.info("Camera reading thread started")
        
        # Start processing thread (detection + tracking + prediction) with minimal queue for low latency
        # Queue size of 1 ensures we always process the latest frame, dropping old ones
        processing_queue_size = max(1, Config.DETECTION_QUEUE_SIZE)  # At least 1
        self.processing_thread = DetectionProcessingThread(
            self.detector,
            self.tracker,
            self.prediction_horizon_ms,
            self.processing_signals,
            max_queue_size=processing_queue_size,
            detection_frame_skip=Config.DETECTION_FRAME_SKIP
        )
        self.processing_thread.start()
        logger.info(f"Processing thread started (detection every {Config.DETECTION_FRAME_SKIP} frames, track/predict every frame, queue size: {processing_queue_size})")
        
        # Start frame processing only if camera is connected
        if camera_connected:
            frame_interval = int(1000 / Config.UI_REFRESH_RATE)  # Convert to milliseconds
            self.frame_timer.start(frame_interval)
            self.is_running = True
        else:
            self.is_running = False
            logger.info("Frame processing not started - camera not connected")
        
        # Set initial mode in UI
        self.main_window.set_mode(Config.DETECT_MODE)
        
        # Set initial color if in balloon mode
        if Config.DETECT_MODE.lower() == "balloon":
            selected_color = getattr(Config, 'SELECTED_BALLOON_COLOR', 'red')
            # Update blacklist to only include selected color
            Config.BALLOON_BLACKLIST_COLORS = [selected_color]
            self.main_window.set_color(selected_color)
        
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
            # Update blacklist to only include selected color (unless "All" is selected)
            if selected_color.lower() != "all":
                Config.BALLOON_BLACKLIST_COLORS = [selected_color]
            # Ensure color selector is set correctly
            self.main_window.set_color(selected_color)
        
        # Stop current processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Stopping current processing thread...")
            self.processing_thread.stop()
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        # Clear current results
        self.latest_enriched_detections = []
        self.latest_predicted_point = None
        self.latest_tracks = []
        self.last_processing_frame_id = -1
        
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
        
        # Restart processing thread
        processing_queue_size = max(1, Config.DETECTION_QUEUE_SIZE)
        self.processing_thread = DetectionProcessingThread(
            self.detector,
            self.tracker,
            self.prediction_horizon_ms,
            self.processing_signals,
            max_queue_size=processing_queue_size,
            detection_frame_skip=Config.DETECTION_FRAME_SKIP
        )
        self.processing_thread.start()
        
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
        # Update processing thread's prediction horizon
        if self.processing_thread:
            self.processing_thread.prediction_horizon_ms = horizon_ms
        # Update config for persistence
        Config.PREDICTION_HORIZON_MS = horizon_ms
        self.main_window.get_system_view().add_alert(
            f"Prediction horizon updated to {horizon_ms} ms", "INFO"
        )
    
    def _on_color_changed(self, color: str):
        """
        Handle color selection change from UI.
        
        Args:
            color: Selected color name (can be "All" or a specific color)
        """
        color_lower = color.lower()
        logger.info(f"Balloon color selection changed to: {color}")
        # Update config
        Config.SELECTED_BALLOON_COLOR = color_lower
        
        # If "All" is selected, don't update blacklist (detect all colors)
        # Otherwise, update blacklist to only include selected color
        if color_lower == "all":
            # Don't filter by color - detect all balloons
            # Keep existing blacklist for threat detection purposes
            pass
        else:
            # Update blacklist colors to only include selected color
            Config.BALLOON_BLACKLIST_COLORS = [color_lower]
        
        self.main_window.get_system_view().add_alert(
            f"Balloon detection color set to: {color}", "INFO"
        )
    
    def _on_processing_result(self, frame_id: int, enriched_detections: list, predicted_point: Optional[tuple], tracks: list):
        """
        Handle processing results from background thread (thread-safe callback).
        Updates UI with detection, tracking, and prediction results.
        
        Args:
            frame_id: Frame ID that was processed
            enriched_detections: List of enriched detections with track info
            predicted_point: Predicted point (x, y) or None
            tracks: List of active tracks
        """
        # Update cached results
        self.latest_enriched_detections = enriched_detections
        self.latest_predicted_point = predicted_point
        self.latest_tracks = tracks
        self.last_processing_frame_id = frame_id
        
        # Update detection status
        self.main_window.get_system_view().update_detection_status(len(enriched_detections) > 0)
        
        # Check for blacklist detections and add alerts
        blacklist_detections = self.detector.get_blacklist_detections(enriched_detections)
        if blacklist_detections:
            # Threat detected
            det = blacklist_detections[0]
            color_info = f" ({det.color_class})" if det.color_class else ""
            self.main_window.get_system_view().add_alert(
                f"THREAT DETECTED: {det.class_name}{color_info} at ({det.x}, {det.y})",
                "THREAT"
            )
        elif len(enriched_detections) > 0 and len(blacklist_detections) == 0:
            # Only whitelist detected
            det = enriched_detections[0]
            self.main_window.get_system_view().add_alert(
                f"Whitelist object detected: {det.class_name}",
                "INFO"
            )
        
        # Calculate and log distance between prediction point and camera center
        if predicted_point and self.main_window.get_operator_view().current_frame is not None:
            frame = self.main_window.get_operator_view().current_frame
            frame_height, frame_width = frame.shape[:2]
            camera_center_x = frame_width / 2.0
            camera_center_y = frame_height / 2.0
            
            pred_x, pred_y = predicted_point
            distance_x = pred_x - camera_center_x
            distance_y = pred_y - camera_center_y
            
            # Log distances to alert log (throttled to avoid spam)
            distance_changed = False
            if self.last_prediction_distance is None:
                distance_changed = True
            else:
                last_dist_x, last_dist_y = self.last_prediction_distance
                if abs(distance_x - last_dist_x) > 10.0 or abs(distance_y - last_dist_y) > 10.0:
                    distance_changed = True
            
            current_time = time.time()
            time_since_last_log = current_time - self.last_prediction_distance_log_time
            if distance_changed or time_since_last_log >= self.prediction_distance_log_interval:
                self.main_window.get_system_view().add_alert(
                    f"Prediction distance from center: X={distance_x:.1f} pixels, Y={distance_y:.1f} pixels",
                    "INFO"
                )
                self.last_prediction_distance = (distance_x, distance_y)
                self.last_prediction_distance_log_time = current_time
        
        # PTU auto-tracking: Move PTU to align camera center with predicted point
        if (self.ptu_tracking_enabled and 
            predicted_point is not None and 
            self.main_window.get_operator_view().current_frame is not None):
            self._update_ptu_tracking(predicted_point)
        
        # Update UI with latest results (will use current frame from operator view)
        if self.main_window.get_operator_view().current_frame is not None:
            try:
                frame = self.main_window.get_operator_view().current_frame
                servo_crosshair = None  # Can be calculated if needed
                
                # Update operator view with enriched detections and predictions
                self.main_window.get_operator_view().update_frame(
                    frame,
                    enriched_detections,
                    predicted_point,
                    servo_crosshair,
                    self.prediction_horizon_ms
                )
            except Exception as e:
                logger.error(f"Error updating UI with processing results: {str(e)}")
                # Continue even if UI update fails
    
    def process_frame(self):
        """
        Process a single frame - display immediately, detection runs asynchronously.
        All camera I/O happens in camera_reading_thread to prevent UI blocking.
        """
        if not self.is_running:
            return
        
        # Check if camera is still connected
        if not self.camera.is_connected():
            self.is_running = False
            self.frame_timer.stop()
            self.main_window.get_system_view().update_camera_status(False)
            return
        
        # Get frame from camera reading thread queue (non-blocking)
        try:
            frame, timestamp = self.camera_frame_queue.get_nowait()
        except Empty:
            # No frame available yet, skip this cycle (camera thread will provide next frame)
            return
        
        # Update camera status
        self.main_window.get_system_view().update_camera_status(True)
        
        # CRITICAL: Display raw frame IMMEDIATELY before any processing
        # This ensures the UI shows the latest frame as soon as it's captured
        # Processing results will update the frame later via signal callback
        try:
            self.main_window.get_operator_view().update_frame(
                frame,
                self.latest_enriched_detections,  # Use latest results if available
                self.latest_predicted_point,  # Use latest prediction if available
                None,  # No servo crosshair yet
                self.prediction_horizon_ms
            )
        except Exception as e:
            logger.error(f"Error updating frame display: {str(e)}")
            # Continue processing even if display update fails
        
        # Queue frame for processing (detection + tracking + prediction) in background thread
        if self.processing_thread:
            try:
                self.processing_thread.add_frame(frame, timestamp)
                self.detection_fps_count += 1
            except Exception as e:
                logger.error(f"Error queuing frame for processing: {str(e)}")
        
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
            if self.processing_thread:
                dropped = self.processing_thread.dropped_count
                if dropped > 0:
                    dropped_info = f" (Dropped: {dropped})"
            
            status_msg = f"Display: {display_fps:.1f} FPS | Detection: {detection_fps:.1f} FPS{dropped_info}"
            self.main_window.statusBar().showMessage(status_msg)
    
    def run(self):
        """Run the application."""
        if not self.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Show main window maximized (shows taskbar and toolbar)
        self.main_window.showMaximized()
        
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
        
        # Stop camera reading thread first (stops frame production)
        if self.camera_reading_thread and self.camera_reading_thread.is_alive():
            logger.info("Stopping camera reading thread...")
            self.camera_reading_thread.stop()
            self.camera_reading_thread.join(timeout=2.0)
        
        # Stop processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.stop()
            self.processing_thread.join(timeout=2.0)
        
        # Stop timer
        self.frame_timer.stop()
        
        # Release camera
        self.camera.release()
        
        # Disconnect PTU and cleanup thread
        if self.ptu.is_connected:
            self.ptu.disconnect()
        # Cleanup PTU thread
        if hasattr(self.ptu, 'cleanup'):
            self.ptu.cleanup()
        
        logger.info("Cleanup complete")
    
    def _update_ptu_ports(self):
        """Update available PTU serial ports in the UI."""
        ports = self.ptu.get_available_ports()
        self.main_window.get_ptu_control_view().update_available_ports(ports)
    
    def _on_ptu_connect(self, port: str, baud_rate: int):
        """Handle PTU connection request."""
        logger.info(f"Connecting to PTU on {port} at {baud_rate} baud")
        success = self.ptu.connect(port, baud_rate)
        
        if success:
            self.main_window.get_ptu_control_view().update_connection_status(True, port)
            self.main_window.get_ptu_control_view().update_position(0.0, 0.0)  # Reset position to zero
            self.main_window.get_system_view().add_alert(
                f"PTU connected on {port}", "INFO"
            )
        else:
            self.main_window.get_ptu_control_view().update_connection_status(False)
            self.main_window.get_system_view().add_alert(
                f"Failed to connect to PTU on {port}", "ERROR"
            )
    
    def _on_ptu_disconnect(self):
        """Handle PTU disconnection request."""
        logger.info("Disconnecting from PTU")
        self.ptu.disconnect()
        self.main_window.get_ptu_control_view().update_connection_status(False)
        self.main_window.get_system_view().add_alert("PTU disconnected", "INFO")
    
    def _on_ptu_move_to_position(self, azimuth: float, pitch: float, speed: int):
        """Handle PTU move to absolute position."""
        if self.ptu.is_connected:
            success = self.ptu.move_to_position(azimuth, pitch, speed)
            if success:
                self.main_window.get_ptu_control_view().update_position(azimuth, pitch)
            else:
                self.main_window.get_system_view().add_alert(
                    f"Failed to move PTU to ({azimuth:.1f}°, {pitch:.1f}°)", "ERROR"
                )
    
    def _on_ptu_move_relative(self, delta_azimuth: float, delta_pitch: float, speed: int):
        """Handle PTU relative movement."""
        if self.ptu.is_connected:
            success = self.ptu.move_relative(delta_azimuth, delta_pitch, speed)
            if success:
                azimuth, pitch = self.ptu.get_position()
                self.main_window.get_ptu_control_view().update_position(azimuth, pitch)
    
    def _on_ptu_move_directional(self, direction: str, speed: int):
        """Handle PTU directional movement using H61/H62/H63/H64 commands."""
        if self.ptu.is_connected:
            self.ptu.move_directional(direction, speed)
    
    def _on_ptu_stop(self):
        """Handle PTU stop request."""
        if self.ptu.is_connected:
            self.ptu.stop()
    
    def _on_ptu_go_to_zero(self, speed: int):
        """Handle PTU go to zero position."""
        if self.ptu.is_connected:
            success = self.ptu.go_to_zero(speed)
            if success:
                self.main_window.get_ptu_control_view().update_position(0.0, 0.0)
    
    def _on_ptu_set_speed(self, speed: int):
        """Handle PTU speed setting."""
        if self.ptu.is_connected:
            self.ptu.set_speed(speed)
    
    def _on_ptu_set_acceleration(self, acceleration: int):
        """Handle PTU acceleration setting."""
        if self.ptu.is_connected:
            self.ptu.set_acceleration(acceleration)
    
    def _on_ptu_get_position(self):
        """Handle PTU get position request."""
        if self.ptu.is_connected:
            # Get position from PTU (queries H10 and H20 commands)
            azimuth, pitch = self.ptu.get_position()
            # Update the textboxes with retrieved values
            self.main_window.get_ptu_control_view().update_position(azimuth, pitch)
            logger.info(f"Get Position: Updated textboxes - Azimuth={azimuth:.2f}°, Pitch={pitch:.2f}°")
            self.main_window.get_system_view().add_alert(
                f"Position retrieved: Azimuth={azimuth:.2f}°, Pitch={pitch:.2f}°", "INFO"
            )
        else:
            self.main_window.get_ptu_control_view().add_command_history(
                "Cannot get position: PTU not connected", "error"
            )
            self.main_window.get_system_view().add_alert(
                "Cannot get position: PTU not connected", "ERROR"
            )
    
    def _on_ptu_send_raw_command(self, command: str):
        """Handle raw command sending (without waiting for Done response)."""
        if self.ptu.is_connected:
            self.ptu.send_raw_command(command)
    
    def _move_ptu_to_point(self, point: Tuple[float, float]):
        """
        Move PTU to track a predicted point.
        
        Args:
            point: (x, y) pixel coordinates
        """
        if not self.ptu.is_connected or not self.coordinate_converter:
            return
        
        pixel_x, pixel_y = point
        current_azimuth, current_pitch = self.ptu.get_position()
        
        # Convert pixel to angle
        new_azimuth, new_pitch = self.coordinate_converter.pixel_to_angle(
            pixel_x, pixel_y, current_azimuth, current_pitch
        )
        
        # Move PTU (with speed from UI)
        ptu_view = self.main_window.get_ptu_control_view()
        speed = getattr(ptu_view, 'current_speed', 20)
        
        success = self.ptu.move_to_position(new_azimuth, new_pitch, speed)
        if success:
            self.main_window.get_ptu_control_view().update_position(new_azimuth, new_pitch)
    
    def enable_ptu_tracking(self, enable: bool):
        """
        Enable/disable automatic PTU tracking of predicted points.
        
        Args:
            enable: True to enable tracking, False to disable
        """
        self.ptu_tracking_enabled = enable
        logger.info(f"PTU tracking {'enabled' if enable else 'disabled'}")
    
    def _update_ptu_tracking(self, predicted_point: Tuple[float, float]):
        """
        Update PTU position to align camera center with predicted point using smooth incremental movements.
        
        This method calculates the pixel offset between the predicted point (reference)
        and the camera center, then moves the PTU in small incremental steps to minimize
        the deviation iteratively. Each step moves a fraction of the calculated offset,
        then recalculates the deviation and moves again until convergence.
        
        Args:
            predicted_point: (x, y) pixel coordinates of the predicted point
        """
        if not self.ptu.is_connected:
            return
        
        # Throttle movements to avoid excessive commands
        current_time = time.time()
        if current_time - self.last_ptu_tracking_time < self.PTU_TRACKING_MIN_INTERVAL:
            return
        self.last_ptu_tracking_time = current_time
        
        # Get current frame to determine camera center
        if self.main_window.get_operator_view().current_frame is None:
            return
        
        frame = self.main_window.get_operator_view().current_frame
        frame_height, frame_width = frame.shape[:2]
        
        # Camera center (reference point)
        camera_center_x = frame_width / 2.0
        camera_center_y = frame_height / 2.0
        
        # Predicted point (target)
        pred_x, pred_y = predicted_point
        
        # Calculate pixel offset (positive = target is right/below center)
        offset_x = pred_x - camera_center_x  # Positive = target is to the right
        offset_y = pred_y - camera_center_y  # Positive = target is below center
        
        # Calculate total deviation magnitude
        total_deviation = (offset_x**2 + offset_y**2)**0.5
        
        # Check if we've converged (deviation is small enough)
        if total_deviation < self.PTU_TRACKING_CONVERGENCE_THRESHOLD:
            return  # Already aligned, no movement needed
        
        # Convert full pixel offset to degrees using calibration data
        # Calibration: 1° azimuth → 29 pixels right, 1° pitch → 37 pixels down
        # To bring target to center: if target is RIGHT, move PTU LEFT (inverse direction)
        full_delta_azimuth_deg = -offset_x / self.PTU_PIXELS_PER_DEGREE_AZIMUTH  # Inverted: move opposite direction
        full_delta_pitch_deg = -offset_y / self.PTU_PIXELS_PER_DEGREE_PITCH  # Inverted: move opposite direction
        
        # Calculate step size: move a percentage of the full offset, but cap at maximum step size
        step_azimuth_deg = full_delta_azimuth_deg * self.PTU_TRACKING_STEP_PERCENTAGE
        step_pitch_deg = full_delta_pitch_deg * self.PTU_TRACKING_STEP_PERCENTAGE
        
        # Apply maximum step size limit to prevent overshooting
        if abs(step_azimuth_deg) > self.PTU_TRACKING_MAX_STEP_DEGREES:
            step_azimuth_deg = self.PTU_TRACKING_MAX_STEP_DEGREES * (1.0 if step_azimuth_deg > 0 else -1.0)
        if abs(step_pitch_deg) > self.PTU_TRACKING_MAX_STEP_DEGREES:
            step_pitch_deg = self.PTU_TRACKING_MAX_STEP_DEGREES * (1.0 if step_pitch_deg > 0 else -1.0)
        
        # Get current speed from PTU control view
        ptu_view = self.main_window.get_ptu_control_view()
        speed = getattr(ptu_view, 'current_speed', 20)
        
        # Move PTU relative to current position (incremental step)
        success = self.ptu.move_relative(step_azimuth_deg, step_pitch_deg, speed)
        
        if success:
            # Calculate expected remaining deviation after this step
            # Convert step back to pixels to estimate remaining offset
            remaining_offset_x = offset_x + (step_azimuth_deg * self.PTU_PIXELS_PER_DEGREE_AZIMUTH)
            remaining_offset_y = offset_y + (step_pitch_deg * self.PTU_PIXELS_PER_DEGREE_PITCH)
            remaining_deviation = (remaining_offset_x**2 + remaining_offset_y**2)**0.5
            
            logger.info(
                f"PTU auto-tracking (smooth): pred=({pred_x:.1f}, {pred_y:.1f}), center=({camera_center_x:.1f}, {camera_center_y:.1f}), "
                f"offset=({offset_x:.1f}, {offset_y:.1f}) px, "
                f"step=({step_azimuth_deg:.3f}°, {step_pitch_deg:.3f}°), "
                f"remaining_deviation={remaining_deviation:.1f} px"
            )
        else:
            logger.warning("PTU auto-tracking: Failed to move PTU")


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

