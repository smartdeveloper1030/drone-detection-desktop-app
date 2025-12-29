"""
Operator View - Live video feed with detection visualization.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import cv2
import numpy as np
from typing import List, Optional
from detection import Detection


class OperatorView(QWidget):
    """Operator view showing live video feed with detections."""
    
    def __init__(self, parent=None):
        """Initialize the operator view."""
        super().__init__(parent)
        self.setup_ui()
        self.current_frame: Optional[np.ndarray] = None
        self.detections: List[Detection] = []
        self.predicted_point: Optional[tuple] = None  # (x, y)
        self.servo_crosshair: Optional[tuple] = None  # (x, y)
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start(1000)  # Update every second
        self.last_time = None
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setText("Waiting for video feed...")
        layout.addWidget(self.video_label)
        
        # Info bar
        info_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: green; font-weight: bold;")
        info_layout.addWidget(self.fps_label)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet("color: white;")
        info_layout.addWidget(self.detection_count_label)
        
        self.classification_label = QLabel("Classification: -")
        self.classification_label.setStyleSheet("color: yellow;")
        info_layout.addWidget(self.classification_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        self.setLayout(layout)
    
    def update_frame(self, frame: np.ndarray, detections: List[Detection], 
                    predicted_point: Optional[tuple] = None,
                    servo_crosshair: Optional[tuple] = None,
                    prediction_horizon_ms: int = 500):
        """
        Update the video frame with detections.
        
        Args:
            frame: Video frame (BGR)
            detections: List of detections
            predicted_point: Predicted position (x, y)
            servo_crosshair: Servo crosshair position (x, y)
            prediction_horizon_ms: Prediction horizon in milliseconds
        """
        self.current_frame = frame.copy()
        self.detections = detections
        self.predicted_point = predicted_point
        self.servo_crosshair = servo_crosshair
        
        # Draw on frame
        display_frame = self.draw_detections(frame.copy(), detections, predicted_point, servo_crosshair, prediction_horizon_ms)
        
        # Convert to QImage and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update info labels
        self.detection_count_label.setText(f"Detections: {len(detections)}")
        
        # Update classification label
        blacklist_detections = [d for d in detections if self._is_blacklist(d)]
        if blacklist_detections:
            det = blacklist_detections[0]
            color_info = f" ({det.color_class})" if det.color_class else ""
            self.classification_label.setText(f"Classification: BLACKLIST - {det.class_name}{color_info}")
            self.classification_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            whitelist_detections = [d for d in detections if not self._is_blacklist(d)]
            if whitelist_detections:
                det = whitelist_detections[0]
                color_info = f" ({det.color_class})" if det.color_class else ""
                self.classification_label.setText(f"Classification: WHITELIST - {det.class_name}{color_info}")
                self.classification_label.setStyleSheet("color: green;")
            else:
                self.classification_label.setText("Classification: -")
                self.classification_label.setStyleSheet("color: yellow;")
        
        # Update FPS counter
        self.frame_count += 1
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                       predicted_point: Optional[tuple] = None,
                       servo_crosshair: Optional[tuple] = None,
                       prediction_horizon_ms: int = 500) -> np.ndarray:
        """
        Draw detections, predicted point, and servo crosshair on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            predicted_point: Predicted position
            servo_crosshair: Servo crosshair position
            prediction_horizon_ms: Prediction horizon in milliseconds
        
        Returns:
            np.ndarray: Frame with drawings
        """
        # Calculate adaptive scale based on frame resolution
        # Base scale for 640x480, scale up for higher resolutions
        frame_height, frame_width = frame.shape[:2]
        base_resolution = 480  # Reference height
        scale_factor = max(0.6, min(2.0, frame_height / base_resolution * 0.6))
        
        # Scale drawing parameters
        font_scale = scale_factor
        line_thickness = max(1, int(scale_factor * 2))
        box_thickness_blacklist = max(2, int(scale_factor * 3))
        box_thickness_whitelist = max(1, int(scale_factor * 2))
        center_point_radius = max(3, int(scale_factor * 5))
        label_padding = max(3, int(scale_factor * 5))
        
        # Draw bounding boxes
        for det in detections:
            x1 = det.x - det.width // 2
            y1 = det.y - det.height // 2
            x2 = det.x + det.width // 2
            y2 = det.y + det.height // 2
            
            # Determine color based on classification
            is_blacklist = self._is_blacklist(det)
            color = (0, 0, 255) if is_blacklist else (0, 255, 0)  # Red for blacklist, green for whitelist
            thickness = box_thickness_blacklist if is_blacklist else box_thickness_whitelist
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Build label with ID, class, confidence, and speed
            label_parts = []
            if det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            label_parts.append(det.class_name)
            label_parts.append(f"{det.confidence:.2f}")
            
            # Add speed if available
            if det.velocity is not None:
                vx, vy = det.velocity
                speed = np.sqrt(vx**2 + vy**2)
                label_parts.append(f"Speed:{speed:.1f}px/s")
            
            if det.color_class:
                label_parts.append(f"[{det.color_class}]")
            
            label = " ".join(label_parts)
            
            # Label background with adaptive sizing
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - label_padding),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Label text with adaptive font size
            cv2.putText(
                frame,
                label,
                (x1, y1 - label_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness
            )
            
            # Draw center point with adaptive size
            cv2.circle(frame, (det.x, det.y), center_point_radius, color, -1)
        
        # Draw predicted point with adaptive sizing (red circle like in image)
        if predicted_point:
            pred_x, pred_y = predicted_point
            pred_radius = max(15, int(scale_factor * 15))
            pred_thickness = max(2, int(scale_factor * 3))
            pred_font_scale = font_scale * 0.7
            pred_color = (0, 0, 255)  # Red (BGR)
            trajectory_color = (0, 165, 255)  # Orange (BGR)
            
            # Find the detection that corresponds to this prediction (primary track)
            # The primary prediction prioritizes blacklist/threat objects, then most confident
            primary_detection = None
            if detections:
                # First, try to find blacklist/threat detections
                blacklist_detections = [det for det in detections 
                                       if det.track_id is not None and det.velocity is not None 
                                       and self._is_blacklist(det)]
                if blacklist_detections:
                    # Use the most confident blacklist detection
                    primary_detection = max(blacklist_detections, key=lambda d: d.confidence)
                else:
                    # Otherwise, use the most confident tracked detection
                    tracked_detections = [det for det in detections 
                                        if det.track_id is not None and det.velocity is not None]
                    if tracked_detections:
                        primary_detection = max(tracked_detections, key=lambda d: d.confidence)
            
            # Draw trajectory line from detection center to predicted point
            if primary_detection:
                trajectory_thickness = max(1, int(scale_factor * 2))
                cv2.line(frame, (primary_detection.x, primary_detection.y), 
                        (int(pred_x), int(pred_y)), trajectory_color, trajectory_thickness)
            
            # Draw outer circle
            cv2.circle(frame, (int(pred_x), int(pred_y)), pred_radius, pred_color, pred_thickness)
            # Draw inner dot
            cv2.circle(frame, (int(pred_x), int(pred_y)), max(2, int(scale_factor * 3)), pred_color, -1)
            
            # Draw prediction label
            pred_label = f"Pred {prediction_horizon_ms}ms"
            (text_width, text_height), baseline = cv2.getTextSize(
                pred_label, cv2.FONT_HERSHEY_SIMPLEX, pred_font_scale, line_thickness
            )
            # Position label to the right of the circle
            label_x = int(pred_x) + pred_radius + max(5, int(scale_factor * 8))
            label_y = int(pred_y)
            cv2.putText(
                frame,
                pred_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                pred_font_scale,
                pred_color,
                line_thickness
            )
        
        # Draw servo crosshair with adaptive sizing
        if servo_crosshair:
            x, y = servo_crosshair
            x, y = int(x), int(y)
            crosshair_size = max(10, int(scale_factor * 20))
            crosshair_thickness = max(1, int(scale_factor * 2))
            servo_font_scale = font_scale * 0.8
            # Horizontal line
            cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 255), crosshair_thickness)
            # Vertical line
            cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 255), crosshair_thickness)
            # Center circle
            cv2.circle(frame, (x, y), max(3, int(scale_factor * 5)), (0, 255, 255), crosshair_thickness)
            cv2.putText(
                frame,
                "SERVO",
                (x + max(8, int(scale_factor * 15)), y - max(8, int(scale_factor * 15))),
                cv2.FONT_HERSHEY_SIMPLEX,
                servo_font_scale,
                (0, 255, 255),
                line_thickness
            )
        
        return frame
    
    def _is_blacklist(self, detection: Detection) -> bool:
        """Check if detection is blacklist."""
        # Drones are always blacklist
        if 'drone' in detection.class_name.lower():
            return True
        # Balloons: check color
        if 'balloon' in detection.class_name.lower() and detection.color_class:
            from detection import ColorClassifier
            return ColorClassifier.is_blacklist(detection.color_class)
        return False
    
    def update_fps_display(self):
        """Update FPS display."""
        self.fps = self.frame_count
        self.frame_count = 0
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
    
    def resizeEvent(self, event):
        """Handle resize event to update video display."""
        super().resizeEvent(event)
        if self.current_frame is not None:
            # Redraw the frame at new size
            self.update_frame(
                self.current_frame,
                self.detections,
                self.predicted_point,
                self.servo_crosshair
            )

