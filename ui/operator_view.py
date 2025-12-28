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
                    servo_crosshair: Optional[tuple] = None):
        """
        Update the video frame with detections.
        
        Args:
            frame: Video frame (BGR)
            detections: List of detections
            predicted_point: Predicted position (x, y)
            servo_crosshair: Servo crosshair position (x, y)
        """
        self.current_frame = frame.copy()
        self.detections = detections
        self.predicted_point = predicted_point
        self.servo_crosshair = servo_crosshair
        
        # Draw on frame
        display_frame = self.draw_detections(frame.copy(), detections, predicted_point, servo_crosshair)
        
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
                       servo_crosshair: Optional[tuple] = None) -> np.ndarray:
        """
        Draw detections, predicted point, and servo crosshair on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            predicted_point: Predicted position
            servo_crosshair: Servo crosshair position
        
        Returns:
            np.ndarray: Frame with drawings
        """
        # Draw bounding boxes
        for det in detections:
            x1 = det.x - det.width // 2
            y1 = det.y - det.height // 2
            x2 = det.x + det.width // 2
            y2 = det.y + det.height // 2
            
            # Determine color based on classification
            is_blacklist = self._is_blacklist(det)
            color = (0, 0, 255) if is_blacklist else (0, 255, 0)  # Red for blacklist, green for whitelist
            thickness = 3 if is_blacklist else 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.color_class:
                label += f" [{det.color_class}]"
            
            # Label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            cv2.circle(frame, (det.x, det.y), 5, color, -1)
        
        # Draw predicted point
        if predicted_point:
            x, y = predicted_point
            cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 0), 2)  # Yellow circle
            cv2.putText(
                frame,
                "PRED",
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )
        
        # Draw servo crosshair
        if servo_crosshair:
            x, y = servo_crosshair
            x, y = int(x), int(y)
            crosshair_size = 20
            # Horizontal line
            cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 255), 2)
            # Vertical line
            cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 255), 2)
            # Center circle
            cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)
            cv2.putText(
                frame,
                "SERVO",
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
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

