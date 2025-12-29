"""
System View - System status, alerts, and configuration display.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QGroupBox, QGridLayout, QPushButton)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont
from datetime import datetime
from typing import List, Optional
from config import Config


class SystemView(QWidget):
    """System view showing status, alerts, and configuration."""
    
    def __init__(self, parent=None):
        """Initialize the system view."""
        super().__init__(parent)
        self.setup_ui()
        self.alert_log: List[str] = []
        self.max_alerts = 100
        self.heartbeat_active = False
        self.estop_status = False
        
        # Heartbeat timer
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.toggle_heartbeat)
        self.heartbeat_timer.start(1000)  # Toggle every second
        
    def setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout()
        
        # Status group
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout()
        
        # Heartbeat indicator
        heartbeat_label = QLabel("Heartbeat:")
        heartbeat_label.setStyleSheet("color: white;")
        status_layout.addWidget(heartbeat_label, 0, 0)
        self.heartbeat_label = QLabel("●")
        self.heartbeat_label.setStyleSheet("color: red; font-size: 20px;")
        status_layout.addWidget(self.heartbeat_label, 0, 1)
        
        # E-Stop status
        estop_label = QLabel("E-Stop:")
        estop_label.setStyleSheet("color: white;")
        status_layout.addWidget(estop_label, 1, 0)
        self.estop_label = QLabel("ACTIVE")
        self.estop_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.estop_label, 1, 1)
        
        # Camera status
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("color: white;")
        status_layout.addWidget(camera_label, 2, 0)
        self.camera_status_label = QLabel("Disconnected")
        self.camera_status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.camera_status_label, 2, 1)
        
        # Detection status
        detection_label = QLabel("Detection:")
        detection_label.setStyleSheet("color: white;")
        status_layout.addWidget(detection_label, 3, 0)
        self.detection_status_label = QLabel("Inactive")
        self.detection_status_label.setStyleSheet("color: yellow;")
        status_layout.addWidget(self.detection_status_label, 3, 1)
        
        # Mode
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: white;")
        status_layout.addWidget(mode_label, 4, 0)
        self.mode_label = QLabel("Drone/Balloon")
        self.mode_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.mode_label, 4, 1)
        
        # Prediction horizon
        prediction_label = QLabel("Prediction Horizon:")
        prediction_label.setStyleSheet("color: white;")
        status_layout.addWidget(prediction_label, 5, 0)
        self.prediction_horizon_label = QLabel("0 ms")
        self.prediction_horizon_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.prediction_horizon_label, 5, 1)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        test_mode_label = QLabel("Test Mode:")
        test_mode_label.setStyleSheet("color: white;")
        config_layout.addWidget(test_mode_label, 0, 0)
        test_mode_text = "Enabled" if Config.TEST_OPTION else "Disabled"
        test_mode_value = QLabel(test_mode_text)
        test_mode_value.setStyleSheet("color: yellow;" if Config.TEST_OPTION else "color: green;")
        config_layout.addWidget(test_mode_value, 0, 1)
        
        camera_type_label = QLabel("Camera Type:")
        camera_type_label.setStyleSheet("color: white;")
        config_layout.addWidget(camera_type_label, 1, 0)
        camera_type_value = QLabel(Config.CAMERA_TYPE.upper())
        camera_type_value.setStyleSheet("color: white;")
        config_layout.addWidget(camera_type_value, 1, 1)
        
        yolo_model_label = QLabel("YOLO Model:")
        yolo_model_label.setStyleSheet("color: white;")
        config_layout.addWidget(yolo_model_label, 2, 0)
        model_label = QLabel(Config.YOLO_MODEL_PATH)
        model_label.setStyleSheet("color: white;")
        config_layout.addWidget(model_label, 2, 1)
        
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setStyleSheet("color: white;")
        config_layout.addWidget(conf_label, 3, 0)
        conf_value = QLabel(f"{Config.YOLO_CONFIDENCE_THRESHOLD:.2f}")
        conf_value.setStyleSheet("color: white;")
        config_layout.addWidget(conf_value, 3, 1)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Alerts group
        alerts_group = QGroupBox("Alert Log")
        alerts_layout = QVBoxLayout()
        
        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setMaximumHeight(200)
        self.alert_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")  # Green text
        alerts_layout.addWidget(self.alert_text)
        
        # Clear button
        clear_button = QPushButton("Clear Log")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:pressed {
                background-color: #555;
            }
        """)
        clear_button.clicked.connect(self.clear_alerts)
        alerts_layout.addWidget(clear_button)
        
        alerts_group.setLayout(alerts_layout)
        main_layout.addWidget(alerts_group)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
        
        # Set dark theme with white text
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
    
    def toggle_heartbeat(self):
        """Toggle heartbeat indicator."""
        self.heartbeat_active = not self.heartbeat_active
        if self.heartbeat_active:
            self.heartbeat_label.setStyleSheet("color: green; font-size: 20px;")
        else:
            self.heartbeat_label.setStyleSheet("color: red; font-size: 20px;")
    
    def update_camera_status(self, connected: bool):
        """Update camera status."""
        if connected:
            self.camera_status_label.setText("Connected")
            self.camera_status_label.setStyleSheet("color: green;")
        else:
            self.camera_status_label.setText("Disconnected")
            self.camera_status_label.setStyleSheet("color: red;")
    
    def update_detection_status(self, active: bool):
        """Update detection status."""
        if active:
            self.detection_status_label.setText("Active")
            self.detection_status_label.setStyleSheet("color: green;")
        else:
            self.detection_status_label.setText("Inactive")
            self.detection_status_label.setStyleSheet("color: yellow;")
    
    def update_prediction_horizon(self, horizon_ms: int):
        """Update prediction horizon."""
        self.prediction_horizon_label.setText(f"{horizon_ms} ms")
    
    def update_mode(self, mode: str):
        """Update detection mode."""
        self.mode_label.setText(mode)
    
    def update_estop_status(self, active: bool):
        """Update E-Stop status."""
        self.estop_status = active
        if active:
            self.estop_label.setText("ACTIVE")
            self.estop_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.estop_label.setText("INACTIVE")
            self.estop_label.setStyleSheet("color: green; font-weight: bold;")
    
    def add_alert(self, message: str, alert_type: str = "INFO"):
        """
        Add an alert to the log.
        
        Args:
            message: Alert message
            alert_type: Type of alert (INFO, WARNING, ERROR, THREAT)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color coding
        color_map = {
            "INFO": "#00ff00",  # Green
            "WARNING": "#ffaa00",  # Orange (keep for visibility)
            "ERROR": "#ff0000",  # Red (keep for visibility)
            "THREAT": "#ff0080"  # Pink (keep for visibility)
        }
        color = color_map.get(alert_type, "#00ff00")  # Default to green
        
        alert_text = f'<span style="color: {color};">[{timestamp}] [{alert_type}] {message}</span>'
        self.alert_log.append(alert_text)
        
        # Limit log size
        if len(self.alert_log) > self.max_alerts:
            self.alert_log.pop(0)
        
        # Update display
        self.alert_text.setHtml("<br>".join(self.alert_log))
        # Auto-scroll to bottom
        self.alert_text.verticalScrollBar().setValue(
            self.alert_text.verticalScrollBar().maximum()
        )
    
    def clear_alerts(self):
        """Clear the alert log."""
        self.alert_log.clear()
        self.alert_text.clear()

