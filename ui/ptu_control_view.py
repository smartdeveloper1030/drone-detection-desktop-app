"""
PTU Gimbal Control View - Tab for servo motor control.
Matches the PTU control interface design from the manual.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QLineEdit, QGroupBox,
                             QGridLayout, QSlider, QCheckBox, QTextEdit,
                             QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PTUControlView(QWidget):
    """PTU Gimbal Control View - Tab for servo motor control."""
    
    # Signals
    connect_requested = pyqtSignal(str, int)  # port, baud_rate
    disconnect_requested = pyqtSignal()
    move_to_position = pyqtSignal(float, float, int)  # azimuth, pitch, speed
    move_relative = pyqtSignal(float, float, int)  # delta_azimuth, delta_pitch, speed
    stop_requested = pyqtSignal()
    go_to_zero = pyqtSignal(int)  # speed
    set_speed = pyqtSignal(int)  # speed
    set_acceleration = pyqtSignal(int)  # acceleration percentage
    tracking_enabled_changed = pyqtSignal(bool)  # enable/disable automatic tracking
    
    def __init__(self, parent=None):
        """Initialize the PTU control view."""
        super().__init__(parent)
        self.setup_ui()
        self.is_connected = False
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        self.current_speed = 5
        self.current_acceleration = 5
    
    def setup_ui(self):
        """Set up the UI components matching the PTU control interface."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left and right panels layout
        panels_layout = QHBoxLayout()
        
        # Left Panel - PTU Control
        left_panel = self.create_control_panel()
        panels_layout.addWidget(left_panel, 2)  # 2/3 width
        
        # Right Panel - Programming
        right_panel = self.create_programming_panel()
        panels_layout.addWidget(right_panel, 1)  # 1/3 width
        
        main_layout.addLayout(panels_layout)
        
        # Bottom Panel - Serial Communication
        bottom_panel = self.create_serial_panel()
        main_layout.addWidget(bottom_panel)
        
        self.setLayout(main_layout)
        
        # Set dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
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
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #666;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QLineEdit:read-only {
                background-color: #2b2b2b;
                color: #aaa;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QSlider::groove:horizontal {
                background: #555;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #777;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #999;
            }
        """)
    
    def create_control_panel(self) -> QGroupBox:
        """Create the left control panel."""
        panel = QGroupBox("PTU Control")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Pitch angle input (read-only)
        pitch_label = QLabel("Pitch Angle:")
        pitch_label.setStyleSheet("color: white;")
        layout.addWidget(pitch_label, 0, 0)
        
        self.pitch_input = QLineEdit()
        self.pitch_input.setReadOnly(True)
        self.pitch_input.setText("0.00°")
        layout.addWidget(self.pitch_input, 0, 1, 1, 2)
        
        # Azimuth angle input (read-only)
        azimuth_label = QLabel("Azimuth:")
        azimuth_label.setStyleSheet("color: white;")
        layout.addWidget(azimuth_label, 1, 0)
        
        self.azimuth_input = QLineEdit()
        self.azimuth_input.setReadOnly(True)
        self.azimuth_input.setText("0.00°")
        layout.addWidget(self.azimuth_input, 1, 1, 1, 2)
        
        # Speed slider
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: white;")
        layout.addWidget(speed_label, 5, 0)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self.speed_slider, 5, 1, 1, 2)
        
        self.speed_label = QLabel("5%")
        self.speed_label.setStyleSheet("color: white; min-width: 50px;")
        layout.addWidget(self.speed_label, 5, 3)
        
        # Acceleration slider
        accel_label = QLabel("Acceleration:")
        accel_label.setStyleSheet("color: white;")
        layout.addWidget(accel_label, 6, 0)
        
        self.accel_slider = QSlider(Qt.Horizontal)
        self.accel_slider.setRange(0, 100)
        self.accel_slider.setValue(5)
        self.accel_slider.valueChanged.connect(self._on_acceleration_changed)
        layout.addWidget(self.accel_slider, 6, 1, 1, 2)
        
        self.accel_label = QLabel("5%")
        self.accel_label.setStyleSheet("color: white; min-width: 50px;")
        layout.addWidget(self.accel_label, 6, 3)
        
        # Directional control buttons (3x3 grid)
        # Top row
        up_btn = QPushButton("Up ↑")
        up_btn.clicked.connect(lambda: self._on_directional_move(0, -5))
        layout.addWidget(up_btn, 2, 1)
        
        # Middle row
        left_btn = QPushButton("← Left")
        left_btn.clicked.connect(lambda: self._on_directional_move(-5, 0))
        layout.addWidget(left_btn, 3, 0)
        
        pause_btn = QPushButton("Pause")
        pause_btn.clicked.connect(self._on_pause)
        layout.addWidget(pause_btn, 3, 1)
        
        right_btn = QPushButton("Right →")
        right_btn.clicked.connect(lambda: self._on_directional_move(5, 0))
        layout.addWidget(right_btn, 3, 2)
        
        # Bottom row
        down_btn = QPushButton("Down ↓")
        down_btn.clicked.connect(lambda: self._on_directional_move(0, 5))
        layout.addWidget(down_btn, 4, 1)
        
        # Automatic tracking checkbox
        self.auto_tracking_checkbox = QCheckBox("Enable Automatic Tracking")
        self.auto_tracking_checkbox.setToolTip(
            "When enabled, PTU will automatically move to track predicted object positions"
        )
        self.auto_tracking_checkbox.setChecked(False)
        self.auto_tracking_checkbox.setStyleSheet("color: white;")
        self.auto_tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)
        layout.addWidget(self.auto_tracking_checkbox, 7, 0, 1, 3)
        
        # Keyboard control checkbox
        self.keyboard_control_checkbox = QCheckBox(
            "Keyboard arrow keys are allowed for control, but this will affect the focus of other controls."
        )
        self.keyboard_control_checkbox.setChecked(True)
        self.keyboard_control_checkbox.setStyleSheet("color: white;")
        layout.addWidget(self.keyboard_control_checkbox, 8, 0, 1, 3)
        
        # Store button references for enabling/disabling
        self.control_buttons = [
            up_btn, left_btn, right_btn, down_btn, pause_btn
        ]
        
        panel.setLayout(layout)
        return panel
    
    def create_programming_panel(self) -> QGroupBox:
        """Create the right command history panel."""
        panel = QGroupBox("Command History")
        layout = QVBoxLayout()
        
        # Command history display
        self.command_history_text = QTextEdit()
        self.command_history_text.setReadOnly(True)
        self.command_history_text.setPlaceholderText("Command history will appear here...")
        self.command_history_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.command_history_text)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self._on_clear_history)
        buttons_layout.addWidget(clear_history_btn)
        
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_serial_panel(self) -> QGroupBox:
        """Create the bottom serial communication panel."""
        panel = QGroupBox("Serial Communication")
        layout = QHBoxLayout()
        
        # Serial port selection
        port_label = QLabel("Serial Number:")
        port_label.setStyleSheet("color: white;")
        layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        self.port_combo.addItem("COM3")  # Default, will be updated
        layout.addWidget(self.port_combo)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn)
        
        # Baud rate selection
        baud_label = QLabel("Baud Rate:")
        baud_label.setStyleSheet("color: white;")
        layout.addWidget(baud_label)
        
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("9600")
        self.baud_combo.setMinimumWidth(100)
        layout.addWidget(self.baud_combo)
        
        layout.addStretch()
        
        # Send command
        send_label = QLabel("Send:")
        send_label.setStyleSheet("color: white;")
        layout.addWidget(send_label)
        
        self.send_input = QLineEdit()
        self.send_input.setPlaceholderText("Enter command (e.g., H12,45,30,20E)")
        self.send_input.returnPressed.connect(self._on_send_command)
        layout.addWidget(self.send_input)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_send_command)
        layout.addWidget(send_btn)
        
        panel.setLayout(layout)
        return panel
    
    # Event handlers
    def _on_connect_clicked(self):
        """Handle connect/disconnect button click."""
        if self.is_connected:
            self.disconnect_requested.emit()
        else:
            port = self.port_combo.currentText()
            baud_rate = int(self.baud_combo.currentText())
            self.connect_requested.emit(port, baud_rate)
    
    def _on_speed_changed(self, value: int):
        """Handle speed slider change."""
        self.current_speed = value
        self.speed_label.setText(f"{value}%")
        self.set_speed.emit(value)
    
    def _on_acceleration_changed(self, value: int):
        """Handle acceleration slider change."""
        self.current_acceleration = value
        self.accel_label.setText(f"{value}%")
        self.set_acceleration.emit(value)
    
    def _on_directional_move(self, delta_azimuth: float, delta_pitch: float):
        """Handle directional movement."""
        self.move_relative.emit(delta_azimuth, delta_pitch, self.current_speed)
    
    def _on_pause(self):
        """Handle pause button."""
        self.stop_requested.emit()
    
    
    def _on_program_run(self):
        """Handle program run."""
        program = self.program_editor.toPlainText()
        self.add_output(f"Running program...\n{program}")
        # Program execution would be handled by main application
    
    def _on_program_stop(self):
        """Handle program stop."""
        self.stop_requested.emit()
        self.add_output("Program stopped")
    
    def _on_program_keep(self):
        """Handle program keep."""
        self.add_output("Program kept")
    
    def _on_program_open(self):
        """Handle program open."""
        self.add_output("Open program file (not implemented)")
    
    def _on_take_over(self):
        """Handle take over."""
        self.add_output("Taking over control")
    
    def _on_program_clear(self):
        """Handle program clear."""
        self.program_editor.clear()
    
    def _on_illustrate(self):
        """Handle illustrate."""
        self.add_output("Illustrate (not implemented)")
    
    def _on_setup(self):
        """Handle setup."""
        self.add_output("Setup (not implemented)")
    
    def _on_send_command(self):
        """Handle send command."""
        command = self.send_input.text()
        if command:
            self.add_output(f"Sent: {command}")
            # Command sending would be handled by main application
            self.send_input.clear()
    
    def _on_tracking_toggled(self, state):
        """Handle automatic tracking checkbox toggle."""
        enabled = (state == Qt.Checked)
        self.tracking_enabled_changed.emit(enabled)
        status = "enabled" if enabled else "disabled"
        self.add_output(f"Automatic tracking {status}")
    
    # Public methods
    def update_connection_status(self, connected: bool, port: str = ""):
        """Update connection status."""
        self.is_connected = connected
        if connected:
            self.connect_btn.setText("Disconnect")
            self.connect_btn.setStyleSheet("background-color: #d32f2f;")
            for btn in self.control_buttons:
                btn.setEnabled(True)
        else:
            self.connect_btn.setText("Connect")
            self.connect_btn.setStyleSheet("")
            for btn in self.control_buttons:
                btn.setEnabled(False)
    
    def update_available_ports(self, ports: list):
        """Update available serial ports."""
        self.port_combo.clear()
        for port in ports:
            self.port_combo.addItem(port)
        if ports:
            self.port_combo.setCurrentIndex(0)
    
    def update_position(self, azimuth: float, pitch: float):
        """Update current position display."""
        self.current_azimuth = azimuth
        self.current_pitch = pitch
        self.azimuth_input.setText(f"{azimuth:.2f}°")
        self.pitch_input.setText(f"{pitch:.2f}°")
    
    def add_output(self, message: str):
        """Add message to output area (deprecated - use add_command_history instead)."""
        # This method is kept for backward compatibility but now adds to command history
        self.add_command_history(message)
    
    def add_command_history(self, command: str, status: str = "sent", response: str = ""):
        """
        Add command to history display.
        
        Args:
            command: Command string that was sent
            status: Status of command ('sent', 'done', 'timeout', 'error')
            response: Response from PTU (if any)
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        
        # Format the history entry
        if status == "done" and response:
            entry = f"[{timestamp}] {command} -> {response}"
        elif status == "timeout":
            entry = f"[{timestamp}] {command} -> TIMEOUT"
        elif status == "error":
            entry = f"[{timestamp}] {command} -> ERROR: {response}"
        else:
            entry = f"[{timestamp}] {command}"
        
        self.command_history_text.append(entry)
        
        # Auto-scroll to bottom
        scrollbar = self.command_history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_clear_history(self):
        """Clear command history display."""
        self.command_history_text.clear()