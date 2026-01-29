"""
Zoom Control View - Tab for camera zoom in/out via ISAPI.
Matches layout: Camera Connection, Zoom Speed (slider), Zoom Control (In/Out/Stop), Status.
"""
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QLineEdit,
    QSlider,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ZoomView(QWidget):
    """Zoom control view for ISAPI-based zoom in/out."""

    connect_requested = pyqtSignal(str, str, str)  # ip, username, password
    disconnect_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal(int)   # speed 1–100
    zoom_out_requested = pyqtSignal(int)
    zoom_stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._zoom_speed = 50
        self._connected = False

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_layout.addWidget(self._create_connection_panel())
        main_layout.addWidget(self._create_speed_panel())
        main_layout.addWidget(self._create_control_panel())
        main_layout.addWidget(self._create_status_panel())

        main_layout.addStretch()
        self.setLayout(main_layout)
        self.update_connection_status(False)

        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; }
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
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #555; }
            QPushButton:disabled { background-color: #2b2b2b; color: #666; }
            QLineEdit {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #777;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover { background: #999; }
        """)

    def _create_connection_panel(self) -> QGroupBox:
        panel = QGroupBox("Camera Connection")
        layout = QGridLayout()

        layout.addWidget(QLabel("IP Address:"), 0, 0)
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.64")
        self.ip_edit.setText("192.168.1.64")
        layout.addWidget(self.ip_edit, 0, 1)

        layout.addWidget(QLabel("Username:"), 1, 0)
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("admin")
        self.user_edit.setText("admin")
        layout.addWidget(self.user_edit, 1, 1)

        layout.addWidget(QLabel("Password:"), 2, 0)
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.pass_edit.setPlaceholderText("(optional)")
        layout.addWidget(self.pass_edit, 2, 1)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn, 3, 0, 1, 2)

        panel.setLayout(layout)
        return panel

    def _create_speed_panel(self) -> QGroupBox:
        panel = QGroupBox("Zoom Speed")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.speed_slider.setMinimumWidth(120)
        layout.addWidget(self.speed_slider, 1)

        self.speed_label = QLabel("50")
        self.speed_label.setMinimumWidth(28)
        self.speed_label.setStyleSheet("color: white;")
        layout.addWidget(self.speed_label)

        panel.setLayout(layout)
        return panel

    def _create_control_panel(self) -> QGroupBox:
        panel = QGroupBox("Zoom Control")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self.zoom_in_btn = QPushButton("ZOOM IN ▲")
        self.zoom_in_btn.setMinimumHeight(44)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("ZOOM OUT ▼")
        self.zoom_out_btn.setMinimumHeight(44)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        layout.addWidget(self.zoom_out_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.clicked.connect(self._on_zoom_stop)
        layout.addWidget(self.stop_btn)

        panel.setLayout(layout)
        return panel

    def _create_status_panel(self) -> QGroupBox:
        panel = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.action_label = QLabel("Action: None")
        self.action_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.action_label)

        panel.setLayout(layout)
        return panel

    def _on_connect_clicked(self):
        if self._connected:
            self.disconnect_requested.emit()
            return
        ip = (self.ip_edit.text() or "").strip() or "192.168.1.64"
        user = (self.user_edit.text() or "").strip() or "admin"
        pass_ = self.pass_edit.text() or ""
        self.connect_requested.emit(ip, user, pass_)

    def _on_speed_changed(self, v: int):
        self._zoom_speed = v
        self.speed_label.setText(str(v))

    def _on_zoom_in(self):
        self.zoom_in_requested.emit(self._zoom_speed)
        self._set_action("Zooming In")

    def _on_zoom_out(self):
        self.zoom_out_requested.emit(self._zoom_speed)
        self._set_action("Zooming Out")

    def _on_zoom_stop(self):
        self.zoom_stop_requested.emit()
        self._set_action("None")

    def _set_action(self, text: str):
        self.action_label.setText(f"Action: {text}")

    def update_connection_status(self, connected: bool, message: str = ""):
        self._connected = connected
        if connected:
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.connect_btn.setText("Disconnect")
            for b in (self.zoom_in_btn, self.zoom_out_btn, self.stop_btn):
                b.setEnabled(True)
        else:
            self.status_label.setText("Status: Disconnected" + (f" — {message}" if message else ""))
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.connect_btn.setText("Connect")
            for b in (self.zoom_in_btn, self.zoom_out_btn, self.stop_btn):
                b.setEnabled(False)
        if message and not connected:
            logger.warning("Zoom connection: %s", message)

    def update_action(self, text: str):
        self.action_label.setText(f"Action: {text}")

    def get_speed(self) -> int:
        return self._zoom_speed
