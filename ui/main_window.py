"""
Main window with camera view on left and tabs on right.
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QStatusBar, QComboBox, QLabel, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.camera_view import CameraView
from ui.system_view import SystemView
from ui.ptu_control_view import PTUControlView
from config import Config


class MainWindow(QMainWindow):
    """Main application window."""
    
    # Signal emitted when detection mode changes
    mode_changed = pyqtSignal(str)  # Emits "balloon", "drone", or "person"
    
    # Signal emitted when color selection changes
    color_changed = pyqtSignal(str)  # Emits color name (e.g., "red", "blue")
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Drone Detection System")
        self.setGeometry(100, 100, 1600, 900)  # Larger window for new layout
        
        # Disable animations for lower latency if configured
        if Config.UI_DISABLE_ANIMATIONS:
            # Disable window animations and effects
            self.setAttribute(Qt.WA_TranslucentBackground, False)
            self.setUpdatesEnabled(True)  # Keep updates enabled but disable animations
        
        # Create central widget with horizontal layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create toolbar with mode selector
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        mode_label = QLabel("Detection Mode:")
        mode_label.setProperty("class", "mode-label")
        toolbar_layout.addWidget(mode_label)
        
        # Mode selector (balloon/drone/person)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Drone", "Balloon", "Person"])
        self.mode_combo.setCurrentText(Config.DETECT_MODE.capitalize())
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar_layout.addWidget(self.mode_combo)
        
        # Color selector (only visible when Balloon mode is selected)
        color_label = QLabel("Color:")
        color_label.setProperty("class", "color-label")
        self.color_label = color_label
        toolbar_layout.addWidget(color_label)
        
        self.color_combo = QComboBox()
        # Available colors from ColorClassifier, with "All" option first
        self.color_combo.addItems(["All", "red", "white", "green", "blue", "black", "orange", "yellow"])
        self.color_combo.setCurrentText("red")  # Default to red
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        toolbar_layout.addWidget(self.color_combo)
        
        # Initially hide color selector if not in balloon mode
        self._update_color_selector_visibility()
        
        toolbar_layout.addStretch()  # Push mode selector to the left
        
        main_layout.addLayout(toolbar_layout)
        
        # Create main horizontal splitter: Left (camera) | Right (tabs)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Camera view
        self.camera_view = CameraView()
        
        # Right side: Tab widget
        self.tab_widget = QTabWidget()
        
        # Create views
        self.system_view = SystemView()
        self.ptu_control_view = PTUControlView()
        
        # Add tabs
        self.tab_widget.addTab(self.system_view, "Status")
        self.tab_widget.addTab(self.ptu_control_view, "PTU")
        
        # Add to splitter: Left (camera) | Right (tabs)
        main_splitter.addWidget(self.camera_view)
        main_splitter.addWidget(self.tab_widget)
        
        # Set minimum width for tab widget to prevent it from being too narrow
        self.tab_widget.setMinimumWidth(400)
        
        # Set splitter proportions (70% camera, 30% tabs)
        # First set initial sizes explicitly to ensure correct initial split
        # Calculate based on window width minus margins (approximately 1600px - 20px margins = 1580px usable)
        window_width = 1600
        camera_width = int(window_width * 0.70)  # 70% = 1120px
        tabs_width = int(window_width * 0.30)    # 30% = 480px
        main_splitter.setSizes([camera_width, tabs_width])
        
        # Then set stretch factors for resizing behavior
        main_splitter.setStretchFactor(0, 7)  # Camera: 70%
        main_splitter.setStretchFactor(1, 3)  # Tabs: 30%
        
        main_layout.addWidget(main_splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Ensure sizes are applied after window is shown (use showEvent override)
        self.main_splitter = main_splitter
        self.camera_width = camera_width
        self.tabs_width = tabs_width
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _on_mode_changed(self, mode_text: str):
        """Handle mode selection change."""
        mode = mode_text.lower()
        self.mode_changed.emit(mode)
        self._update_color_selector_visibility()
    
    def _update_color_selector_visibility(self):
        """Show/hide color selector based on current mode."""
        is_balloon_mode = self.mode_combo.currentText().lower() == "balloon"
        self.color_label.setVisible(is_balloon_mode)
        self.color_combo.setVisible(is_balloon_mode)
    
    def _on_color_changed(self, color: str):
        """Handle color selection change."""
        self.color_changed.emit(color.lower())
    
    def get_selected_color(self) -> str:
        """Get currently selected color."""
        return self.color_combo.currentText().lower()
    
    def set_color(self, color: str):
        """Set color programmatically."""
        color_lower = color.lower()
        if color_lower == "all":
            self.color_combo.setCurrentText("All")
        else:
            color_capitalized = color.capitalize()
            if color_capitalized in ["Red", "White", "Green", "Blue", "Black", "Orange", "Yellow"]:
                self.color_combo.setCurrentText(color_capitalized)
    
    def get_current_mode(self) -> str:
        """Get current detection mode."""
        return self.mode_combo.currentText().lower()
    
    def set_mode(self, mode: str):
        """Set detection mode programmatically."""
        mode_capitalized = mode.capitalize()
        if mode_capitalized in ["Drone", "Balloon", "Person"]:
            self.mode_combo.setCurrentText(mode_capitalized)
    
    def get_operator_view(self) -> CameraView:
        """Get the camera view (backward compatibility - returns camera_view)."""
        return self.camera_view
    
    def get_camera_view(self) -> CameraView:
        """Get the camera view."""
        return self.camera_view
    
    def get_system_view(self) -> SystemView:
        """Get the system view."""
        return self.system_view
    
    def get_ptu_control_view(self) -> PTUControlView:
        """Get the PTU control view."""
        return self.ptu_control_view
    
    def showEvent(self, event):
        """Override showEvent to set splitter sizes after window is shown."""
        super().showEvent(event)
        # Set splitter sizes after window is visible to ensure correct proportions
        if hasattr(self, 'main_splitter'):
            self.main_splitter.setSizes([self.camera_width, self.tabs_width])

