"""
Main window with side-by-side Operator View and System View.
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QStatusBar, QComboBox, QLabel, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.operator_view import OperatorView
from ui.system_view import SystemView
from config import Config


class MainWindow(QMainWindow):
    """Main application window."""
    
    # Signal emitted when detection mode changes
    mode_changed = pyqtSignal(str)  # Emits "balloon", "drone", or "person"
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Drone Detection System")
        self.setGeometry(100, 100, 1200, 500)  # Wider window for side-by-side layout
        
        # Create central widget with horizontal layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create toolbar with mode selector
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        mode_label = QLabel("Detection Mode:")
        mode_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        toolbar_layout.addWidget(mode_label)
        
        # Mode selector (balloon/drone/person)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Drone", "Balloon", "Person"])
        self.mode_combo.setCurrentText(Config.DETECT_MODE.capitalize())
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 120px;
                font-size: 12px;
            }
            QComboBox:hover {
                background-color: #444;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #3b3b3b;
                color: white;
                selection-background-color: #555;
                border: 1px solid #555;
            }
        """)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar_layout.addWidget(self.mode_combo)
        
        toolbar_layout.addStretch()  # Push mode selector to the left
        
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for side-by-side views
        splitter = QSplitter(Qt.Horizontal)
        
        # Create views
        self.operator_view = OperatorView()
        self.system_view = SystemView()
        
        # Add views to splitter (left: operator, right: system)
        splitter.addWidget(self.operator_view)
        splitter.addWidget(self.system_view)
        
        # Set splitter proportions (65% operator, 35% system)
        splitter.setStretchFactor(0, 13)  # Operator view: 65% (13/20)
        splitter.setStretchFactor(1, 7)   # System view: 35% (7/20)
        # Initial sizes: 65% and 35% of 1200px window width
        splitter.setSizes([780, 420])     # 65% = 780px, 35% = 420px
        
        # Set splitter style
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #666;
            }
        """)
        
        main_layout.addWidget(splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        # Set status bar text color to green
        self.statusBar().setStyleSheet("""
            QStatusBar {
                color: #00aa00;
                background-color: #2b2b2b;
            }
            QStatusBar::item {
                color: #00ff00;
            }
        """)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b00;
            }
            QWidget {
                background-color: #2b2b00;
            }
        """)
    
    def _on_mode_changed(self, mode_text: str):
        """Handle mode selection change."""
        mode = mode_text.lower()
        self.mode_changed.emit(mode)
    
    def get_current_mode(self) -> str:
        """Get current detection mode."""
        return self.mode_combo.currentText().lower()
    
    def set_mode(self, mode: str):
        """Set detection mode programmatically."""
        mode_capitalized = mode.capitalize()
        if mode_capitalized in ["Drone", "Balloon", "Person"]:
            self.mode_combo.setCurrentText(mode_capitalized)
    
    def get_operator_view(self) -> OperatorView:
        """Get the operator view."""
        return self.operator_view
    
    def get_system_view(self) -> SystemView:
        """Get the system view."""
        return self.system_view

