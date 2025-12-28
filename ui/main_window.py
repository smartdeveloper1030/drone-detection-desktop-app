"""
Main window with tabs for Operator View and System View.
"""
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QStatusBar
from PyQt5.QtCore import Qt
from ui.operator_view import OperatorView
from ui.system_view import SystemView


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Drone Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create views
        self.operator_view = OperatorView()
        self.system_view = SystemView()
        
        # Add tabs
        self.tab_widget.addTab(self.operator_view, "Operator View")
        self.tab_widget.addTab(self.system_view, "System View")
        
        # Set central widget
        self.setCentralWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3b3b3b;
                color: white;
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #555;
            }
            QTabBar::tab:hover {
                background-color: #444;
            }
        """)
    
    def get_operator_view(self) -> OperatorView:
        """Get the operator view."""
        return self.operator_view
    
    def get_system_view(self) -> SystemView:
        """Get the system view."""
        return self.system_view

