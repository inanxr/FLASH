"""
Main Window for FLASH Studio - Minimal Glassmorphism Design
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QMenu, QStatusBar, QToolBar,
    QMessageBox, QLabel
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

from .training_tab import TrainingTab


class MainWindow(QMainWindow):
    """Main application window with glassmorphism design."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLASH Studio")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
        self._apply_glassmorphism_theme()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        
        # Add tabs
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Training")
        
        # Placeholder tabs
        viewer_placeholder = QWidget()
        export_placeholder = QWidget()
        self.tabs.addTab(viewer_placeholder, "Viewer")
        self.tabs.addTab(export_placeholder, "Export")
        
        layout.addWidget(self.tabs)
    
    def _create_header(self):
        """Create minimal header."""
        header = QWidget()
        header.setFixedHeight(60)
        header.setObjectName("header")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)
        
        # Logo/Title
        title = QLabel("FLASH")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 2px;
        """)
        layout.addWidget(title)
        
        subtitle = QLabel("Fast Learning for Accurate Scene Hashing")
        subtitle.setStyleSheet("""
            font-size: 11px;
            color: rgba(255, 255, 255, 0.5);
            margin-left: 16px;
        """)
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        return header
    
    def _create_menu_bar(self):
        """Create minimal menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About FLASH", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    
    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Update periodically
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(2000)
    
    def _update_status(self):
        """Update status bar."""
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.status_bar.showMessage(f"Memory: {memory_mb:.0f} MB")
    
    def _apply_glassmorphism_theme(self):
        """Apply minimal glassmorphism theme."""
        self.setStyleSheet("""
        QMainWindow {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0a0a0a,
                stop:1 #1a1a1a
            );
        }
        
        QWidget#header {
            background: rgba(30, 30, 30, 0.7);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        QTabWidget::pane {
            border: none;
            background: transparent;
        }
        
        QTabBar::tab {
            background: rgba(40, 40, 40, 0.3);
            color: rgba(255, 255, 255, 0.6);
            padding: 12px 24px;
            margin-right: 4px;
            border: none;
            border-radius: 8px 8px 0 0;
            font-size: 13px;
            font-weight: 500;
        }
        
        QTabBar::tab:selected {
            background: rgba(60, 60, 60, 0.5);
            color: #ffffff;
            backdrop-filter: blur(10px);
        }
        
        QTabBar::tab:hover {
            background: rgba(50, 50, 50, 0.4);
        }
        
        QMenuBar {
            background: transparent;
            color: rgba(255, 255, 255, 0.9);
            padding: 4px;
        }
        
        QMenuBar::item:selected {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        QMenu {
            background: rgba(30, 30, 30, 0.95);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px;
        }
        
        QMenu::item {
            padding: 8px 24px;
            border-radius: 4px;
        }
        
        QMenu::item:selected {
            background: rgba(255, 255, 255, 0.1);
        }
        
        QToolBar {
            background: rgba(30, 30, 30, 0.5);
            border: none;
            spacing: 12px;
            padding: 8px 16px;
        }
        
        QToolBar QAction {
            color: #ffffff;
        }
        
        QStatusBar {
            background: rgba(20, 20, 20, 0.8);
            color: rgba(255, 255, 255, 0.6);
            font-size: 11px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        QPushButton {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
        }
        
        QPushButton:hover {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        QPushButton:pressed {
            background: rgba(255, 255, 255, 0.05);
        }
        
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: rgba(255, 255, 255, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        QLabel {
            color: rgba(255, 255, 255, 0.9);
        }
        
        QGroupBox {
            background: rgba(40, 40, 40, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            margin-top: 12px;
            padding-top: 16px;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.9);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px;
            color: rgba(255, 255, 255, 0.7);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        QComboBox, QSpinBox {
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 12px;
            border-radius: 6px;
        }
        
        QComboBox:hover, QSpinBox:hover {
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        QSlider::groove:horizontal {
            background: rgba(255, 255, 255, 0.1);
            height: 4px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff,
                stop:1 #cccccc
            );
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        QSlider::handle:horizontal:hover {
            background: #ffffff;
        }
        
        QCheckBox {
            color: rgba(255, 255, 255, 0.8);
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.05);
        }
        
        QCheckBox::indicator:hover {
            border: 2px solid rgba(255, 255, 255, 0.5);
            background: rgba(255, 255, 255, 0.1);
        }
        
        QCheckBox::indicator:checked {
            background: rgba(100, 200, 255, 0.6);
            border: 2px solid rgba(100, 200, 255, 0.8);
            image: url(none);
        }
        
        QCheckBox::indicator:checked::after {
            content: "✓";
            color: white;
        }
        """)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FLASH Studio",
            "<h2>FLASH Studio v1.0</h2>"
            "<p><b>Fast Learning for Accurate Scene Hashing</b></p>"
            "<p>High-performance NeRF training and visualization</p>"
            "<p>Built with PyQt6 and Instant-NGP</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        if hasattr(self.training_tab, 'training_worker'):
            if self.training_tab.training_worker and self.training_tab.training_worker.isRunning():
                reply = QMessageBox.question(
                    self,
                    "Training in Progress",
                    "Training is running. Stop and exit?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.training_tab.stop_training()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()
        else:
            event.accept()
