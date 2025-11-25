"""
Training Tab - Minimal Glassmorphism Design
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QComboBox, QPushButton,
    QGridLayout, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
import pyqtgraph as pg
import numpy as np
import qtawesome as qta


class TrainingTab(QWidget):
    """Minimal training interface."""
    
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.training_worker = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup minimal UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Top: Controls (single row)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._create_dataset_group(), 1)
        top_layout.addWidget(self._create_params_group(), 2)
        layout.addLayout(top_layout)
        
        # Middle: Training controls
        layout.addWidget(self._create_control_group())
        
        # Bottom: Monitoring
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._create_loss_plot(), 1)
        bottom_layout.addWidget(self._create_preview(), 1)
        layout.addLayout(bottom_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.5);
            font-size: 11px;
            padding: 8px 0;
        """)
        layout.addWidget(self.status_label)
    
    def _create_dataset_group(self):
        """Dataset settings."""
        group = QGroupBox("Dataset")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["lego", "ship", "drums", "ficus", "hotdog"])
        layout.addWidget(self.dataset_combo)
        
        # Load custom dataset button
        custom_btn = QPushButton(" Load Custom Dataset")
        custom_btn.setIcon(qta.icon('fa5s.folder-open', color='white'))
        custom_btn.clicked.connect(self.load_custom_dataset)
        custom_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 200, 255, 0.2);
                border: 1px solid rgba(100, 200, 255, 0.4);
            }
            QPushButton:hover {
                background: rgba(100, 200, 255, 0.3);
            }
        """)
        layout.addWidget(custom_btn)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(100, 800)
        self.img_size_spin.setValue(400)
        self.img_size_spin.setSuffix(" px")
        layout.addWidget(self.img_size_spin)
        
        self.quick_test_check = QCheckBox("Quick Test")
        layout.addWidget(self.quick_test_check)
        
        layout.addStretch()
        return group
    
    def _create_params_group(self):
        """Training parameters."""
        group = QGroupBox("Parameters")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(1, 1)
        
        # Iterations
        layout.addWidget(QLabel("Iterations"), 0, 0)
        self.iters_slider = self._create_slider(1000, 20000, 5000)
        self.iters_value = QLabel("5000")
        self.iters_value.setStyleSheet("color: rgba(255, 255, 255, 0.7); min-width: 50px;")
        self.iters_slider.valueChanged.connect(lambda v: self.iters_value.setText(str(v)))
        layout.addWidget(self.iters_slider, 0, 1)
        layout.addWidget(self.iters_value, 0, 2)
        
        # Batch size
        layout.addWidget(QLabel("Batch Size"), 1, 0)
        self.batch_slider = self._create_slider(4096, 32768, 16384, 4096)
        self.batch_value = QLabel("16384")
        self.batch_value.setStyleSheet("color: rgba(255, 255, 255, 0.7); min-width: 50px;")
        self.batch_slider.valueChanged.connect(lambda v: self.batch_value.setText(str(v)))
        layout.addWidget(self.batch_slider, 1, 1)
        layout.addWidget(self.batch_value, 1, 2)
        
        # Hash levels
        layout.addWidget(QLabel("Hash Levels"), 2, 0)
        self.levels_slider = self._create_slider(12, 24, 20)
        self.levels_value = QLabel("20")
        self.levels_value.setStyleSheet("color: rgba(255, 255, 255, 0.7); min-width: 50px;")
        self.levels_slider.valueChanged.connect(lambda v: self.levels_value.setText(str(v)))
        layout.addWidget(self.levels_slider, 2, 1)
        layout.addWidget(self.levels_value, 2, 2)
        
        return group
    
    def _create_slider(self, min_val, max_val, default, step=1):
        """Create slider."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        return slider
    
    def _create_control_group(self):
        """Control buttons."""
        group = QGroupBox("Controls")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)
        
        self.start_btn = QPushButton(" Start Training")
        self.start_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.start_btn.setMinimumHeight(44)
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.setIcon(qta.icon('fa5s.stop', color='white'))
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        layout.addWidget(self.start_btn, 3)
        layout.addWidget(self.stop_btn, 1)
        
        return group
    
    def _create_loss_plot(self):
        """Loss plot."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((20, 20, 20, 150))
        
        # Style the plot
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=(255,255,255,128), width=1))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=(255,255,255,128), width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=(255,255,255,180)))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=(255,255,255,180)))
        
        self.plot_widget.setLabel('left', 'Loss', color='#ffffff', **{'font-size': '11pt'})
        self.plot_widget.setLabel('bottom', 'Iteration', color='#ffffff', **{'font-size': '11pt'})
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # Create curve with bright color
        self.loss_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#00D9FF', width=3),
            name='Loss',
            antialias=True
        )
        
        self.iterations = []
        self.losses = []
        
        layout.addWidget(self.plot_widget)
        
        # Stats
        stats = QHBoxLayout()
        self.loss_label = QLabel("Loss: --")
        self.loss_label.setStyleSheet("color: #00D9FF; font-size: 13px; font-weight: 600;")
        self.psnr_label = QLabel("PSNR: --")
        self.psnr_label.setStyleSheet("color: #00FF88; font-size: 13px; font-weight: 600;")
        self.iter_label = QLabel("Iteration: 0/0")
        self.iter_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 11px;")
        stats.addWidget(self.loss_label)
        stats.addWidget(self.psnr_label)
        stats.addStretch()
        stats.addWidget(self.iter_label)
        layout.addLayout(stats)
        
        return group
    
    def _create_preview(self):
        """Preview display."""
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        
        self.preview_widget = pg.ImageView()
        self.preview_widget.ui.roiBtn.hide()
        self.preview_widget.ui.menuBtn.hide()
        
        layout.addWidget(self.preview_widget)
        
        placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
        placeholder[180:220, 180:220] = [40, 40, 40]
        self.preview_widget.setImage(placeholder)
        
        return group
    
    def start_training(self):
        """Start training."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.iterations.clear()
        self.losses.clear()
        
        self.status_label.setText("Initializing...")
        
        config_dict = {
            'dataset': self.dataset_combo.currentText(),
            'img_size': self.img_size_spin.value(),
            'iterations': self.iters_slider.value(),
            'batch_size': self.batch_slider.value(),
            'num_levels': self.levels_slider.value(),
            'features_per_level': 4,
            'num_samples': 128,
            'quick_test': self.quick_test_check.isChecked()
        }
        
        try:
            from workers.training_worker import TrainingWorker
            self.training_worker = TrainingWorker(config_dict)
            
            self.training_worker.progress_update.connect(self.update_progress)
            self.training_worker.preview_update.connect(self.update_preview)
            self.training_worker.status_update.connect(
                lambda msg: self.status_label.setText(msg)
            )
            self.training_worker.training_complete.connect(self._on_complete)
            self.training_worker.training_error.connect(self._on_error)
            
            self.training_worker.start()
            self.training_started.emit()
        except Exception as e:
            self._on_error(str(e))
    
    def stop_training(self):
        """Stop training."""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        self.training_stopped.emit()
    
    def _on_complete(self):
        """Handle completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Complete")
    
    def _on_error(self, error_msg):
        """Handle error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {error_msg}")
        
        QMessageBox.critical(self, "Training Error", f"Error:\n\n{error_msg}")
    
    @pyqtSlot(int, float, float)
    def update_progress(self, iteration, loss, psnr):
        """Update progress."""
        self.iterations.append(iteration)
        self.losses.append(loss)
        
        self.loss_curve.setData(self.iterations, self.losses)
        
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.psnr_label.setText(f"PSNR: {psnr:.2f} dB")
        max_iters = self.iters_slider.value()
        self.iter_label.setText(f"{iteration}/{max_iters}")
        self.status_label.setText(f"Training... {iteration}/{max_iters}")
    
    @pyqtSlot(np.ndarray)
    def update_preview(self, image):
        """Update preview."""
        self.preview_widget.setImage(image.transpose(1, 0, 2))
    
    def load_custom_dataset(self):
        """Load and process custom dataset from photos."""
        from PyQt6.QtWidgets import QFileDialog, QProgressDialog
        import os
        
        # Select folder
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Photos",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return
        
        # Validate images
        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(image_files) < 10:
            QMessageBox.warning(
                self,
                "Not Enough Images",
                f"Found only {len(image_files)} images.\nNeed at least 10-20 images for good results.\n\nTips:\n- Walk around the object\n- Take photos from different angles\n- Keep good overlap between views"
            )
            return
        
        # Get dataset name
        dataset_name = os.path.basename(folder)
        output_dir = os.path.join("data", "custom", dataset_name)
        
        # Create progress dialog
        progress = QProgressDialog(
            "Processing images with COLMAP...",
            "Cancel",
            0, 0,
            self
        )
        progress.setWindowTitle("Creating 3D Dataset")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Create and start COLMAP worker
        from workers.colmap_worker import ColmapWorker
        self.colmap_worker = ColmapWorker(folder, output_dir, dataset_name)
        
        # Connect signals
        self.colmap_worker.progress_update.connect(
            lambda msg: progress.setLabelText(f"Processing: {msg}")
        )
        self.colmap_worker.processing_complete.connect(
            lambda name, path: self._on_colmap_complete(name, path, progress)
        )
        self.colmap_worker.processing_error.connect(
            lambda err: self._on_colmap_error(err, progress)
        )
        
        # Start
        self.colmap_worker.start()
        
        # Show progress
        progress.exec()
    
    def _on_colmap_complete(self, dataset_name, output_dir, progress):
        """Handle successful COLMAP processing."""
        progress.close()
        
        # Add to dropdown
        if self.dataset_combo.findText(dataset_name) == -1:
            self.dataset_combo.addItem(dataset_name)
        
        # Select it
        self.dataset_combo.setCurrentText(dataset_name)
        
        QMessageBox.information(
            self,
            "Dataset Created!",
            f"Successfully processed dataset: {dataset_name}\n\n"
            f"Saved to: {output_dir}\n\n"
            f"Ready to train! Click 'Start Training' to begin."
        )
    
    def _on_colmap_error(self, error_msg, progress):
        """Handle COLMAP processing error."""
        progress.close()
        
        QMessageBox.critical(
            self,
            "Processing Failed",
            f"Failed to process images:\n\n{error_msg}\n\n"
            f"Tips:\n"
            f"- Install pycolmap: pip install pycolmap\n"
            f"- Take more photos (20-100+)\n"
            f"- Ensure good overlap between views\n"
            f"- Use consistent lighting"
        )
