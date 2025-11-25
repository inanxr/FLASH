"""
COLMAP Worker for Background Processing

Runs COLMAP processing in a background thread to keep UI responsive.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColmapWorker(QThread):
    """Background worker for COLMAP processing."""
    
    progress_update = pyqtSignal(str)  # status message
    processing_complete = pyqtSignal(str, str)  # (dataset_name, output_dir)
    processing_error = pyqtSignal(str)  # error message
    
    def __init__(self, image_folder, output_dir, dataset_name):
        super().__init__()
        self.image_folder = image_folder
        self.output_dir = output_dir
        self.dataset_name = dataset_name
    
    def run(self):
        """Run COLMAP processing."""
        try:
            from utils.colmap_processor import ColmapProcessor
            
            processor = ColmapProcessor()
            
            # Process images
            success, message = processor.process_images(
                self.image_folder,
                self.output_dir,
                self.progress_update.emit
            )
            
            if success:
                self.processing_complete.emit(self.dataset_name, self.output_dir)
            else:
                self.processing_error.emit(message)
                
        except Exception as e:
            self.processing_error.emit(f"Processing failed: {str(e)}")
