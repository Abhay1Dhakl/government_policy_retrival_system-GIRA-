"""
File Cleanup Module for GIRA AI
Handles automatic deletion of temporary files and MinIO objects.
"""

import os
import threading
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class cleanup_manager:
    """Manages scheduled cleanup of files"""
    
    def __init__(self, cleanup_delay: int = 3600):
        self.cleanup_delay = cleanup_delay
        self.cleanup_timers: Dict[str, threading.Timer] = {}
        
    def schedule_cleanup(self, file_path: str, minio_client: Any, minio_object: str = None, delay: int = None):
        """Schedule automatic file cleanup after specified delay."""
        if delay is None:
            delay = self.cleanup_delay
            
        def _cleanup_task():
            try:
                # Clean up local file
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Auto-deleted local file: {file_path}")
                
                # Clean up MinIO file
                if minio_object and minio_client:
                    minio_client.delete_file(minio_object)
                
                # Remove from tracking
                cleanup_key = self._get_key(file_path, minio_object)
                if cleanup_key in self.cleanup_timers:
                    del self.cleanup_timers[cleanup_key]
                    
            except Exception as e:
                logger.error(f"Error during auto-cleanup of {file_path}: {e}")
        
        # Cancel existing timer if any
        cleanup_key = self._get_key(file_path, minio_object)
        self.cancel_cleanup(file_path, minio_object)
        
        # Schedule new
        timer = threading.Timer(delay, _cleanup_task)
        timer.start()
        self.cleanup_timers[cleanup_key] = timer
        logger.info(f"Scheduled cleanup for {file_path} in {delay}s")

    def cancel_cleanup(self, file_path: str, minio_object: str = None):
        """Cancel scheduled cleanup for a specific file."""
        key = self._get_key(file_path, minio_object)
        if key in self.cleanup_timers:
            self.cleanup_timers[key].cancel()
            del self.cleanup_timers[key]
            logger.info(f"Cancelled cleanup for {file_path}")

    def _get_key(self, file_path: str, minio_object: str) -> str:
        return f"{file_path}|{minio_object}" if minio_object else file_path
