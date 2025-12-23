"""
MinIO Client Module for GIRA AI
Handles file uploads, downloads, and bucket management.
"""

import os
import tempfile
from typing import Optional, List
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)

class MinioClientWrapper:
    """Wrapper for MinIO client operations"""
    
    def __init__(self, 
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 secure: bool = False,
                 documents_bucket: str = "government-policies",
                 highlighted_bucket: str = "highlighted-policies"):
        
        self.client = None
        self.documents_bucket = documents_bucket
        self.highlighted_bucket = highlighted_bucket
        
        if endpoint and access_key and secret_key:
            try:
                self.client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure
                )
                logger.info(f"MinIO client initialized for endpoint: {endpoint}")
                self._ensure_bucket_exists(highlighted_bucket)
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {e}")
                self.client = None
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure MinIO bucket exists, create if it doesn't."""
        if not self.client: return
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created MinIO bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Error checking/creating bucket {bucket_name}: {e}")

    def upload_file(self, local_path: str, object_name: str, bucket: str = None) -> Optional[str]:
        """Upload a file to MinIO and return the object URL."""
        if not self.client:
            return None
        
        bucket_name = bucket or self.highlighted_bucket
        
        try:
            self.client.fput_object(bucket_name, object_name, local_path)
            minio_url = f"minio://{bucket_name}/{object_name}"
            logger.info(f"Uploaded to MinIO: {minio_url}")
            return minio_url
        except Exception as e:
            logger.error(f"Failed to upload to MinIO: {e}")
            return None

    def download_file(self, object_name: str, bucket: str = None) -> tempfile.NamedTemporaryFile:
        """Download file from MinIO to a temporary file."""
        if not self.client:
            raise ValueError("MinIO client not initialized")
        
        bucket_name = bucket or self.documents_bucket
        
        try:
            temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            self.client.fget_object(bucket_name, object_name, temp.name)
            logger.info(f"Downloaded {object_name} from bucket {bucket_name}")
            return temp
        except S3Error as e:
            raise ValueError(f"Failed to download from MinIO: {e}")

    def delete_file(self, object_name: str, bucket: str = None):
        """Delete a file from MinIO."""
        if not self.client: return
        
        bucket_name = bucket or self.highlighted_bucket
        
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"Deleted from MinIO: {bucket_name}/{object_name}")
        except Exception as e:
            logger.error(f"Failed to delete from MinIO: {e}")
            
    def bucket_exists(self, bucket_name: str) -> bool:
        if not self.client: return False
        try:
            return self.client.bucket_exists(bucket_name)
        except:
            return False
