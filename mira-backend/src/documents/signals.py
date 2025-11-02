import json
import os
import requests
import logging
import threading
import time
import tempfile
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Document

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

logger = logging.getLogger(__name__)

@receiver(post_save, sender=Document)
def document_created_handler(sender, instance, created, **kwargs):
    """
    Signal handler that triggers when a Document is created.
    This only calls ingest API for documents that weren't processed through the serializer
    (i.e., documents created directly in the database without going through the API).
    """
    if created:  # Only for newly created documents
        logger.info(f"Document created via signal: {instance.id}")
        
        # Check if this document was created through the API (serializer handles ingestion)
        # vs created directly in DB (needs signal-based ingestion)
        # The serializer always sets minio_object_name, so we can use that as a flag
        
        # Add a delay to allow any ongoing serializer processing to complete
        def delayed_ingest():
            time.sleep(3)  # Wait 3 seconds for any serializer processing
            
            # Re-fetch the instance to get the latest state
            try:
                fresh_instance = Document.objects.get(id=instance.id)
                
                logger.info(f"Post-delay check for document {fresh_instance.id}: minio_object_name={fresh_instance.minio_object_name}, source_type={fresh_instance.source_type}")
                # If minio_object_name exists and source_type is upload, 
                # the serializer likely already handled ingestion
                if (fresh_instance.source_type == 'upload' and 
                    fresh_instance.minio_object_name and 
                    not fresh_instance.manual_text):
                    logger.info(f"Document {fresh_instance.id} appears to have been processed by serializer, skipping signal ingestion")
                    return
                
                # Otherwise, proceed with signal-based ingestion
                call_ingest_api_for_document(fresh_instance)
                
            except Document.DoesNotExist:
                logger.error(f"Document {instance.id} not found during delayed ingestion")
        
        # Run in background thread to avoid blocking the request
        thread = threading.Thread(target=delayed_ingest)
        thread.daemon = True
        thread.start()

def call_ingest_api_for_document(document):
    """
    Call the ingest API for a given document instance
    """
    ingest_base_url = os.getenv('INGEST_BASE_URL', 'http://mira-agent:8081')
    url = f"{ingest_base_url}/admin/documents/ingest/"
    
    # Build payload from document instance with proper validation
    source_type = (getattr(document, 'source_type', 'upload') or 'upload').lower()
    
    # Ensure source_type is valid
    if source_type not in ['api', 'upload', 'manual']:
        source_type = 'upload'
    
    payload = {
        "instance_id": getattr(document, 'instance_id', '') or str(document.id),
        "document_type": document.document_type or '',
        "source_type": source_type,
        "document_metadata": {
            "title": document.title or 'Untitled Document',
            "language": getattr(document, 'language', '') or 'en',
            "region": getattr(document, 'region', '') or 'US',
            "author": getattr(document, 'author', '') or 'Unknown',
            "tags": [tag.strip() for tag in getattr(document, 'tags', '').split(',') if tag.strip()] if getattr(document, 'tags', '') else [],
        },
        "api_connection_info": {
            "auth_type": getattr(document, 'auth_type', '') or '',
            "client_id": getattr(document, 'client_id', '') or '',
            "client_secret": getattr(document, 'client_secret', '') or '',
            "token_url": getattr(document, 'token_url', '') or '',
            "data_url": getattr(document, 'data_url', '') or '',
        },
        "manual_text": getattr(document, 'manual_text', '') or '',
        "document_id": str(document.id),
        "file_name": document.minio_object_name or f"{document.id}.pdf"
    }
    
    form_data = {"data": json.dumps(payload)}
    
    # Try to get the file if it exists
    files = None
    file_found = False
    
    logger.info(f"Starting file retrieval for document {document.id}, minio_object_name: {document.minio_object_name}")
    
    # DEBUG: Add print statement to see signal handler execution
    print(f"DEBUG SIGNALS: Processing document {document.id} with minio_object_name: {document.minio_object_name}")
    
    # Method 1: Check if document has a direct file field
    if hasattr(document, 'file') and document.file:
        try:
            file_path = document.file.path
            logger.info(f"Method 1: Checking direct file field at {file_path}")
            if os.path.exists(file_path):
                files = {"file_upload": open(file_path, 'rb')}
                file_found = True
                logger.info(f"Method 1: File found and opened from {file_path}")
        except Exception as e:
            logger.warning(f"Method 1: Failed to access direct file field: {e}")
    
    # Method 2: Check if document has related files
    if not file_found and hasattr(document, 'files') and document.files.exists():
        try:
            file_instance = document.files.first()
            if file_instance and file_instance.file:
                file_path = file_instance.file.path
                logger.info(f"Method 2: Checking related file at {file_path}")
                if os.path.exists(file_path):
                    files = {"file_upload": open(file_path, 'rb')}
                    file_found = True
                    logger.info(f"Method 2: File found and opened from {file_path}")
        except Exception as e:
            logger.warning(f"Method 2: Failed to access related file: {e}")
    
    # Method 3: Download file from MinIO temporarily
    temp_file_to_cleanup = None
    if not file_found and document.minio_object_name and MINIO_AVAILABLE:
        try:
            # MinIO client configuration - use same config as utils.py
            minio_client = Minio(
                os.getenv('MINIO_ENDPOINT', 'minio:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
                secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
                secure=os.getenv('MINIO_SECURE', 'False').lower() == 'true'
            )
            
            bucket_name = 'medical-documents'
            object_name = document.minio_object_name
            
            logger.info(f"Attempting to retrieve file from MinIO: bucket={bucket_name}, object={object_name}")
            
            # Check if object exists in MinIO
            try:
                stat_info = minio_client.stat_object(bucket_name, object_name)
                logger.info(f"File found in MinIO: {stat_info}")
                
                # Download to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                minio_client.fget_object(bucket_name, object_name, temp_file_path)
                files = {"file_upload": open(temp_file_path, 'rb')}
                file_found = True
                
                # Store temp file path for cleanup later
                temp_file_to_cleanup = temp_file_path
                logger.info(f"Successfully downloaded file from MinIO to {temp_file_path}")
                
            except Exception as e:
                logger.error(f"Failed to retrieve file from MinIO: {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
    
    # Method 4: Look for file in uploads directory using sequential numbering (fallback)
    if not file_found:
        upload_dir = "/app/uploads"
        logger.info(f"Method 4: Checking uploads directory at {upload_dir}")
        
        # Try to find file using the sequential filename stored in minio_object_name
        if document.minio_object_name:
            sequential_file_path = f"{upload_dir}/{document.minio_object_name}"
            logger.info(f"Method 4a: Checking sequential file path {sequential_file_path}")
            if os.path.exists(sequential_file_path):
                try:
                    files = {"file_upload": open(sequential_file_path, 'rb')}
                    file_found = True
                    logger.info(f"Method 4a: File found at {sequential_file_path}")
                except Exception as e:
                    logger.warning(f"Method 4a: Failed to open file {sequential_file_path}: {e}")
        
        # Fallback: Check common file extensions with just document ID
        if not file_found:
            logger.info(f"Method 4b: Checking fallback paths for document ID {document.id}")
            for ext in ['.pdf', '.docx', '.doc', '.txt']:
                potential_path = f"{upload_dir}/{document.id}{ext}"
                if os.path.exists(potential_path):
                    try:
                        files = {"file_upload": open(potential_path, 'rb')}
                        file_found = True
                        logger.info(f"Method 4b: File found at {potential_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Method 4b: Failed to open file {potential_path}: {e}")
    
    # If no file found and we have manual_text, change source_type to 'manual'
    if not file_found and payload.get("manual_text"):
        logger.info(f"No file found, but manual_text exists. Switching to manual source_type")
        payload["source_type"] = "manual"
        form_data = {"data": json.dumps(payload)}
    elif not file_found:
        logger.error(f"No file found for document {document.id} with any method. MinIO object: {document.minio_object_name}")
    else:
        logger.info(f"File successfully located for document {document.id}")
    
    try:
        response = requests.post(url, data=form_data, files=files, timeout=60)
        
        if response.status_code == 200:
            logger.info(f"Signal ingest API successful for document {document.id}")
        else:
            logger.error(f"Signal ingest API failed for document {document.id}: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Signal ingest API connection error: {e}")
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Signal ingest API timeout: {e}")
        
    except Exception as e:
        logger.error(f"Signal ingest API unexpected error: {e}")
    
    finally:
        # Close file if it was opened
        if files and 'file_upload' in files:
            try:
                files['file_upload'].close()
            except Exception:
                pass
        
        # Clean up temporary file if it was created
        if 'temp_file_to_cleanup' in locals() and temp_file_to_cleanup:
            try:
                os.unlink(temp_file_to_cleanup)
            except Exception:
                pass
