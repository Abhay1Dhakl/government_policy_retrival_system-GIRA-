import logging
import os

logger = logging.getLogger(__name__)

# Cache the debug flag to avoid repeated environment lookups
DEBUG_DOCUMENTS = os.getenv('DEBUG_DOCUMENTS', 'false').lower() == 'true'

class DocumentDebugMiddleware:
    """
    Optimized middleware for document request tracking.
    Only logs when DEBUG_DOCUMENTS=true in environment.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Pre-compile path patterns for faster lookup
        self.document_paths = frozenset([
            '/documents/',
            '/api/documents/',
            '/admin/documents/',
            '/admin-ui/documents/',
            '/document/',
            '/api/document/'
        ])

    def __call__(self, request):
        # Early exit if debug logging is disabled
        if not DEBUG_DOCUMENTS:
            return self.get_response(request)
        
        # Fast path check using startswith for better performance
        is_document_request = self._is_document_request_fast(request.path)
        
        if is_document_request:
            self._log_request_details(request)
        
        response = self.get_response(request)
        
        if is_document_request:
            self._log_response_details(request, response)
        
        return response
    
    def _is_document_request_fast(self, path):
        """Optimized document path checking using startswith"""
        return any(path.startswith(doc_path) for doc_path in self.document_paths)
    
    def _log_request_details(self, request):
        """Optimized request logging with minimal overhead"""
        # Use single log entry with structured data
        log_data = {
            'method': request.method,
            'path': request.path,
            'content_type': getattr(request, 'content_type', None)
        }
        
        # Only log files if present (avoid empty checks)
        if request.FILES:
            log_data['files'] = {
                key: {
                    'name': getattr(file_obj, 'name', 'unknown'),
                    'size': getattr(file_obj, 'size', 0)
                }
                for key, file_obj in request.FILES.items()
            }
        
        # Only log non-empty POST data, exclude sensitive fields
        if request.method == 'POST' and request.POST:
            sensitive_fields = {'password', 'secret', 'token', 'key'}
            log_data['post_data'] = {
                k: '[REDACTED]' if k.lower() in sensitive_fields else v
                for k, v in request.POST.items()
            }
        
        logger.info(f"Document Request: {log_data}")
    
    def _log_response_details(self, request, response):
        """Optimized response logging"""
        status_code = response.status_code
        
        # Single log entry for successful responses
        if status_code < 400:
            logger.info(f"Document Response: {request.method} {request.path} -> {status_code}")
        else:
            # Detailed logging only for errors
            error_data = {
                'method': request.method,
                'path': request.path,
                'status': status_code
            }
            
            # Only decode content for errors and limit size
            if hasattr(response, 'content') and response.content:
                try:
                    error_data['content'] = response.content.decode('utf-8')[:200]
                except UnicodeDecodeError:
                    error_data['content'] = '[Binary content]'
            
            logger.error(f"Document request failed: {error_data}")