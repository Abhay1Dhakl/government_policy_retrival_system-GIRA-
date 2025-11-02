"""
Request/Response logging middleware
Logs all HTTP requests with timing information
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from logging_config import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses
    Includes timing information and request/response details
    """
    
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"→ {request.method} {request.url.path}")
        logger.debug(f"Headers: {dict(request.headers)}")
        
        # Get request body for POST/PUT (be careful with large payloads)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) < 1000:  # Only log small bodies
                    logger.debug(f"Body: {body.decode()[:500]}")
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"← {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {duration:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"✗ {request.method} {request.url.path} "
                f"Failed after {duration:.3f}s: {e}",
                exc_info=True
            )
            raise


class ResponseTimingMiddleware(BaseHTTPMiddleware):
    """
    Simple middleware to add timing headers to responses
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        return response