"""
CORS (Cross-Origin Resource Sharing) middleware
Configures allowed origins, methods, and headers
"""
from typing import List
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def get_cors_config() -> dict:
    """
    Get CORS configuration based on environment
    
    Returns:
        Dictionary with CORS settings for FastAPI CORSMiddleware
    """
    origins = settings.cors_origins
    
    logger.info(f"CORS configured for environment: {settings.ENVIRONMENT}")
    logger.info(f"Allowed origins: {origins}")
    
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "User-Agent",
            "DNT",
            "Cache-Control",
            "X-Mx-ReqToken",
            "Keep-Alive",
            "X-Requested-With",
            "X-User-ID",
            "X-Page-ID",
            "X-Country",
            "X-Page-Title",
            "X-Page-URL",
        ],
        "expose_headers": ["*"],
    }


def validate_origin(origin: str) -> bool:
    """
    Validate if origin is allowed
    
    Args:
        origin: Origin header value
        
    Returns:
        True if allowed, False otherwise
    """
    allowed_origins = settings.cors_origins
    
    # Check exact match
    if origin in allowed_origins:
        return True
    
    # Check wildcard patterns (if needed in future)
    # TODO: Implement pattern matching for production
    
    logger.warning(f"Origin not allowed: {origin}")
    return False