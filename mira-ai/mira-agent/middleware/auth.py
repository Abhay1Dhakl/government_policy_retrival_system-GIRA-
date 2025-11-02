"""
Authentication and authorization middleware
Handles JWT token decoding and user identification
"""
from typing import Optional, Tuple
from fastapi import Request
import jwt
from logging_config import get_logger

logger = get_logger(__name__)


def decode_user_id_from_header(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """
    Decode user ID and country from request headers
    
    Supports multiple authentication methods:
    1. Authorization Bearer token (JWT)
    2. X-User-ID header
    
    Args:
        request: FastAPI request object
        
    Returns:
        Tuple of (user_id, country) or (None, None) if not found
    """
    try:
        # Method 1: Check Authorization header for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.debug(f"Decoding JWT token")
            
            try:
                # Decode JWT token (without verification for now)
                # In production, verify with your secret key:
                # decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                decoded_payload = jwt.decode(token, options={"verify_signature": False})
                
                user_id = decoded_payload.get("user_id")
                country = decoded_payload.get("country")
                
                if user_id and country:
                    logger.info(f"User authenticated via JWT: {user_id}")
                    return str(user_id), str(country)
                    
            except jwt.DecodeError as e:
                logger.warning(f"JWT decode error: {e}")
            except Exception as e:
                logger.error(f"JWT processing error: {e}")
        
        # Method 2: Check X-User-ID header (fallback)
        user_id_header = request.headers.get("X-User-ID")
        country_header = request.headers.get("X-Country", "US")
        
        if user_id_header:
            logger.info(f"User authenticated via X-User-ID header: {user_id_header}")
            return user_id_header, country_header
        
        logger.warning("No authentication credentials found in request")
        return None, None

    except Exception as e:
        logger.error(f"Unexpected error in decode_user_id_from_header: {e}", exc_info=True)
        return None, None