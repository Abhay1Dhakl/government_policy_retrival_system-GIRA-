"""
Middleware package
Contains all cross-cutting concerns for the application
"""
from .auth import decode_user_id_from_header
from .cors import get_cors_config
from .logging import RequestLoggingMiddleware, ResponseTimingMiddleware
from .error_handler import setup_exception_handlers

__all__ = [
    "decode_user_id_from_header",
    "get_cors_config",
    "RequestLoggingMiddleware",
    "ResponseTimingMiddleware",
    "setup_exception_handlers",
]