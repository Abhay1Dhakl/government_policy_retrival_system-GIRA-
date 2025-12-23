"""
Configuration package for GIRA AI Agent
Re-exports for backward compatibility
"""

from config.settings import Settings, settings
from config.logging import (
    setup_logging,
    get_logger,
    LoggerFactory,
    log_execution_time
)

__all__ = [
    "Settings",
    "settings",
    "setup_logging",
    "get_logger",
    "LoggerFactory",
    "log_execution_time",
]
