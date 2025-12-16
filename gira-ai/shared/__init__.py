"""
GIRA Shared Module
Utilities and common code shared between gira-agent and gira-mcp-server
"""

from .exceptions import (
    GIRAException,
    EmbeddingError,
    SearchError,
    DocumentProcessingError,
    LLMError,
)

__all__ = [
    "GIRAException",
    "EmbeddingError",
    "SearchError",
    "DocumentProcessingError",
    "LLMError",
]
