"""Core server and configuration modules."""

from .server import mcp, initialize_server
from .constants import (
    STOPWORDS,
    DOCUMENT_TYPE_SYNONYMS,
    SECTION_PRIORITY_WEIGHTS,
    REGION_ALIASES,
)

__all__ = [
    "mcp",
    "initialize_server",
    "STOPWORDS",
    "DOCUMENT_TYPE_SYNONYMS",
    "SECTION_PRIORITY_WEIGHTS",
    "REGION_ALIASES",
]
