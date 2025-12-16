"""
Shared type definitions and constants
"""

from enum import Enum
from typing import Literal


class DocumentType(str, Enum):
    """Supported document types"""
    PIS = "pis"  # Prescribing Information Sheet
    LRD = "lrd"  # Literature Review Document
    CLINICAL_GUIDELINE = "clinical_guideline"
    POLICY_DOCUMENT = "policy_document"
    RESEARCH_PAPER = "research_paper"


class EmbeddingModel(str, Enum):
    """Supported embedding models"""
    GEMINI = "gemini"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Response status constants
class ResponseStatus(str, Enum):
    """Response status types"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


# Search configuration defaults
DEFAULT_SEARCH_CONFIG = {
    "top_k": 10,
    "alpha": 0.5,  # Hybrid search weight
    "chunk_size": 500,
    "chunk_overlap": 100,
}

# Embedding configuration defaults
DEFAULT_EMBEDDING_CONFIG = {
    "dimension": 768,
    "model": "sentence-transformers/all-mpnet-base-v2",
    "batch_size": 32,
}

# Common stopwords
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "shall",
    "under", "when", "into", "upon", "such", "which", "been", "were", "your", "their",
    "than", "about", "each", "within", "while", "those", "these", "there", "after",
    "before", "during", "because", "other", "where", "should", "could", "would",
}

# Error messages
ERROR_MESSAGES = {
    "invalid_query": "Query cannot be empty or None",
    "invalid_document": "Invalid or unsupported document format",
    "embedding_failed": "Failed to generate embeddings",
    "search_failed": "Search operation failed",
    "llm_failed": "LLM service failed",
    "database_error": "Database operation failed",
}
