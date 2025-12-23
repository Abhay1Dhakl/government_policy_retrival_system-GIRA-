"""
Embeddings service package for GIRA AI
"""

from services.embeddings.gemini import (
    initialize_gemini,
    get_gemini_embedding,
    get_gemini_embedding_async,
    test_gemini_embeddings
)

__all__ = [
    "initialize_gemini",
    "get_gemini_embedding",
    "get_gemini_embedding_async",
    "test_gemini_embeddings",
]
