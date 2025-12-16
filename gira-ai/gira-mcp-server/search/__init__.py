"""Search module initialization."""

from .engine import (
    execute_hybrid_search,
    execute_pinecone_query,
    execute_pinecone_past_query,
    expand_document_type,
    normalize_region_filter,
)
from .scoring import (
    apply_quality_scoring,
    compute_quality_bonus,
    tokenize_text,
)
from .parsing import parse_pinecone_response, _process_search_matches

__all__ = [
    "execute_hybrid_search",
    "execute_pinecone_query",
    "execute_pinecone_past_query",
    "expand_document_type",
    "normalize_region_filter",
    "apply_quality_scoring",
    "compute_quality_bonus",
    "tokenize_text",
    "parse_pinecone_response",
]
