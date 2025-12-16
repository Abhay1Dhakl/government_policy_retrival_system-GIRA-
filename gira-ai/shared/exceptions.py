"""
Custom exceptions for GIRA system
"""


class GIRAException(Exception):
    """Base exception for all GIRA-related errors"""
    pass


class EmbeddingError(GIRAException):
    """Raised when embedding generation fails"""
    pass


class SearchError(GIRAException):
    """Raised when search operation fails"""
    pass


class DocumentProcessingError(GIRAException):
    """Raised when document processing fails"""
    pass


class LLMError(GIRAException):
    """Raised when LLM service fails"""
    pass


class ConfigurationError(GIRAException):
    """Raised when configuration is invalid"""
    pass


class ValidationError(GIRAException):
    """Raised when validation fails"""
    pass
