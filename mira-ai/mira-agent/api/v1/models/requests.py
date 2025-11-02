"""
API Request Models
Pydantic models for API request validation
"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class QueryRequest(BaseModel):
    """Standard query request"""
    page_id: Optional[str] = None
    tools: List[str]
    llm: str
    user_query: str
    session_id: Optional[str] = None


class StreamingQueryRequest(BaseModel):
    """Streaming query request with stream type options"""
    page_id: Optional[str] = None
    tools: List[str]
    llm: str
    user_query: str
    session_id: Optional[str] = None
    stream_type: Optional[str] = "token"  # "word", "token", or "sentence"


class RegenerationRequest(BaseModel):
    """Request to regenerate a previous response"""
    page_id: Optional[str] = None
    conversation_id: str
    tools: List[str]
    llm: str


class RegisterPageRequest(BaseModel):
    """Register a new page or update existing page"""
    page_id: str
    page_title: str
    page_url: str
    page_type: Optional[str] = "general"
    page_metadata: Optional[Dict] = None


class PageHistoryRequest(BaseModel):
    """Request conversation history for a page"""
    user_id: str
    page_id: str
    session_id: Optional[str] = None
    limit: int = 50


class CrossPageSearchRequest(BaseModel):
    """Search across multiple pages"""
    query: str
    page_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    limit: int = 20


class PageAnalyticsRequest(BaseModel):
    """Request analytics for a page"""
    page_id: str


class HighlightTextRequest(BaseModel):
    """Request to highlight text in PDF"""
    pdf_path: str
    texts_to_highlight: List[Dict]
    output_filename: Optional[str] = None
    auto_cleanup: bool = True
    cleanup_delay: Optional[int] = None
    return_file: bool = False


class HighlightCoordinatesRequest(BaseModel):
    """Request to highlight specific coordinates in PDF"""
    pdf_path: str
    highlight_coordinates: List[Dict]  # [{"page": 0, "x": 20, "y": 200, "width": 50, "height": 20}]
    output_filename: Optional[str] = None
    auto_cleanup: bool = True
    cleanup_delay: Optional[int] = None


class CancelCleanupRequest(BaseModel):
    """Request to cancel PDF cleanup"""
    filename: str


class FeedbackRequest(BaseModel):
    """User feedback for RLHF training"""
    conversation_id: str
    user_query: str
    assistant_response: str
    feedback: int
    feedback_reason: str
