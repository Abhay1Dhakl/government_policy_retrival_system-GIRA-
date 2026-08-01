"""
Title Generation Service
Generates concise titles for government policy queries.
"""
import re
from config.logging import get_logger

logger = get_logger(__name__)


def _fallback_title(user_query: str) -> str:
    text = re.sub(r"\s+", " ", (user_query or "").strip())
    if not text:
        return "Untitled Query"

    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = text.rstrip("?.!,;:")
    words = text.split()
    if not words:
        return "Untitled Query"

    title = " ".join(words[:8])
    return title[:80] or "Untitled Query"


def generate_title(user_query: str) -> str:
    """
    Generate a concise title for the user's query without external API calls.
    
    Args:
        user_query: The user's government policy question
        
    Returns:
        Generated title or "Untitled Query" on error
    """
    try:
        title = _fallback_title(user_query)
        logger.info(f"Generated title: {title}")
        return title
    
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return "Untitled Query"
