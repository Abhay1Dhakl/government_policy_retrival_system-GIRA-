"""
Google Gemini API Embeddings Module
Uses Gemini's text-embedding-004 model for high-quality embeddings
"""

import os
import sys
from typing import List, Optional
import google.generativeai as genai

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_gemini_initialized = False

def initialize_gemini():
    """Initialize Gemini API with the provided key"""
    global _gemini_initialized
    
    if _gemini_initialized:
        return True
    
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not found in environment", file=sys.stderr)
        return False
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_initialized = True
        print("✅ Gemini API initialized successfully", file=sys.stderr)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}", file=sys.stderr)
        return False

def get_gemini_embedding(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """
    Get multilingual embeddings from Gemini API
    
    Args:
        text: Text to embed (supports ANY language - English, Arabic, French, Spanish, Chinese, etc.)
        task_type: One of:
            - "retrieval_query" (for search queries)
            - "retrieval_document" (for documents to be retrieved)
            - "semantic_similarity" (for comparing similarity)
            - "classification" (for text classification)
            - "clustering" (for clustering tasks)
    
    Returns:
        List of floats (1024 dimensions) or None on error
        
    Note: Gemini text-embedding-004 model supports 100+ languages natively
    """
    if not initialize_gemini():
        return None
    
    try:
        # Use text-embedding-004 model (768 dimensions, multilingual)
        # Supports: English, Arabic, French, Spanish, German, Chinese, Japanese, Korean, 
        # Russian, Portuguese, Italian, Dutch, Turkish, Polish, and 100+ more languages
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type,
            title=None  # Optional title for retrieval_document tasks
        )
        
        embedding = result['embedding']
        
        # Gemini returns 768 dimensions, we need 1024 for Pinecone
        # Pad with zeros to reach 1024 dimensions
        if len(embedding) < 1024:
            padding = [0.0] * (1024 - len(embedding))
            embedding.extend(padding)
        
        return embedding
        
    except Exception as e:
        print(f"⚠️ Gemini embedding failed: {e}", file=sys.stderr)
        return None

async def get_gemini_embedding_async(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """
    Async wrapper for Gemini embeddings
    Note: Gemini SDK doesn't have native async, so we use the sync version
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_gemini_embedding, text, task_type)

def test_gemini_embeddings():
    """Test function to verify Gemini embeddings work"""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set")
        return False
    
    try:
        embedding = get_gemini_embedding("test government policy query about education reform")
        if embedding and len(embedding) == 1024:
            print(f"✅ Gemini embeddings working! Dimension: {len(embedding)}")
            return True
        else:
            print(f"❌ Unexpected embedding dimension: {len(embedding) if embedding else 'None'}")
            return False
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    # Test when run directly
    test_gemini_embeddings()
