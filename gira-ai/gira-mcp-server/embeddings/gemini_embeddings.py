"""
Government Information Retrieval System (GIRS) - Gemini API Embeddings

Uses Google Gemini's text-embedding-004 model for generating embeddings.
"""

import os
import sys
from typing import List, Optional
from google import genai
from google.genai import types

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_client = None

def initialize_gemini():
    """Initialize Gemini API with the provided key"""
    global _client
    
    if _client:
        return True
    
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not found in environment", file=sys.stderr)
        return False
    
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API initialized successfully", file=sys.stderr)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}", file=sys.stderr)
        return False

def get_gemini_embedding(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """
    Get embeddings from Gemini API using google-genai package
    
    Args:
        text: Text to embed
        task_type: Task type for embedding
    
    Returns:
        List of floats (padded to 1024 dimensions) or None on error
    """
    if not initialize_gemini():
        return None
    
    try:
        result = _client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                title=None
            ),
        )
        
        if not result.embeddings:
            return None
            
        embedding = result.embeddings[0].values
        
        # Gemini returns 768 dimensions, we need 1024 for Pinecone
        # Pad with zeros to reach 1024 dimensions
        if len(embedding) < 1024:
            padding = [0.0] * (1024 - len(embedding))
            embedding.extend(padding)
        elif len(embedding) > 1024:
            embedding = embedding[:1024]
        
        return embedding
        
    except Exception as e:
        print(f"⚠️ Gemini embedding failed: {e}", file=sys.stderr)
        return None

async def get_gemini_embedding_async(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """
    Async wrapper for Gemini embeddings
    Note: Gemini SDK doesn't have native async, so we use the sync version in executor
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
        # task_type must be valid enum or string supported by API
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
