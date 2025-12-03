# Standard library imports
import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
# Third-party imports
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
# from mcp_use import MCPClient
# from mcp_use.adapters.langchain_adapter import LangChainAdapter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Local imports - Core
from config import settings
from logging_config import setup_logging, get_logger

# Local imports - Middleware
from middleware.cors import get_cors_config
from middleware.error_handler import setup_exception_handlers
from middleware.logging import RequestLoggingMiddleware


# Local imports - Other
from pdf_highlighter import MedicalPDFHighlighter
from document_upload.app.api.v1.routes_ingestion import router as ingestion_router

# Local imports - API Routes
from api.v1.routes.query import router as query_router
from api.v1.routes.feedback import router as feedback_router
from api.v1.routes.pdf import router as pdf_router
from api.v1.routes.pages import router as pages_router

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)
# Initialize environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variable for lazy PDF highlighter initialization
_pdf_highlighter = None

def get_pdf_highlighter():
    """Lazily initialize and return the PDF highlighter."""
    global _pdf_highlighter
    if _pdf_highlighter is None:
        print("Initializing PDF highlighter...")
        _pdf_highlighter = MedicalPDFHighlighter(
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_access_key=settings.MINIO_ACCESS_KEY,
            minio_secret_key=settings.MINIO_SECRET_KEY,
            minio_bucket=settings.MINIO_BUCKET,
            minio_secure=settings.MINIO_SECURE,
            cleanup_delay=settings.PDF_CLEANUP_DELAY
        )
        print("PDF highlighter initialized successfully!")
    return _pdf_highlighter

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize FastAPI app
app = FastAPI(
    title="GIRA AI Medical Query Agent",
    description="AI-powered medical query assistant",
    version="1.0.0"
)

# Setup CORS middleware using new config
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# Setup request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Setup exception handlers
setup_exception_handlers(app)

# Initialize Pinecone with settings (SECURITY FIX - no more hardcoded key!)
logger.info("Initializing Pinecone...")
pc = Pinecone(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT
)

combined_vector_to_upsert = []
# Create a dense index with integrated embedding
index_name = "quickstart-py"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", 
                            region="us-east-1")
    )

index = pc.Index(index_name)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Citation validation control (set to "false" to disable strict validation for testing)
STRICT_CITATION_VALIDATION = os.getenv("STRICT_CITATION_VALIDATION", "true").lower() == "true"

# Set Windows event loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  
# Note: Using OpenAI API for embeddings - no local model needed

# INCLUDE API ROUTERS
# Admin routes
app.include_router(ingestion_router, prefix="/admin/documents", tags=["Ingestion"])

# API v1 routes
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(feedback_router, prefix="/api/v1", tags=["Feedback"])
app.include_router(pdf_router, prefix="/api/v1", tags=["PDF"])
app.include_router(pages_router, prefix="/api/v1", tags=["Pages"])

logger.info("All API routers registered successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck and monitoring"""
    return {
        "status": "healthy",
        "service": "gira-agent",
        "timestamp": datetime.utcnow().isoformat()
    }

# Removed get_embedding_model() - now using OpenAI embeddings via API
# See document_upload/app/models/document.py for embedding implementation

@app.delete("/delete_all_pinecone_records")
async def delete_all_pinecone_records():
    """
    Delete all records from Pinecone vector database.
    This will completely clear the vector database.
    Use with caution as this action is irreversible.
    """
    try:
        # Get the index
        index = pc.Index(index_name)
        
        # Delete all records from the index
        # This deletes all vectors in all namespaces
        delete_response = index.delete(delete_all=True)
        
        return JSONResponse(
            content={
                "message": "All Pinecone records deleted successfully",
                "index_name": index_name,
                "delete_response": delete_response,
                "status": "success"
            },
            status_code=200
        )
        
    except Exception as e:
        print(f"[delete_all_pinecone_records] Error: {e}")
        return JSONResponse(
            content={
                "error": f"Failed to delete Pinecone records: {str(e)}",
                "status": "error"
            },
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info"
    )
