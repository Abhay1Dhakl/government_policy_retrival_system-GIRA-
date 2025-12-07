"""
Query processing endpoints
Handles streaming queries, regeneration, and MCP interactions
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import AsyncGenerator
import uuid
import json
from datetime import datetime

from api.v1.models.requests import StreamingQueryRequest, RegenerationRequest, QueryRequest
from middleware.auth import decode_user_id_from_header
from database.services import DatabaseService
from database.config import create_tables
from logging_config import get_logger
import os
import re
from services.streaming_service import stream_ai_response

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/stream_query")
async def stream_query(req: StreamingQueryRequest, request: Request):
    """
    Stream a government policy query response in real-time
    
    - Streams response tokens/words/sentences
    - Stores conversation history
    """
    logger.info(f"Received streaming query: {req.user_query[:50]}...")
    create_tables()
    
    # Decode user
    user_id, country = decode_user_id_from_header(request)
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"},
            status_code=400
        )

    logger.info(f"Using user_id: {user_id}")

    # Get page_id
    header_page_id = request.headers.get("X-Page-ID")
    final_page_id = header_page_id or req.page_id

    # If no page_id provided, generate a new one (new chat session)
    if not final_page_id:
        final_page_id = str(uuid.uuid4())
        logger.info(f"Generated new page_id for new chat: {final_page_id}")

    logger.info(f"Using page_id: {final_page_id}")
    
    # Create streaming response with HTTPS-compatible headers and SSE format
    return StreamingResponse(
        stream_ai_response(req, user_id, country, final_page_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for streaming
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Expose-Headers": "Content-Type"
        }
    )


@router.post("/regenerate_response")
async def regenerate_response(req: RegenerationRequest, request: Request):
    """
    Regenerate a previous response with streaming
    
    - Retrieves original conversation
    - Generates new response with same parameters
    """
    logger.info(f"Regenerating response for conversation: {req.conversation_id}")
    create_tables()
    
    # Get original conversation
    conv_data = await DatabaseService.get_conversation(req.conversation_id)
    if not conv_data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Extract original query
    user_query = conv_data.get("user_query", "")
    
    # Decode user
    user_id, country = decode_user_id_from_header(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    # Get page_id from request or conversation data
    page_id = req.page_id or conv_data.get("page_id", str(uuid.uuid4()))
    
    # Create a new streaming query request with the original query
    streaming_req = StreamingQueryRequest(
        page_id=page_id,
        tools=req.tools,
        llm=req.llm,
        user_query=user_query,
        stream_type="token"
    )
    
    logger.info(f"Regenerating with query: {user_query[:50]}..., tools: {req.tools}, llm: {req.llm}")
    
    # Return streaming response
    return StreamingResponse(
        stream_ai_response(streaming_req, user_id, country, page_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Expose-Headers": "Content-Type"
        }
    )