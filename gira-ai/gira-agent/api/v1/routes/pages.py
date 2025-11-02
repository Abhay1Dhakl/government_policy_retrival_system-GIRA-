"""
Page management endpoints
Handles page registration, analytics, and metadata
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import re
from api.v1.models.requests import RegisterPageRequest
from middleware.auth import decode_user_id_from_header
from database.services import DatabaseService
from logging_config import get_logger
from typing import Optional
logger = get_logger(__name__)
router = APIRouter(prefix="/pages", tags=["Pages"])


@router.post("/register_page")
async def register_page(req: RegisterPageRequest, request: Request):
    """
    Register a new page or update existing page
    
    - Tracks page metadata
    - Links pages to users
    - Used for conversation context
    """
    logger.info(f"Registering page: {req.page_id}")
    
    # Decode user
    user_id, _ = decode_user_id_from_header(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    try:
        logger.info(f"Registering page: {req.page_id} for user: {user_id}")
        
        page_data = await DatabaseService.register_page(
            user_id=user_id,
            page_id=req.page_id,
            page_title=req.page_title,
            page_url=req.page_url,
            page_type=req.page_type or "general"
        )
        
        logger.info(f"Page registered successfully: {page_data}")
        return JSONResponse(content={
            "success": True,
            "message": "Page registered successfully",
            "page_data": page_data
        })
        
    except Exception as e:
        logger.error(f"Page registration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to register page: {str(e)}")


@router.get("/list")
async def list_pages(request: Request):
    """Get all registered pages for the user"""
    user_id, _ = decode_user_id_from_header(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    try:
        logger.info(f"Fetching pages for user: {user_id}")
        pages = await DatabaseService.get_registered_pages(user_id)
        logger.info(f"Found {len(pages) if isinstance(pages, list) else 'N/A'} pages")
        
        return JSONResponse(content={
            "success": True,
            "pages": pages
        })
        
    except Exception as e:
        logger.error(f"Failed to list pages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pages: {str(e)}")


@router.get("/get_stored_chat_history")
async def get_stored_chat_history(
    page_id: str,
    request: Request,
    session_id: Optional[str] = None,
    limit: int = 50
):
    """Get conversation history for a specific page"""
    user_id, _ = decode_user_id_from_header(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    if not page_id:
        raise HTTPException(status_code=400, detail="Page ID is required")
    
    try:
        logger.info(f"Fetching conversation history for page: {page_id}, user: {user_id}")
        
        data = await DatabaseService.get_page_conversation_history(
            user_id=user_id,
            page_id=page_id,
            session_id=session_id,
            limit=limit,
        )
        
        logger.info(f"Retrieved {len(data) if isinstance(data, list) else 'N/A'} conversations")
        
        return JSONResponse(content=data, status_code=200)
        
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.get("/{page_id}/analytics")
async def get_page_analytics(page_id: str, request: Request):
    """Get analytics for a specific page"""
    user_id, _ = decode_user_id_from_header(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    try:
        # TODO: Implement analytics retrieval
        return JSONResponse(content={
            "page_id": page_id,
            "total_queries": 0,
            "total_conversations": 0
        })
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")
    
   
@router.get("/get_page_data")
async def get_page_data():
    try:
        data = await DatabaseService.get_registered_pages()
        return JSONResponse(content=data, status_code=200)
    except Exception as e:
        print(f"[get_page_data] Error registering page: {e}")
        return JSONResponse(content={"error": "Failed to register page"}, status_code=500)


@router.get("/get_user_chat_sessions")
async def get_user_chat_sessions(
    request: Request,
    limit: int = 50
):
    # Decode user ID from headers
    final_user_id, country = decode_user_id_from_header(request)

    if not final_user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    try:
        data = await DatabaseService.get_user_chat_sessions(
            user_id=final_user_id,
            limit=limit,
        )
        return JSONResponse(content=data, status_code=200)
    
    except Exception as e:
        print(f"[get_user_chat_sessions] Error retrieving chat sessions: {e}")
        return JSONResponse(content={"error": "Failed to retrieve chat sessions"}, status_code=500)