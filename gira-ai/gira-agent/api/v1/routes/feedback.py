"""
Feedback collection endpoints
Handles user feedback for RLHF training
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import re
from api.v1.models.requests import FeedbackRequest
from middleware.auth import decode_user_id_from_header
from database.services import DatabaseService
from logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/store_feedback")
async def store_feedback(req: FeedbackRequest, request: Request):
    try:
        # Decode user ID from headers
        user_id, country = decode_user_id_from_header(request)

        if not user_id:
            return JSONResponse(
                content={"error": "User ID not found in headers"}, 
                status_code=400
            )
        
        print(f"[store_feedback] Using user_id: {user_id} (from header)")

        feedback_data = await DatabaseService.store_rlhf_feedback(
            user_id=user_id,
            conversation_id=req.conversation_id,
            user_query=req.user_query,
            assistant_response=req.assistant_response,
            feedback=req.feedback,
            feedback_reason=req.feedback_reason
        )
        
        print(f"[store_feedback] Stored feedback: {feedback_data}")
        return JSONResponse(content={
            "success": True,
            "feedback_record": feedback_data
        }, status_code=200)
    
    except Exception as e:
        print(f"[store_feedback] Error: {e}")
        return JSONResponse(content={"error": "Failed to store feedback"}, status_code=500)
 