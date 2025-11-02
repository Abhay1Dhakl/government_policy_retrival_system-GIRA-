from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func
from datetime import datetime, timedelta
import json
import uuid

from .models import (
     RegisteredPage, PageConversation, DPO_RLHF
     
     # PageConversation, ChainOfThought, 
    # PageAnalytics, ConversationEmbedding, ChatSession
)
from .config import get_db_session


class DatabaseService:
    @staticmethod
    async def store_rlhf_feedback(
        user_id: str,  
        conversation_id: str,
        user_query: str,
        assistant_response: str,
        feedback: int,
        feedback_reason: str) -> Dict:
        """Store RLHF feedback for a specific conversation turn"""
        with get_db_session() as db:
            rlhf_entry = DPO_RLHF(
                user_id=user_id,
                conversation_id=conversation_id,
                user_query=user_query,
                assistant_response=assistant_response,
                feedback=feedback,
                feedback_reason=feedback_reason,
            )
            db.add(rlhf_entry)
            db.flush()
            return {
                "rlhf_id": rlhf_entry.rlhf_id,
                "conversation_id": rlhf_entry.conversation_id,
                "turn_id": rlhf_entry.turn_id,
                "feedback": rlhf_entry.feedback,
                "created_at": rlhf_entry.created_at.isoformat()
            }
            
    @staticmethod
    async def register_page(
        user_id: str,
        page_id: str,
        page_title: str, 
        page_url: str,
        page_type: str = "general"
    ) -> Dict:
        """Register a new page or update existing one"""
        with get_db_session() as db:
            # Check if page already exists
            existing_page = db.query(RegisteredPage).filter(
                RegisteredPage.page_id == page_id
            ).first()
            
            if existing_page:
                # Update existing page
                existing_page.user_id = user_id
                existing_page.page_url = page_url
                existing_page.page_type = page_type
                existing_page.updated_at = datetime.utcnow()
                page = existing_page
            else:
                # Create new page
                page = RegisteredPage(
                    user_id=user_id,
                    page_id=page_id,
                    page_title=page_title,
                    page_url=page_url,
                    page_type=page_type,
                )
                db.add(page)
            
            db.flush()
            return {
                "user_id": user_id,
                "page_id": page.page_id,
                "page_title": page.page_title,
                "page_url": page.page_url,
                "page_type": page.page_type,
                "registered_at": page.registered_at.isoformat()
            }
    
    @staticmethod
    async def get_registered_pages() -> Dict:
        """Get all registered pages"""
        with get_db_session() as db:
            pages = db.query(RegisteredPage).filter(
                RegisteredPage.is_active == True
            ).all()
            
            return {
                "pages": [
                    {
                        "page_id": page.page_id,
                        "page_title": page.page_title,
                        "page_url": page.page_url,
                        "page_type": page.page_type,
                        "registered_at": page.registered_at.isoformat()
                    }
                    for page in pages
                ],
                "count": len(pages)
            }
    
    @staticmethod
    async def get_page_info(page_id: str) -> Optional[Dict]:
        """Get information for a specific page"""
        with get_db_session() as db:
            page = db.query(RegisteredPage).filter(
                RegisteredPage.page_id == page_id,
                RegisteredPage.is_active == True
            ).first()
            
            if not page:
                return None
                
            return {
                "page_id": page.page_id,
                "page_title": page.page_title,
                "page_url": page.page_url,
                "page_type": page.page_type,
                "metadata": page.metadata,
                "registered_at": page.registered_at.isoformat()
            }
    
    @staticmethod
    async def store_page_conversation(
        user_id: str,
        user_query: str,
        conversation_id: str,
        assistant_response: str,
        page_context: Dict,
        session_id: str
    ) -> Dict:
        """Store a page-based conversation with dynamic context"""
        with get_db_session() as db:
            # Create conversation record
            conversation = PageConversation(
                user_id=user_id,
                conversation_id=conversation_id,
                page_id=page_context.get("page_id"),
                session_id=session_id,
                user_query=user_query,
                assistant_response=assistant_response,
            )
            
            db.add(conversation)
            db.flush()
            
            # # Store chain of thought if provided
            # if chain_of_thought:
            #     for i, thought in enumerate(chain_of_thought):
            #         cot = ChainOfThought(
            #             conversation_id=conversation_id,
            #             reasoning_step=thought.get("reasoning_step", ""),
            #             step_number=i + 1,
            #             confidence_score=thought.get("confidence_score", 0.0),
            #             step_type=thought.get("step_type", "reasoning")
            #         )
            #         db.add(cot)
            
            # # Update session tracking
            # await DatabaseService._update_session_activity(db, session_id)
            
            # # Update page analytics
            # await DatabaseService._update_page_analytics(db, page_context.get("page_id"))
            
            return {
                "conversation_id": conversation_id,
                "status": "stored",
                "page_id": page_context.get("page_id"),
                "session_id": session_id
            }
    
    @staticmethod
    async def get_conversation(conversation_id: str) -> Dict:
        """Get conversation details for a given conversation"""
        with get_db_session() as db:
            conversation = db.query(PageConversation).filter(
                PageConversation.conversation_id == conversation_id
            ).first()
            
            if not conversation:
                return {"error": "Conversation not found"}
            
            # Logic to regenerate response could go here
            # For now, just return existing response
            return {
                "conversation_id": conversation.conversation_id,
                "user_query": conversation.user_query,
                "assistant_response": conversation.assistant_response,
                "created_at": conversation.created_at.isoformat()
            }

    @staticmethod
    async def get_page_conversation_history(
        page_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> Dict:
        """Get conversation history for a specific page"""
        with get_db_session() as db:
            # Join with RegisteredPage to get page_title
            query = db.query(PageConversation, RegisteredPage.page_title).join(
                RegisteredPage, PageConversation.page_id == RegisteredPage.page_id
            ).filter(
                PageConversation.page_id == page_id
            )
            
            if user_id:
                query = query.filter(PageConversation.user_id == user_id)
            
            if session_id:
                query = query.filter(PageConversation.session_id == session_id)
            
            conversations = query.order_by(
                desc(PageConversation.created_at)
            ).limit(limit).all()
            
            result = []
            page_title = None
            for conv, title in conversations:
                if page_title is None:
                    page_title = title
                
                conv_data = {
                    "conversation_id": conv.conversation_id,
                    "user_query": conv.user_query,
                    "assistant_response": conv.assistant_response,
                    "created_at": conv.created_at.isoformat(),
                    "session_id": conv.session_id,
                    "user_id": conv.user_id,
                }
                
                # if include_chain_of_thought:
                #     thoughts = db.query(ChainOfThought).filter(
                #         ChainOfThought.conversation_id == conv.conversation_id
                #     ).order_by(ChainOfThought.step_number).all()
                    
                #     conv_data["chain_of_thought"] = [
                #         {
                #             "step_number": thought.step_number,
                #             "reasoning_step": thought.reasoning_step,
                #             "confidence_score": thought.confidence_score,
                #             "step_type": thought.step_type
                #         }
                #         for thought in thoughts
                #     ]
                
                result.append(conv_data)
            
            # Get page title even if no conversations exist
            if page_title is None:
                # Fetch page title directly from RegisteredPage
                page = db.query(RegisteredPage).filter(
                    RegisteredPage.page_id == page_id
                ).first()
                page_title = page.page_title if page else None
            
            return {
                "page_id": page_id,
                "page_title": page_title,
                "conversations": result,
                "count": len(result),
                "session_id": session_id,
                "user_id": user_id
            }
    
    @staticmethod
    async def get_user_chat_sessions(
        user_id: str,
        limit: int = 50
    ) -> Dict:
        """Get all chat sessions for a user, grouped by page_id"""
        with get_db_session() as db:
            # Get distinct page_ids with latest conversation timestamp and page info
            subquery = db.query(
                PageConversation.page_id,
                func.max(PageConversation.created_at).label('last_activity'),
                func.count(PageConversation.conversation_id).label('message_count'),
                func.min(PageConversation.user_query).label('first_message')
            ).filter(
                PageConversation.user_id == user_id
            ).group_by(
                PageConversation.page_id
            ).subquery()
            
            # Join with RegisteredPage to get page titles
            sessions = db.query(
                subquery.c.page_id,
                subquery.c.last_activity,
                subquery.c.message_count,
                subquery.c.first_message,
                RegisteredPage.page_title
            ).join(
                RegisteredPage, subquery.c.page_id == RegisteredPage.page_id
            ).order_by(
                desc(subquery.c.last_activity)
            ).limit(limit).all()
            
            result = []
            for session in sessions:
                session_data = {
                    "page_id": session.page_id,
                    "page_title": session.page_title or session.first_message[:50] + "..." if len(session.first_message) > 50 else session.first_message,
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": session.message_count,
                    "first_message": session.first_message
                }
                result.append(session_data)
            
            return {
                "user_id": user_id,
                "sessions": result,
                "count": len(result)
            }
    
    # @staticmethod
    # async def search_conversations(
    #     query: str,
    #     page_ids: Optional[List[str]] = None,
    #     session_id: Optional[str] = None,
    #     limit: int = 20
    # ) -> Dict:
    #     """Search conversations across pages using text matching"""
    #     with get_db_session() as db:
    #         search_query = db.query(PageConversation)
            
    #         # Text search in user queries and assistant responses
    #         search_term = f"%{query}%"
    #         search_query = search_query.filter(
    #             or_(
    #                 PageConversation.user_query.ilike(search_term),
    #                 PageConversation.assistant_response.ilike(search_term)
    #             )
    #         )
            
    #         if page_ids:
    #             search_query = search_query.filter(
    #                 PageConversation.page_id.in_(page_ids)
    #             )
            
    #         if session_id:
    #             search_query = search_query.filter(
    #                 PageConversation.session_id == session_id
    #             )
            
    #         conversations = search_query.order_by(
    #             desc(PageConversation.created_at)
    #         ).limit(limit).all()
            
    #         results = [
    #             {
    #                 "conversation_id": conv.conversation_id,
    #                 "page_id": conv.page_id,
    #                 "user_query": conv.user_query,
    #                 "assistant_response": conv.assistant_response[:200] + "..." if len(conv.assistant_response) > 200 else conv.assistant_response,
    #                 "query_category": conv.query_category,
    #                 "priority": conv.priority,
    #                 "created_at": conv.created_at.isoformat(),
    #                 "session_id": conv.session_id
    #             }
    #             for conv in conversations
    #         ]
            
    #         return {
    #             "query": query,
    #             "results": results,
    #             "count": len(results),
    #             "page_ids": page_ids,
    #             "session_id": session_id
    #         }
    
    # @staticmethod
    # async def get_page_analytics(page_id: str) -> Dict:
    #     """Get analytics for a specific page"""
    #     with get_db_session() as db:
    #         # Get or create analytics record
    #         analytics = db.query(PageAnalytics).filter(
    #             PageAnalytics.page_id == page_id
    #         ).first()
            
    #         if not analytics:
    #             # Calculate analytics from conversations
    #             analytics = await DatabaseService._calculate_page_analytics(db, page_id)
            
    #         # Get recent activity (last 30 days)
    #         thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    #         recent_conversations = db.query(PageConversation).filter(
    #             and_(
    #                 PageConversation.page_id == page_id,
    #                 PageConversation.created_at >= thirty_days_ago
    #             )
    #         ).count()
            
    #         return {
    #             "page_id": page_id,
    #             "total_conversations": analytics.total_conversations if analytics else 0,
    #             "total_queries": analytics.total_queries if analytics else 0,
    #             "avg_query_length": analytics.avg_query_length if analytics else 0.0,
    #             "most_common_category": analytics.most_common_category if analytics else None,
    #             "recent_activity_30d": recent_conversations,
    #             "calculated_at": analytics.calculated_at.isoformat() if analytics else None
    #         }
    
    # @staticmethod
    # async def _update_session_activity(db: Session, session_id: str):
    #     """Update session activity tracking"""
    #     session = db.query(ChatSession).filter(
    #         ChatSession.session_id == session_id
    #     ).first()
        
    #     if not session:
    #         session = ChatSession(
    #             session_id=session_id,
    #             total_interactions=1
    #         )
    #         db.add(session)
    #     else:
    #         session.last_activity = datetime.utcnow()
    #         session.total_interactions += 1
    
    # @staticmethod
    # async def _update_page_analytics(db: Session, page_id: str):
    #     """Update page analytics"""
    #     # This could be done asynchronously in production
    #     analytics = await DatabaseService._calculate_page_analytics(db, page_id)
    
    # @staticmethod
    # async def _calculate_page_analytics(db: Session, page_id: str) -> PageAnalytics:
    #     """Calculate analytics for a page"""
    #     # Get existing analytics or create new
    #     analytics = db.query(PageAnalytics).filter(
    #         PageAnalytics.page_id == page_id
    #     ).first()
        
    #     # Calculate stats from conversations
    #     conversations = db.query(PageConversation).filter(
    #         PageConversation.page_id == page_id
    #     ).all()
        
    #     total_conversations = len(conversations)
    #     total_queries = total_conversations
    #     avg_query_length = sum(len(c.user_query) for c in conversations) / total_conversations if total_conversations > 0 else 0
        
    #     # Most common category
    #     categories = [c.query_category for c in conversations if c.query_category]
    #     most_common_category = max(set(categories), key=categories.count) if categories else None
        
    #     if analytics:
    #         analytics.total_conversations = total_conversations
    #         analytics.total_queries = total_queries
    #         analytics.avg_query_length = avg_query_length
    #         analytics.most_common_category = most_common_category
    #         analytics.calculated_at = datetime.utcnow()
    #     else:
    #         analytics = PageAnalytics(
    #             page_id=page_id,
    #             total_conversations=total_conversations,
    #             total_queries=total_queries,
    #             avg_query_length=avg_query_length,
    #             most_common_category=most_common_category
    #         )
    #         db.add(analytics)
        
    #     return analytics
