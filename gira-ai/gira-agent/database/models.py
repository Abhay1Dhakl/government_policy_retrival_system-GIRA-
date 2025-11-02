from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class DPO_RLHF(Base):
    __tablename__ = "rlhf_feedback"
    rlhf_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True)
    conversation_id = Column(String(255), nullable=False)
    turn_id = Column(String(36), default=lambda: str(uuid.uuid4()), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    feedback = Column(Integer, nullable=True)
    feedback_reason = Column(Text, nullable=True)
    used_in_training = Column(Boolean, default=False)  # Track if feedback was used in training
    created_at = Column(DateTime, default=datetime.utcnow)
    

class RegisteredPage(Base):
    __tablename__ = "registered_pages"
    user_id = Column(String(255), nullable=False) 
    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(String(255), unique=True, index=True, nullable=False)
    page_title = Column(String(500), nullable=False)
    page_url = Column(Text, nullable=False)

    page_type = Column(String(100), default="general")
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    conversations = relationship("PageConversation", back_populates="page")
#     analytics = relationship("PageAnalytics", back_populates="page")

class ModelRegistry(Base):
    """Registry for tracking fine-tuned models"""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), unique=True, index=True, nullable=False)
    base_model = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    model_metadata = Column(JSON, nullable=True)  # For storing additional model info
    performance_metrics = Column(JSON, nullable=True)  # For storing model performance data
    training_data_version = Column(String(255), nullable=True)  # Version of training data used
    
    def __repr__(self):
        return f"<ModelRegistry(model_id='{self.model_id}', is_active={self.is_active})>"

class PageConversation(Base):
    __tablename__ = "page_conversations"
    id = Column(Integer, primary_key=True, index=True)   # PK for relationships
    conversation_id = Column(String(255), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    page_id = Column(String(255), ForeignKey("registered_pages.page_id"), nullable=False) 
    session_id = Column(String(255), nullable=True, index=True)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String(255), nullable=True, index=True)
    page = relationship("RegisteredPage", back_populates="conversations")
    # chain_of_thoughts = relationship("ChainOfThought", back_populates="conversation")
    # embeddings = relationship("ConversationEmbedding", back_populates="conversation")


# class ChainOfThought(Base):
#     __tablename__ = "chain_of_thoughts"
    
#     id = Column(Integer, primary_key=True, index=True)
#     conversation_id = Column(String(255), ForeignKey("page_conversations.conversation_id"), nullable=False)
    
#     # Chain of thought content
#     reasoning_step = Column(Text, nullable=False)
#     step_number = Column(Integer, nullable=False)
#     confidence_score = Column(Float, default=0.0)
#     step_type = Column(String(100), nullable=True)  # analysis, reasoning, conclusion
    
#     # Metadata
#     generated_at = Column(DateTime, default=datetime.utcnow)
    
#     # Relationships
#     conversation = relationship("PageConversation", back_populates="chain_of_thoughts")

# class PageAnalytics(Base):
#     __tablename__ = "page_analytics"
    
#     id = Column(Integer, primary_key=True, index=True)
#     page_id = Column(String(255), ForeignKey("registered_pages.page_id"), nullable=False)
    
#     # Analytics data
#     total_conversations = Column(Integer, default=0)
#     total_queries = Column(Integer, default=0)
#     avg_query_length = Column(Float, default=0.0)
#     most_common_category = Column(String(100), nullable=True)
#     most_mentioned_drugs = Column(JSON, default=[])
    
#     # Time-based analytics
#     daily_activity = Column(JSON, default={})
#     hourly_activity = Column(JSON, default={})
    
#     # Last updated
#     calculated_at = Column(DateTime, default=datetime.utcnow)
    
#     # Relationships
#     page = relationship("RegisteredPage", back_populates="analytics")

# class ConversationEmbedding(Base):
#     __tablename__ = "conversation_embeddings"
    
#     id = Column(Integer, primary_key=True, index=True)
#     conversation_id = Column(String(255), ForeignKey("page_conversations.conversation_id"), nullable=False)
    
#     # Embeddings (stored as JSON array)
#     query_embedding = Column(JSON, nullable=False)
#     response_embedding = Column(JSON, nullable=True)
    
#     # Embedding metadata
#     embedding_model = Column(String(255), default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     embedding_dimension = Column(Integer, default=384)
    
#     # Timestamps
#     created_at = Column(DateTime, default=datetime.utcnow)

# class ChatSession(Base):
#     __tablename__ = "chat_sessions"
    
#     id = Column(Integer, primary_key=True, index=True)
#     session_id = Column(String(255), unique=True, index=True, nullable=False)
    
#     # Session metadata
#     user_id = Column(String(255), nullable=True)
#     session_metadata = Column(JSON, default={})
    
#     # Activity tracking
#     first_activity = Column(DateTime, default=datetime.utcnow)
#     last_activity = Column(DateTime, default=datetime.utcnow)
#     total_interactions = Column(Integer, default=0)
    
#     # Session status
#     is_active = Column(Boolean, default=True)
#     ended_at = Column(DateTime, nullable=True)
