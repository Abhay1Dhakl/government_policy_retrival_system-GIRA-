import sys
import os
import uuid
import json
import requests
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from prompts.system_prompt import generate_system_prompt
from document_upload.app.api.v1.routes_ingestion import router as ingestion_router
import uvicorn
from typing import Dict, List, Optional, AsyncGenerator
from database.config import engine, create_tables, drop_tables
from database.services import DatabaseService
from prompts.system_prompt import generate_system_prompt
from llm_options.llm_choose import  choose_llm
from pinecone import Pinecone, ServerlessSpec
import dotenv
from datetime import datetime
import jwt
from pdf_highlighter import MedicalPDFHighlighter
import tempfile
import re
from fastapi.responses import FileResponse
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PII_DETECTION_AVAILABLE = True
    print("Presidio PII detection module loaded successfully")
except ImportError as e:
    print(f"PII detection not available: {e}")
    PII_DETECTION_AVAILABLE = False

_analyzer_engine = None

def get_analyzer_engine():
    """Lazily initialize and return the Presidio AnalyzerEngine."""
    global _analyzer_engine, PII_DETECTION_AVAILABLE
    if _analyzer_engine is None and PII_DETECTION_AVAILABLE:
        print("Loading Presidio Analyzer Engine (one-time setup)...")
        try:
            # Configure NLP engine with a SpaCy model
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            })
            _analyzer_engine = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=["en"]
            )
            print("Presidio Analyzer Engine loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize Presidio Analyzer Engine: {e}")
            PII_DETECTION_AVAILABLE = False
            _analyzer_engine = None
    return _analyzer_engine


def is_likely_english(text: str) -> bool:
    """
    Lightweight language heuristic: prefer using an installed language detector if available,
    otherwise fall back to a simple heuristic that looks for English stopwords and Latin script.
    Returns True if text is likely English.
    """
    if not text or not isinstance(text, str):
        return False

    # Try to use langdetect if it's installed for better accuracy
    try:
        from langdetect import detect
        lang = detect(text)
        return lang == "en"
    except Exception:
        pass

    # Heuristic fallback:
    # - If text contains non-Latin scripts (e.g., Devanagari, Cyrillic), assume non-English
    if re.search(r"[\u0400-\u04FF\u0900-\u097F\u4E00-\u9FFF]", text):
        return False

    # Count tokens that match a small set of common English words
    tokens = re.findall(r"[A-Za-z']{2,}", text)
    if not tokens:
        return False

    common_english = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "that", "this", "it", "as", "are", "was", "were", "be", "by", "or", "from", "what", "who", "when", "where"}
    matches = sum(1 for t in tokens if t.lower() in common_english)
    ratio = matches / len(tokens)

    # If at least 20% of tokens are common English words, consider it English
    return ratio >= 0.20

# Initialize the analyzer at startup
get_analyzer_engine()

# Initialize the database
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()

# Initialize PDF Highlighter with MinIO configuration
pdf_highlighter = MedicalPDFHighlighter(
    minio_endpoint=os.getenv("MINIO_ENDPOINT"),
    minio_access_key=os.getenv("MINIO_ACCESS_KEY"),
    minio_secret_key=os.getenv("MINIO_SECRET_KEY"),
    minio_bucket=os.getenv("MINIO_BUCKET", "medical-documents"),
    minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",  # Default to HTTP
    cleanup_delay=int(os.getenv("PDF_CLEANUP_DELAY", "3600"))  # 1 hour default
)

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

# Get CORS origins from environment variables
def get_cors_origins():
    """
    Get CORS origins based on the environment.
    Returns a list of allowed origins for CORS.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        origins_str = os.getenv("CORS_ORIGINS_PROD", "")
    elif environment == "development":
        origins_str =  "http://localhost:8000,http://localhost:3535,http://localhost:3000"


    # Split by comma and strip whitespace
    origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    
    # Fallback to localhost if no origins are configured
    if not origins:
        origins = ["https://gira-backend.medgentics.com", "https://gira.medgentics.com"]
    
    # Add internal Docker network origins for production
    if environment == "production":
        internal_origins = [
            "http://gira-backend:8082",  # Internal Docker service name
            "http://172.21.0.6:8082",   # Internal Docker IP from logs
            "http://gira-backend",       # Docker service name without port
            "https://gira-backend.medgentics.com",  # External backend domain
            "http://gira-backend.medgentics.com",   # HTTP version
        ]
        origins.extend(internal_origins)
    
    print(f"[CORS] Environment: {environment}")
    print(f"[CORS] Allowed origins: {origins}")
    
    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "Accept", 
        "Origin", 
        "User-Agent", 
        "DNT", 
        "Cache-Control", 
        "X-Mx-ReqToken", 
        "Keep-Alive", 
        "X-Requested-With",
        "X-User-ID",
        "X-Page-ID"
    ],
    expose_headers=["*"],
)

pc = Pinecone(
    # api_key=os.getenv("PINECONE_API_KEY"))
    api_key= "pcsk_2RGA3Z_LVfVmxNQ7A7DX7w5BuhEW4MTCGmGuSghX7GmMwizqWqVCumyrWCcMdtE1jDxgav",
    environment="aped-4627-b74a"  )

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
  
# Load Sentence Transformer for query vectorization
_embedding_model = None

# Define the input schema
class QueryRequest(BaseModel):
    page_id: Optional[str] = None  # Optional since we get it from headers
    tools: List[str]
    llm: str
    user_query: str
    session_id: Optional[str] = None

class RegenerationRequest(BaseModel):
    page_id: Optional[str] = None  # Optional since we get it from headers
    conversation_id: str
    tools: List[str]
    llm: str

class RegisterPageRequest(BaseModel):
    page_id: str
    page_title: str
    page_url: str
    page_type: Optional[str] = "general"
    page_metadata: Optional[Dict] = None

class PageHistoryRequest(BaseModel):
    user_id: str
    page_id: str
    session_id: Optional[str] = None
    limit: int = 50

class CrossPageSearchRequest(BaseModel):
    query: str
    page_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    limit: int = 20

class PageAnalyticsRequest(BaseModel):
    page_id: str

# PDF Highlighting Request Models
class HighlightTextRequest(BaseModel):
    pdf_path: str
    texts_to_highlight: List[Dict]
    output_filename: Optional[str] = None
    auto_cleanup: bool = True  # Whether to auto-delete after delay
    cleanup_delay: Optional[int] = None  # Custom cleanup delay in seconds
    return_file: bool = False  # Whether to return PDF file directly or metadata

class HighlightCoordinatesRequest(BaseModel):
    pdf_path: str  # Can be local path, URL, or MinIO object path (bucket/object_name)
    highlight_coordinates: List[Dict]  # [{"page": 0, "x": 20, "y": 200, "width": 50, "height": 20}]
    output_filename: Optional[str] = None
    auto_cleanup: bool = True  # Whether to auto-delete after delay
    cleanup_delay: Optional[int] = None  # Custom cleanup delay in seconds

class CancelCleanupRequest(BaseModel):
    filename: str  # Filename to cancel cleanup for

class FeedbackRequest(BaseModel):
    conversation_id: str
    user_query: str
    assistant_response: str
    feedback: int
    feedback_reason: str

class StreamingQueryRequest(BaseModel):
    page_id: Optional[str] = None
    tools: List[str]
    llm: str
    user_query: str
    session_id: Optional[str] = None
    stream_type: Optional[str] = "token"  # "word", "token", or "sentence"
    
app.include_router(ingestion_router, prefix="/admin/documents", tags=["Ingestion"])

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (one-time setup)...")
        _embedding_model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
        print("Embedding model loaded successfully!")
    return _embedding_model

def generate_title(user_query: str) -> str:
    """
    Generate a concise title for the user's query using OpenAI's GPT-3.5-turbo model.
    
    Args:
        user_query (str): The user's medical question
        
    Returns:
        str: Generated title
    """
    try:
        if not OPENAI_API_KEY or not OPENAI_BASE_URL:
            raise ValueError("OpenAI API key or base URL is not set in environment variables.")
        print("openai api key:", OPENAI_API_KEY)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        prompt = f"Generate a concise title (max 10 words) for the following medical question:\n\n{user_query}\n\nTitle:"
        
        payload = {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates concise titles."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 20,
            "temperature": 0.5,
            "n": 1,
            "stop": ["\n"]
        }
        
        response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        print("OpenAI response status:", response.status_code)
        response.raise_for_status()
        
        data = response.json()
        title = data['choices'][0]['message']['content'].strip()
        print("Generated title:", title)
        return title
    
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Untitled Query"
    
    
def decode_user_id_from_header(request: Request) -> Optional[str]:
    """
    Decode user ID from various header formats.
    
    This function attempts to extract and decode the user ID from request headers.
    It supports multiple formats:
    1. Authorization Bearer token (JWT)
    2. X-User-ID header (direct or base64 encoded)
    3. X-Auth-Token header
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        Optional[str]: Decoded user ID or None if not found/invalid
    """
    try:
        # Method 1: Check Authorization header for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            print(f"*******([decode_user_id] JWT decoded payload: {token}")
            try:
                # Try to decode JWT token (without verification for now)
                # In production, you should verify with your secret key
                decoded_payload = jwt.decode(token, options={"verify_signature": False})
                print(f"[decode_user_id] JWT decoded payload: {decoded_payload}")
                user_id = decoded_payload.get("user_id")
                country = decoded_payload.get("country")
                if user_id and country:
                    print(f"[decode_user_id] Found user_id in JWT: {user_id}")
                    return str(user_id), str(country)
            except Exception as e:
                print(f"[decode_user_id] JWT decode error: {e}")
                return None, None

    except Exception as e:
        print(f"[decode_user_id] Unexpected error: {e}")
        return None, None 


def detect_pii(text: str) -> dict:
    """
    Detect personally identifiable information in text using Presidio Analyzer.
    Falls back to simple regex patterns if Presidio is unavailable.
    
    Args:
        text (str): Text to analyze for PII
        
    Returns:
        dict: Contains 'has_high_risk_pii', 'risk_level', 'warning_message', and 'details'
    """
    if not text:
        return {
            "has_high_risk_pii": False,
            "risk_level": 0,
            "warning_message": None,
            "details": [],
            "flagged_entities": []
        }

    analyzer = get_analyzer_engine()

    # If Presidio is available, use it — but only if the input text is likely English.
    if PII_DETECTION_AVAILABLE and analyzer:
        try:
            if not is_likely_english(text):
                print("[PII] Input not detected as English. Skipping Presidio analysis to avoid false positives.")
            else:
                print("[PII] Analyzing text with Presidio...")
                analyzer_results = analyzer.analyze(text=text, language='en')

                # Filter results to only include PERSON entities to avoid flagging medical terms as organizations.
                original_count = len(analyzer_results)
                allowed_entities = ["PERSON"]
                analyzer_results = [res for res in analyzer_results if res.entity_type in allowed_entities]
                if len(analyzer_results) < original_count:
                    print(f"[PII] Filtered Presidio results to only include {', '.join(allowed_entities)} entities.")
                
                if analyzer_results:
                    print(f"[PII] Presidio detected {len(analyzer_results)} PII entities after filtering.")
                    detected_types = list(set([res.entity_type for res in analyzer_results]))
                    details = []
                    flagged_entities = []
                    
                    for res in analyzer_results:
                        details.append({
                            'type': res.entity_type,
                            'text': text[res.start:res.end],
                            'risk_level': 5,  # Assign a high risk level for any Presidio finding
                            'score': res.score
                        })
                        flagged_entities.append(f"{res.entity_type}='{text[res.start:res.end]}'")

                    warning_message = (
                        f"PII/PHI detected in your message. Please remove the following before proceeding: {', '.join(detected_types)}. "
                        "For your privacy and HIPAA compliance, we cannot process requests containing personal information."
                    )

                    return {
                        "has_high_risk_pii": True,
                        "risk_level": 5,
                        "warning_message": warning_message,
                        "details": details,
                        "flagged_entities": flagged_entities
                    }
                else:
                    print("[PII] Presidio analysis complete - No PII entities of allowed types found.")
                    return {
                        "has_high_risk_pii": False,
                        "risk_level": 0,
                        "warning_message": None,
                        "details": [],
                        "flagged_entities": []
                    }
        except Exception as e:
            print(f"[PII] Presidio analysis failed: {e}. Falling back to regex.")
            # Fall through to regex method if Presidio fails at runtime

    # Fallback to simple regex patterns if Presidio is not available or failed
    print("[PII] Using fallback regex-based detection.")
    try:
        # Simple regex patterns for common PII types
        patterns = {
            'EMAIL_ADDRESS': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'PERSON': r'(?i)\b(dr\.?|doctor|prof\.?|professor)\s+([a-zA-Z][a-zA-Z\s]{1,30})\b'
        }
        
        detected = []
        flagged_entities = []
        
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    match_text = ' '.join(match).strip() if isinstance(match, tuple) else match
                    detected.append({
                        'type': pii_type,
                        'text': match_text,
                        'risk_level': 4
                    })
                    flagged_entities.append(f"{pii_type.upper()}='{match_text}'")
        
        if detected:
            detected_types = list(set([d['type'] for d in detected]))
            warning_message = (
                f"PII/PHI detected in your message. Please remove the following before proceeding: {', '.join(detected_types)}. "
                "For your privacy and HIPAA compliance, we cannot process requests containing personal information."
            )
            return {
                "has_high_risk_pii": True,
                "risk_level": 4,
                "warning_message": warning_message,
                "details": detected,
                "flagged_entities": flagged_entities
            }
        else:
            return {
                "has_high_risk_pii": False,
                "risk_level": 0,
                "warning_message": None,
                "details": [],
                "flagged_entities": []
            }
            
    except Exception as e:
        print(f"[PII] Fallback regex detection failed: {e}")
        return {
            "has_high_risk_pii": True,
            "risk_level": 5,
            "warning_message": "PII detection system error. Request blocked as a safety measure.",
            "details": [],
            "error": str(e),
            "flagged_entities": []
        }


# @app.get("/llm/providers")
# async def get_llm_providers():
#     """Get available LLM providers"""
#     try:
#         providers = [provider.value for provider in LLMProvider]
#         return JSONResponse(content={
#             "providers": providers,
#             "default": "openai"
#         }, status_code=200)
#     except Exception as e:
#         print(f"[get_llm_providers] Error: {e}")
#         return JSONResponse(content={"error": "Failed to get LLM providers"}, status_code=500)

@app.post("/store_feedback")
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
    

@app.post("/regenerate_response")
async def regenerate_response(req: RegenerationRequest, request: Request):
    try:
        conv_data = await DatabaseService.get_conversation(
            conversation_id=req.conversation_id
        )

        if not conv_data:
            return JSONResponse(
                content={"error": "Conversation not found"}, 
                status_code=404
            )
    
    except Exception as e:
        print(f"[regenerate_response] Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve conversation"}, status_code=500)

    if isinstance(conv_data, dict):
        user_query = (
            conv_data.get("user_query") 
        )
        assistant_response = conv_data.get("assistant_response", "")
        user_query = user_query + assistant_response

    print(f"[regenerate_response] Original user query: {user_query}")
    
    # PII Detection - Check the original query for sensitive information
    pii_result = detect_pii(user_query)
    
    if pii_result["has_high_risk_pii"]:
        print(f"[regenerate_response] PII detected in original query - blocking regeneration")
        return JSONResponse(
            content={
                "error": "PII/PHI detected in original query. Please start a new conversation without personal information.",
                "warning": "The original conversation contains sensitive information. Please start a new conversation without personal details.",
                "pii_detected": True,
                "risk_level": pii_result["risk_level"],
                "details": pii_result["details"],
                "flagged_entities": pii_result.get("flagged_entities", [])
            }, 
            status_code=400
        )

    # print(f"[regenerate_response] Retrieved conversation: {conv_data}")
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)

    # Use decoded user ID if available, otherwise fallback to request body
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers or request body"}, 
            status_code=400
        )
    
    print(f"[handle_query] Using user_id: {user_id} (from {'header' if user_id else 'request body'})")

    tools = req.tools
    llm = req.llm
    session_id = str(uuid.uuid4())
    
    
    page_context = {
        "page_id": req.page_id,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "client_ip": request.client.host,
        "page_title": request.headers.get("X-Page-Title", "unknown"),
        "page_url": request.headers.get("X-Page-URL", "unknown")
    }
    try:
        page_data = await DatabaseService.register_page(
        user_id=user_id,  # Use the decoded user_id
        page_id=req.page_id,  # Use the final page_id (header or request body)
        page_title=request.headers.get("X-Page-Title", "unknown"),
        page_url=request.headers.get("X-Page-URL", "unknown"),
        )
        
        print(f"[regenerate_response] Registered page: {page_data}")

    except Exception as e:
        print(f"[regenerate_response] Error registering page: {e}")
        return JSONResponse(content={"error": "Failed to register page"}, status_code=500)
    
    try:
        response = await process_query(user_query, llm, tools, country, user_id)  # Add user_id parameter
        conversation_id = str(uuid.uuid4())
        store_conversation = await DatabaseService.store_page_conversation(
            user_id=user_id,  # Use the decoded user_id instead of hardcoded "1234"
            conversation_id=conversation_id,
            user_query=user_query,
            assistant_response=response,
            session_id=session_id,
            page_context=page_context,
        )

        print(f"[handle_query] Stored conversation: {store_conversation}")
        return JSONResponse(content={
            "response": response,
            "session_id": session_id,
            "page_context": page_data,
            "storage_result": store_conversation
        })
    
    except Exception as e:
        print(f"[regenerate_response] Error: {e}")
        return JSONResponse(content={"error": "Failed to regenerate response"}, status_code=500)
        

@app.get("/get_stored_chat_history")
async def get_stored_chat_history(
    request: Request,
    page_id: str,
    session_id: Optional[str] = None,
    limit: int = 50
):
    # Decode user ID from headers first, then fallback to query parameter
    final_user_id, country = decode_user_id_from_header(request)

    if not final_user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers or query parameters"}, 
            status_code=400
        )
    
    if not page_id:
        return JSONResponse(
            content={"error": "Page ID not found in headers or query parameters"}, 
            status_code=400
        )
    
    try:
        data = await DatabaseService.get_page_conversation_history(
            user_id=final_user_id,
            page_id=page_id,
            session_id=session_id,
            limit=limit,
        )
        return JSONResponse(content=data, status_code=200)
    
    except Exception as e:
        print(f"[get_stored_chat_history] Error retrieving chat history: {e}")
        return JSONResponse(content={"error": "Failed to retrieve chat history"}, status_code=500)

    
@app.get("/get_page_data")
async def get_page_data():
    try:
        data = await DatabaseService.get_registered_pages()
        return JSONResponse(content=data, status_code=200)
    except Exception as e:
        print(f"[get_page_data] Error registering page: {e}")
        return JSONResponse(content={"error": "Failed to register page"}, status_code=500)

# PDF Highlighting API Endpoints
@app.post("/highlight_pdf_text")
async def highlight_pdf_text(req: HighlightTextRequest, request: Request):
    """
    Highlight specific text in a PDF and return the highlighted PDF directly.
    Supports local files, URLs, and MinIO object paths.
    
    Request body example:
    {
        "pdf_path": "medical-docs/azithromycin_study.pdf",  // MinIO object path
        "texts_to_highlight": [
            {
                "text": "azithromycin dosage",
                "type": "drug_info",
                "page": 1
            }
        ],
        "output_filename": "highlighted_document.pdf",
        "auto_cleanup": true,
        "cleanup_delay": 3600,
        "return_file": true  // Set to true to return PDF directly, false for metadata
    }
    """
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    try:
        # Highlight text in PDF using global highlighter
        result = pdf_highlighter.highlight_text_in_pdf(
            input_pdf_path=req.pdf_path,
            texts_to_highlight=req.texts_to_highlight,
            user_id=user_id,
            output_filename=req.output_filename,
            auto_cleanup=req.auto_cleanup,
            cleanup_delay=req.cleanup_delay
        )
        
        if result["success"]:
            # Check if we should return the file directly or metadata
            return_file = getattr(req, 'return_file', False)
            
            if return_file:
                # Get the local file path from the result first
                output_path = result.get("local_file_path")
                
                # If local file exists, return it directly
                if output_path and os.path.exists(output_path):
                    return FileResponse(
                        path=output_path,
                        filename=result["output_filename"],
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f"inline; filename={result['output_filename']}",
                            "X-Total-Highlights": str(result["total_highlights"]),
                            "X-Source-Type": result["source_type"],
                            "X-Original-Filename": result["original_filename"]
                        }
                    )        
            else:
                try:
                    import tempfile
                    # Download from MinIO to temporary file
                    minio_object_name = result["minio_object_name"]
                    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                    
                    # Download from MinIO highlighted bucket
                    pdf_highlighter.minio_client.fget_object(
                        pdf_highlighter.highlighted_bucket, 
                        minio_object_name, 
                        temp_file.name
                    )
                    
                    print(f"Downloaded {minio_object_name} from MinIO for direct file return")
                    
                    # Return the downloaded file
                    return FileResponse(
                        path=temp_file.name,
                        filename=result["output_filename"],
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f"inline; filename={result['output_filename']}",
                            "X-Total-Highlights": str(result["total_highlights"]),
                            "X-Source-Type": "minio",
                            "X-Original-Filename": result["original_filename"]
                        }
                    )
                    
                except Exception as e:
                    print(f"Failed to download from MinIO: {e}")
                    return JSONResponse(
                        content={"error": f"Failed to download highlighted PDF from MinIO: {str(e)}"}, 
                        status_code=500
                    )
                
        else:
            return JSONResponse(
                content={"error": result["error"]}, 
                status_code=500
            )
            
    except Exception as e:
        print(f"[highlight_pdf_text] Error: {e}")
        return JSONResponse(
            content={"error": "Failed to highlight PDF"}, 
            status_code=500
        )

# @app.post("/highlight_pdf_coordinates")
# async def highlight_pdf_coordinates(req: HighlightCoordinatesRequest, request: Request):
#     """
#     Highlight PDF using pre-calculated coordinates.
#     Supports local files, URLs, and MinIO object paths.
    
#     Request body example:
#     {
#         "pdf_path": "medical-docs/research_paper.pdf",  // MinIO object path
#         "highlight_coordinates": [
#             {
#                 "page": 0,
#                 "x": 20,
#                 "y": 200,
#                 "width": 150,
#                 "height": 20,
#                 "type": "drug_info",
#                 "citation": "Dosage information"
#             }
#         ],
#         "auto_cleanup": true,
#         "cleanup_delay": 7200
#     }
#     """
#     # Decode user ID from headers
#     user_id, country = decode_user_id_from_header(request)
    
#     if not user_id:
#         return JSONResponse(
#             content={"error": "User ID not found in headers"}, 
#             status_code=400
#         )
    
#     try:
#         # Highlight PDF using coordinates with global highlighter
#         result = pdf_highlighter.highlight_passages_from_coordinates(
#             input_pdf_path=req.pdf_path,
#             highlight_coordinates=req.highlight_coordinates,
#             user_id=user_id,
#             output_filename=req.output_filename,
#             auto_cleanup=req.auto_cleanup,
#             cleanup_delay=req.cleanup_delay
#         )
        
#         if result["success"]:
#             return JSONResponse(content={
#                 "success": True,
#                 "highlighted_pdf_url": result["highlighted_pdf_url"],
#                 "original_filename": result["original_filename"],
#                 "output_filename": result["output_filename"],
#                 "total_highlights": result["total_highlights"],
#                 "auto_cleanup": result["auto_cleanup"],
#                 "cleanup_delay": result["cleanup_delay"],
#                 "source_type": result["source_type"],
#                 "message": f"PDF highlighted successfully using coordinates from {result['source_type']} source"
#             })
#         else:
#             return JSONResponse(
#                 content={"error": result["error"]}, 
#                 status_code=500
#             )
            
#     except Exception as e:
#         print(f"[highlight_pdf_coordinates] Error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to highlight PDF with coordinates"}, 
#             status_code=500
#         )

# @app.get("/downloads/highlighted/{filename}")
# async def download_highlighted_pdf(filename: str, request: Request):
#     """
#     Download highlighted PDF file.
#     """
#     # Decode user ID for security
#     user_id, country = decode_user_id_from_header(request)
    
#     if not user_id:
#         return JSONResponse(
#             content={"error": "Authentication required"}, 
#             status_code=401
#         )
    
#     try:
#         file_path = os.path.join("uploads/highlighted", filename)
        
#         if not os.path.exists(file_path):
#             return JSONResponse(
#                 content={"error": "File not found"}, 
#                 status_code=404
#             )
        
#         # Security: Check if filename contains user_id to prevent unauthorized access
#         if user_id not in filename:
#             return JSONResponse(
#                 content={"error": "Unauthorized access to file"}, 
#                 status_code=403
#             )
        
#         return FileResponse(
#             path=file_path,
#             filename=filename,
#             media_type="application/pdf"
#         )
        
#     except Exception as e:
#         print(f"[download_highlighted_pdf] Error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to download file"}, 
#             status_code=500
#         )

@app.post("/cancel_pdf_cleanup")
async def cancel_pdf_cleanup(req: CancelCleanupRequest, request: Request):
    """
    Cancel scheduled automatic cleanup for a specific highlighted PDF.
    """
    # Decode user ID for security
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "Authentication required"}, 
            status_code=401
        )
    
    try:
        # Security: Check if filename contains user_id
        if user_id not in req.filename:
            return JSONResponse(
                content={"error": "Unauthorized access to file"}, 
                status_code=403
            )
        
        file_path = os.path.join("uploads/highlighted", req.filename)
        pdf_highlighter.cancel_file_cleanup(file_path)
        
        return JSONResponse(content={
            "success": True,
            "filename": req.filename,
            "message": "Automatic cleanup cancelled for file"
        })
        
    except Exception as e:
        print(f"[cancel_pdf_cleanup] Error: {e}")
        return JSONResponse(
            content={"error": "Failed to cancel cleanup"}, 
            status_code=500
        )

@app.delete("/cleanup_all_pdfs")
async def cleanup_all_pdfs(request: Request):
    """
    Immediately cleanup all highlighted PDFs and cancel all scheduled cleanups.
    Admin endpoint for maintenance.
    """
    # Decode user ID for security
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "Authentication required"}, 
            status_code=401
        )
    
    try:
        pdf_highlighter.cleanup_all_files()
        
        return JSONResponse(content={
            "success": True,
            "message": "All highlighted PDFs cleaned up and timers cancelled"
        })
        
    except Exception as e:
        print(f"[cleanup_all_pdfs] Error: {e}")
        return JSONResponse(
            content={"error": "Failed to cleanup files"}, 
            status_code=500
        )

@app.post("/query")
async def handle_query(req: QueryRequest, request: Request):
    create_tables()
    
    # PII Detection - Check for sensitive information before processing
    pii_result = detect_pii(req.user_query)
    
    if pii_result["has_high_risk_pii"]:
        print(f"[handle_query] PII detected - blocking request. Details: {pii_result['details']}")
        return JSONResponse(
            content={
                "error": "PII/PHI detected. Please remove personal information before proceeding.",
                "warning": pii_result["warning_message"],
                "pii_detected": True,
                "risk_level": pii_result["risk_level"],
                "details": pii_result["details"],
                "flagged_entities": pii_result.get("flagged_entities", [])
            }, 
            status_code=400
        )

    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)

    # Use decoded user ID if available, otherwise fallback to request body
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers or request body"}, 
            status_code=400
        )
    
    print(f"[handle_query] Using user_id: {user_id} (from {'header' if user_id else 'request body'})")
    
    tools = req.tools
    llm = req.llm
    user_query = req.user_query
    
    # Get page_id from headers if available, otherwise use request body
    header_page_id = request.headers.get("X-Page-ID")
    final_page_id = header_page_id or req.page_id
    
    # If no page_id provided, generate a new one (new chat session)
    if not final_page_id:
        final_page_id = str(uuid.uuid4())
        print(f"[handle_query] Generated new page_id for new chat: {final_page_id}")
    
    # For existing page_id, reuse session_id. For new page_id, create new session_id
    session_id = req.session_id or final_page_id  # Use page_id as session_id for consistency
    
    title = generate_title(user_query)

    page_context = {
        "page_id": final_page_id,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "client_ip": request.client.host,
        "page_title": title,
        "page_url": request.headers.get("X-Page-URL", "unknown")
    }
    print(f"[handle_query] Using page_id: {final_page_id} (from {'header' if header_page_id else 'request body'})")
    print(f"[handle_query] Generated title: {title}")
    try:
        data = await DatabaseService.register_page(
        user_id=user_id,  # Use the decoded user_id
        page_id=final_page_id,  # Use the final page_id (header or request body)
        page_title= title,
        page_url=request.headers.get("X-Page-URL", "unknown"),
        )
        
        print(f"[handle_query] Registered page: {data}")

    except Exception as e:
        print(f"[handle_query] Error registering page: {e}")
        return JSONResponse(content={"error": "Failed to register page"}, status_code=500)
    if not user_query:
        return JSONResponse(content={"error": "Invalid input. Query is required."}, status_code=400)
    
    try:
        response = await process_query(user_query, llm, tools, country, user_id)  # Use the decoded user_id

        metadata = {
        "user_id": user_id,  # Use the decoded user_id
        "question": req.user_query,
        "document_type":"past_cases",
        "answer": response,
        "timestamp": datetime.utcnow().isoformat()
             }
        
        model = get_embedding_model()
        
        embedding = model.encode(response)
        
        index.upsert([{
        "id": user_id,  # Use the decoded user_id
        "values": embedding,
        "metadata": metadata
    }])

        conversation_id = str(uuid.uuid4())
        store_conversation = await DatabaseService.store_page_conversation(
            user_id=user_id,  # Use the decoded user_id instead of hardcoded "1234"
            user_query=user_query,
            conversation_id=conversation_id,
            assistant_response=response,
            page_context=page_context,
            session_id=session_id
        )

        print(f"[handle_query] Stored conversation: {store_conversation}")
        return JSONResponse(content={
            "conversation_id": conversation_id,
            "response": response,
            "session_id": session_id,
            "page_context": data,
            "storage_result": store_conversation
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        return JSONResponse(content={"error": "An error occurred while processing the query."}, status_code=500)
    

def process_mcp_response(result: any, tool_name: str) -> str:
    """
    Process MCP server response and extract meaningful medical content for the LLM.
    
    Args:
        result: The raw result from MCP server tool
        tool_name: Name of the tool that was called
        
    Returns:
        str: Processed content that the LLM can understand
    """
    try:
        print(f"[process_mcp_response] Processing {tool_name} result type: {type(result)}", file=sys.stderr)
        print(f"[process_mcp_response] Raw result content (first 500 chars): {str(result)[:500]}", file=sys.stderr)
        
        # Handle different result formats
        if isinstance(result, str):
            print(f"[process_mcp_response] {tool_name}: Received string result, attempting JSON parse", file=sys.stderr)
            # If it's already a string, try to parse as JSON
            try:
                import json
                parsed_result = json.loads(result)
                print(f"[process_mcp_response] {tool_name}: Successfully parsed JSON", file=sys.stderr)
                return process_structured_response(parsed_result, tool_name)
            except json.JSONDecodeError as e:
                print(f"[process_mcp_response] {tool_name}: JSON parse failed: {e}", file=sys.stderr)
                # If not JSON, return as is
                return result
                
        elif isinstance(result, dict):
            print(f"[process_mcp_response] {tool_name}: Received dict result", file=sys.stderr)
            # If it's a dictionary, process it
            return process_structured_response(result, tool_name)
            
        else:
            print(f"[process_mcp_response] {tool_name}: Converting {type(result)} to string", file=sys.stderr)
            # Convert other types to string
            return str(result)
            
    except Exception as e:
        print(f"[process_mcp_response] Error processing response: {e}", file=sys.stderr)
        return f"Error processing {tool_name} response: {str(e)}"

def process_structured_response(data: dict, tool_name: str) -> str:
    """
    Process structured response data from MCP server.
    
    Args:
        data: Dictionary containing the response data
        tool_name: Name of the tool
        
    Returns:
        str: Formatted content for LLM
    """
    try:
        print(f"[process_structured_response] Processing {tool_name} data: {type(data)}", file=sys.stderr)
        print(f"[process_structured_response] Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}", file=sys.stderr)

        # SPECIAL CASE: past_cases may contain previous assistant outputs embedded in metadata.answer
        # If that embedded answer is a previous assistant fallback (the three-line fallback JSON), ignore it
        if tool_name == "past_cases":
            try:
                matches = data.get("matches", []) if isinstance(data, dict) else []
                past_texts = []
                for match in matches[:5]:
                    if not isinstance(match, dict):
                        continue
                    md = match.get("metadata", {}) or {}
                    ans = md.get("answer") or md.get("text") or ""
                    if not ans:
                        continue
                    # Remove possible code fences
                    clean = re.sub(r"```[\s\S]*?```", "", str(ans)).strip()
                    # If it contains JSON, try to parse
                    try:
                        parsed = json.loads(clean)
                        inner = parsed.get("answer", "") if isinstance(parsed, dict) else clean
                    except Exception:
                        inner = clean
                    inner = str(inner).strip()
                    # If inner contains the known fallback lines, skip it
                    fallback_detect = ("No PI document available" in inner and "No LRD document available" in inner)
                    if not fallback_detect and inner:
                        past_texts.append(inner[:200])
                if past_texts:
                    return "Past cases found:\n\n" + "\n\n".join(past_texts)
                else:
                    return "No additional documents or past cases available for this question."
            except Exception as e:
                print(f"[process_structured_response] past_cases parsing error: {e}", file=sys.stderr)
                # Fall through to generic handling

        # Check if this is our new simplified format
        if "matches" in data and "total_found" in data:
            matches = data.get("matches", [])
            total_found = data.get("total_found", 0)

            # CRITICAL: Filter matches by document_type to ensure tool alignment
            target_doc_type = tool_name.lower().strip()
            if matches and target_doc_type:
                filtered_matches = []
                for match in matches:
                    doc_type = str(match.get("document_type", "")).lower().strip()
                    if doc_type == target_doc_type:
                        filtered_matches.append(match)
                if filtered_matches:
                    matches = filtered_matches
                    total_found = len(filtered_matches)
                    print(f"[process_structured_response] {tool_name}: Filtered to {len(filtered_matches)} matches with document_type='{target_doc_type}'", file=sys.stderr)
                else:
                    print(f"[process_structured_response] {tool_name}: No matches with document_type='{target_doc_type}', returning fallback", file=sys.stderr)
                    return f"No {tool_name.upper()} documents found for this query."
            
            # Limit matches processing for faster performance
            matches_to_process = matches[:15]  # Process only first 15 matches maximum for faster processing

            print(f"[process_structured_response] {tool_name}: total_found={total_found}, matches_len={len(matches)}", file=sys.stderr)

            # More detailed check - ensure we have actual content, not just empty matches
            has_content = False
            for match in matches_to_process:  # Use limited matches
                if match.get("text", "").strip():
                    has_content = True
                    break

            print(f"[process_structured_response] {tool_name}: has_content={has_content}", file=sys.stderr)

            if total_found == 0 or not matches or not has_content:
                return f"No {tool_name.upper()} documents found for this query."

            # CRITICAL: Implement page-based consolidation for PDF highlighting
            # Group matches by (source, page_number) to consolidate references
            page_groups = {}
            for match in matches_to_process:  # Use limited matches for faster processing
                source = match.get("source", "")
                page_number = match.get("page_number", "")
                
                # Create unique key for same source + page
                group_key = f"{source}_page_{page_number}" if page_number else f"{source}_no_page"
                
                if group_key not in page_groups:
                    page_groups[group_key] = []
                page_groups[group_key].append(match)

            print(f"[process_structured_response] {tool_name}: Grouped {len(matches_to_process)} matches into {len(page_groups)} page groups", file=sys.stderr)

            # Format the consolidated groups
            formatted_content = f"Found {total_found} relevant {tool_name.upper()} documents (consolidated by page):\n\n"

            group_index = 1
            for group_key, group_matches in page_groups.items():
                if not group_matches:
                    continue
                    
                # Use the first match for header info
                first_match = group_matches[0]
                doc_type = first_match.get("document_type", "")
                region = first_match.get("region", "")
                page_number = first_match.get("page_number", "")
                source = first_match.get("source", "")
                
                # Calculate average score for the group
                scores = [match.get("score", 0.0) for match in group_matches]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Include page number and source in document header if available
                page_info = f", Page: {page_number}" if page_number else ""
                source_info = f"Source: {source}" if source else ""
                formatted_content += f"Document from {source_info}{page_info} (Type: {doc_type}, {len(group_matches)} chunks, Avg Score: {avg_score:.3f}):\n\n"
                
                # Add all text chunks from this page group
                chunk_index = 1
                for match in group_matches:
                    text = match.get("text", "")
                    chunk_score = match.get("score", 0.0)
                    
                    print(f"[process_structured_response] Group {group_index}, Chunk {chunk_index}: text_len={len(text)}, score={chunk_score}", file=sys.stderr)
                    
                    if text and text.strip():
                        # Remove excessive whitespace and format nicely while preserving full content
                        clean_text = " ".join(text.split())
                        
                        # Add metadata to the chunk itself
                        chunk_metadata = f"Source: {match.get('source', 'N/A')}, Page: {match.get('page_number', 'N/A')}, Chunk: {match.get('chunk_index', 'N/A')}, Score: {chunk_score:.3f}"
                        formatted_content += f"Text: {clean_text}\nMetadata: {chunk_metadata}\n\n"
                        chunk_index += 1
                
                formatted_content += "---\n\n"  # Separator between page groups
                group_index += 1
                
                # Prevent runaway payloads but allow full chunk text for synthesis
                if len(formatted_content) > 25000:
                    formatted_content += "\n[RESPONSE TRUNCATED - TOO LARGE]\n"
                    break

            result = formatted_content.strip()
            print(f"[process_structured_response] {tool_name}: returning content length={len(result)}", file=sys.stderr)
            return result

        # Handle legacy format or error responses
        elif "error" in data:
            return f"Error from {tool_name}: {data.get('error', 'Unknown error')}"

        else:
            # Try to extract any text content from legacy format
            if "matches" in data:
                matches = data["matches"]
                if matches:
                    content_parts = []
                    for match in matches[:5]:  # Limit to 5 matches
                        if isinstance(match, dict):
                            metadata = match.get("metadata", {})
                            # Prefer 'text' but fall back to 'answer' for past_cases legacy
                            text = metadata.get("text", "") or metadata.get("answer", "")
                            if text:
                                # If the text is JSON string, try parse and extract inner answer
                                try:
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict) and parsed.get("answer"):
                                        content_parts.append(parsed.get("answer")[:250])
                                    else:
                                        content_parts.append(str(text)[:250])
                                except Exception:
                                    content_parts.append(str(text)[:250])  # Limit text length for faster processing

                    if content_parts:
                        return f"{tool_name.upper()} documents found:\n\n" + "\n\n".join(content_parts)

            return f"No relevant {tool_name.upper()} content found in response."

    except Exception as e:
        print(f"[process_structured_response] Error: {e}")
        return f"Error processing {tool_name} structured response: {str(e)}"

async def query_mcp(user_query: str, llm: str, tools: List[str], country: str, user_id: str) -> str:
    """
    Query the MCP server to retrieve medical information from azithromycin documents.
    
    This function connects to the MCP server, uses AI to select the most relevant
    document(s) based on the user's query, and returns a comprehensive medical
    response with proper citations.
    
    Args:
        user_query (str): The user's medical question about azithromycin
        llm (str): The LLM provider to use
        tools (List[str]): List of tools to use
        country (str): User's country
        
    Returns:
        str: Medical information response with citations and references
        
    Raises:
        Exception: If MCP server connection fails or query processing encounters errors
    """
    # Get dynamic MCP configuration from environment variables
    try:
        # Create a temporary config file for MCP client
        import tempfile
        import json
        
        environment = os.getenv("ENVIRONMENT", "development").lower()
        print(f"[MCP] Environment: {environment}")
        # Select URL based on environment
        if environment == "production":
            config_file = "mcp_server_config/config_production.json"
        else:
            config_file = "mcp_server_config/config_development.json"

        print(f"[MCP] Using config file: {config_file}")
        
        try:
            client = MCPClient.from_config_file(config_file)
            print(f"[MCP] Client created successfully")
        except Exception as e:
            print(f"[MCP] ERROR: Failed to create MCP client: {e}")
            # Return a fallback response when MCP server is not available
            return f"I apologize, but I'm currently unable to access the medical document database to answer your question about '{user_query}'. The document service appears to be temporarily unavailable. Please try again in a few moments, or contact support if the issue persists."
        
        # Initialize LLM using the chooser
        llm_instance = choose_llm(llm, temperature=0.01)

        # System prompt - clear instruction 
        system_prompt = generate_system_prompt(user_query, country, tools)
        print(f"System Prompt: {system_prompt}")
        adapter = LangChainAdapter()
        
        try:
            all_tools = await adapter.create_tools(client)
            print(f"[MCP] Available tools: {[tool.name for tool in all_tools]}")
        except Exception as e:
            print(f"[MCP] ERROR: Failed to create tools from MCP client: {e}")
            # Return a fallback response when tools can't be created
            return f"I apologize, but I'm currently unable to access the medical document database to answer your question about '{user_query}'. The document service appears to be temporarily unavailable. Please try again in a few moments, or contact support if the issue persists."
        
        # CRITICAL: Filter tools to only include the ones passed in the tools parameter
        filtered_tools = []
        for tool in all_tools:
            if tool.name in tools:
                filtered_tools.append(tool)
                print(f"Including tool: {tool.name}")
            else:
                print(f"Excluding tool: {tool.name} (not in requested tools: {tools})")
        
        print(f"Filtered tools to use: {[tool.name for tool in filtered_tools]}")
        
        if not filtered_tools:
            print(" WARNING: No valid tools found after filtering!")
            return "Error: No valid tools available for the requested tool list."
        
        # Create a custom LangChain agent with only the filtered tools
        llm_with_tools = llm_instance.bind_tools(filtered_tools)
        from langchain_core.messages import ToolMessage
        
        # Step 1: Initial LLM call without timeout
        response = await llm_with_tools.ainvoke([system_prompt, user_query])
        # print("this is result",result)
        # print(type(result))
        tool_messages = []
        if response.tool_calls:
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                
                # CRITICAL: Validate that the tool being called is in the allowed tools list
                if tool_name not in tools:
                    print(f"BLOCKING unauthorized tool call: {tool_name}. Not in allowed tools: {tools}")
                    continue  # Skip this tool call entirely
                
                # Add country parameter to tool arguments
                if "country" not in tool_args:
                    tool_args["country"] = country
                
                if "user_id" not in tool_args:
                    tool_args["user_id"] = user_id 
                    
                print(f"Calling authorized tool {tool_name} with args: {tool_args}")
                
                # Find the tool in our filtered tools list
                tool_found = False
                for tool in filtered_tools:
                    if tool.name == tool_name:
                        try:
                            print(f"[query_mcp] Invoking tool: {tool_name}", file=sys.stderr)
                            result = await tool.ainvoke(tool_args)
                            print(f"[query_mcp] {tool_name} raw result type: {type(result)}", file=sys.stderr)
                            print(f"[query_mcp] {tool_name} raw result preview: {str(result)[:200]}", file=sys.stderr)
                            
                            # CRITICAL FIX: Process the MCP server response properly
                            processed_content = process_mcp_response(result, tool_name)
                            print(f"[query_mcp] {tool_name} processed content length: {len(processed_content)}", file=sys.stderr)
                            
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": processed_content
                            })
                            tool_found = True
                            break
                        except Exception as e:
                            print(f"[query_mcp] ERROR invoking tool {tool_name}: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc()
                            
                            # Add an error message for this tool
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": f"Error calling {tool_name}: {str(e)}"
                            })
                            tool_found = True
                            break
                
                if not tool_found:
                    print(f"Tool {tool_name} not found in filtered tools list")

        print(f"Tool calls executed: {len(tool_messages)} out of {len(response.tool_calls) if response.tool_calls else 0} requested")

        # Step 3: Feed full conversation + tool output back to LLM without timeout
        final_response = await llm_with_tools.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": response.tool_calls
            },
            *tool_messages
        ])

        print(final_response.content)
            
        # Handle different response formats
        if isinstance(final_response, dict):
            result = final_response.get("filter", str(final_response))
        else:
            result = final_response.content
            
        # Fallback for string responses
        return result
        
    except Exception as e:
        # Log the error and return a default or raise
        print(f"Error in query_mcp: {str(e)}")
        return "Error detected"

   
async def process_query(user_query, llm, tools, country, user_id):
    """
    Process a medical query and return the final response.
    
    This is the main query processing function that coordinates with the MCP server
    to provide medical information about azithromycin.
    
    Args:
        user_query (str): The user's medical question
        
    Returns:
        str: The processed medical information response with citations
    """

    # Get the result from MCP
    mcp_result = await query_mcp(user_query, llm, tools, country, user_id) 
    
    return mcp_result

@app.post("/highlight_pdf_text_file")
async def highlight_pdf_text_file(req: HighlightTextRequest, request: Request):
    """
    Highlight specific text in a PDF and return the highlighted PDF file directly.
    This endpoint always returns the PDF file, not metadata.
    """
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    try:
        # Highlight text in PDF using global highlighter
        result = pdf_highlighter.highlight_text_in_pdf(
            input_pdf_path=req.pdf_path,
            texts_to_highlight=req.texts_to_highlight,
            user_id=user_id,
            output_filename=req.output_filename,
            auto_cleanup=req.auto_cleanup,
            cleanup_delay=req.cleanup_delay
        )
        
        if result["success"]:
            # Always return the file directly - try local first, then MinIO
            output_path = result.get("local_file_path")
            
            # If local file exists, return it directly
            if output_path and os.path.exists(output_path):
                return FileResponse(
                    path=output_path,
                    filename=result["output_filename"],
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"inline; filename={result['output_filename']}",
                        "X-Total-Highlights": str(result["total_highlights"]),
                        "X-Source-Type": result["source_type"],
                        "X-Original-Filename": result["original_filename"],
                        "X-Highlighted-Regions": str(len(result.get("highlighted_regions", []))),
                        "Access-Control-Expose-Headers": "X-Total-Highlights,X-Source-Type,X-Original-Filename,X-Highlighted-Regions"
                    }
                )
            
            # If local file doesn't exist, try to download from MinIO
            elif result.get("minio_object_name") and pdf_highlighter.minio_client:
                try:
                    import tempfile
                    # Download from MinIO to temporary file
                    minio_object_name = result["minio_object_name"]
                    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                    
                    # Download from MinIO highlighted bucket
                    pdf_highlighter.minio_client.fget_object(
                        pdf_highlighter.highlighted_bucket, 
                        minio_object_name, 
                        temp_file.name
                    )
                    
                    print(f"Downloaded {minio_object_name} from MinIO for direct file return")
                    
                    # Return the downloaded file
                    return FileResponse(
                        path=temp_file.name,
                        filename=result["output_filename"],
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f"inline; filename={result['output_filename']}",
                            "X-Total-Highlights": str(result["total_highlights"]),
                            "X-Source-Type": "minio",
                            "X-Original-Filename": result["original_filename"],
                            "X-Highlighted-Regions": str(len(result.get("highlighted_regions", []))),
                            "Access-Control-Expose-Headers": "X-Total-Highlights,X-Source-Type,X-Original-Filename,X-Highlighted-Regions"
                        }
                    )
                    
                except Exception as e:
                    print(f"Failed to download from MinIO: {e}")
                    return JSONResponse(
                        content={"error": f"Failed to download highlighted PDF from MinIO: {str(e)}"}, 
                        status_code=500
                    )
            else:
                return JSONResponse(
                    content={"error": "Highlighted PDF file not found locally or in MinIO"}, 
                    status_code=500
                )
        else:
            return JSONResponse(
                content={"error": result["error"]}, 
                status_code=500
            )
            
    except Exception as e:
        print(f"[highlight_pdf_text_file] Error: {e}")
        return JSONResponse(
            content={"error": "Failed to highlight PDF"}, 
            status_code=500
        )

@app.post("/stream_query")
async def handle_query(req: StreamingQueryRequest, request: Request):
    print(f"[stream_query] Received streaming request for query: {req.user_query[:50]}...")
    create_tables()
    
    # PII Detection - Check for sensitive information before processing
    pii_result = detect_pii(req.user_query)
    
    if pii_result["has_high_risk_pii"]:
        print(f"[stream_query] PII detected - blocking request. Details: {pii_result['details']}")
        return JSONResponse(
            content={
                "error": "PII/PHI detected. Please remove personal information before proceeding.",
                "warning": pii_result["warning_message"],
                "pii_detected": True,
                "risk_level": pii_result["risk_level"],
                "details": pii_result["details"],
                "flagged_entities": pii_result.get("flagged_entities", [])
            }, 
            status_code=200
        )
    
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    print(f"[stream_query] Using user_id: {user_id}")
    
    # Get page_id from headers if available, otherwise use request body
    header_page_id = request.headers.get("X-Page-ID")
    final_page_id = header_page_id or req.page_id
    
    # If no page_id provided, generate a new one (new chat session)
    if not final_page_id:
        final_page_id = str(uuid.uuid4())
        print(f"[stream_query] Generated new page_id for new chat: {final_page_id}")
    
    print(f"[stream_query] Using page_id: {final_page_id}")
    
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

async def stream_ai_response(
    req: StreamingQueryRequest, 
    user_id: str, 
    country: str, 
    page_id: str, 
    request: Request
) -> AsyncGenerator[str, None]:
    """
    Generator function that streams AI responses word by word
    """
    try:
        # Generate title and register page
        title = generate_title(req.user_query)
        
        page_context = {
            "page_id": page_id,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "client_ip": request.client.host,
            "page_title": title,
            "page_url": request.headers.get("X-Page-URL", "unknown")
        }
        
        try:
            await DatabaseService.register_page(
                user_id=user_id,
                page_id=page_id,
                page_title=title,
                page_url=request.headers.get("X-Page-URL", "unknown"),
            )
        except Exception as e:
            print(f"[stream_query] Error registering page: {e}")
        
        # Send initial metadata
        conversation_id = str(uuid.uuid4())
        session_id = page_id  # Use page_id as session_id for consistency
        
        initial_data = {
            "type": "metadata",
            "conversation_id": conversation_id,
            "session_id": session_id,
            "page_id": page_id,
            "page_title": title,
            "timestamp": datetime.utcnow().isoformat()
        }
        yield f"data: {json.dumps(initial_data)}\n\n"
        
        # Get full response from MCP with error handling
        try:
            full_response = await query_mcp(req.user_query, req.llm, req.tools, country, user_id)
            print(f"[stream_query] Full MCP response: {full_response[:500]}...")
        except Exception as e:
            print(f"[stream_query] ERROR: MCP query failed: {e}")
            full_response = f"I apologize, but I'm currently unable to access the medical document database to answer your question. The service appears to be temporarily unavailable. Please try again in a few moments."
        
        # Parse the response to extract references and other metadata
        references = []
        flagging_value = ""
        clean_answer = full_response
        
        # Try to extract JSON from the response
        clean_answer = full_response  # Default fallback
        references = []
        flagging_value = ""
        
        if full_response and "{" in full_response and "}" in full_response:
            try:
                # Try to parse the response directly as JSON first
                parsed_response = json.loads(full_response)
                
                if isinstance(parsed_response, dict):
                    clean_answer = parsed_response.get("answer", full_response)
                    references = parsed_response.get("references", [])
                    flagging_value = parsed_response.get("flagging_value", "")
                    print(f"[stream_query] Successfully parsed JSON response")
                    print(f"[stream_query] Extracted clean_answer length: {len(clean_answer)}")
                    print(f"[stream_query] Clean answer preview: {clean_answer[:200]}...")
                else:
                    clean_answer = full_response
                    print(f"[stream_query] Response is not a dict, using full response")
                    
            except json.JSONDecodeError as e:
                print(f"[stream_query] Error parsing response as JSON: {e}")
                # Try regex extraction as fallback
                try:
                    import re
                    json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        parsed_response = json.loads(json_str)
                        
                        if isinstance(parsed_response, dict):
                            clean_answer = parsed_response.get("answer", full_response)
                            references = parsed_response.get("references", [])
                            flagging_value = parsed_response.get("flagging_value", "")
                            print(f"[stream_query] Successfully parsed JSON via regex")
                        else:
                            clean_answer = full_response
                            print(f"[stream_query] Regex extracted JSON is not a dict")
                    else:
                        clean_answer = full_response
                        print(f"[stream_query] No JSON match found via regex")
                except Exception as e2:
                    print(f"[stream_query] Fallback JSON parsing also failed: {e2}")
                    clean_answer = full_response
            except Exception as e:
                print(f"[stream_query] Error parsing response JSON: {e}")
                clean_answer = full_response
        else:
            print(f"[stream_query] No JSON structure found in response, using full response")
            clean_answer = full_response
        
        # Stream the response based on stream_type
        print(f"[stream_query] Starting to stream. clean_answer length: {len(clean_answer)}")
        print(f"[stream_query] Stream type: {req.stream_type}")
        if req.stream_type == "word":
            async for chunk in stream_by_words(clean_answer):
                yield chunk
        elif req.stream_type == "sentence":
            async for chunk in stream_by_sentences(clean_answer):
                yield chunk
        else:  # default to token-like streaming
            async for chunk in stream_by_tokens(clean_answer):
                yield chunk
        
        # Send final data with references
        if references or flagging_value:
            final_data = {
                "type": "final",
                "references": references,
                "flagging_value": flagging_value,
                "conversation_id": conversation_id
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        # Store conversation after streaming
        try:
            await DatabaseService.store_page_conversation(
                user_id=user_id,
                user_query=req.user_query,
                conversation_id=conversation_id,
                assistant_response=full_response,  # Store original response with JSON
                page_context=page_context,
                session_id=session_id
            )
        except Exception as e:
            print(f"[stream_query] Error storing conversation: {e}")
        
        # Send completion signal
        completion_data = {
            "type": "complete",
            "conversation_id": conversation_id,
            "total_length": len(full_response)
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        yield f"data: {json.dumps(error_data)}\n\n"

async def stream_by_words(text: str) -> AsyncGenerator[str, None]:
    """Stream text word by word with paragraph detection"""
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    
    # Split by paragraphs
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
    
    total_words = sum(len(p.split()) for p in paragraphs)
    word_index = 0
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Stream words
        words = paragraph.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            data = {
                "type": "chunk",
                "content": chunk,
                "index": word_index,
                "total_words": total_words
            }
            yield f"data: {json.dumps(data)}\n\n"
            word_index += 1
            await asyncio.sleep(0.1)
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"

async def stream_by_sentences(text: str) -> AsyncGenerator[str, None]:
    """Stream text sentence by sentence with paragraph detection"""
    import re
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    
    # Split by paragraphs first
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
    
    sentence_index = 0
    total_sentences = sum(len([s for s in re.split(r'[.!?]+', p) if s.strip()]) for p in paragraphs)
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Split paragraph into sentences
        sentences = re.split(r'[.!?]+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Add punctuation back except for last sentence
            if i < len(sentences) - 1:
                sentence += ". "
            
            data = {
                "type": "chunk", 
                "content": sentence,
                "index": sentence_index,
                "total_sentences": total_sentences
            }
            yield f"data: {json.dumps(data)}\n\n"
            sentence_index += 1
            await asyncio.sleep(0.3)
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"

async def stream_by_tokens(text: str) -> AsyncGenerator[str, None]:
    """Stream text token by token with realistic delays and paragraph detection"""
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n for paragraph detection
    escaped_pattern = '\\n\\n'
    actual_pattern = '\n\n'
    print(f"[STREAMING DEBUG] Original text contains {escaped_pattern}: {escaped_pattern in text}")
    print(f"[STREAMING DEBUG] Original text first 200 chars: {repr(text[:200])}")
    
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    print(f"[STREAMING DEBUG] After conversion, contains actual newlines: {actual_pattern in processed_text}")
    print(f"[STREAMING DEBUG] Processed text first 200 chars: {repr(processed_text[:200])}")
    
    # Split by actual newlines to detect paragraphs
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    print(f"[STREAMING DEBUG] Found {len(paragraphs)} paragraphs after splitting")
    for i, para in enumerate(paragraphs):
        print(f"[STREAMING DEBUG] Paragraph {i}: {para[:20]}...")
    
    # If no paragraphs found (old format), treat as single paragraph
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
        print(f"[STREAMING DEBUG] Fallback to single paragraph mode")
    
    total_words = sum(len(p.split()) for p in paragraphs)
    word_index = 0
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Detect source type based on content
        source_type = "unknown"
        para_lower = paragraph.lower()
        
        if para_idx == 0:
            source_type = "PIS"
        elif para_idx == 1:
            source_type = "LRD"
        elif para_idx == 2:
            source_type = "OTHER"
        else:
            source_type = "OTHER"
        
        # Enhanced source detection
        if "no source 1" in para_lower or "no pi document" in para_lower:
            source_type = "PIS"
        elif "no source 2" in para_lower or "no lrd document" in para_lower:
            source_type = "LRD"
        elif "no source 3" in para_lower or "past cases" in para_lower:
            source_type = "OTHER"
        
        # Stream words in this paragraph
        words = paragraph.split()
        for i, word in enumerate(words):
            token = word if i == 0 else f" {word}"
            
            data = {
                "type": "chunk",
                "content": token,
                "index": word_index,
                "total_tokens": total_words,
                "is_citation": bool(re.search(r'\[\d+\.\d+\]', token))
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            word_index += 1
            
            # Variable delay to simulate thinking
            if word_index < 5:  # Start slower
                await asyncio.sleep(0.2)
            elif any(punct in word for punct in '.!?'):  # Pause at sentence endings
                await asyncio.sleep(0.3)
            else:
                await asyncio.sleep(0.08)  # Normal speed
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"
        
        # Add slight delay between paragraphs
        if para_idx < len(paragraphs) - 1:
            await asyncio.sleep(0.5)

@app.post("/stream_query_sse")
async def handle_sse_streaming_query(req: StreamingQueryRequest, request: Request):
    """
    Server-Sent Events streaming endpoint for better browser compatibility
    """
    create_tables()
    
    # PII Detection - Check for sensitive information before processing
    pii_result = detect_pii(req.user_query)
    
    if pii_result["has_high_risk_pii"]:
        print(f"[stream_query_sse] PII detected - blocking request. Details: {pii_result['details']}")
        return JSONResponse(
            content={
                "error": "PII/PHI detected. Please remove personal information before proceeding.",
                "warning": pii_result["warning_message"],
                "pii_detected": True,
                "risk_level": pii_result["risk_level"],
                "details": pii_result["details"],
                "flagged_entities": pii_result.get("flagged_entities", [])
            }, 
            status_code=400
        )
    
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    # Get page_id from headers if available, otherwise use request body
    header_page_id = request.headers.get("X-Page-ID")
    final_page_id = header_page_id or req.page_id
    
    if not final_page_id:
        return JSONResponse(
            content={"error": "Page ID not found in headers or request body"}, 
            status_code=400
        )
    
    # Create SSE streaming response
    return StreamingResponse(
        sse_stream_ai_response(req, user_id, country, final_page_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

async def sse_stream_ai_response(
    req: StreamingQueryRequest, 
    user_id: str, 
    country: str, 
    page_id: str, 
    request: Request
) -> AsyncGenerator[str, None]:
    """
    Server-Sent Events generator for streaming AI responses
    """
    try:
        # Send connection established event
        yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"
        
        # Generate response
        full_response = await query_mcp(req.user_query, req.llm, req.tools, country, user_id)
        
        # Stream character by character for smooth typing effect
        for i, char in enumerate(full_response):
            event_data = {
                "char": char,
                "position": i,
                "total_length": len(full_response)
            }
            yield f"event: char\ndata: {json.dumps(event_data)}\n\n"
            await asyncio.sleep(0.03)  # Adjust typing speed
        
        # Send completion event
        yield f"event: complete\ndata: {json.dumps({'message': 'Response complete'})}\n\n"
        
    except Exception as e:
        error_event = {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

@app.post("/test_pii")
async def test_pii_detection(request: dict):
    """
    Test endpoint to verify PII detection is working correctly.
    Send POST request with: {"text": "My name is John Doe and my email is john@example.com"}
    """
    text = request.get("text", "")
    if not text:
        return JSONResponse(
            content={"error": "Text field is required"}, 
            status_code=400
        )
    
    pii_result = detect_pii(text)
    
    return JSONResponse(content={
        "original_text": text,
        "pii_detection_available": PII_DETECTION_AVAILABLE,
        "has_high_risk_pii": pii_result["has_high_risk_pii"],
        "risk_level": pii_result["risk_level"],
        "warning_message": pii_result["warning_message"],
        "details": pii_result["details"],
        "sanitized_text": pii_result.get("sanitized_text", text),
        "would_block_request": pii_result["has_high_risk_pii"]
    })

@app.get("/get_user_chat_sessions")
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

def debug_paragraph_detection(text: str) -> dict:
    """Debug function to show how paragraph detection works"""
    print(f"[DEBUG] Original text: {repr(text[:200])}...")
    
    # Step 1: Convert escaped newlines
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    print(f"[DEBUG] After newline conversion: {repr(processed_text[:200])}...")
    
    # Step 2: Split by paragraphs
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    print(f"[DEBUG] Found {len(paragraphs)} paragraphs:")
    for i, para in enumerate(paragraphs):
        print(f"[DEBUG] Paragraph {i}: {para[:20]}...")
    
    return {
        "original_length": len(text),
        "processed_length": len(processed_text),
        "paragraph_count": len(paragraphs),
        "paragraphs": [p[:20] + "..." if len(p) > 20 else p for p in paragraphs]
    }