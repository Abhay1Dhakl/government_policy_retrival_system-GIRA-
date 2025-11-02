from mcp.server.fastmcp import FastMCP 
import os
import json
import asyncio
import time
import re
import copy
from functools import lru_cache
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sklearn.decomposition import PCA
import numpy as np
import pickle

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import Pinecone
from concept_expander import build_expanded_queries

# Import Gemini embeddings
try:
    from gemini_embeddings import get_gemini_embedding_async, initialize_gemini
    gemini_available = initialize_gemini()
    if gemini_available:
        print("✅ Gemini API initialized successfully", file=sys.stderr)
    else:
        print("⚠️ Gemini API not available", file=sys.stderr)
except ImportError as e:
    gemini_available = False
    print(f"⚠️ Gemini embeddings module not available: {e}", file=sys.stderr)

# Optional imports with graceful error handling
bm25_encoder = None
rank_bm25 = None

try:
    from pinecone_text.sparse import BM25Encoder
    from rank_bm25 import BM25Okapi
    rank_bm25 = BM25Okapi
    print("✓ BM25 encoder available", file=sys.stderr)
except ImportError as e:
    print(f"⚠ BM25 encoder not available: {e} - semantic search only", file=sys.stderr)
except Exception as e:
    print(f"⚠ BM25 encoder initialization failed: {e} - semantic search only", file=sys.stderr)

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY", "pcsk_2RGA3Z_LVfVmxNQ7A7DX7w5BuhEW4MTCGmGuSghX7GmMwizqWqVCumyrWCcMdtE1jDxgav"),
    environment="aped-4627-b74a"
)

mcp = FastMCP(
    name="mcp_server",
    host="0.0.0.0",
    port=8001,
    debug=False
)

document_index_host = pc.describe_index(name="quickstart-py").host
document_index = pc.Index(host=document_index_host, grpc_config=GRPCClientConfig(secure=False))

# Global thread pool for CPU-intensive tasks
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Global variables for corpus
_medical_corpus = []
_corpus_last_updated = None
_corpus_update_interval = int(os.getenv("CORPUS_UPDATE_INTERVAL", "3600"))  # 1 hour default

# Note: No longer using local embedding model - using Gemini API

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "shall",
    "under", "when", "into", "upon", "such", "which", "been", "were", "your", "their",
    "than", "about", "each", "within", "while", "those", "these", "there", "after",
    "before", "during", "because", "other", "where", "patients", "patient", "should",
    "could", "would", "might", "effect", "effects", "used", "using", "use", "dose",
    "doses", "per", "day", "days", "week", "weeks", "month", "months", "or", "of",
    "via", "into", "onto"
}

DOCUMENT_TYPE_SYNONYMS = {
    "pis": {
        "pis", "pi", "prescribing_information", "prescribing-information",
        "prescribing information", "product_information", "product-information",
        "product information"
    },
    "lrd": {
        "lrd", "label_repository_data", "label repository data",
        "label_repository_document", "label repository document", "label-data"
    },
    "hpl": {
        "hpl", "health_product_label", "health product label", "product label",
        "product_label"
    },
    "past_cases": {"past_cases", "past-cases", "history", "user_history"}
}

SECTION_PRIORITY_WEIGHTS = {
    "warning": 0.25,
    "contraindication": 0.25,
    "safety": 0.22,
    "adverse": 0.2,
    "reaction": 0.18,
    "overdose": 0.18,
    "dosage": 0.15,
    "pediatric": 0.15,
    "geriatric": 0.12
}

REGION_ALIASES = {
    "us": {"US", "USA", "UNITED STATES", "UNITED_STATES"},
    "eu": {"EU", "EUROPE", "EMA"},
    "uk": {"UK", "UNITED KINGDOM", "UNITED_KINGDOM", "GB"},
    "np": {"NP", "NEPAL"},
    "ca": {"CA", "CANADA"},
    "global": {"GLOBAL", "WORLDWIDE"}
}


def expand_document_type(document_type: Optional[str]) -> Tuple[Optional[str], List[str]]:
    if not document_type:
        return None, []
    normalized = document_type.strip().lower()
    for canonical, synonyms in DOCUMENT_TYPE_SYNONYMS.items():
        if normalized == canonical or normalized in synonyms:
            expanded = {canonical}
            expanded.update({s.lower() for s in synonyms})
            expanded.update({s.upper() for s in synonyms})
            expanded.add(canonical.upper())
            return canonical, sorted(expanded)
    return normalized, [normalized, normalized.upper()]


def normalize_region_filter(country: Optional[str]) -> Tuple[Optional[str], List[str]]:
    if not country:
        return None, []
    normalized = country.strip().lower()
    if normalized in REGION_ALIASES:
        variants = set(REGION_ALIASES[normalized])
        variants.add(normalized.upper())
        return normalized.upper(), sorted(variants)
    return country.upper(), [country.upper()]


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def extract_prf_terms(matches: List[Dict[str, Any]], query: str, max_terms: int = 5) -> List[str]:
    if not matches:
        return []
    query_tokens = set(tokenize_text(query))
    tokens: Counter = Counter()
    for match in matches[:3]:
        metadata = match.get("metadata", {}) or {}
        text_content = str(metadata.get("text", ""))
        for token in tokenize_text(text_content):
            if token in STOPWORDS or token in query_tokens:
                continue
            tokens[token] += 1
    return [term for term, _ in tokens.most_common(max_terms)]


def compute_quality_bonus(metadata: Dict[str, Any], text: str, query_tokens: List[str]) -> Tuple[float, List[str]]:
    bonus = 0.0
    factors: List[str] = []
    text_length = len(text)
    if 400 <= text_length <= 1200:
        bonus += 0.2
        factors.append("ideal_length")
    elif text_length < 200:
        bonus -= 0.15
        factors.append("too_short")
    elif text_length > 1600:
        bonus -= 0.1
        factors.append("too_long")

    # Basic query-term coverage
    if query_tokens:
        coverage_hits = sum(1 for token in query_tokens if token in text)
        if coverage_hits:
            coverage_bonus = min(0.2, (coverage_hits / len(query_tokens)) * 0.25)
            bonus += coverage_bonus
            factors.append(f"query_coverage_{coverage_hits}/{len(query_tokens)}")
        else:
            bonus -= 0.18
            factors.append("no_query_terms")

    # Pediatric-aware scoring
    section_title = str(metadata.get("section_title", "")).lower()
    chunk_type = str(metadata.get("chunk_type", "")).lower()
    is_pediatric_meta = str(metadata.get("is_pediatric", "")).lower() in {"1", "true", "yes"} or metadata.get("is_pediatric") is True
    pediatric_focus = any(token in {"child", "children", "pediatric", "paediatric", "infant", "neonate"} for token in query_tokens)
    
    # Pregnancy-aware scoring
    pregnancy_focus = any(token in {"pregnant", "pregnancy", "fetal", "fetus", "teratogenic", "birth", "defect"} for token in query_tokens)

    # Section keyword/label weighting (generic)
    section_boost_applied = None
    for keyword, weight in SECTION_PRIORITY_WEIGHTS.items():
        if keyword in section_title or keyword in chunk_type:
            bonus += weight
            section_boost_applied = (keyword, weight)
            factors.append(f"section_{keyword}")
            break

    # Specific boosts/penalties for pediatric queries
    if pediatric_focus:
        # Strong boost if metadata says pediatric or section numbering 8.4
        if is_pediatric_meta:
            bonus += 0.35
            factors.append("is_pediatric_meta")
        # Numbered pediatric section (8.4 Pediatric Use)
        if re.search(r"^\s*8(?:\.\d+)*\b", section_title) and ("pediatric" in section_title or "paediatric" in section_title):
            bonus += 0.3
            factors.append("section_8.x_pediatric")
        # Textual pediatric indicators
        if any(term in text for term in ["pediatric", "paediatric", "children", "child", "infant", "neonate", "adolescent", "under 16", "years of age"]):
            bonus += 0.2
            factors.append("text_pediatric_terms")
        # Dosage patterns common in pediatrics
        if re.search(r"\b\d+\s*mg\s*/\s*kg\b", text):
            bonus += 0.15
            factors.append("dose_mg_per_kg")
        # Down-weight unrelated sections for pediatric-focused queries
        if any(term in section_title for term in ["lactation", "pregnancy", "nonclinical", "geriatric"]):
            bonus -= 0.2
            factors.append("section_not_pediatric")

    # Specific boosts for pregnancy queries
    if pregnancy_focus:
        # Boost pregnancy-related sections
        if any(term in section_title for term in ["pregnancy", "fetal", "teratogenic", "reproduction", "developmental"]):
            bonus += 0.35
            factors.append("pregnancy_section")
        # Textual pregnancy indicators
        if any(term in text for term in ["pregnancy", "pregnant", "fetal", "fetus", "teratogenic", "birth defect", "congenital", "developmental toxicity"]):
            bonus += 0.25
            factors.append("pregnancy_content")
        # Animal study references (common in pregnancy sections)
        if any(term in text for term in ["rat", "mouse", "rabbit", "animal", "embryo", "fetus", "developmental"]):
            bonus += 0.15
            factors.append("pregnancy_studies")
        # Down-weight unrelated sections for pregnancy-focused queries
        if any(term in section_title for term in ["pediatric", "geriatric", "nonclinical"]):
            bonus -= 0.15
            factors.append("section_not_pregnancy")

    # Clamp
    bonus = max(-0.4, min(bonus, 0.85))

    # Attach which section boost was applied for observability
    if section_boost_applied:
        metadata["section_boost"] = section_boost_applied[1]
    return bonus, factors


def apply_quality_scoring(matches: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not matches:
        return matches
    query_tokens = [token for token in tokenize_text(query) if token not in STOPWORDS]
    for match in matches:
        metadata = match.get("metadata", {}) or {}
        text_content = str(metadata.get("text", "")).lower()
        base_score = match.get("hybrid_score", match.get("score", 0.0)) or 0.0
        bonus, factors = compute_quality_bonus(metadata, text_content, query_tokens)
        adjusted_score = base_score * (1 + bonus)
        match["quality_score"] = round(bonus, 4)
        match["quality_factors"] = factors
        match["score"] = adjusted_score
        # Surface section_boost if the scorer flagged one
        if isinstance(metadata, dict) and "section_boost" in metadata:
            match["section_boost"] = metadata.get("section_boost")
    matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return matches

@lru_cache(maxsize=500)
def get_cached_gemini_embedding(query: str, task_type: str = "retrieval_query") -> Optional[List[float]]:
    """Cache Gemini embeddings for frequently used queries"""
    if not gemini_available:
        return None
    
    from gemini_embeddings import get_gemini_embedding
    return get_gemini_embedding(query, task_type=task_type)

async def get_embedding_async(query: str, task_type: str = "retrieval_query") -> List[float]:
    """Get embedding asynchronously using Gemini API (384 dimensions)"""
    
    if not gemini_available:
        print("⚠️ Gemini API not available", file=sys.stderr)
        # Return zero vector as fallback
        return [0.0] * 384
    
    try:
        # Try cache first
        cached_result = get_cached_gemini_embedding(query, task_type)
        if cached_result:
            return cached_result
        
        # If cache miss, get from API
        embedding = await get_gemini_embedding_async(query, task_type)
        
        if embedding:
            return embedding
        else:
            print(f"⚠️ Failed to get Gemini embedding for query", file=sys.stderr)
            return [0.0] * 384
            
    except Exception as e:
        print(f"❌ Error getting embedding: {e}", file=sys.stderr)
        return [0.0] * 384

async def execute_pinecone_query_async(query_vector: list, filter_dict: dict, top_k: int = 10):
    """Execute Pinecone query in thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    
    def query_pinecone():
        print(f" PINECONE QUERY DEBUG:", file=sys.stderr)
        print(f"   Filter: {filter_dict}", file=sys.stderr)
        print(f"   Top K: {top_k}", file=sys.stderr)
        
        result = document_index.query(
            top_k=top_k,
            include_values=False,  # Don't include values to reduce payload
            include_metadata=True,
            vector=query_vector,
            filter=filter_dict
        )
        
        print(f"📊 PINECONE RESULT DEBUG:", file=sys.stderr)
        print(f"   Total matches: {len(result.get('matches', []))}", file=sys.stderr)
        
        # Print first match details for debugging
        if result.get('matches'):
            first_match = result['matches'][0]
            print(f"   First match ID: {first_match.get('id', 'NO_ID')}", file=sys.stderr)
            print(f"   First match score: {first_match.get('score', 'NO_SCORE')}", file=sys.stderr)
            
            metadata = first_match.get('metadata', {})
            print(f"   First match metadata keys: {list(metadata.keys())}", file=sys.stderr)
            
            # Check for source fields
            source_fields = ["file_name", "source", "filename", "document_name", "doc_name"]
            found_sources = {}
            for field in source_fields:
                if field in metadata:
                    found_sources[field] = metadata[field]
            print(f"   Found source fields: {found_sources}", file=sys.stderr)
            
            # Specifically highlight file_name since that's what we expect
            if "file_name" in metadata:
                print(f"   ✅ FOUND FILE_NAME: {metadata['file_name']}", file=sys.stderr)
            else:
                print(f"    NO FILE_NAME FIELD FOUND!", file=sys.stderr)
            
            # Print document type and region
            print(f"   Document type: {metadata.get('document_type', 'NO_DOC_TYPE')}", file=sys.stderr)
            print(f"   Region: {metadata.get('region', 'NO_REGION')}", file=sys.stderr)
            
            # Print text preview
            text_preview = str(metadata.get('text', 'NO_TEXT'))[:20]
            print(f"   Text preview: {text_preview}...", file=sys.stderr)
        else:
            print(f"   ⚠️ NO MATCHES FOUND!", file=sys.stderr)
        
        return result
    
    return await loop.run_in_executor(_thread_pool, query_pinecone)

def get_bm25_encoder():
    """Get BM25 encoder with graceful error handling"""
    global bm25_encoder
    if bm25_encoder is None and rank_bm25 is not None:
        try:
            from pinecone_text.sparse import BM25Encoder
            bm25_encoder = BM25Encoder.default()
            print("✓ BM25 encoder initialized", file=sys.stderr)
        except ImportError:
            print("⚠ BM25 encoder not available", file=sys.stderr)
        except Exception as e:
            print(f"⚠ BM25 encoder initialization failed: {e}", file=sys.stderr)
    return bm25_encoder

def extract_medical_corpus_from_documents(documents: List[Dict]) -> List[str]:
    """Extract medical terms and phrases from actual documents using comprehensive patterns"""
    corpus_terms = set()
    
    # Enhanced medical patterns with more comprehensive coverage
    medical_patterns = [
        # Drug names and formulations
        r'\b[A-Z][a-z]+(?:cillin|mycin|floxacin|prazole|sartan|statin|ide|ine|ole|ate|ium)\b',
        # Medical conditions with common suffixes
        r'\b\w*(?:itis|osis|emia|pathy|trophy|plasia|sclerosis|stenosis|megaly|algia|dynia)\b',
        # Dosages and measurements
        r'\b\d+\s*(?:mg|mcg|g|ml|L|units?|tablets?|capsules?|doses?)\b',
        # Medical procedures and tests
        r'\b(?:CT|MRI|X-ray|ECG|EKG|ultrasound|biopsy|endoscopy|surgery)\b',
        # Body systems and organs
        r'\b(?:cardiovascular|respiratory|hepatic|renal|neurological|dermatological|gastrointestinal)\b',
        # Side effects and symptoms
        r'\b(?:nausea|vomiting|diarrhea|constipation|dizziness|headache|rash|itching|swelling)\b',
        # Cardiac-specific terms
        r'\b(?:cardiotoxic|cardiac|arrhythm|QT|QTc|torsades|ventricular|atrial|bradycard|tachycard)\w*\b',
        r'\b(?:heart|cardiac|cardiovascular|electrocardiogram|palpitation|chest pain)\b',
        # Medical abbreviations
        r'\b(?:QID|BID|TID|PRN|PO|IV|IM|SC|q\d+h)\b',
        # Contraindications and warnings
        r'\b(?:contraindicated|warning|caution|adverse|reaction|interaction)\b'
    ]
    
    for doc in documents:
        text_content = doc.get('metadata', {}).get('text', '')
        if text_content:
            # Extract using patterns
            for pattern in medical_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                corpus_terms.update([match.lower() for match in matches])
            
            # Extract noun phrases (simple approach)
            words = re.findall(r'\b[A-Za-z]{3,}\b', text_content)
            for i, word in enumerate(words[:-1]):
                if word.lower() in ['side', 'adverse', 'drug', 'contraindication', 'indication', 'cardiac', 'heart', 'cardiovascular', 'qt', 'arrhythm']:
                    # Capture following medical terms
                    if i + 1 < len(words):
                        corpus_terms.add(f"{word.lower()} {words[i+1].lower()}")
                        
            # Extract specific cardiac combinations
            cardiac_patterns = [
                r'(?:qt|qtc)\s+(?:prolongation|interval|extension)',
                r'cardiac\s+(?:effects|toxicity|arrhythm|monitoring)',
                r'heart\s+(?:rhythm|rate|effects|problems)',
                r'(?:ventricular|atrial)\s+(?:arrhythm|tachycard|fibrillation)',
                r'torsades\s+de\s+pointes'
            ]
            for pattern in cardiac_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                corpus_terms.update([match.lower() for match in matches])
    
    return list(corpus_terms)

async def build_dynamic_corpus():
    """Build medical corpus from actual documents in Pinecone"""
    global _medical_corpus, _corpus_last_updated
    
    try:
        # Use specific medical queries to find relevant documents
        medical_queries = [
            "side effects", "contraindications", "dosage", "warnings", 
            "adverse reactions", "drug interactions", "toxicity", "overdose",
            "pharmacokinetics", "metabolism", "excretion", "absorption",
            "cardiotoxicity", "hepatotoxicity", "nephrotoxicity", "neurotoxicity",
            "cardiac effects", "heart effects", "QT prolongation", "QTc prolongation",
            "arrhythmia", "cardiac arrhythmia", "heart rhythm", "ventricular arrhythmia",
            "torsades de pointes", "cardiac toxicity", "cardiovascular effects",
            "electrocardiogram", "ECG changes", "cardiac monitoring"
        ]
        
        all_documents = []
        
        for query in medical_queries:
            query_vector = await get_embedding_async(query)
            
            # Query each document type
            for doc_type in ["pis", "lrd", "hpl"]:
                try:
                    response = await execute_pinecone_query_async(
                        query_vector=query_vector,
                        filter_dict={"document_type": doc_type},
                        top_k=50
                    )
                    all_documents.extend(response.get("matches", []))
                except Exception as e:
                    pass  # Continue with other document types
        
        # Extract medical corpus from documents
        _medical_corpus = extract_medical_corpus_from_documents(all_documents)
        _corpus_last_updated = datetime.now()
        
        return _medical_corpus
        
    except Exception as e:
        # Fallback to basic medical terms
        _medical_corpus = [
            "side effects", "contraindications", "dosage", "warnings", "adverse reactions",
            "cardiotoxicity", "hepatotoxicity", "drug interactions", "toxicity",
            "cardiac effects", "heart effects", "QT prolongation", "QTc prolongation",
            "arrhythmia", "cardiac arrhythmia", "ventricular arrhythmia", "torsades de pointes",
            "cardiac toxicity", "cardiovascular effects", "ECG changes", "cardiac monitoring"
        ]
        return _medical_corpus

async def update_bm25_with_dynamic_corpus():
    """Update BM25 encoder with dynamic medical corpus"""
    if not rank_bm25:
        return None
    
    try:
        # Check if corpus needs updating
        if (_corpus_last_updated is None or 
            datetime.now() - _corpus_last_updated > timedelta(seconds=_corpus_update_interval)):
            await build_dynamic_corpus()
        
        # Tokenize corpus for BM25
        tokenized_corpus = [term.split() for term in _medical_corpus]
        bm25 = rank_bm25(tokenized_corpus)
        return bm25
        
    except ImportError:
        return None
    except Exception as e:
        return None

async def get_bm25_scores(query: str, corpus: List[str]) -> Dict[str, float]:
    """Get BM25 scores for query against corpus"""
    if not rank_bm25 or not corpus:
        return {}
    
    try:
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = rank_bm25(tokenized_corpus)
        
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Return as dictionary with terms and scores
        return {term: score for term, score in zip(corpus, scores) if score > 0}
    except Exception as e:
        return {}

async def execute_hybrid_search(query: str, document_type: str, country: str = None, user_id: str = None, top_k: int = 30, alpha: float = None) -> Dict[str, Any]:
    """Execute hybrid search combining dense embeddings, BM25 signals, and quality scoring."""
    start_time = time.time()

    try:
        # Adaptive alpha determination
        from adaptive_alpha import get_alpha_recommendation
        alpha_recommendation = get_alpha_recommendation(query, {
            'document_type': document_type,
            'country': country
        })
        actual_alpha = alpha if alpha is not None else alpha_recommendation.alpha

        embedding_start = time.time()
        # Use retrieval_query task type for search queries
        query_vector = await get_embedding_async(query, task_type="retrieval_query")
        embedding_time = time.time() - embedding_start

        bm25_start = time.time()
        bm25_scores: Dict[str, float] = {}
        if _medical_corpus:
            bm25_scores = await get_bm25_scores(query, _medical_corpus)
        bm25_time = time.time() - bm25_start

        canonical_doc_type, doc_type_variants = expand_document_type(document_type)
        _, region_variants = normalize_region_filter(country)

        filter_dict: Dict[str, Any] = {}
        if doc_type_variants:
            filter_dict["document_type"] = {"$in": doc_type_variants} if len(doc_type_variants) > 1 else doc_type_variants[0]
        if region_variants:
            filter_dict["region"] = {"$in": region_variants} if len(region_variants) > 1 else region_variants[0]
        if user_id:
            filter_dict["user_id"] = user_id

        active_filter = copy.deepcopy(filter_dict)
        pinecone_start = time.time()
        pinecone_response = await execute_pinecone_query_async(query_vector, active_filter, top_k)
        matches = list(pinecone_response.get("matches", []))

        query_variants_detail: List[Dict[str, Any]] = [{
            "source": "base",
            "query": query,
            "result_count": len(matches),
            "filter": copy.deepcopy(active_filter)
        }]

        fallback_region_used = False
        fallback_document_type_used = False
        allow_doc_type_fallback = canonical_doc_type not in {"pis", "lrd", "hpl", "past_cases"}

        if not matches and "region" in active_filter:
            fallback_filter = copy.deepcopy(active_filter)
            fallback_filter.pop("region", None)
            fallback_response = await execute_pinecone_query_async(query_vector, fallback_filter, top_k)
            fallback_matches = list(fallback_response.get("matches", []))
            query_variants_detail.append({
                "source": "fallback_region",
                "query": query,
                "result_count": len(fallback_matches),
                "filter": copy.deepcopy(fallback_filter)
            })
            if fallback_matches:
                matches = fallback_matches
                active_filter = fallback_filter
                fallback_region_used = True

        if not matches and allow_doc_type_fallback and "document_type" in active_filter:
            fallback_filter = copy.deepcopy(active_filter)
            fallback_filter.pop("document_type", None)
            fallback_response = await execute_pinecone_query_async(query_vector, fallback_filter, top_k)
            fallback_matches = list(fallback_response.get("matches", []))
            query_variants_detail.append({
                "source": "fallback_document_type",
                "query": query,
                "result_count": len(fallback_matches),
                "filter": copy.deepcopy(fallback_filter)
            })
            if fallback_matches:
                matches = fallback_matches
                active_filter = fallback_filter
                fallback_document_type_used = True

        all_results = list(matches)
        seen_ids = {match.get("id") for match in all_results if match}
        seen_query_variants = {query.lower()}

        expanded_queries = build_expanded_queries(query) or []
        concept_variants: List[str] = []
        seen_concepts = set()
        for alt_query in expanded_queries:
            candidate = alt_query.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered == query.lower() or lowered in seen_concepts:
                continue
            concept_variants.append(candidate)
            seen_concepts.add(lowered)

        prf_terms = extract_prf_terms(matches, query)

        bm25_terms_used: List[str] = []
        if bm25_scores and len(all_results) < max(3, top_k // 2):
            top_bm25_terms = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[:5]
            bm25_terms_used = [term for term, _ in top_bm25_terms]

        variant_backlog: List[Tuple[str, str]] = []
        for variant in concept_variants:
            variant_backlog.append(("concept", variant))
        for term in prf_terms:
            variant_backlog.append(("prf", f"{query} {term}"))
        for term in bm25_terms_used:
            variant_backlog.append(("bm25", f"{query} {term}"))

        for source_label, variant_query in variant_backlog:
            trimmed = variant_query.strip()
            if not trimmed:
                continue
            lowered = trimmed.lower()
            if lowered in seen_query_variants:
                continue
            # Use retrieval_query task type for query variants
            variant_vector = await get_embedding_async(trimmed, task_type="retrieval_query")
            variant_response = await execute_pinecone_query_async(variant_vector, active_filter, top_k)
            variant_matches = list(variant_response.get("matches", []))
            query_variants_detail.append({
                "source": source_label,
                "query": trimmed,
                "result_count": len(variant_matches),
                "filter": copy.deepcopy(active_filter)
            })
            seen_query_variants.add(lowered)
            for match in variant_matches:
                match_id = match.get("id") if match else None
                if match_id and match_id not in seen_ids:
                    all_results.append(match)
                    seen_ids.add(match_id)

        pinecone_time = time.time() - pinecone_start
        matches = all_results

        if bm25_scores and matches:
            for match in matches:
                metadata = match.get("metadata", {}) or {}
                text_content = str(metadata.get("text", "")).lower()
                bm25_boost = 0.0
                for term, score_value in bm25_scores.items():
                    if term in text_content:
                        bm25_boost += score_value
                original_score = match.get("score", 0.0) or 0.0
                match["bm25_boost"] = bm25_boost
                match["hybrid_score"] = actual_alpha * original_score + (1 - actual_alpha) * (bm25_boost / 10)
            matches.sort(key=lambda item: item.get("hybrid_score", item.get("score", 0.0)), reverse=True)

        matches = apply_quality_scoring(matches, query)

        # Graph-RAG expansion: expand from seed chunks using entity relationships
        graph_expansion_start = time.time()
        graph_chunks = []
        if GRAPH_AVAILABLE and GRAPH_EXPANSION_ENABLED and matches:
            try:
                # Extract seed chunks for graph expansion (use more than top_k to get better graph coverage)
                seed_chunks = matches[:min(len(matches), top_k * 2)]
                graph_chunks = await graph_expand_candidates(
                    query,
                    seed_chunks,
                    k_hop=GRAPH_MAX_HOPS,
                    max_neighbors=GRAPH_MAX_NEIGHBORS,
                    graph_boost=GRAPH_BOOST_WEIGHT
                )
                print(f"📊 Graph expansion: {len(graph_chunks)} chunks from {len(seed_chunks)} seeds")
            except Exception as e:
                print(f"⚠️ Graph expansion failed: {e}")
                graph_chunks = []
        graph_expansion_time = time.time() - graph_expansion_start

        # Fuse graph-expanded chunks with original results
        if graph_chunks and GRAPH_EXPANSION_ENABLED:
            try:
                matches = fuse_with_graph_expansion(matches, graph_chunks, graph_boost=GRAPH_BOOST_WEIGHT)
                print(f"🔗 Graph fusion complete: {len(matches)} total chunks after fusion")
            except Exception as e:
                print(f"⚠️ Graph fusion failed: {e}")

        matches = matches[:top_k]

        final_document_filter_terms = doc_type_variants if "document_type" in active_filter else []
        final_region_filter_terms = region_variants if "region" in active_filter else []

        total_time = time.time() - start_time
        processed_response = parse_pinecone_response({"matches": matches})

        return {
            "matches": processed_response["matches"],
            "search_metadata": {
                "total_time": round(total_time, 3),
                "embedding_time": round(embedding_time, 3),
                "bm25_time": round(bm25_time, 3),
                "pinecone_time": round(pinecone_time, 3),
                "bm25_terms_found": len(bm25_scores),
                "hybrid_ranking_applied": bool(bm25_scores and matches),
                "prf_terms": prf_terms,
                "concept_variants": concept_variants,
                "bm25_terms_used": bm25_terms_used,
                "fallback_region_used": fallback_region_used,
                "fallback_document_type_used": fallback_document_type_used,
                "document_filter_terms": final_document_filter_terms,
                "region_filter_terms": final_region_filter_terms,
                "query_variants_executed": query_variants_detail,
                "quality_scoring_applied": bool(matches),
                "graph_expansion_applied": GRAPH_AVAILABLE and bool(matches),
                "graph_expansion_time": round(graph_expansion_time, 3) if GRAPH_AVAILABLE else 0.0,
                "graph_chunks_added": len(graph_chunks) if GRAPH_AVAILABLE else 0,
                "adaptive_alpha": {
                    "value": round(actual_alpha, 3),
                    "recommended": round(alpha_recommendation.alpha, 3),
                    "query_type": alpha_recommendation.query_type.value,
                    "confidence": round(alpha_recommendation.confidence, 3),
                    "reasoning": alpha_recommendation.reasoning
                }
            }
        }

    except Exception as e:
        error_time = time.time() - start_time
        return {
            "error": str(e),
            "matches": [],
            "search_metadata": {
                "total_time": round(error_time, 3),
                "error": True,
                "error_message": str(e)
            }
        }
async def execute_pinecone_past_query(query: str, document_type: str, user_id: str = None, top_k: int = 20) -> Dict[str, Any]:
    """Execute a query for past cases with hybrid search"""
    return await execute_hybrid_search(query, document_type, country=None, user_id=user_id, top_k=top_k)

async def execute_pinecone_query(query: str, document_type: str, country: str = None, user_id: str = None, top_k: int = 20) -> Dict[str, Any]:
    """Execute a query against the Pinecone index with hybrid search"""
    return await execute_hybrid_search(query, document_type, country, user_id, top_k)

def parse_pinecone_response(pinecone_response):
    """Parse Pinecone response to extract metadata (optimized)"""
    if not pinecone_response or "matches" not in pinecone_response:
        return {"matches": []}
    
    # Use list comprehension with proper None handling
    matches = []
    for match in pinecone_response["matches"]:
        if match is None:
            continue
            
        # Ensure all values are serializable (no None values)
        processed_match = {
            "id": match.get("id", ""),
            "score": match.get("score", 0.0),
            "metadata": {}
        }
        
        # Process metadata safely
        raw_metadata = match.get("metadata", {})
        if raw_metadata:
            for key, value in raw_metadata.items():
                # Replace None with appropriate defaults
                if value is None:
                    processed_match["metadata"][key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    processed_match["metadata"][key] = value
                else:
                    # Convert other types to string
                    processed_match["metadata"][key] = str(value)
        
        # Add hybrid scoring fields if present
        if "hybrid_score" in match:
            processed_match["hybrid_score"] = match.get("hybrid_score", 0.0)
        if "bm25_boost" in match:
            processed_match["bm25_boost"] = match.get("bm25_boost", 0.0)
        if "quality_score" in match:
            processed_match["quality_score"] = match.get("quality_score", 0.0)
        if "quality_factors" in match:
            processed_match["quality_factors"] = match.get("quality_factors", [])
            
        matches.append(processed_match)
    
    return {"matches": matches}

def format_response(data: Any, status: str = "success") -> Any:
    """Format response data for MCP tools - return the data directly"""
    if status == "error":
        return {"error": str(data), "matches": []}
    return data

# Enhanced tool functions with hybrid search
async def execute_tool_with_timing(tool_name: str, query: str, document_type: str, country: str = None, user_id: str = None):
    """Execute tool with hybrid search and performance timing"""
    start_time = time.time()
    
    try:
        result = await execute_hybrid_search(query, document_type=document_type, country=country, user_id=user_id, top_k=20)
        
        total_time = time.time() - start_time
        result_count = len(result.get("matches", []))
        
        # Ensure result is properly serializable
        if result is None:
            return {"matches": [], "error": "No result returned"}
        
        # Validate structure
        if not isinstance(result, dict):
            return {"matches": [], "error": f"Invalid result type: {type(result)}"}
        
        # Ensure matches exist and are a list
        if "matches" not in result:
            result["matches"] = []
        elif not isinstance(result["matches"], list):
            result["matches"] = []
        
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        return {
            "matches": [],
            "error": str(e),
            "search_metadata": {
                "total_time": round(total_time, 3),
                "error": True
            }
        }

@mcp.tool(name="system_status", description="Check system status and available features")
async def system_status() -> Dict[str, Any]:
    """Get system status and available features"""
    return {
        "gemini_available": gemini_available,
        "bm25_available": rank_bm25 is not None,
        "corpus_size": len(_medical_corpus),
        "corpus_last_updated": _corpus_last_updated.isoformat() if _corpus_last_updated else None,
        "features": {
            "hybrid_search": gemini_available and rank_bm25 is not None,
            "semantic_search": gemini_available,  # Available via Gemini API
            "dynamic_corpus": True,
            "bm25_ranking": rank_bm25 is not None
        }
    }

@mcp.tool(name="rebuild_corpus", description="Manually rebuild the dynamic medical corpus")
async def rebuild_corpus() -> Dict[str, Any]:
    """Manually rebuild the dynamic medical corpus"""
    try:
        corpus = await build_dynamic_corpus()
        return {
            "status": "success",
            "corpus_size": len(corpus),
            "sample_terms": corpus[:10] if corpus else [],
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _extract_metadata_field(metadata: Dict[str, Any], field_names: List[str], default: str = "") -> str:
    """
    Extract a field from metadata trying multiple possible field names.
    
    Args:
        metadata: The metadata dictionary
        field_names: List of possible field names to try
        default: Default value if no field is found
        
    Returns:
        The first found field value as string, or default if none found
    """
    print(f"🔎 EXTRACTING METADATA FIELD:", file=sys.stderr)
    print(f"   Looking for fields: {field_names}", file=sys.stderr)
    print(f"   Available metadata keys: {list(metadata.keys())}", file=sys.stderr)
    
    for field_name in field_names:
        value = metadata.get(field_name)
        print(f"   Checking '{field_name}': {value}", file=sys.stderr)
        if value:
            result = str(value)
            print(f"   ✅ Found value for '{field_name}': {result}", file=sys.stderr)
            return result
    
    print(f"    No field found, using default: {default}", file=sys.stderr)
    return default

def _process_search_matches(matches: List[Dict[str, Any]], max_matches: int = 20) -> List[Dict[str, Any]]:
    """
    Process search matches into a standardized format with comprehensive metadata extraction.
    
    Args:
        matches: Raw matches from search results
        max_matches: Maximum number of matches to process
        
    Returns:
        List of processed matches with standardized metadata
    """
    print(f"📝 PROCESSING {len(matches)} MATCHES (max {max_matches}):", file=sys.stderr)
    processed_matches = []
    
    for i, match in enumerate(matches[:max_matches]):
        print(f"   Processing match {i+1}:", file=sys.stderr)
        
        if not match or not isinstance(match, dict):
            print(f"    Match {i+1} is invalid type: {type(match)}", file=sys.stderr)
            continue
            
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            print(f"    Match {i+1} has invalid metadata: {type(metadata)}", file=sys.stderr)
            continue
        
        print(f"   Match {i+1} metadata keys: {list(metadata.keys())}", file=sys.stderr)
        
        # Extract source filename using multiple possible field names
        print(f"    Extracting source filename for match {i+1}:", file=sys.stderr)
        document_type = metadata.get("document_type", "")
        
        # Handle different document types appropriately
        if document_type == "past_cases":
            # For past cases, use a generic identifier instead of filename
            source_filename = f"past_case_{match.get('id', 'unknown')}"
            print(f"    Past case document - using generic source: {source_filename}", file=sys.stderr)
        else:
            # For regular documents (PI, LRD, HPL), extract actual filename
            source_filename = _extract_metadata_field(
                metadata, 
                ["file_name", "source", "filename", "document_name", "doc_name"]
            )
            print(f"    Document type '{document_type}' - extracted filename: {source_filename}", file=sys.stderr)
        
        # Extract page number using multiple possible field names
        page_number = _extract_metadata_field(
            metadata,
            ["page_number", "page", "page_num", "chunk_page", "section_page"]
        )
        
        # Extract chunk index for precise referencing
        chunk_index = _extract_metadata_field(
            metadata,
            ["chunk_index", "chunk_id", "index", "section_index"]
        )
        
        # Extract section information for context
        section_info = _extract_metadata_field(
            metadata,
            ["section_title", "section_name", "chunk_type", "section_type"]
        )
        
        # Extract text content based on document type
        if document_type == "past_cases":
            # For past cases, combine question and answer for context
            question = metadata.get("question", "")
            answer = metadata.get("answer", "")
            text_content = f"Previous Question: {question}\nPrevious Answer: {answer}"
            region = "user_history"  # Override region for past cases
        else:
            # For regular documents, use the text field
            text_content = str(metadata.get("text", ""))
            region = str(metadata.get("region", ""))
        
        processed_match = {
            "id": str(match.get("id", "")),
            "score": float(match.get("score", 0.0)),
            "text": text_content[:1500],  # Increased text length for better context
            "document_type": document_type,
            "region": region,
            "source": source_filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "section_info": section_info,
            # Include additional scoring metadata if available
            "hybrid_score": match.get("hybrid_score"),
            "bm25_boost": match.get("bm25_boost"),
            "section_boost": match.get("section_boost")
        }
        
        print(f"   ✅ Match {i+1} processed - source: '{source_filename}'", file=sys.stderr)
        processed_matches.append(processed_match)
    
    print(f"📊 PROCESSING COMPLETE: {len(processed_matches)} matches processed", file=sys.stderr)
    return processed_matches

async def _execute_document_search(tool_name: str, query: str, document_type: str, country: str = None) -> Dict[str, Any]:
    """
    Execute document search with standardized error handling and response formatting.
    
    Args:
        tool_name: Name of the tool for logging/debugging
        query: Search query
        document_type: Type of document to search (lrd, pis, hpl)
        country: Optional country filter
        
    Returns:
        Standardized search response with processed matches
    """
    try:
        result = await execute_tool_with_timing(tool_name, query, document_type, country)
        
        # Initialize standardized response structure
        response = {
            "matches": [],
            "total_found": 0,
            "query_processed": query[:20],  # Increased limit for better debugging
            "country_filter": country or "none",
            "document_type": document_type,
            "search_completed": True,
            "search_metadata": result.get("search_metadata", {}),
            "sources_found": []  # Add list of source filenames found
        }
        
        # Process search results if available
        if result and isinstance(result, dict) and "matches" in result:
            matches = result["matches"]
            if matches and isinstance(matches, list):
                response["total_found"] = len(matches)
                processed_matches = _process_search_matches(matches)
                response["matches"] = processed_matches
                
                # Extract unique source filenames from processed matches
                sources = set()
                for match in processed_matches:
                    source = match.get("source", "")
                    if source and source.strip():
                        sources.add(source)
                response["sources_found"] = sorted(list(sources))
        print("the response is", response, file=sys.stderr)
        return response
        
    except Exception as e:
        # Standardized error response
        return {
            "matches": [],
            "total_found": 0,
            "query_processed": query[:20],
            "country_filter": country or "none",
            "document_type": document_type,
            "search_completed": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool(name="lrd", description="Get LRD (Label Repository Data) with hybrid search and comprehensive metadata")
async def document1(query: str, country: str = None) -> Dict[str, Any]:
    """
    Retrieve LRD documents using hybrid search with country filtering.
    
    Args:
        query: Medical search query
        country: Optional country/region filter (e.g., 'NP', 'US', 'EU')
        
    Returns:
        Structured response with processed matches including source filenames
    """
    return await _execute_document_search("LRD", query, "lrd", country)

@mcp.tool(name="pis", description="Get PI (Prescribing Information) data with hybrid search and comprehensive metadata") 
async def document2(query: str, country: str = None) -> Dict[str, Any]:
    """
    Retrieve PI documents using hybrid search with country filtering.
    
    Args:
        query: Medical search query
        country: Optional country/region filter (e.g., 'NP', 'US', 'EU')
        
    Returns:
        Structured response with processed matches including source filenames
    """
    return await _execute_document_search("PI", query, "pis", country)

@mcp.tool(name="hpl", description="Get HPL (Health Product Label) data with hybrid search and comprehensive metadata")
async def document3(query: str, country: str = None) -> Dict[str, Any]:
    """
    Retrieve HPL documents using hybrid search with country filtering.
    
    Args:
        query: Medical search query
        country: Optional country/region filter (e.g., 'NP', 'US', 'EU')
        
    Returns:
        Structured response with processed matches including source filenames
    """
    return await _execute_document_search("HPL", query, "hpl", country)

@mcp.tool(name="past_cases", description="Get past cases with hybrid search")
async def past_cases(query: str, user_id: str = None) -> Dict[str, Any]:
    """Get past_cases data from Pinecone with hybrid search and user_id filtering"""
    return await execute_tool_with_timing("past_cases", query, "past_cases", user_id=user_id)

@mcp.tool(name="debug_regions", description="Check what regions/countries are available in database")
async def debug_regions() -> Dict[str, Any]:
    """Debug tool to see what regions are available in the database"""
    try:
        start_time = time.time()
        
        # Use a simple query
        query_vector = await get_embedding_async("test query")

        pinecone_response = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={},  # No filter to get all documents
            top_k=20
        )

        regions = set()
        document_types = set()
        
        for match in pinecone_response.get("matches", []):
            metadata = match.get("metadata", {})
            if "region" in metadata:
                regions.add(metadata["region"])
            if "document_type" in metadata:
                document_types.add(metadata["document_type"])
        
        result = {
            "available_regions": sorted(list(regions)),
            "available_document_types": sorted(list(document_types)),
            "total_documents_checked": len(pinecone_response.get("matches", []))
        }
        
        debug_time = time.time() - start_time
        return result
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool(name="debug_search", description="Debug search results to see what's actually being returned")
async def debug_search(query: str = "azithromycin", document_type: str = "lrd", country: str = "NP") -> Dict[str, Any]:
    """Debug tool to see the actual search results and data structure"""
    try:
        result = await execute_hybrid_search(query, document_type, country, top_k=10)
        
        # Return a simplified structure for testing
        return {
            "debug_info": {
                "result_type": str(type(result)),
                "has_matches": "matches" in result,
                "matches_count": len(result.get("matches", [])),
                "search_metadata": result.get("search_metadata", {}),
                "error": result.get("error")
            },
            "sample_match": result.get("matches", [{}])[0] if result.get("matches") else {},
            "full_result": result
        }
        
    except Exception as e:
        return {"error": str(e), "debug_info": {"error": True}}

@mcp.tool(name="inspect_database", description="Comprehensive inspection of all documents in Pinecone database")
async def inspect_database(top_k: int = 50) -> Dict[str, Any]:
    """Comprehensive tool to inspect all documents in the database."""
    try:
        # Use a generic query to fetch documents from all types
        query_vector = await get_embedding_async("medical information")
        
        response = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={},  # No filter to get all documents
            top_k=top_k
        )
        
        matches = response.get("matches", [])
        if not matches:
            return {"error": "No documents found in database"}
            
        # Comprehensive analysis
        document_types = {}
        all_sources = set()
        regions = set()
        metadata_fields = set()
        
        for match in matches:
            metadata = match.get("metadata", {})
            if not metadata:
                continue
                
            # Track document types
            doc_type = metadata.get("document_type", "unknown")
            if doc_type not in document_types:
                document_types[doc_type] = {"count": 0, "sources": set()}
            document_types[doc_type]["count"] += 1
            
            # Collect all metadata fields
            for key in metadata.keys():
                metadata_fields.add(key)
            
            # Collect sources and regions
            for source_field in ["file_name", "source", "filename", "document_name", "doc_name"]:
                if source_field in metadata and metadata[source_field]:
                    source_name = str(metadata[source_field])
                    all_sources.add(source_name)
                    document_types[doc_type]["sources"].add(source_name)
            
            if "region" in metadata and metadata["region"]:
                regions.add(metadata["region"])
        
        # Format results
        formatted_doc_types = {}
        for doc_type, info in document_types.items():
            formatted_doc_types[doc_type] = {
                "count": info["count"],
                "sources": sorted(list(info["sources"]))
            }
        
        return {
            "total_documents_found": len(matches),
            "document_types": formatted_doc_types,
            "all_source_filenames": sorted(list(all_sources)),
            "available_regions": sorted(list(regions)),
            "metadata_fields_found": sorted(list(metadata_fields)),
            "sample_documents": matches[:3]  # Show first 3 for inspection
        }

    except Exception as e:
        return {"error": str(e)}

@mcp.tool(name="debug_document_type", description="Debug a specific document type to check its metadata structure")
async def debug_document_type(document_type: str = "pis", top_k: int = 20) -> Dict[str, Any]:
    """Debug tool to inspect the metadata of a given document type."""
    try:
        # Use a generic query to fetch some documents
        query_vector = await get_embedding_async("drug information")
        
        response = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={"document_type": document_type},
            top_k=top_k
        )
        
        matches = response.get("matches", [])
        if not matches:
            return {"error": f"No documents found for document_type='{document_type}'"}
            
        # Analyze metadata fields
        field_counts = {}
        region_values = set()
        section_title_samples = set()
        chunk_type_samples = set()
        source_filenames = set()  # NEW: Track actual source filenames

        for match in matches:
            metadata = match.get("metadata", {})
            if not metadata:
                continue
            
            for key, value in metadata.items():
                field_counts[key] = field_counts.get(key, 0) + 1
                if value:
                    if key == "region":
                        region_values.add(value)
                    if key == "section_title" and len(section_title_samples) < 10:
                        section_title_samples.add(str(value))
                    if key == "chunk_type" and len(chunk_type_samples) < 10:
                        chunk_type_samples.add(str(value))
                    # NEW: Collect actual source filenames
                    if key in ["file_name", "source", "filename", "document_name", "doc_name"] and len(source_filenames) < 20:
                        source_filenames.add(str(value))

        return {
            "document_type_analyzed": document_type,
            "documents_checked": len(matches),
            "metadata_field_counts": field_counts,
            "distinct_regions_found": sorted(list(region_values)),
            "sample_section_titles": sorted(list(section_title_samples)),
            "sample_chunk_types": sorted(list(chunk_type_samples)),
            "actual_source_filenames": sorted(list(source_filenames)),  # NEW: Show real filenames
            "sample_first_match": matches[0] if matches else {}
        }

    except Exception as e:
        return {"error": str(e)}

# Graph-RAG configuration
GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true"
GRAPH_EXPANSION_ENABLED = os.getenv("GRAPH_EXPANSION_ENABLED", "true").lower() == "true"
GRAPH_RERANKING_ENABLED = os.getenv("GRAPH_RERANKING_ENABLED", "false").lower() == "true"
GRAPH_CACHE_ENABLED = os.getenv("GRAPH_CACHE_ENABLED", "true").lower() == "true"
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "2"))
GRAPH_MAX_NEIGHBORS = int(os.getenv("GRAPH_MAX_NEIGHBORS", "10"))
GRAPH_BOOST_WEIGHT = float(os.getenv("GRAPH_BOOST_WEIGHT", "0.15"))

print(f"🔗 Graph-RAG enabled: {GRAPH_RAG_ENABLED}", file=sys.stderr)
print(f"🚀 Graph expansion enabled: {GRAPH_EXPANSION_ENABLED}", file=sys.stderr)
print(f"📊 Graph reranking enabled: {GRAPH_RERANKING_ENABLED}", file=sys.stderr)

# Graph-RAG imports
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../mira-agent/document_upload/app/services'))
    from graph_builder import get_knowledge_graph, graph_expand_candidates, fuse_with_graph_expansion
    from entity_extraction import get_entity_extractor
    GRAPH_AVAILABLE = True

    # Add caching for performance
    if GRAPH_CACHE_ENABLED:
        _graph_cache = {}

except ImportError:
    GRAPH_AVAILABLE = False
    print("⚠️ Graph-RAG modules not available", file=sys.stderr)

# Enhanced startup with system initialization
async def startup():
    """Pre-warm expensive resources and initialize hybrid search system"""
    print("MIRA MCP Server starting up...", file=sys.stderr)

    # System status report
    print("System Status:", file=sys.stderr)
    print(f" Gemini API: {'Available ✅' if gemini_available else 'Unavailable ❌'}", file=sys.stderr)
    print(f" BM25: {'Available' if rank_bm25 else 'Unavailable (semantic search only)'}", file=sys.stderr)
    print(f" Pinecone: Connected", file=sys.stderr)

    # Pre-compute a test embedding to warm up the Gemini API connection
    if gemini_available:
        print(" Warming up Gemini API...", file=sys.stderr)
        test_embedding = await get_embedding_async("medical terminology test query")
        if test_embedding:
            print(f" ✅ Gemini API ready (embedding dimension: {len(test_embedding)})", file=sys.stderr)
        else:
            print(" ⚠️ Gemini API test failed", file=sys.stderr)

    # Build initial dynamic corpus
    try:
        await build_dynamic_corpus()
        print(f" Dynamic corpus: {len(_medical_corpus)} terms loaded", file=sys.stderr)
    except Exception as e:
        print(f" Dynamic corpus initialization failed: {e}", file=sys.stderr)

    # Initialize BM25 if available
    if rank_bm25:
        try:
            await update_bm25_with_dynamic_corpus()
            print(" BM25 encoder: Initialized with medical corpus", file=sys.stderr)
        except Exception as e:
            print(f" BM25 initialization failed: {e}", file=sys.stderr)

    print("MIRA MCP Server startup complete - hybrid search ready!", file=sys.stderr)

if __name__ == "__main__":
    import asyncio
    print("Initializing advanced MIRA MCP server with hybrid search...", file=sys.stderr)
    
    try:
        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run startup tasks
        loop.run_until_complete(startup())
        
        # Start the server
        print("Starting MCP server on host=0.0.0.0, port=8001", file=sys.stderr)
        loop.run_until_complete(mcp.run(transport='sse'))
        
    except KeyboardInterrupt:
        print("MCP Server shutting down gracefully...", file=sys.stderr)
    except Exception as e:
        print(f" Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup thread pool
        print("Cleaning up resources...", file=sys.stderr)      
        _thread_pool.shutdown(wait=True)
        print("MCP Server shutdown complete", file=sys.stderr)