"""GIRS Embeddings Management Module

Manages vector embeddings for government documents using Gemini API,
with corpus building and BM25 integration for hybrid search.
"""

import sys
from functools import lru_cache
from typing import Optional, List
from datetime import datetime, timedelta
import re

# Import from gemini embeddings module (local file in embeddings folder)
try:
    from embeddings.gemini_embeddings import get_gemini_embedding_async, initialize_gemini, get_gemini_embedding
    gemini_available = initialize_gemini()
    if gemini_available:
        print("✅ Gemini API initialized successfully", file=sys.stderr)
    else:
        print("⚠️ Gemini API not available", file=sys.stderr)
except ImportError as e:
    gemini_available = False
    print(f"⚠️ Gemini embeddings module not available: {e}", file=sys.stderr)

from _utils import rank_bm25, _policy_corpus, _corpus_last_updated, _corpus_update_interval


@lru_cache(maxsize=500)
def get_cached_gemini_embedding(query: str, task_type: str = "retrieval_query") -> Optional[List[float]]:
    """Cache Gemini embeddings for frequently used queries"""
    if not gemini_available:
        return None
    
    return get_gemini_embedding(query, task_type=task_type)


async def get_embedding_async(query: str, task_type: str = "retrieval_query") -> List[float]:
    """Get embedding asynchronously using Gemini API"""
    
    if not gemini_available:
        print("⚠️ Gemini API not available", file=sys.stderr)
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


def extract_policy_corpus_from_documents(documents: list) -> List[str]:
    """Extract governmental terms and phrases from documents"""
    corpus_terms = set()
    
    policy_patterns = [
        r'\b[A-Z][a-z]+(?:ment|tion|ance|ence|ure|ness|ship|hood)\b',
        r'\b\w*(?:lation|ative|atory|ible|able|ful|less|ward|wise)\b',
        r'\b\d+\s*(?:section|article|clause|chapter|part|division|title)\b',
        r'\b(?:Act|Amendment|Regulation|Directive|Policy|Statute|Ordinance|Bylaw)\b',
        r'\b(?:Legislative|Executive|Judicial|Administrative|Municipal|Federal|State|National)\b',
        r'\b(?:penalty|fine|sanction|prohibition|restriction|mandate|requirement|obligation)\b',
        r'\b(?:enforcement|compliance|jurisdiction|authority|power|right|duty|liability)\b',
        r'\b(?:government|ministry|department|agency|commission|board|committee|council)\b',
        r'\b(?:shall|must|may|should|cannot|prohibited|required|forbidden|allowed)\b',
        r'\b(?:amendment|provision|exemption|exception|waiver|appeal|dispute)\b'
    ]
    
    for doc in documents:
        text_content = doc.get('metadata', {}).get('text', '')
        if text_content:
            for pattern in policy_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                corpus_terms.update([match.lower() for match in matches])
            
            words = re.findall(r'\b[A-Za-z]{3,}\b', text_content)
            for i, word in enumerate(words[:-1]):
                if word.lower() in ['section', 'article', 'amendment', 'statute', 'regulation', 'legislation', 'authority', 'jurisdiction']:
                    if i + 1 < len(words):
                        corpus_terms.add(f"{word.lower()} {words[i+1].lower()}")
    
    return list(corpus_terms)


async def build_dynamic_corpus():
    """Build government policy corpus from actual documents in Pinecone"""
    global _policy_corpus, _corpus_last_updated
    
    try:
        from search.engine import execute_pinecone_query_async
        
        policy_queries = [
            "statutory provisions", "regulatory requirements", "enforcement authority", "compliance obligations", 
            "legislative amendments", "policy directives", "government regulations", "legal penalties",
            "jurisdiction and authority", "administrative procedures", "appeal mechanisms", "exemptions and exceptions",
            "government agencies", "ministerial powers", "enforcement mechanisms", "stakeholder obligations",
            "section provisions", "article requirements", "clause definitions", "penalty provisions",
            "constitutional rights", "fundamental freedom", "federal structure", "judicial review",
            "education policy", "school management", "university grants Commission",
            "public health", "medical registration", "government directive"
        ]
        
        all_documents = []
        
        # Search for documents. 
        # Note: Currently many documents are indexed as 'pis' in the database.
        # We search with no filter for the corpus build to catch everything.
        for query in policy_queries:
            query_vector = await get_embedding_async(query)
            
            try:
                response = await execute_pinecone_query_async(
                    query_vector=query_vector,
                    filter_dict={},  # Use no filter to catch all current documents (pis, act, etc)
                    top_k=50
                )
                all_documents.extend(response.get("matches", []))
            except Exception as e:
                pass
        
        _policy_corpus = extract_policy_corpus_from_documents(all_documents)
        _corpus_last_updated = datetime.now()
        
        return _policy_corpus
        
    except Exception as e:
        _policy_corpus = [
            "statutory provisions", "regulatory requirements", "enforcement authority", "compliance obligations",
            "legislative amendments", "policy directives", "government regulations", "legal penalties",
            "jurisdiction provisions", "administrative procedures", "appeal mechanisms", "exemption clauses",
            "government agencies", "ministerial authority", "enforcement powers", "stakeholder responsibilities",
            "penalty provisions", "compliance requirements", "authority limits", "mandatory obligations"
        ]
        return _policy_corpus


async def update_bm25_with_dynamic_corpus():
    """Update BM25 encoder with dynamic government policy corpus"""
    if not rank_bm25:
        return None
    
    try:
        if (_corpus_last_updated is None or 
            datetime.now() - _corpus_last_updated > timedelta(seconds=_corpus_update_interval)):
            await build_dynamic_corpus()
        
        tokenized_corpus = [term.split() for term in _policy_corpus]
        bm25 = rank_bm25(tokenized_corpus)
        return bm25
        
    except ImportError:
        return None
    except Exception as e:
        return None
