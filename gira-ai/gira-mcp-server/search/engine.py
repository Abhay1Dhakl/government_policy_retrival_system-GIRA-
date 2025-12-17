"""Government Information Retrieval System (GIRS) - Hybrid Search Engine

Core hybrid search execution combining dense embeddings and BM25 sparse search
for government policy document retrieval.
"""

import asyncio
import time
import copy
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.constants import (
    STOPWORDS,
    DOCUMENT_TYPE_SYNONYMS,
    SECTION_PRIORITY_WEIGHTS,
    REGION_ALIASES,
)
from search.scoring import apply_quality_scoring, compute_quality_bonus, tokenize_text, extract_prf_terms
from search.parsing import parse_pinecone_response
from optimization.concept_expander import build_expanded_queries
from optimization.adaptive_alpha import get_alpha_recommendation
from embeddings.manager import get_embedding_async

# Import global instances
from _utils import document_index, rank_bm25, _policy_corpus, _corpus_last_updated, _corpus_update_interval


def expand_document_type(document_type: Optional[str]):
    """Expand document type with synonyms"""
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


def normalize_region_filter(country: Optional[str]):
    """Normalize region filter"""
    if not country:
        return None, []
    normalized = country.strip().lower()
    if normalized in REGION_ALIASES:
        variants = set(REGION_ALIASES[normalized])
        variants.add(normalized.upper())
        return normalized.upper(), sorted(variants)
    return country.upper(), [country.upper()]


async def execute_pinecone_query_async(query_vector: list, filter_dict: dict, top_k: int = 10):
    """Execute Pinecone query in thread pool to avoid blocking"""
    import sys
    from concurrent.futures import ThreadPoolExecutor
    
    if document_index is None:
        print("⚠️  Pinecone not initialized, returning empty results", file=sys.stderr)
        return {"matches": []}
    
    loop = asyncio.get_event_loop()
    _thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def query_pinecone():
        print(f" PINECONE QUERY DEBUG:", file=sys.stderr)
        print(f"   Filter: {filter_dict}", file=sys.stderr)
        print(f"   Top K: {top_k}", file=sys.stderr)
        
        result = document_index.query(
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            vector=query_vector,
            filter=filter_dict
        )
        
        print(f"📊 PINECONE RESULT DEBUG:", file=sys.stderr)
        print(f"   Total matches: {len(result.get('matches', []))}", file=sys.stderr)
        
        if result.get('matches'):
            first_match = result['matches'][0]
            print(f"   First match ID: {first_match.get('id', 'NO_ID')}", file=sys.stderr)
            print(f"   First match score: {first_match.get('score', 'NO_SCORE')}", file=sys.stderr)
        else:
            print(f"   ⚠️ NO MATCHES FOUND!", file=sys.stderr)
        
        return result
    
    return await loop.run_in_executor(_thread_pool, query_pinecone)


async def get_bm25_scores(query: str, corpus: List[str]) -> Dict[str, float]:
    """Get BM25 scores for query against corpus"""
    if not rank_bm25 or not corpus:
        return {}
    
    try:
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = rank_bm25(tokenized_corpus)
        
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        return {term: score for term, score in zip(corpus, scores) if score > 0}
    except Exception as e:
        return {}


async def execute_hybrid_search(query: str, document_type: str, country: str = None, user_id: str = None, top_k: int = 30, alpha: float = None) -> Dict[str, Any]:
    """Execute hybrid search combining dense embeddings, BM25 signals, and quality scoring."""
    import sys
    
    start_time = time.time()

    try:
        # Adaptive alpha determination
        alpha_recommendation = get_alpha_recommendation(query, {
            'document_type': document_type,
            'country': country
        })
        actual_alpha = alpha if alpha is not None else alpha_recommendation.alpha

        embedding_start = time.time()
        query_vector = await get_embedding_async(query, task_type="retrieval_query")
        embedding_time = time.time() - embedding_start

        bm25_start = time.time()
        bm25_scores: Dict[str, float] = {}
        if _policy_corpus:
            bm25_scores = await get_bm25_scores(query, _policy_corpus)
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

        # Fallback handling for region and document type
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

        # Apply hybrid scoring
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
                "pinecone_time": time.time() - pinecone_start,
                "bm25_terms_found": len(bm25_scores),
                "hybrid_ranking_applied": bool(bm25_scores and matches),
                "prf_terms": prf_terms,
                "concept_variants": concept_variants,
                "bm25_terms_used": bm25_terms_used,
                "fallback_region_used": fallback_region_used,
                "fallback_document_type_used": fallback_document_type_used,
                "document_filter_terms": final_document_filter_terms,
                "region_filter_terms": final_region_filter_terms,
                "adaptive_alpha": {
                    "value": round(actual_alpha, 3),
                    "recommended": round(alpha_recommendation.alpha, 3),
                    "confidence": round(alpha_recommendation.confidence, 3),
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
