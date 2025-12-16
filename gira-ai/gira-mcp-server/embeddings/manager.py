"""Embeddings management module."""

import sys
from functools import lru_cache
from typing import Optional, List
from datetime import datetime, timedelta
import re

# Import from gemini embeddings module (local file in embeddings folder)
try:
    from .gemini_embeddings import get_gemini_embedding_async, initialize_gemini, get_gemini_embedding
    gemini_available = initialize_gemini()
    if gemini_available:
        print("✅ Gemini API initialized successfully", file=sys.stderr)
    else:
        print("⚠️ Gemini API not available", file=sys.stderr)
except ImportError as e:
    gemini_available = False
    print(f"⚠️ Gemini embeddings module not available: {e}", file=sys.stderr)

from .._utils import rank_bm25, _medical_corpus, _corpus_last_updated, _corpus_update_interval


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


def extract_medical_corpus_from_documents(documents: list) -> List[str]:
    """Extract medical terms and phrases from documents"""
    corpus_terms = set()
    
    medical_patterns = [
        r'\b[A-Z][a-z]+(?:cillin|mycin|floxacin|prazole|sartan|statin|ide|ine|ole|ate|ium)\b',
        r'\b\w*(?:itis|osis|emia|pathy|trophy|plasia|sclerosis|stenosis|megaly|algia|dynia)\b',
        r'\b\d+\s*(?:mg|mcg|g|ml|L|units?|tablets?|capsules?|doses?)\b',
        r'\b(?:CT|MRI|X-ray|ECG|EKG|ultrasound|biopsy|endoscopy|surgery)\b',
        r'\b(?:cardiovascular|respiratory|hepatic|renal|neurological|dermatological|gastrointestinal)\b',
        r'\b(?:nausea|vomiting|diarrhea|constipation|dizziness|headache|rash|itching|swelling)\b',
        r'\b(?:cardiotoxic|cardiac|arrhythm|QT|QTc|torsades|ventricular|atrial|bradycard|tachycard)\w*\b',
        r'\b(?:heart|cardiac|cardiovascular|electrocardiogram|palpitation|chest pain)\b',
        r'\b(?:QID|BID|TID|PRN|PO|IV|IM|SC|q\d+h)\b',
        r'\b(?:contraindicated|warning|caution|adverse|reaction|interaction)\b'
    ]
    
    for doc in documents:
        text_content = doc.get('metadata', {}).get('text', '')
        if text_content:
            for pattern in medical_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                corpus_terms.update([match.lower() for match in matches])
            
            words = re.findall(r'\b[A-Za-z]{3,}\b', text_content)
            for i, word in enumerate(words[:-1]):
                if word.lower() in ['side', 'adverse', 'drug', 'contraindication', 'indication', 'cardiac', 'heart', 'cardiovascular', 'qt', 'arrhythm']:
                    if i + 1 < len(words):
                        corpus_terms.add(f"{word.lower()} {words[i+1].lower()}")
                        
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
        from .engine import execute_pinecone_query_async
        
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
            
            for doc_type in ["pis", "lrd", "hpl"]:
                try:
                    response = await execute_pinecone_query_async(
                        query_vector=query_vector,
                        filter_dict={"document_type": doc_type},
                        top_k=50
                    )
                    all_documents.extend(response.get("matches", []))
                except Exception as e:
                    pass
        
        _medical_corpus = extract_medical_corpus_from_documents(all_documents)
        _corpus_last_updated = datetime.now()
        
        return _medical_corpus
        
    except Exception as e:
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
        if (_corpus_last_updated is None or 
            datetime.now() - _corpus_last_updated > timedelta(seconds=_corpus_update_interval)):
            await build_dynamic_corpus()
        
        tokenized_corpus = [term.split() for term in _medical_corpus]
        bm25 = rank_bm25(tokenized_corpus)
        return bm25
        
    except ImportError:
        return None
    except Exception as e:
        return None
