"""GIRS Quality Scoring and Ranking Module

Implements domain-specific quality scoring for government policy documents,
including section-based weighting and relevance re-ranking.
"""

import re
from typing import Dict, Any, List, Tuple
from collections import Counter

from core.constants import STOPWORDS, SECTION_PRIORITY_WEIGHTS


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def extract_prf_terms(matches: List[Dict[str, Any]], query: str, max_terms: int = 5) -> List[str]:
    """Extract pseudo-relevance feedback terms from top matches."""
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
    """Compute quality bonus for a match based on various factors."""
    bonus = 0.0
    factors: List[str] = []
    text_length = len(text)
    
    # Text length scoring
    if 400 <= text_length <= 1200:
        bonus += 0.2
        factors.append("ideal_length")
    elif text_length < 200:
        bonus -= 0.15
        factors.append("too_short")
    elif text_length > 1600:
        bonus -= 0.1
        factors.append("too_long")

    # Query term coverage
    if query_tokens:
        coverage_hits = sum(1 for token in query_tokens if token in text)
        if coverage_hits:
            coverage_bonus = min(0.2, (coverage_hits / len(query_tokens)) * 0.25)
            bonus += coverage_bonus
            factors.append(f"query_coverage_{coverage_hits}/{len(query_tokens)}")
        else:
            bonus -= 0.18
            factors.append("no_query_terms")

    # Section keyword weighting
    section_title = str(metadata.get("section_title", "")).lower()
    chunk_type = str(metadata.get("chunk_type", "")).lower()
    section_boost_applied = None
    
    for keyword, weight in SECTION_PRIORITY_WEIGHTS.items():
        if keyword in section_title or keyword in chunk_type:
            bonus += weight
            section_boost_applied = (keyword, weight)
            factors.append(f"section_{keyword}")
            break

    # Pediatric-aware scoring
    is_pediatric_meta = str(metadata.get("is_pediatric", "")).lower() in {"1", "true", "yes"} or metadata.get("is_pediatric") is True
    pediatric_focus = any(token in {"child", "children", "pediatric", "paediatric", "infant", "neonate"} for token in query_tokens)
    
    if pediatric_focus:
        if is_pediatric_meta:
            bonus += 0.35
            factors.append("is_pediatric_meta")
        if re.search(r"^\s*8(?:\.\d+)*\b", section_title) and ("amendment" in section_title or "provision" in section_title):
            bonus += 0.3
            factors.append("section_8.x_amendment")
        if any(term in text for term in ["amendment", "provision", "statute", "regulation", "legislative"]):
            bonus += 0.2
            factors.append("text_legislative_terms")
        if re.search(r"\b\d+\s*(?:section|article|clause)\b", text):
            bonus += 0.15
        if any(term in section_title.lower() for term in ["penalty", "penalty_clause", "enforcement", "penalties", "violation", "compliance_requirement"]):
            factors.append("legal_penalty_term")
        if any(term in section_title for term in ["exemption", "exclusion", "exception", "not_applicable"]):
            bonus -= 0.2
            factors.append("section_exclusion")

    # Compliance-focused scoring
    compliance_focus = any(token in {"compliance", "compliance_requirement", "authority", "jurisdiction", "regulatory", "enforcement"} for token in query_tokens)
    
    if compliance_focus:
        if any(term in section_title for term in ["compliance", "authority", "jurisdiction", "enforcement"]):
            bonus += 0.35
            factors.append("compliance_section")
        if any(term in text for term in ["compliance", "authority", "jurisdiction", "regulatory", "enforcement"]):
            bonus += 0.25
            factors.append("compliance_content")
        if any(term in text for term in ["government", "state", "federal", "regional", "national", "agency"]):
            bonus += 0.15
            factors.append("compliance_government_context")
        if any(term in section_title for term in ["exemption", "exclusion", "exception"]):
            bonus -= 0.15
            factors.append("section_not_compliance")

    # Clamp bonus value
    bonus = max(-0.4, min(bonus, 0.85))

    # Attach section boost metadata
    if section_boost_applied:
        metadata["section_boost"] = section_boost_applied[1]
        
    return bonus, factors


def apply_quality_scoring(matches: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Apply quality scoring to matches and re-rank them."""
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
        
        if isinstance(metadata, dict) and "section_boost" in metadata:
            match["section_boost"] = metadata.get("section_boost")
    
    matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return matches
