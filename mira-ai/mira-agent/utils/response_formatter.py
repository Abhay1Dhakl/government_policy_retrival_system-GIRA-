"""
Response formatting utilities for MIRA AI
Handles formatting of MCP server responses and document processing
"""
import sys
import json
import re
from typing import Dict, List, Any, Optional
from config import settings


def format_page_groups(page_groups: Dict[str, List[Dict]], tool_name: str) -> str:
    """
    Format page groups into consolidated response text with optimized string building
    
    Args:
        page_groups: Dictionary of page groups with matches
        tool_name: Name of the tool being processed
    
    Returns:
        Formatted content string
    """
    # Use list for efficient string building
    content_parts = [f"Found {sum(len(matches) for matches in page_groups.values())} relevant {tool_name.upper()} documents (consolidated by page):\n\n"]
    
    group_index = 1
    for group_key, group_matches in page_groups.items():
        if not group_matches:
            continue
        
        # Use the first match for header info
        first_match = group_matches[0]
        doc_type = first_match.get("document_type", "")
        page_number = first_match.get("page_number", "")
        source = first_match.get("source", "")
        
        # Calculate average score for the group
        scores = [match.get("score", 0.0) for match in group_matches]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Build document header
        page_info = f", Page: {page_number}" if page_number else ""
        source_info = f"Source: {source}" if source else ""
        content_parts.append(
            f"Document from {source_info}{page_info} "
            f"(Type: {doc_type}, {len(group_matches)} chunks, Avg Score: {avg_score:.3f}):\n\n"
        )
        
        # Add all text chunks from this page group
        chunk_parts = []
        for chunk_index, match in enumerate(group_matches, 1):
            text = match.get("text", "")
            chunk_score = match.get("score", 0.0)
            
            if text and text.strip():
                # Remove excessive whitespace
                clean_text = " ".join(text.split())
                
                # Add metadata to the chunk
                chunk_metadata = (
                    f"Source: {match.get('source', 'N/A')}, "
                    f"Page: {match.get('page_number', 'N/A')}, "
                    f"Chunk: {match.get('chunk_index', 'N/A')}, "
                    f"Score: {chunk_score:.3f}"
                )
                chunk_parts.append(f"Text: {clean_text}\nMetadata: {chunk_metadata}")
        
        # Join all chunks for this group
        if chunk_parts:
            content_parts.append("\n\n".join(chunk_parts))
            content_parts.append("\n\n---\n\n")  # Separator between page groups
        
        group_index += 1
        
        # Prevent runaway payloads
        current_length = sum(len(part) for part in content_parts)
        if current_length > settings.MAX_RESPONSE_LENGTH:
            content_parts.append("\n[RESPONSE TRUNCATED - TOO LARGE]\n")
            break
    
    return "".join(content_parts).strip()


def consolidate_matches_by_page(matches: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group matches by (source, page_number) for consolidated references
    
    Args:
        matches: List of match dictionaries
    
    Returns:
        Dictionary with page groups
    """
    page_groups = {}
    
    for match in matches[:settings.MAX_MATCHES_TO_PROCESS]:
        source = match.get("source", "")
        page_number = match.get("page_number", "")
        
        # Create unique key for same source + page
        group_key = f"{source}_page_{page_number}" if page_number else f"{source}_no_page"
        
        if group_key not in page_groups:
            page_groups[group_key] = []
        page_groups[group_key].append(match)
    
    return page_groups


def extract_past_cases(data: Dict, tool_name: str) -> Optional[str]:
    """
    Extract and format past cases from response data
    
    Args:
        data: Response data dictionary
        tool_name: Name of the tool
    
    Returns:
        Formatted past cases string or None
    """
    if tool_name != "past_cases":
        return None
    
    try:
        matches = data.get("matches", [])
        if not matches:
            return "No additional documents or past cases available for this question."
        
        past_texts = []
        for match in matches[:5]:
            if not isinstance(match, dict):
                continue
            
            md = match.get("metadata", {}) or {}
            ans = md.get("answer") or md.get("text") or ""
            if not ans:
                continue
            
            # Remove code fences
            clean = re.sub(r"```[\s\S]*?```", "", str(ans)).strip()
            
            # Try to parse JSON
            try:
                parsed = json.loads(clean)
                inner = parsed.get("answer", "") if isinstance(parsed, dict) else clean
            except Exception:
                inner = clean
            
            inner = str(inner).strip()
            
            # Filter out fallback messages
            fallback_detect = ("No PI document available" in inner and "No LRD document available" in inner)
            if not fallback_detect and inner:
                past_texts.append(inner[:settings.PAST_TEXT_PREVIEW_LENGTH])
        
        if past_texts:
            return "Past cases found:\n\n" + "\n\n".join(past_texts)
        return "No additional documents or past cases available for this question."
    
    except Exception as e:
        print(f"[extract_past_cases] Error: {e}", file=sys.stderr)
        return None


def has_content_in_matches(matches: List[Dict]) -> bool:
    """
    Check if matches contain actual content
    
    Args:
        matches: List of match dictionaries
    
    Returns:
        True if content exists, False otherwise
    """
    for match in matches[:settings.MAX_MATCHES_TO_PROCESS]:
        if match.get("text", "").strip():
            return True
    return False


def filter_matches_by_document_type(matches: List[Dict], target_doc_type: str) -> List[Dict]:
    """
    Filter matches by document type to ensure tool alignment
    
    Args:
        matches: List of match dictionaries
        target_doc_type: Target document type to filter by
    
    Returns:
        Filtered list of matches
    """
    if not target_doc_type:
        return matches
    
    filtered_matches = []
    for match in matches:
        doc_type = str(match.get("document_type", "")).lower().strip()
        if doc_type == target_doc_type.lower().strip():
            filtered_matches.append(match)
    
    return filtered_matches


def format_legacy_matches(matches: List[Dict], tool_name: str) -> str:
    """
    Format matches using legacy format
    
    Args:
        matches: List of match dictionaries
        tool_name: Name of the tool
    
    Returns:
        Formatted content string
    """
    content_parts = []
    
    for match in matches[:5]:  # Limit to 5 matches
        if isinstance(match, dict):
            metadata = match.get("metadata", {})
            # Prefer 'text' but fall back to 'answer' for past_cases legacy
            text = metadata.get("text", "") or metadata.get("answer", "")
            
            if text:
                # Try to parse JSON string
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and parsed.get("answer"):
                        content_parts.append(parsed.get("answer")[:250])
                    else:
                        content_parts.append(str(text)[:250])
                except Exception:
                    content_parts.append(str(text)[:250])
    
    if content_parts:
        return f"{tool_name.upper()} documents found:\n\n" + "\n\n".join(content_parts)
    
    return f"No relevant {tool_name.upper()} content found in response."
