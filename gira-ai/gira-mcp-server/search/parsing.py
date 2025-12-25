"""GIRS Search Response Parsing and Document Formatting

Handles Pinecone response parsing and standardization of government document
metadata for consistent presentation to clients.
"""

from typing import Dict, Any, List


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
    for field_name in field_names:
        value = metadata.get(field_name)
        if value:
            return str(value)
    return default


def parse_pinecone_response(pinecone_response):
    """Parse Pinecone response to extract metadata (optimized)"""
    if not pinecone_response or "matches" not in pinecone_response:
        return {"matches": []}
    
    matches = []
    for match in pinecone_response["matches"]:
        if match is None:
            continue
            
        processed_match = {
            "id": match.get("id", ""),
            "score": match.get("score", 0.0),
            "metadata": {}
        }
        
        raw_metadata = match.get("metadata", {})
        if raw_metadata:
            for key, value in raw_metadata.items():
                if value is None:
                    processed_match["metadata"][key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    processed_match["metadata"][key] = value
                else:
                    processed_match["metadata"][key] = str(value)
        
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


def _process_search_matches(matches: List[Dict[str, Any]], max_matches: int = 20) -> List[Dict[str, Any]]:
    """
    Process search matches into a standardized format with comprehensive metadata extraction.
    
    Args:
        matches: Raw matches from search results
        max_matches: Maximum number of matches to process
        
    Returns:
        List of processed matches with standardized metadata
    """
    processed_matches = []
    
    for i, match in enumerate(matches[:max_matches]):
        if not match or not isinstance(match, dict):
            continue
            
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        
        document_type = metadata.get("document_type", "")
        
        if document_type == "past_cases":
            source_filename = f"past_case_{match.get('id', 'unknown')}"
        else:
            source_filename = _extract_metadata_field(
                metadata, 
                ["file_name", "source", "filename", "document_name", "doc_name"]
            )
        
        page_number = _extract_metadata_field(
            metadata,
            ["page_number", "page", "page_num", "chunk_page", "section_page"]
        )
        
        chunk_index = _extract_metadata_field(
            metadata,
            ["chunk_index", "chunk_id", "index", "section_index"]
        )
        
        section_info = _extract_metadata_field(
            metadata,
            ["section_title", "section_name", "chunk_type", "section_type"]
        )
        
        processed_match = {
            "id": match.get("id", ""),
            "score": match.get("score", 0.0),
            "source": source_filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "section_info": section_info,
            "text": str(metadata.get("text", "")),
            "document_type": document_type,
            "region": metadata.get("region", ""),
            "content_preview": str(metadata.get("text", ""))[:200],
            "full_metadata": metadata
        }
        
        if "hybrid_score" in match:
            processed_match["hybrid_score"] = match.get("hybrid_score", 0.0)
        if "bm25_boost" in match:
            processed_match["bm25_boost"] = match.get("bm25_boost", 0.0)
        if "quality_score" in match:
            processed_match["quality_score"] = match.get("quality_score", 0.0)
            
        processed_matches.append(processed_match)
    
    return processed_matches
