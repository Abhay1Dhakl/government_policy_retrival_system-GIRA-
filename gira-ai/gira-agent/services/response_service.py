import json
import sys
import re

def process_mcp_response(result: any, tool_name: str) -> tuple[str, list]:
    """
    Process MCP server response and extract meaningful government policy content for the LLM.
    
    Args:
        result: The raw result from MCP server tool
        tool_name: Name of the tool that was called
        
    Returns:
        tuple: (processed_content, chunk_metadata_list)
        - processed_content: Formatted content for LLM
        - chunk_metadata_list: List of chunk metadata for reference creation
    """
    try:
        print(f"[process_mcp_response] Processing {tool_name} result type: {type(result)}", file=sys.stderr)
        print(f"[process_mcp_response] Raw result content (first 500 chars): {str(result)[:500]}", file=sys.stderr)
        
        # Handle different result formats
        if isinstance(result, list):
            # MCP SDK returns list of TextContent objects
            print(f"[process_mcp_response] {tool_name}: Received list result (MCP SDK format)", file=sys.stderr)
            for item in result:
                if hasattr(item, 'text'):
                    try:
                        import json
                        parsed_result = json.loads(item.text)
                        print(f"[process_mcp_response] {tool_name}: Successfully parsed JSON from TextContent", file=sys.stderr)
                        content, chunks = process_structured_response(parsed_result, tool_name)
                        return content, chunks
                    except json.JSONDecodeError as e:
                        print(f"[process_mcp_response] {tool_name}: JSON parse failed: {e}", file=sys.stderr)
                        # If not JSON, return the text as is
                        return item.text, []
            # If no TextContent found, convert list to string
            print(f"[process_mcp_response] {tool_name}: No TextContent in list, converting to string", file=sys.stderr)
            return str(result), []
            
        elif isinstance(result, str):
            print(f"[process_mcp_response] {tool_name}: Received string result, attempting JSON parse", file=sys.stderr)
            # If it's already a string, try to parse as JSON
            try:
                import json
                parsed_result = json.loads(result)
                print(f"[process_mcp_response] {tool_name}: Successfully parsed JSON", file=sys.stderr)
                content, chunks = process_structured_response(parsed_result, tool_name)
                return content, chunks
            except json.JSONDecodeError as e:
                print(f"[process_mcp_response] {tool_name}: JSON parse failed: {e}", file=sys.stderr)
                # If not JSON, return as is
                return result, []
                
        elif isinstance(result, dict):
            print(f"[process_mcp_response] {tool_name}: Received dict result", file=sys.stderr)
            # If it's a dictionary, process it
            content, chunks = process_structured_response(result, tool_name)
            return content, chunks
            
        else:
            print(f"[process_mcp_response] {tool_name}: Converting {type(result)} to string", file=sys.stderr)
            # Convert other types to string
            return str(result), []
            
    except Exception as e:
        print(f"[process_mcp_response] Error processing response: {e}", file=sys.stderr)
        return f"Error processing {tool_name} response: {str(e)}", []

def process_structured_response(data: dict, tool_name: str) -> tuple[str, list]:
    """
    Process structured response data from MCP server.
    
    Args:
        data: Dictionary containing the response data
        tool_name: Name of the tool
        
    Returns:
        tuple: (formatted_content, chunk_metadata_list)
        - formatted_content: Formatted content for LLM
        - chunk_metadata_list: List of chunk metadata for reference creation
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

            # Note: Removed strict document_type filtering to allow cross-document information
            # (e.g., drug interactions mentioning the query drug from other drug PIs)
            print(f"[process_structured_response] {tool_name}: Processing {len(matches)} matches without document_type filtering", file=sys.stderr)
            
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

        # Check if this is our new simplified format
        if "matches" in data and "total_found" in data:
            matches = data.get("matches", [])
            total_found = data.get("total_found", 0)

            # Note: Removed strict document_type filtering to allow cross-document information
            # (e.g., drug interactions mentioning the query drug from other drug PIs)
            print(f"[process_structured_response] {tool_name}: Processing {len(matches)} matches without document_type filtering", file=sys.stderr)
            
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
                return f"No {tool_name.upper()} documents found for this query.", []

            # Collect chunk metadata for reference creation
            chunk_metadata_list = []
            
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
                        
                        # Collect chunk metadata for reference creation
                        chunk_metadata_list.append({
                            "text": clean_text,
                            "source": match.get("source", ""),
                            "page_number": match.get("page_number", ""),
                            "chunk_index": match.get("chunk_index", ""),
                            "score": chunk_score,
                            "document_type": match.get("document_type", ""),
                            "region": match.get("region", ""),
                            "tool_name": tool_name  # Add tool name for filtering
                        })
                        
                        chunk_index += 1
                
                formatted_content += "---\n\n"  # Separator between page groups
                group_index += 1
                
            # Prevent runaway payloads but allow full chunk text for synthesis
            if len(formatted_content) > 25000:
                formatted_content += "\n[RESPONSE TRUNCATED - TOO LARGE]\n"

            result = formatted_content.strip()
            print(f"[process_structured_response] {tool_name}: returning content length={len(result)}, chunks={len(chunk_metadata_list)}", file=sys.stderr)
            return result, chunk_metadata_list        # Handle legacy format or error responses
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
