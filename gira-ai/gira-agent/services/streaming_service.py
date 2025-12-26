import uuid
import json
from datetime import datetime
from typing import AsyncGenerator
from database.services import DatabaseService
from services.mcp_service import query_mcp
from services.title_service import generate_title
from services.streaming_utils import stream_by_words, stream_by_sentences, stream_by_tokens
import re

def _parse_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None

def _redistribute_trailing_citations(answer_text: str) -> str:
    """
    Move grouped end-of-paragraph citations so each attaches to the last
    N sentences in that paragraph (one per sentence), preserving order.

    This helps enforce "one citation per chunk explanation" when the model
    grouped citations at the end.
    """
    try:
        if not answer_text:
            return answer_text
        paragraphs = answer_text.split("\n\n")
        fixed = []
        trail_re = re.compile(r"(\s*(?:\[(\d+)\.(\d+)\]\s*)+)\s*$")
        cite_re = re.compile(r"\[(\d+)\.(\d+)\]")
        for para in paragraphs:
            m = trail_re.search(para)
            if not m:
                fixed.append(para)
                continue
            # Extract trailing citations and remove them from paragraph
            trail = m.group(1)
            cites = cite_re.findall(trail)
            body = para[: m.start()].rstrip()

            # Split body into sentences
            sentences = re.split(r"(?<=[.!?])\s+", body)
            sentences = [s for s in sentences if s.strip()]
            if not sentences or not cites:
                fixed.append(para)
                continue

            n = len(cites)
            s = len(sentences)
            # Attach citations to the last n sentences (or all if fewer)
            attach_start = max(0, s - n)
            idx = 0
            for i in range(attach_start, s):
                if idx >= n:
                    break
                doc, pos = cites[idx]
                if not sentences[i].rstrip().endswith('.'):  # keep punctuation natural
                    sentences[i] = sentences[i].rstrip()
                sentences[i] = sentences[i].rstrip() + f" [{doc}.{pos}]"
                idx += 1
            # If extra citations remain, append them to the last sentence spaced out
            while idx < n:
                doc, pos = cites[idx]
                sentences[-1] = sentences[-1].rstrip() + f" [{doc}.{pos}]"
                idx += 1

            fixed.append(" ".join(sentences))

        return "\n\n".join(fixed)
    except Exception:
        return answer_text
def extract_citations_from_answer(answer_text):
    """
    Extract citation references like [1.1], [2.3], etc. from the answer text.
    
    Args:
        answer_text: The answer text containing citations
        
    Returns:
        set: Set of tuples (doc_num, chunk_num) for cited chunks
    """
    # Find all citations in format [X.Y]
    citation_pattern = r'\[(\d+)\.(\d+)\]'
    matches = re.findall(citation_pattern, answer_text)
    
    cited_chunks = set()
    for match in matches:
        try:
            doc_num = int(match[0])
            chunk_num = int(match[1])
            cited_chunks.add((doc_num, chunk_num))
        except ValueError:
            continue
    
    return cited_chunks

def create_precise_references(answer_text: str, filtered_chunks):
    """
    Create references ONLY for chunks that were actually cited (100% precision),
    and renumber per-document citations based on usage order in the answer text.

    Args:
        answer_text: Full answer text containing [doc.chunk] citations
        filtered_chunks: Dict with 'pi_chunks', 'lrd_chunks', 'other_chunks'

    Returns:
        list: List of reference objects for cited chunks only, ordered by
              document number ascending and usage order within each document.
    """
    references = []

    # Build a mapping from (doc_num, doc_chunk_position) -> chunk data
    doc_position_to_chunk = {}
    for tool_type in ['pi_chunks', 'lrd_chunks', 'other_chunks']:
        for chunk in filtered_chunks.get(tool_type, []):
            doc_num = _parse_int(chunk.get('assigned_doc_num'))
            position = _parse_int(chunk.get('doc_chunk_position', chunk.get('chunk_index')))
            if doc_num is None or position is None:
                continue
            doc_position_to_chunk[(doc_num, position)] = chunk

    print(f"[create_precise_references] Built mapping with {len(doc_position_to_chunk)} positions")

    # Find citations in order of appearance (keep first occurrence per unique [doc,pos])
    citation_iter = re.finditer(r"\[(\d+)\.(\d+)\]", answer_text or "")
    seen = set()
    ordered_keys = []
    usage_snippet_by_key = {}
    for m in citation_iter:
        try:
            d = int(m.group(1))
            p = int(m.group(2))
        except Exception:
            continue
        key = (d, p)
        if key in seen:
            continue
        seen.add(key)
        ordered_keys.append(key)

        # Extract a concise sentence-level snippet around the citation location
        # Find sentence boundaries within +/- 200 chars
        start_idx = m.start()
        end_idx = m.end()
        left = max(0, start_idx - 200)
        right = min(len(answer_text), end_idx + 200)
        window = answer_text[left:right]

        # Determine local positions
        local_cite_start = start_idx - left

        # Find previous sentence boundary
        prev_boundary = window.rfind('.', 0, local_cite_start)
        if prev_boundary == -1:
            prev_boundary = window.rfind('\n', 0, local_cite_start)
        if prev_boundary == -1:
            prev_boundary = 0
        else:
            prev_boundary += 1  # move past the period

        # Find next sentence boundary
        next_boundary = window.find('.', local_cite_start)
        if next_boundary == -1:
            next_boundary = window.find('\n', local_cite_start)
        if next_boundary == -1:
            next_boundary = len(window)

        snippet = window[prev_boundary:next_boundary].strip()
        # Shorten overly long snippets
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + '...'
        usage_snippet_by_key[key] = snippet

    for doc_num, chunk_position in ordered_keys:
        key = (doc_num, chunk_position)
        if key not in doc_position_to_chunk:
            print(f"[create_precise_references] WARNING: Citation [{doc_num}.{chunk_position}] not found in chunk mapping")
            continue
        chunk = doc_position_to_chunk[key]

        document_type = str(chunk.get('document_type', '')).lower()
        if document_type in ['pis', 'pi', 'prescribing_information', 'prescribing information', 'policy', 'act', 'bill', 'main_policy']:
            tool_label = "Policy Document"
        elif document_type in ['lrd', 'labeling', 'regulatory', 'label', 'regulation', 'amendment', 'gazette', 'guideline']:
            tool_label = "Regulatory Guideline"
        else:
            tool_label = "Other Document"

        answer_segment = (
            f"{tool_label}: {chunk.get('source', 'Unknown')} "
            f"(Page {chunk.get('page_number', 'N/A')}, Chunk {chunk.get('chunk_index', 'N/A')})"
        )

        references.append({
            "answer_segment": answer_segment,
            "original_text": chunk.get("text", ""),
            "answer_snippet": usage_snippet_by_key.get(key, ""),
            "page_number": chunk.get("page_number", ""),
            "chunk_index": chunk.get("chunk_index", ""),
            "source": chunk.get("source", ""),
            "reference_number": f"[{doc_num}.{chunk_position}]",
            "original_citation": f"[{doc_num}.{chunk_position}]",
        })
        print(
            f"[create_precise_references] Created reference [{doc_num}.{chunk_position}] "
            f"for {chunk.get('source', 'Unknown')}"
        )

    return references

def filter_and_group_chunks(all_chunk_metadata, tools):
    """
    Group by document_type buckets and assign document numbers based on
    first appearance order across all chunks.

    Returns dict with 'pi_chunks', 'lrd_chunks', 'other_chunks', where each
    chunk includes:
      - assigned_doc_num: global document index across all paragraphs
      - doc_chunk_position: chunk index used in citations
    """
    # Group chunks by document_type (NOT tool name)
    pi_chunks: list = []
    lrd_chunks: list = []
    other_chunks: list = []

    doc_counter = 1
    document_mapping = {}  # source -> doc_num

    for chunk in all_chunk_metadata:
        src = chunk.get('source', 'Unknown')
        if src not in document_mapping:
            document_mapping[src] = doc_counter
            doc_counter += 1

        doc_num = document_mapping[src]
        chunk_position = _parse_int(chunk.get('chunk_index'))

        enriched = dict(chunk)
        enriched['assigned_doc_num'] = doc_num
        if chunk_position is not None:
            enriched['doc_chunk_position'] = chunk_position

        document_type = str(chunk.get('document_type', '')).lower()
        source = str(src).lower()
        
        # Broaden categorization for government documents
        # Policy/Primary bucket (PI equivalent)
        if any(term in document_type for term in ['constitution', 'policy', 'act', 'bill', 'law', 'statute', 'pis', 'main']) or \
           any(term in source for term in ['constitution', 'act', 'policy']):
            pi_chunks.append(enriched)
        # Regulatory/Guideline bucket (LRD equivalent)
        elif any(term in document_type for term in ['regulation', 'directive', 'amendment', 'gazette', 'guideline', 'rule', 'bylaw', 'lrd']) or \
             any(term in source for term in ['regulation', 'directive', 'rule']):
            lrd_chunks.append(enriched)
        else:
            other_chunks.append(enriched)

    return {
        'pi_chunks': pi_chunks,
        'lrd_chunks': lrd_chunks,
        'other_chunks': other_chunks,
        'document_mapping': document_mapping,
    }
async def stream_ai_response(
    req, user_id: str, country: str, page_id: str, request
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
            "page_url": request.headers.get("X-Page-URL", "unknown"),
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
            "timestamp": datetime.utcnow().isoformat(),
        }
        yield f"data: {json.dumps(initial_data)}\n\n"

        # Get full response from MCP with error handling
        try:
            mcp_result = await query_mcp(req.user_query, req.llm, req.tools, country, user_id)
            
            # Handle tuple return (response, chunk_metadata)
            if isinstance(mcp_result, tuple) and len(mcp_result) == 2:
                full_response, all_chunk_metadata = mcp_result
                print(f"[stream_query] Full MCP response: {full_response[:500]}...")
                print(f"[stream_query] Retrieved {len(all_chunk_metadata)} filtered chunks from MCP")
                
                # Reconstruct filtered chunks deterministically based on document_type
                # and assign document numbers/positions to match MCP prompt numbering
                filtered_grouped = filter_and_group_chunks(all_chunk_metadata, req.tools)
                filtered_chunks = {
                    'pi_chunks': filtered_grouped.get('pi_chunks', []),
                    'lrd_chunks': filtered_grouped.get('lrd_chunks', []),
                    'other_chunks': filtered_grouped.get('other_chunks', []),
                }
                print(
                    f"[stream_query] Grouped chunks (by document_type) - PI: {len(filtered_chunks['pi_chunks'])}, "
                    f"LRD: {len(filtered_chunks['lrd_chunks'])}, Other: {len(filtered_chunks['other_chunks'])}"
                )
                
            else:
                # Fallback for old format
                full_response = mcp_result
                all_chunk_metadata = []
                filtered_chunks = {'pi_chunks': [], 'lrd_chunks': [], 'other_chunks': []}
                print(f"[stream_query] Full MCP response: {full_response[:500]}...")
        except Exception as e:
            print(f"[stream_query] ERROR: MCP query failed: {e}")
            full_response = (
                "I apologize, but I'm currently unable to access the government policy document database to answer your question. "
                "The service appears to be temporarily unavailable. Please try again in a few moments."
            )
            all_chunk_metadata = []
            filtered_chunks = {'pi_chunks': [], 'lrd_chunks': [], 'other_chunks': []}

        # Parse the response to extract references and other metadata
        references = []
        flagging_value = ""
        clean_answer = full_response

        # Try to extract JSON from the response
        if full_response and "{" in full_response and "}" in full_response:
            try:
                parsed_response = json.loads(full_response)

                if isinstance(parsed_response, dict):
                    clean_answer = parsed_response.get("answer", full_response)
                    llm_references = parsed_response.get("references", [])
                    flagging_value = parsed_response.get("flagging_value", "")
                    print(f"[stream_query] Successfully parsed JSON response with {len(llm_references)} LLM references")
                    
                    # Fix grouped end-of-paragraph citations → attach one per sentence/chunk
                    clean_answer = _redistribute_trailing_citations(clean_answer)

                    # Extract citations (for logging only)
                    cited_chunks = extract_citations_from_answer(clean_answer)
                    print(f"[stream_query] Found {len(cited_chunks)} citations in answer: {cited_chunks}")

                    # Create references ONLY for cited chunks (100% precision),
                    # and renumber chunk positions by usage order
                    if clean_answer and filtered_chunks:
                        references.extend(create_precise_references(clean_answer, filtered_chunks))
                        print(f"[stream_query] Created {len(references)} precise references (LLM: {len(llm_references)}, Cited chunks: {len(cited_chunks)})")
                        
                    else:
                        # Fallback to LLM references if no citations found
                        references.extend(llm_references)
                        print(f"[stream_query] No citations found, using {len(llm_references)} LLM references")
                    
                else:
                    clean_answer = full_response
            except json.JSONDecodeError as e:
                print(f"[stream_query] Error parsing response as JSON: {e}")
                clean_answer = full_response
        
        if not references and clean_answer and filtered_chunks:
            # Fallback: extract citations even if JSON parsing failed or produced no references
            clean_answer = _redistribute_trailing_citations(clean_answer)
            cited_chunks = extract_citations_from_answer(clean_answer)
            if cited_chunks:
                references.extend(create_precise_references(clean_answer, filtered_chunks))
                print(f"[stream_query] Fallback created {len(references)} references from raw answer text")

        # Stream the response based on stream_type
        print(f"[stream_query] Starting to stream. clean_answer length: {len(clean_answer)}")
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
                "conversation_id": conversation_id,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        # Store conversation after streaming
        try:
            # Persist the final, cleaned answer with resolved references so it survives refresh
            stored_payload = {
                "answer": clean_answer,
                "references": references,
                "flagging_value": flagging_value,
            }

            await DatabaseService.store_page_conversation(
                user_id=user_id,
                user_query=req.user_query,
                conversation_id=conversation_id,
                assistant_response=json.dumps(stored_payload),
                page_context=page_context,
                session_id=session_id,
            )
        except Exception as e:
            print(f"[stream_query] Error storing conversation: {e}")

        # Send completion signal
        completion_data = {
            "type": "complete",
            "conversation_id": conversation_id,
            "total_length": len(full_response),
        }
        yield f"data: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        yield f"data: {json.dumps(error_data)}\n\n"
