import os
import json
from typing import List
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from services.llm_service import choose_llm
from services.prompt_service import generate_system_prompt
from services.response_service import process_mcp_response
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("query_mcp")

async def query_mcp(user_query: str, llm: str, tools: List[str], country: str, user_id: str) -> tuple[str, list]:
    """
    Query the MCP server using official MCP SDK - No LangChain required!
    
    Args:
        user_query: The user's government policy question
        llm: The LLM provider to use
        tools: List of tool names to use
        country: User's country
        user_id: User ID
        
    Returns:
        tuple: (response, all_chunk_metadata)
        - response: Government policy information response with citations
        - all_chunk_metadata: List of all retrieved chunk metadata
    """
    try:
        # Load MCP server configuration
        environment = os.getenv("ENVIRONMENT", "development").lower()
        print(f"[MCP] Environment: {environment}")
        
        if environment == "production":
            config_file = "mcp_server_config/config_production.json"
        else:
            config_file = "mcp_server_config/config_development.json"

        print(f"[MCP] Using config file: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract server URL from config
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            raise Exception("No MCP servers configured")
        
        # Get first server config
        server_name = list(mcp_servers.keys())[0]
        server_config = mcp_servers[server_name]
        
        # Get SSE transport URL
        server_url = server_config.get("url")
        if not server_url:
            raise Exception(f"No URL configured for server {server_name}")
        
        print(f"[MCP] Connecting to server: {server_name}")
        print(f"[MCP] URL: {server_url}")
        
        # Connect to MCP server via SSE
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                tools_list = await session.list_tools()
                print(f"[MCP] Available tools: {[tool.name for tool in tools_list.tools]}")
                
                # Filter tools based on requested tools
                available_tool_names = {tool.name: tool for tool in tools_list.tools}
                filtered_tool_defs = []
                
                for tool_name in tools:
                    if tool_name in available_tool_names:
                        tool = available_tool_names[tool_name]
                        filtered_tool_defs.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema
                            }
                        })
                        print(f"[MCP] Including tool: {tool_name}")
                
                if not filtered_tool_defs:
                    print("[MCP] WARNING: No valid tools found after filtering!")
                    return "Error: No valid tools available for the requested tool list."
                
                # Initialize LLM
                llm_instance = choose_llm(llm, temperature=0.2)
                
                # Generate system prompt
                system_prompt = generate_system_prompt(user_query, country, tools)
                print(f"[MCP] System Prompt generated: {len(system_prompt)} chars")
                
                # Step 1: Call LLM with tool definitions
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
                
                print(f"[MCP] Calling LLM with {len(filtered_tool_defs)} tools...")
                
                # Call LLM - handle both OpenAI and Anthropic formats
                response = await llm_instance.ainvoke(messages, tools=filtered_tool_defs)
                print(f"[MCP] Initial LLM response type: {type(response)}")
                
                # Extract tool calls from response
                tool_calls = None
                assistant_message = None
                
                # Handle OpenAI format
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    assistant_message = choice.message
                    
                    # Check if there are tool calls
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        tool_calls = assistant_message.tool_calls
                        print(f"[MCP] Found {len(tool_calls)} tool calls from OpenAI")
                    else:
                        # No tool calls, return direct response
                        if assistant_message.content:
                            print(f"[MCP] No tool calls, returning direct response")
                            return assistant_message.content
                        else:
                            print(f"[MCP] WARNING: No tool calls and no content in response")
                            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
                # Handle Anthropic format
                elif hasattr(response, 'content'):
                    # Check for tool use in content blocks
                    tool_calls = []
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == 'tool_use':
                            tool_calls.append({
                                'id': block.id,
                                'function': {
                                    'name': block.name,
                                    'arguments': json.dumps(block.input)
                                }
                            })
                    
                    if tool_calls:
                        print(f"[MCP] Found {len(tool_calls)} tool calls from Anthropic")
                    else:
                        # No tool calls, return text content
                        text_content = ""
                        for block in response.content:
                            if hasattr(block, 'text'):
                                text_content += block.text
                        
                        if text_content:
                            print(f"[MCP] No tool calls, returning direct response")
                            return text_content, []
                        else:
                            print(f"[MCP] WARNING: No tool calls and no text content")
                            return "I apologize, but I couldn't generate a response. Please try rephrasing your question.", []
                
                if not tool_calls:
                    print("[MCP] ERROR: Expected tool calls but got none")
                    return "I apologize, but I couldn't access the government policy database. Please try again.", []
                
                tool_results = []
                all_chunk_metadata = []  # Collect chunk metadata from all tool calls
                
                for tool_call in tool_calls:
                    # Handle OpenAI format
                    if hasattr(tool_call, 'function'):
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    # Handle dict format
                    elif isinstance(tool_call, dict):
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])
                        tool_call_id = tool_call['id']
                    else:
                        print(f"[MCP] WARNING: Unknown tool call format: {type(tool_call)}")
                        continue
                    
                    # Validate tool is in allowed list
                    if tool_name not in tools:
                        print(f"[MCP] BLOCKING unauthorized tool call: {tool_name}")
                        continue
                    
                    # Add context parameters
                    tool_args["country"] = country
                    tool_args["user_id"] = user_id
                    
                    print(f"[MCP] Calling tool {tool_name} with args: {tool_args}")
                    
                    try:
                        # Call MCP tool
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        print(f"[MCP] Tool {tool_name} returned content")
                        
                        # Process the response - now returns (content, chunk_metadata_list)
                        processed_result = process_mcp_response(result.content, tool_name)
                        if isinstance(processed_result, tuple) and len(processed_result) == 2:
                            processed_content, chunk_metadata_list = processed_result
                            # Collect chunk metadata for reference creation
                            all_chunk_metadata.extend(chunk_metadata_list)
                        else:
                            # Fallback for old format
                            processed_content = processed_result
                            print(f"[MCP] WARNING: process_mcp_response returned unexpected format: {type(processed_result)}")
                        
                        print(f"[MCP] Processed content length: {len(processed_content)} chars, chunks: {len(chunk_metadata_list) if 'chunk_metadata_list' in locals() else 0}")
                        
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": processed_content
                        })
                    except Exception as e:
                        print(f"[MCP] ERROR calling tool {tool_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {str(e)}"
                        })
                
                # Collect chunks by tool type for structured processing
                chunks_by_tool = {'pi_chunks': [], 'lrd_chunks': [], 'other_chunks': []}
                
                for tool_result in tool_results:
                    tool_name = tool_result['name']
                    
                    # Find chunks for this tool from all_chunk_metadata
                    tool_chunks = [chunk for chunk in all_chunk_metadata if chunk.get('tool_name') == tool_name]
                    
                    # Categorize chunks by ACTUAL document_type metadata (not tool name)
                    for chunk in tool_chunks:
                        document_type = chunk.get('document_type', '').lower()
                        
                        # Debug: Print chunk classification
                        print(f"[MCP] Chunk from {chunk.get('source', 'Unknown')}: document_type='{document_type}', tool_name='{tool_name}'")
                        
                        # Categorize by document_type metadata
                        if document_type in ['pis', 'pi', 'prescribing_information', 'prescribing information']:
                            chunks_by_tool['pi_chunks'].append(chunk)
                            print(f"[MCP] → Classified as PI chunk")
                        elif document_type in ['lrd', 'labeling', 'regulatory', 'label']:
                            chunks_by_tool['lrd_chunks'].append(chunk)
                            print(f"[MCP] → Classified as LRD chunk")
                        else:
                            chunks_by_tool['other_chunks'].append(chunk)
                            print(f"[MCP] → Classified as Other chunk")
                
                print(f"[MCP] After categorization - PI: {len(chunks_by_tool['pi_chunks'])}, LRD: {len(chunks_by_tool['lrd_chunks'])}, Other: {len(chunks_by_tool['other_chunks'])}")

                
                # Filter to top 5 chunks per tool type (by score) and assign document-based citations
                filtered_chunks = {}
                document_mapping = {}  # Maps (tool_type, source) -> document_number
                doc_counter = 1  # Global counter for unique document numbers
                
                for tool_type, chunks in chunks_by_tool.items():
                    # Sort by score (highest first) and take top 5
                    sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)[:5]
                    
                    # Assign document numbers based on unique sources
                    for chunk in sorted_chunks:
                        source = chunk.get('source', 'Unknown')
                        key = (tool_type, source)
                        
                        if key not in document_mapping:
                            document_mapping[key] = doc_counter
                            doc_counter += 1
                        
                        # Add the assigned document number to the chunk
                        chunk['assigned_doc_num'] = document_mapping[key]
                    
                    filtered_chunks[tool_type] = sorted_chunks
                
                print(f"[MCP] Filtered chunks - PI: {len(filtered_chunks['pi_chunks'])}, LRD: {len(filtered_chunks['lrd_chunks'])}, Other: {len(filtered_chunks['other_chunks'])}")
                print(f"[MCP] Document mapping: {document_mapping}")
                
                # Create structured chunk content for LLM with document-based citations
                chunk_content = "FILTERED CHUNKS BY TOOL TYPE:\n\n"
                
                # Helper function to group chunks by source and format citations
                def format_chunks_by_source(chunks, paragraph_num):
                    if not chunks:
                        return f"PARAGRAPH {paragraph_num} CHUNKS: No chunks available\n\n"
                    
                    content = f"PARAGRAPH {paragraph_num} CHUNKS:\n"
                    
                    # Group chunks by source document
                    chunks_by_source = {}
                    for chunk in chunks:
                        source = chunk.get('source', 'Unknown')
                        if source not in chunks_by_source:
                            chunks_by_source[source] = []
                        chunks_by_source[source].append(chunk)
                    
                    # Format each document's chunks with proper citations
                    chunk_position = 1  # Sequential position within document
                    for source, source_chunks in chunks_by_source.items():
                        doc_num = source_chunks[0]['assigned_doc_num']
                        content += f"\nDocument {doc_num}: {source}\n"
                        
                        for idx, chunk in enumerate(source_chunks, 1):
                            # Add sequential position for mapping later
                            chunk['doc_chunk_position'] = idx
                            content += f"[{doc_num}.{idx}] Page: {chunk.get('page_number', 'N/A')}, Chunk: {chunk.get('chunk_index', 'N/A')}\n"
                            content += f"Text: {chunk.get('text', '')}\n\n"
                    
                    return content
                
                # PI chunks (for Paragraph 1)
                chunk_content += format_chunks_by_source(filtered_chunks['pi_chunks'], 1)
                
                # LRD chunks (for Paragraph 2)
                chunk_content += format_chunks_by_source(filtered_chunks['lrd_chunks'], 2)
                
                # Other chunks (for Paragraph 3)
                chunk_content += format_chunks_by_source(filtered_chunks['other_chunks'], 3)
                
                # Replace tool results with structured chunk content
                final_messages = messages.copy()
                
                # Don't add assistant message with tool_calls - instead add chunk content directly
                # This avoids the OpenAI API requirement for tool messages after tool_calls
                final_messages.append({
                    "role": "user",
                    "content": chunk_content
                })
                
                # Add chunk metadata for reference creation if available
                if all_chunk_metadata:
                    # Create a reference guide for the LLM
                    reference_guide = "AVAILABLE REFERENCES:\n"
                    for i, chunk in enumerate(all_chunk_metadata[:20]):  # Limit to first 20 chunks
                        ref_num = f"[{i+1}]"
                        source = chunk.get('source', 'Unknown')
                        page = chunk.get('page_number', 'N/A')
                        chunk_idx = chunk.get('chunk_index', 'N/A')
                        reference_guide += f"{ref_num} Source: {source}, Page: {page}, Chunk: {chunk_idx}\n"
                    
                    reference_guide += "\nWhen creating citations in your response, use the reference numbers above (e.g., [1], [2], [3]) to cite specific sources.\n\n"
                    
                    # Add reference guide as a system message before the final call
                    final_messages.insert(0, {
                        "role": "system",
                        "content": reference_guide
                    })
                    print(f"[MCP] Added reference guide with {min(len(all_chunk_metadata), 20)} references")
                
                # Make final LLM call
                print(f"[MCP] Making final LLM call with conversation history...")
                final_response = await llm_instance.ainvoke(final_messages)
                print(f"[MCP] Final LLM response type: {type(final_response)}")
                
                # Extract final content
                if hasattr(final_response, 'choices') and final_response.choices:
                    content = final_response.choices[0].message.content
                    if content:
                        print(f"[MCP] Returning final response: {len(content)} chars")
                        return content, all_chunk_metadata
                    else:
                        print("[MCP] WARNING: Final response has no content")
                        # Return tool results as fallback
                        return "\n\n".join([r["content"] for r in tool_results]), all_chunk_metadata
                
                elif hasattr(final_response, 'content'):
                    # Anthropic format
                    text_content = ""
                    for block in final_response.content:
                        if hasattr(block, 'text'):
                            text_content += block.text
                    
                    if text_content:
                        print(f"[MCP] Returning final response: {len(text_content)} chars")
                        return text_content, all_chunk_metadata
                    else:
                        print("[MCP] WARNING: Final response has no text content")
                        return "\n\n".join([r["content"] for r in tool_results]), all_chunk_metadata
                
                else:
                    print(f"[MCP] ERROR: Unknown final response format")
                    return "\n\n".join([r["content"] for r in tool_results])
    
    except Exception as e:
        logger.error(f"[MCP] Error in query_mcp: {e}", exc_info=True)
        return f"I apologize, but an error occurred while processing your request: {str(e)}", []