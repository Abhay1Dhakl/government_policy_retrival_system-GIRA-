import os
import json
from typing import List, Tuple, Any, Optional
import asyncio
import sys
import logging
from mcp import ClientSession
from mcp.client.sse import sse_client
from services.llm_service import choose_llm
from services.prompt_service import generate_system_prompt
from services.response_service import process_mcp_response

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("query_mcp")

# LEGACY TOOL MAPPING (Handles outdated frontend state)
LEGACY_TOOL_MAP = {
    "pis": "search_policies",
    "lrd": "search_constitution",
    "grd": "search_policies",
    "online_db": "search_policies",
    "past_cases": "past_cases"
}

async def query_mcp(user_query: str, llm: str, tools: List[str], country: str, user_id: str) -> Tuple[str, List[Any]]:
    """
    Query the MCP server using official MCP SDK with full government policy support.
    Includes legacy tool mapping to handle stale frontend states.
    """
    print(f"[MCP] query_mcp received tools: {tools}", file=sys.stderr)
    
    # Resolve legacy tools to new tool names
    resolved_tools = []
    for t in tools:
        if t in LEGACY_TOOL_MAP:
            mapped_t = LEGACY_TOOL_MAP[t]
            if mapped_t not in resolved_tools:
                resolved_tools.append(mapped_t)
                print(f"[MCP] Mapped legacy tool {t} -> {mapped_t}", file=sys.stderr)
        else:
            if t not in resolved_tools:
                resolved_tools.append(t)

    print(f"[MCP] Final resolved tools for server: {resolved_tools}", file=sys.stderr)

    final_content = ""
    all_chunk_metadata = []
    collected_tool_results = []
    collected_tool_calls = []
    
    try:
        # Load MCP server configuration
        environment = os.getenv("ENVIRONMENT", "development").lower()
        config_file = "mcp_server_config/config_development.json" if environment != "production" else "mcp_server_config/config_production.json"

        if not os.path.exists(config_file):
            print(f"[MCP] ERROR: Config file {config_file} not found", file=sys.stderr)
            return f"Error: Configuration file not found.", []

        with open(config_file, 'r') as f:
            config = json.load(f)
        
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            raise Exception("No MCP servers configured")
        
        server_name = list(mcp_servers.keys())[0]
        server_url = mcp_servers[server_name].get("url")
        if not server_url:
            raise Exception(f"No URL configured for server {server_name}")
        
        print(f"[MCP] Connecting to server: {server_name} at {server_url}", file=sys.stderr)
        
        # Connect to MCP server via SSE
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List and filter tools
                server_tools_resp = await session.list_tools()
                available_tool_names = {t.name: t for t in server_tools_resp.tools}
                print(f"[MCP] Server reported {len(available_tool_names)} tools: {list(available_tool_names.keys())}", file=sys.stderr)
                
                filtered_tool_defs = []
                for tool_name in resolved_tools:
                    if tool_name in available_tool_names:
                        tool_def = available_tool_names[tool_name]
                        filtered_tool_defs.append({
                            "type": "function",
                            "function": {
                                "name": tool_def.name,
                                "description": tool_def.description or "",
                                "parameters": tool_def.inputSchema
                            }
                        })
                        print(f"[MCP] Including tool: {tool_name}", file=sys.stderr)
                
                if not filtered_tool_defs:
                    print(f"[MCP] WARNING: No matching tools found for {resolved_tools}", file=sys.stderr)
                    return "I'm sorry, I don't have access to the specific tools required to answer that. Please check your document selection.", []

                # Initialize LLM
                llm_instance = choose_llm(llm, temperature=0.1)
                system_prompt = generate_system_prompt(user_query, country, resolved_tools)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
                
                print(f"[MCP] Initial LLM call with {len(filtered_tool_defs)} tools...", file=sys.stderr)
                ai_message = await llm_instance.ainvoke(messages, tools=filtered_tool_defs)
                
                # Extract tool calls
                if hasattr(ai_message, 'choices') and ai_message.choices:
                    msg = ai_message.choices[0].message
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        collected_tool_calls = msg.tool_calls
                    elif msg.content:
                        final_content = msg.content
                elif hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
                    collected_tool_calls = ai_message.tool_calls
                elif hasattr(ai_message, 'content'):
                    if isinstance(ai_message.content, str):
                        final_content = ai_message.content
                    elif isinstance(ai_message.content, list):
                        for block in ai_message.content:
                            if hasattr(block, 'type') and block.type == 'tool_use':
                                collected_tool_calls.append(block)
                            elif hasattr(block, 'text'):
                                final_content += block.text

                # Execute tool calls
                if collected_tool_calls:
                    print(f"[MCP] Executing {len(collected_tool_calls)} tool calls", file=sys.stderr)
                    for tc in collected_tool_calls:
                        if hasattr(tc, 'function'): # OpenAI
                            tc_name = tc.function.name
                            tc_args = json.loads(tc.function.arguments)
                            tc_id = tc.id
                        elif hasattr(tc, 'name'): # Anthropic
                            tc_name = tc.name
                            tc_args = tc.input if hasattr(tc, 'input') else {}
                            tc_id = tc.id
                        else:
                            continue

                        # Add context
                        tc_args["country"] = country
                        tc_args["user_id"] = user_id
                        
                        try:
                            print(f"[MCP] -> Calling: {tc_name}", file=sys.stderr)
                            result = await session.call_tool(tc_name, arguments=tc_args)
                            text_result, chunks = process_mcp_response(result.content, tc_name)
                            all_chunk_metadata.extend(chunks)
                            
                            collected_tool_results.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": tc_name,
                                "content": text_result
                            })
                        except Exception as e:
                            print(f"[MCP] Tool execution error for {tc_name}: {e}", file=sys.stderr)
                            collected_tool_results.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": tc_name,
                                "content": f"Error executing tool: {str(e)}"
                            })
        
        # -- OUTSIDE SSE CONNECTION BLOCK (Session is closed) --
        
        if not collected_tool_results and not final_content:
            return "I couldn't retrieve any relevant information from the policy database.", []

        # Final Synthesis
        if collected_tool_results:
            llm_instance = choose_llm(llm, temperature=0.1)
            system_prompt = generate_system_prompt(user_query, country, resolved_tools)
            
            final_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": None, "tool_calls": collected_tool_calls}
            ] + collected_tool_results
            
            if all_chunk_metadata:
                guide = "\n=== CITATION GUIDE ===\n"
                for i, chunk in enumerate(all_chunk_metadata[:15]):
                    guide += f"Document {i+1}: {chunk.get('source', 'Unknown')} (Page {chunk.get('page_number', 'N/A')})\n"
                guide += "\nCite using [doc_num.chunk_index] format.\n"
                final_messages.insert(0, {"role": "system", "content": guide})

            print(f"[MCP] Final synthesis with {len(collected_tool_results)} results", file=sys.stderr)
            final_ai_msg = await llm_instance.ainvoke(final_messages)
            
            if hasattr(final_ai_msg, 'choices') and final_ai_msg.choices:
                final_content = final_ai_msg.choices[0].message.content or ""
            elif hasattr(final_ai_msg, 'content'):
                if isinstance(final_ai_msg.content, str):
                    final_content = final_ai_msg.content
                else:
                    final_content = "".join([b.text for b in final_ai_msg.content if hasattr(b, 'text')])
                    
        return final_content, all_chunk_metadata

    except Exception as e:
        logger.error(f"[MCP] query_mcp error: {e}", exc_info=True)
        return f"I apologize, but an error occurred while connecting to the policy database: {str(e)}", []