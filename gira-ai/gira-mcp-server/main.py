"""Government Information Retrieval System (GIRS) - MCP Server

A production-ready hybrid search system combining dense embeddings (Gemini API)
and BM25 sparse search for comprehensive government document retrieval.
"""

import os
import sys
import asyncio
import time
from typing import Dict, Any, Optional

from datetime import datetime

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mcp.server.fastmcp import FastMCP

# Import modular components
from core.constants import STOPWORDS, DOCUMENT_TYPE_SYNONYMS, SECTION_PRIORITY_WEIGHTS, REGION_ALIASES
from search.engine import execute_hybrid_search, execute_pinecone_query, execute_pinecone_past_query
from search.parsing import _process_search_matches, parse_pinecone_response
from embeddings.manager import get_embedding_async, build_dynamic_corpus, update_bm25_with_dynamic_corpus
from embeddings.gemini_embeddings import initialize_gemini
from _utils import document_index, rank_bm25, _policy_corpus

# Initialize MCP server
mcp = FastMCP(
    name="mcp_server",
    host="0.0.0.0",
    port=8001,
    debug=False
)

# Initialize Gemini
gemini_available = initialize_gemini()
print(f"Gemini API: {' Available' if gemini_available else '❌ Unavailable'}", file=sys.stderr)
async def execute_tool_with_timing(tool_name: str, query: str, document_type: Optional[str], country: str = None, user_id: str = None):
    """Execute tool with hybrid search and performance timing"""
    start_time = time.time()
    
    try:
        # If document_type is "all" or "general", treat it as None to search everything
        if document_type and document_type.lower() in ["all", "general"]:
            document_type = None
            
        result = await execute_hybrid_search(query, document_type=document_type, country=country, user_id=user_id, top_k=20)
        
        if result is None:
            return {"matches": [], "error": "No result returned"}
        
        if not isinstance(result, dict):
            return {"matches": [], "error": f"Invalid result type: {type(result)}"}
        
        if "matches" not in result:
            result["matches"] = []
        elif not isinstance(result["matches"], list):
            result["matches"] = []
        
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        return {
            "matches": [],
            "error": str(e),
            "search_metadata": {
                "total_time": round(total_time, 3),
                "error": True
            }
        }


async def _execute_document_search(tool_name: str, query: str, document_type: Optional[str], country: str = None) -> Dict[str, Any]:
    """Execute document search with standardized error handling and response formatting."""
    try:
        result = await execute_tool_with_timing(tool_name, query, document_type, country)
        
        response = {
            "matches": [],
            "total_found": 0,
            "query_processed": query[:100],
            "country_filter": country or "none",
            "document_type": document_type or "all",
            "search_completed": True,
            "search_metadata": result.get("search_metadata", {}),
            "sources_found": []
        }
        
        if result and isinstance(result, dict) and "matches" in result:
            matches = result["matches"]
            if matches and isinstance(matches, list):
                response["total_found"] = len(matches)
                processed_matches = _process_search_matches(matches)
                response["matches"] = processed_matches
                
                sources = set()
                for match in processed_matches:
                    source = match.get("source", "")
                    if source and source.strip():
                        sources.add(source)
                response["sources_found"] = sorted(list(sources))
        
        return response
        
    except Exception as e:
        return {
            "matches": [],
            "total_found": 0,
            "query_processed": query[:100],
            "country_filter": country or "none",
            "document_type": document_type or "all",
            "search_completed": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# MCP Tools

@mcp.tool(name="search_policies", description="Search all government policy documents including Constitution, Acts, Laws, Regulations, Education policies, and more")
async def search_policies(query: str, country: str = None) -> Dict[str, Any]:
    """Primary tool to search ALL government policy documents using hybrid search.
    Use this for any policy, law, regulation, constitution, or government document question.
    """
    # Search with NO document_type filter to find everything
    return await _execute_document_search("Policies", query, None, country)


@mcp.tool(name="search_constitution", description="Search constitutional documents and fundamental laws")
async def search_constitution(query: str, country: str = None) -> Dict[str, Any]:
    """Search constitutional documents, fundamental rights, and primary legislation."""
    # Search with NO filter since documents are indexed as 'pis' temporarily
    return await _execute_document_search("Constitution", query, None, country)


@mcp.tool(name="search_education", description="Search education policies, laws, and regulations")
async def search_education(query: str, country: str = None) -> Dict[str, Any]:
    """Search education-related policies, acts, and regulations."""
    return await _execute_document_search("Education", query, None, country)


@mcp.tool(name="search_health", description="Search health policies and medical regulations")
async def search_health(query: str, country: str = None) -> Dict[str, Any]:
    """Search health policies, medical regulations, and healthcare laws."""
    return await _execute_document_search("Health", query, None, country)


@mcp.tool(name="system_status", description="Check system status and available features")
async def system_status() -> Dict[str, Any]:
    """Get system status and available features"""
    return {
        "gemini_available": gemini_available,
        "bm25_available": rank_bm25 is not None,
        "corpus_size": len(_policy_corpus),
        "features": {
            "hybrid_search": gemini_available and rank_bm25 is not None,
            "semantic_search": gemini_available,
            "dynamic_corpus": True,
            "bm25_ranking": rank_bm25 is not None
        }
    }


@mcp.tool(name="rebuild_corpus", description="Manually rebuild the dynamic government policy corpus")
async def rebuild_corpus() -> Dict[str, Any]:
    """Manually rebuild the dynamic government policy corpus"""
    try:
        corpus = await build_dynamic_corpus()
        return {
            "status": "success",
            "corpus_size": len(corpus),
            "sample_terms": corpus[:10] if corpus else [],
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool(name="past_cases", description="Get past cases with hybrid search")
async def past_cases(query: str, user_id: str = None) -> Dict[str, Any]:
    """Get past_cases data from Pinecone with hybrid search and user_id filtering"""
    return await execute_tool_with_timing("past_cases", query, "past_cases", user_id=user_id)


@mcp.tool(name="debug_database", description="Inspect all documents in the database")
async def inspect_database(top_k: int = 50) -> Dict[str, Any]:
    """Comprehensive tool to inspect all documents in the database."""
    try:
        query_vector = await get_embedding_async("government policy information")
        
        from search.engine import execute_pinecone_query_async
        response = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={},
            top_k=top_k
        )
        
        matches = response.get("matches", [])
        if not matches:
            return {"error": "No documents found in database"}
            
        document_types = {}
        all_sources = set()
        regions = set()
        metadata_fields = set()
        
        for match in matches:
            metadata = match.get("metadata", {})
            if not metadata:
                continue
                
            doc_type = metadata.get("document_type", "unknown")
            if doc_type not in document_types:
                document_types[doc_type] = {"count": 0, "sources": set()}
            document_types[doc_type]["count"] += 1
            
            for key in metadata.keys():
                metadata_fields.add(key)
            
            for source_field in ["file_name", "source", "filename", "document_name", "doc_name"]:
                if source_field in metadata and metadata[source_field]:
                    source_name = str(metadata[source_field])
                    all_sources.add(source_name)
                    document_types[doc_type]["sources"].add(source_name)
            
            if "region" in metadata and metadata["region"]:
                regions.add(metadata["region"])
        
        formatted_doc_types = {}
        for doc_type, info in document_types.items():
            formatted_doc_types[doc_type] = {
                "count": info["count"],
                "sources": sorted(list(info["sources"]))
            }
        
        return {
            "total_documents_found": len(matches),
            "document_types": formatted_doc_types,
            "all_source_filenames": sorted(list(all_sources)),
            "available_regions": sorted(list(regions)),
            "metadata_fields_found": sorted(list(metadata_fields))
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool(name="debug_document_type", description="Debug a specific document type")
async def debug_document_type(document_type: str = "pis", top_k: int = 20) -> Dict[str, Any]:
    """Debug tool to inspect the metadata of a given document type."""
    try:
        query_vector = await get_embedding_async("legislation information")
        
        from search.engine import execute_pinecone_query_async
        response = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={"document_type": document_type},
            top_k=top_k
        )
        
        matches = response.get("matches", [])
        if not matches:
            return {"error": f"No documents found for document_type='{document_type}'"}
            
        field_counts = {}
        region_values = set()
        section_title_samples = set()
        chunk_type_samples = set()
        source_filenames = set()

        for match in matches:
            metadata = match.get("metadata", {})
            if not metadata:
                continue
            
            for key, value in metadata.items():
                field_counts[key] = field_counts.get(key, 0) + 1
                if value:
                    if key == "region":
                        region_values.add(value)
                    if key == "section_title" and len(section_title_samples) < 10:
                        section_title_samples.add(str(value))
                    if key == "chunk_type" and len(chunk_type_samples) < 10:
                        chunk_type_samples.add(str(value))
                    if key in ["file_name", "source", "filename", "document_name", "doc_name"] and len(source_filenames) < 20:
                        source_filenames.add(str(value))

        return {
            "document_type_analyzed": document_type,
            "documents_checked": len(matches),
            "metadata_field_counts": field_counts,
            "distinct_regions_found": sorted(list(region_values)),
            "sample_section_titles": sorted(list(section_title_samples)),
            "sample_chunk_types": sorted(list(chunk_type_samples)),
            "actual_source_filenames": sorted(list(source_filenames))
        }

    except Exception as e:
        return {"error": str(e)}


async def startup():
    """Pre-warm expensive resources and initialize hybrid search system"""
    print("GIRA MCP Server starting up...", file=sys.stderr)

    print("System Status:", file=sys.stderr)
    print(f" Gemini API: {'Available ' if gemini_available else 'Unavailable ❌'}", file=sys.stderr)
    print(f" BM25: {'Available' if rank_bm25 else 'Unavailable (semantic search only)'}", file=sys.stderr)
    print(f" Pinecone: {'Connected' if document_index else 'Not connected'}", file=sys.stderr)

    if gemini_available:
        print(" Warming up Gemini API...", file=sys.stderr)
        test_embedding = await get_embedding_async("government terminology test query")
        if test_embedding:
            print(f"  Gemini API ready (embedding dimension: {len(test_embedding)})", file=sys.stderr)
        else:
            print(" ⚠️ Gemini API test failed", file=sys.stderr)

    try:
        await build_dynamic_corpus()
        print(f" Dynamic corpus: {len(_policy_corpus)} terms loaded", file=sys.stderr)
    except Exception as e:
        print(f" Dynamic corpus initialization failed: {e}", file=sys.stderr)

    if rank_bm25:
        try:
            await update_bm25_with_dynamic_corpus()
            print(" BM25 encoder: Initialized with government policy corpus", file=sys.stderr)
        except Exception as e:
            print(f" BM25 initialization failed: {e}", file=sys.stderr)

    print("GIRA MCP Server startup complete - hybrid search ready!", file=sys.stderr)


if __name__ == "__main__":
    print("Initializing GIRA MCP server with hybrid search...", file=sys.stderr)
    
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(startup())
        
        print("Starting MCP server on host=0.0.0.0, port=8001", file=sys.stderr)
        loop.run_until_complete(mcp.run(transport='sse'))
        
    except KeyboardInterrupt:
        print("MCP Server shutting down gracefully...", file=sys.stderr)
    except Exception as e:
        print(f" Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)
