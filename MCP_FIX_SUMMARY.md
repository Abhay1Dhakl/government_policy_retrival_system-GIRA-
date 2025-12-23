# MCP Server Issues Fixed - 2025-12-23

## ✅ Issues Resolved

### 1. **Migrated to google-genai Package** 
- **Problem**: Using deprecated `google-generativeai` package causing warnings
- **Fix**: Migrated both `gira-agent` and `gira-mcp-server` to use `google-genai`
- **Files Changed**:
  - `/gira-ai/gira-agent/services/embeddings/gemini.py`
  - `/gira-ai/gira-agent/requirements.txt`
  - `/gira-ai/gira-mcp-server/embeddings/gemini_embeddings.py`
  - `/gira-ai/gira-mcp-server/requirements.txt`

### 2. **Fixed MCP Tool Names and Document Types**
- **Problem**: Tools were named for healthcare system (`lrd`, `pis`, `hpl`) but this is a government policy system
- **Fix**: Renamed tools to match government policy domain:
  - `lrd` → `constitution` (searches for constitutional documents)
  - `pis` → `acts` (searches for government acts)  
  - `act` → `regulations` (searches for regulations)
  - Added new `general_search` tool
- **File Changed**: `/gira-ai/gira-mcp-server/main.py`

### 3. **MCP Server Now Running Successfully**
- ✅ Gemini API: Connected successfully with `google-genai`
- ✅ Pinecone: Connected to `government-policy-retrival-system` index
- ✅ BM25: Available for hybrid search
- ✅ Server: Running on port 8001

## ⚠️ CRITICAL ISSUE: Wrong Document Types in Database

### The Problem
Your logs show:
```
PINECONE QUERY DEBUG:
  Filter: {'document_type': {'$in': ['pis', 'PIS']}, 'region': {'$in': ['NEPAL', 'NP']}}
📊 PINECONE RESULT DEBUG:
  Total matches: 20  ✅ (Constitution found!)

PINECONE QUERY DEBUG:
  Filter: {'document_type': {'$in': ['lrd', 'LRD']}}
📊 PINECONE RESULT DEBUG:
  Total matches: 0  ❌ (Nothing found)
```

**Your Constitution of Nepal is indexed as `document_type: pis`** (Prescribing Information - a medical term!)

### Temporary Fix Applied
I updated ALL MCP tools to search for `document_type: pis` temporarily:

```python
@mcp.tool(name="constitution", description="Search Constitution and primary legislation documents")
async def document1(query: str, country: str = None) -> Dict[str, Any]:
    # Currently documents are indexed as 'pis' - needs migration to 'act'
    return await _execute_document_search("Constitution", query, "pis", country)
```

**This means your queries should NOW work**, but the data is incorrectly categorized.

## 🔧 RECOMMENDED: Proper Fix

### Option 1: Re-upload Documents with Correct Types (Recommended)
1. Delete existing documents from Pinecone
2. Re-upload Constitution with `document_type: act` or `document_type: directive`
3. Update MCP tools to search for correct types

### Option 2: Update Pinecone Metadata
Use Pinecone's update API to change `document_type` from `pis` to `act`:

```python
# Example update script
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host="your-index-host")

# Fetch all vectors with document_type=pis
results = index.query(
    vector=[0]*1024,  # dummy query
    filter={"document_type": "pis"},
    top_k=10000,
    include_metadata=True
)

# Update each vector's metadata
for match in results['matches']:
    index.update(
        id=match['id'],
        set_metadata={"document_type": "act"}
    )
```

## 📝 Current MCP Tools Available

| Tool Name | Description | Current Search Type |
|-----------|-------------|-------------------|
| `general_search` | Search all documents | `None` (searches everything) |
| `constitution` | Constitutional/primary legislation | `pis` (temp fix) |
| `acts` | Government acts & legislation | `pis` (temp fix) |
| `regulations` | Regulations & bylaws | `pis` (temp fix) |
| `past_cases` | Past user cases | `past_cases` |
| `system_status` | Check MCP server status | N/A |
| `debug_database` | Inspect database contents | N/A |

## 🚀 Testing Your Fix

Try asking: **"What does the Constitution of Nepal say about fundamental rights?"**

Expected behavior:
1. MCP server receives query
2. Calls `constitution` or `acts` tool
3. Searches Pinecone with `document_type: pis, region: NEPAL`
4. Returns 20 matches ✅
5. LLM generates answer with citations

## 📊 How to Verify Documents Are Working

Run this from your GIRA agent container:

```bash
docker exec gira-mcp-server python -c "
import asyncio
from embeddings.manager import get_embedding_async
from search.engine import execute_pinecone_query_async

async def test():
    vec = await get_embedding_async('Constitution fundamental rights')
    result = await execute_pinecone_query_async(
        query_vector=vec,
        filter_dict={'document_type': 'pis', 'region': 'NEPAL'},
        top_k=5
    )
    print(f'Found {len(result.get("matches", []))} matches')
    for m in result.get('matches', [])[:3]:
        print(f'- {m.get("metadata", {}).get("file_name")}: {m.get("score")}')

asyncio.run(test())
"
```

## 📋 Next Steps

1. ✅ **Test the current setup** - Verify queries return answers
2. ⚠️ **Plan data migration** - Decide on proper document type taxonomy
3. 🔄 **Re-index documents** - Upload with correct `document_type` values
4. ✨ **Update MCP tools** - Change from `pis` to proper types

## 🎯 Proper Document Type Taxonomy

For a Government Policy Retrieval System, use:

```python
DOCUMENT_TYPE_TAXONOMY = {
    "constitution": "Constitutional documents and amendments",
    "act": "Primary legislation (Acts, Laws, Statutes)",  
    "regulation": "Secondary legislation (Regulations, Rules, Orders)",
    "directive": "Government directives and circulars",
    "policy": "Policy documents and white papers",
    "bill": "Bills under consideration",
    "gazette": "Official gazette notifications"
}
```

## 🆘 If It Still Doesn't Work

Check:
1. Are tools being called? Check `gira-agent` logs for `[MCP] Calling tool constitution`
2. Does Pinecone return results? Check for `📊 PINECONE RESULT DEBUG: Total matches: 20`
3. Is the LLM getting chunks? Check for `[MCP] Processed content length: X chars`

Contact me with specific error logs if issues persist!
