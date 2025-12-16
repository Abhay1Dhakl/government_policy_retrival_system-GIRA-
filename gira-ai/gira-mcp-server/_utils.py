"""Global instances and utilities module."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import Pinecone

# Initialize Pinecone with error handling
try:
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY", "pcsk_2RGA3Z_LVfVmxNQ7A7DX7w5BuhEW4MTCGmGuSghX7GmMwizqWqVCumyrWCcMdtE1jDxgav"),
        environment="aped-4627-b74a"
    )
    
    document_index_host = pc.describe_index(name=os.getenv("PINECONE_INDEX_NAME", "government-policy-retrival-system")).host
    document_index = pc.Index(host=document_index_host, grpc_config=GRPCClientConfig(secure=False))
    print("✅ Pinecone connection successful", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Pinecone initialization failed: {e}", file=sys.stderr)
    pc = None
    document_index = None

# BM25 initialization
rank_bm25 = None
bm25_encoder = None

try:
    from pinecone_text.sparse import BM25Encoder
    from rank_bm25 import BM25Okapi
    rank_bm25 = BM25Okapi
    print("✓ BM25 encoder available", file=sys.stderr)
except ImportError as e:
    print(f"⚠ BM25 encoder not available: {e} - semantic search only", file=sys.stderr)
except Exception as e:
    print(f"⚠ BM25 encoder initialization failed: {e} - semantic search only", file=sys.stderr)

# Global thread pool
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Global corpus variables
_medical_corpus = []
_corpus_last_updated = None
_corpus_update_interval = int(os.getenv("CORPUS_UPDATE_INTERVAL", "3600"))
