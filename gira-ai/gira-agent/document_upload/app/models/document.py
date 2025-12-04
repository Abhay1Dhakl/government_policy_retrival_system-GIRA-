import uuid
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import dotenv
import os
from gemini_embeddings import get_gemini_embedding

dotenv.load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a dense index with integrated embedding
index_name = os.getenv("PINECONE_INDEX_NAME", "government-policy-retrival-system")
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", 
                            region="us-east-1")
    )

def get_embedding_data(text):
    try:
        if not text or not text.strip():
            return None
            
        # Gemini can handle up to 2048 tokens, truncate if needed
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length]
            
        # Generate Gemini embedding (1024 dimensions for llama-text-embed-v2)
        embedding_list = get_gemini_embedding(text, task_type="retrieval_document")
        
        # Validate embedding
        if embedding_list is None or len(embedding_list) != 1024:
            print(f"Warning: Failed to generate valid Gemini embedding")
            return None
            
        return embedding_list
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

async def store_document(data, text_content=None):
    chunks = data.get("chunks", [])
    if not chunks:
        return {
            "instance_id": data.get("instance_id", "unknown"),
            "status": "error",
            "message": "No chunks to process",
            "chunks_stored": 0
        }
    
    index = pc.Index(index_name)
    
    # Test embedding generation
    test_embedding = get_embedding_data("test text")
    if not test_embedding:
        return {
            "instance_id": data.get("instance_id", "unknown"),
            "status": "error",
            "message": "Embedding model not working",
            "chunks_stored": 0
        }
    
    # Use the provided instance_id or generate one if missing
    instance_id = data.get("instance_id") or str(uuid.uuid4())
    
    # Delete existing vectors for this specific document instance
    try:
        index.delete(filter={"instance_id": {"$eq": instance_id}})
    except Exception as e:
        print(f"Warning: Could not delete existing vectors for instance_id {instance_id}: {e}")
        # Continue anyway - this might be the first time storing this document

    vectors_to_upsert = []
    for chunk in chunks:
        # Unique ID per chunk, but tied to the same instance_id
        vector_id = f"{instance_id}_{chunk.get('page', 0)}_{chunk.get('chunk_index', 0)}"

        metadata = {
            "instance_id": instance_id,
            "document_type": data.get("document_type"),
            "source_type": data.get("source_type"),
            "title": data.get("document_metadata", {}).get("title"),
            "language": data.get("document_metadata", {}).get("language"),
            "region": data.get("document_metadata", {}).get("region"),
            "author": data.get("document_metadata", {}).get("author"),
            "tags": data.get("document_metadata", {}).get("tags"),
            "ingested_at": datetime.utcnow().isoformat(),
            "file_name": data.get("file_name"),
            "mime_type": data.get("mime_type"),
            "page": chunk.get("page"),
            "chunk_index": chunk.get("chunk_index"),
            "text": chunk.get("text")
        }

        for key, value in chunk.items():
            if key not in metadata and value is not None:
                metadata[key] = value
        
        # Embed just this chunk's text
        chunk_text = chunk.get("text", "")
        if not chunk_text or not chunk_text.strip():
            continue
            
        embedding = get_embedding_data(chunk_text)
        
        # Validate embedding
        if not embedding or len(embedding) != 1024:
            continue
        
        # Clean metadata - remove None values and ensure proper types
        clean_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert values to strings if they're not basic types
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif isinstance(value, list):
                    # For lists, ensure all elements are strings
                    clean_metadata[key] = [str(item) for item in value if item is not None]
                else:
                    clean_metadata[key] = str(value)
        
        vector_data = {
            "id": vector_id,
            "values": embedding,
            "metadata": clean_metadata
        }
        
        vectors_to_upsert.append(vector_data)
    
    if not vectors_to_upsert:
        return {
            "instance_id": instance_id,
            "status": "error",
            "message": "No valid chunks to process",
            "chunks_stored": 0
        }
    
    # Upsert all chunks in a single request
    try:
        upsert_response = index.upsert(vectors_to_upsert)
    except Exception as upsert_error:
        print(f"Upsert failed: {upsert_error}")
        raise

    return {
        "instance_id": instance_id,
        "status": "stored",
        "chunks_stored": len(chunks),
        "last_ingested_at": datetime.utcnow().isoformat()
    }
