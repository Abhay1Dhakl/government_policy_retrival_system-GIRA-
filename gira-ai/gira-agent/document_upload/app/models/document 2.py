import uuid
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import dotenv
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

model = SentenceTransformer('BAAI/bge-m3')

# Dimension reduction for BGE-M3 (1024 dims -> 384 dims)
_pca_reducer = None

def _train_pca_reducer():
    """Train PCA reducer for BGE-M3 dimension reduction (1024 -> 384)"""
    global _pca_reducer
    if _pca_reducer is None:
        try:
            print("Training PCA reducer for BGE-M3...")
            
            # Generate sample medical text for training
            sample_texts = [
                "medical treatment and dosage information",
                "drug interactions and contraindications", 
                "side effects and adverse reactions",
                "pediatric dosing and safety guidelines",
                "contraindications and warnings for patients",
                "pharmacokinetics and drug metabolism",
                "cardiac effects and monitoring requirements",
                "hepatic function and dose adjustments",
                "renal impairment considerations",
                "geriatric population safety data",
                "pregnancy and lactation information",
                "clinical trial efficacy data",
                "therapeutic indications and usage",
                "overdose management and treatment",
                "drug monitoring and laboratory tests"
            ] * 20  # Repeat to get more samples
            
            # Generate embeddings for training
            embeddings = []
            for text in sample_texts:
                emb = model.encode(text, convert_to_tensor=False)
                embeddings.append(emb)
            
            # Train PCA
            embedding_matrix = np.array(embeddings)
            _pca_reducer = PCA(n_components=384)
            _pca_reducer.fit(embedding_matrix)
            
            variance_explained = _pca_reducer.explained_variance_ratio_.sum()
            print(f"PCA trained. Variance explained: {variance_explained:.3f}")
            
        except Exception as e:
            print(f"Failed to train PCA reducer: {e}")
            _pca_reducer = None

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

# Initialize PCA reducer
_train_pca_reducer()

def get_embedding_data(text):
    try:
        if not text or not text.strip():
            return None
            
        # BGE-M3 can handle much longer text (up to 8192 tokens)
        max_length = 4000  # Increased for BGE-M3
        if len(text) > max_length:
            text = text[:max_length]
            
        # Generate BGE-M3 embedding (1024 dimensions)
        embedding = model.encode(text, convert_to_tensor=False)
        
        # Apply PCA dimension reduction if available
        if _pca_reducer is not None:
            try:
                # Ensure embedding is 2D for PCA
                emb_array = np.array(embedding)
                if len(emb_array.shape) == 1:
                    emb_array = emb_array.reshape(1, -1)
                
                # Apply PCA reduction: 1024 -> 384
                reduced_embedding = _pca_reducer.transform(emb_array)
                embedding_list = reduced_embedding[0].tolist()
                
                print(f"Applied PCA reduction: {len(embedding)} -> {len(embedding_list)} dimensions")
                
            except Exception as pca_error:
                print(f"PCA reduction failed: {pca_error}")
                # Fallback: truncate to 384 dimensions
                embedding_list = embedding[:384].tolist() if len(embedding) > 384 else embedding.tolist()
        else:
            print("PCA reducer not available, truncating to 384 dimensions")
            # Fallback: truncate to 384 dimensions
            embedding_list = embedding[:384].tolist() if len(embedding) > 384 else embedding.tolist()
        
        # Validate final embedding dimensions
        if len(embedding_list) != 384:
            print(f"Warning: Final embedding has {len(embedding_list)} dimensions, expected 384")
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
        if not embedding or len(embedding) != 384:
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
