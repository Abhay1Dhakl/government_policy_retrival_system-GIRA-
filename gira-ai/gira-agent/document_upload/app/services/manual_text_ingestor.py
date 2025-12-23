from document_upload.app.models.document import store_document
from .chunking_service import healthcare_storage
import uuid

async def ingest_from_manual_text(data: dict):
    """
    Ingest document from manually provided text.
    Processing involves chunking the text and storing it.
    """
    manual_text = data.get("manual_text", "")
    if not manual_text:
        return {
            "status": "error",
            "message": "No manual text provided"
        }

    # Generate a document name if not provided
    document_name = data.get("document_metadata", {}).get("title") or f"manual_entry_{uuid.uuid4().hex[:8]}"
    
    # Process text for chunks
    # Manual text is treated as a single page (page 1)
    chunks = healthcare_storage.process_document_text(
        text=manual_text,
        document_name=document_name,
        page_number=1
    )

    chunk_metadata_with_data = {
        **data,
        "file_name": document_name,
        "mime_type": "text/plain",
        "chunks": chunks,
        "chunking_method": "healthcare_semantic",
        "source_type": "manual"
    }

    return await store_document(chunk_metadata_with_data, None)
