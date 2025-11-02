from document_upload.app.models.document import store_document

async def ingest_from_manual_text(data):
    return await store_document(data, data["manual_text"])
