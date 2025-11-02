from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from document_upload.app.services.api_ingestor import ingest_from_api
from document_upload.app.services.file_ingestor import ingest_from_file
from document_upload.app.services.manual_text_ingestor import ingest_from_manual_text
import json 
router = APIRouter()

class APIConnectionInfo(BaseModel):
    auth_type: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_url: Optional[str] = None
    data_url: Optional[str] = None

class DocumentMetadata(BaseModel):
    title: str
    language: str
    region: str
    author: str
    tags: List[str]

class IngestionRequest(BaseModel):
    instance_id: str
    document_type: str
    source_type: str  # api, upload, manual
    # source_system: str  # veeva, sharepoint, etc.
    document_metadata: DocumentMetadata
    api_connection_info: Optional[APIConnectionInfo] = None
    manual_text: Optional[str] = None

@router.post("/ingest")
async def ingest_document(
    data: str = Form(...),
    file_upload: Optional[UploadFile] = File(None)
):
    try:
        print("Received data:", data)
        data_dict = json.loads(data)
        print("Received data:", data_dict)
        if not isinstance(data_dict, dict):
            raise ValueError("Data must be a valid JSON object")   
        data = IngestionRequest.model_validate(data_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")
    if data.source_type == "api":
        return await ingest_from_api(data_dict)
    elif data.source_type == "upload":
        if not file_upload:
            raise HTTPException(status_code=400, detail="File must be provided")
        return await ingest_from_file(data_dict, file_upload)
    elif data.source_type == "manual":
        if not data_dict["manual_text"]:
            raise HTTPException(status_code=400, detail="Manual text required")
        return await ingest_from_manual_text(data_dict)
    else:
        raise HTTPException(status_code=400, detail="Invalid source_type")
