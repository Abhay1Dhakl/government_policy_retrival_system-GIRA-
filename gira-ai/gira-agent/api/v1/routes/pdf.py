"""
PDF highlighting endpoints
Handles text highlighting in medical documents
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
from api.v1.models.requests import HighlightTextRequest, CancelCleanupRequest
from middleware.auth import decode_user_id_from_header
from logging_config import get_logger
from config import settings
from pdf_highlighter import MedicalPDFHighlighter
import re
logger = get_logger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF"])

# Initialize PDF Highlighter
pdf_highlighter = MedicalPDFHighlighter(
    minio_endpoint=settings.MINIO_ENDPOINT,
    minio_access_key=settings.MINIO_ACCESS_KEY,
    minio_secret_key=settings.MINIO_SECRET_KEY,
    minio_bucket=settings.MINIO_BUCKET,
    minio_secure=settings.MINIO_SECURE,
    cleanup_delay=settings.PDF_CLEANUP_DELAY
)


@router.post("/highlight_pdf_text")
async def highlight_pdf_text(req: HighlightTextRequest, request: Request):
    """
    Highlight specific text in a PDF and return the highlighted PDF directly.
    Supports local files, URLs, and MinIO object paths.
    
    Request body example:
    {
        "pdf_path": "medical-docs/azithromycin_study.pdf",  // MinIO object path
        "texts_to_highlight": [
            {
                "text": "azithromycin dosage",
                "type": "drug_info",
                "page": 1
            }
        ],
        "output_filename": "highlighted_document.pdf",
        "auto_cleanup": true,
        "cleanup_delay": 3600,
        "return_file": true  // Set to true to return PDF directly, false for metadata
    }
    """
    # Decode user ID from headers
    user_id, country = decode_user_id_from_header(request)
    
    if not user_id:
        return JSONResponse(
            content={"error": "User ID not found in headers"}, 
            status_code=400
        )
    
    try:
        # Highlight text in PDF using global highlighter
        result = pdf_highlighter.highlight_text_in_pdf(
            input_pdf_path=req.pdf_path,
            texts_to_highlight=req.texts_to_highlight,
            user_id=user_id,
            output_filename=req.output_filename,
            auto_cleanup=req.auto_cleanup,
            cleanup_delay=req.cleanup_delay
        )
        
        if result["success"]:
            # Check if we should return the file directly or metadata
            return_file = getattr(req, 'return_file', False)
            
            if return_file:
                # Get the local file path from the result first
                output_path = result.get("local_file_path")
                
                # If local file exists, return it directly
                if output_path and os.path.exists(output_path):
                    return FileResponse(
                        path=output_path,
                        filename=result["output_filename"],
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f"inline; filename={result['output_filename']}",
                            "X-Total-Highlights": str(result["total_highlights"]),
                            "X-Source-Type": result["source_type"],
                            "X-Original-Filename": result["original_filename"]
                        }
                    )        
            else:
                try:
                    import tempfile
                    # Download from MinIO to temporary file
                    minio_object_name = result["minio_object_name"]
                    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                    
                    # Download from MinIO highlighted bucket
                    pdf_highlighter.minio_client.fget_object(
                        pdf_highlighter.highlighted_bucket, 
                        minio_object_name, 
                        temp_file.name
                    )
                    
                    print(f"Downloaded {minio_object_name} from MinIO for direct file return")
                    
                    # Return the downloaded file
                    return FileResponse(
                        path=temp_file.name,
                        filename=result["output_filename"],
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f"inline; filename={result['output_filename']}",
                            "X-Total-Highlights": str(result["total_highlights"]),
                            "X-Source-Type": "minio",
                            "X-Original-Filename": result["original_filename"]
                        }
                    )
                    
                except Exception as e:
                    print(f"Failed to download from MinIO: {e}")
                    return JSONResponse(
                        content={"error": f"Failed to download highlighted PDF from MinIO: {str(e)}"}, 
                        status_code=500
                    )
                
        else:
            return JSONResponse(
                content={"error": result["error"]}, 
                status_code=500
            )
            
    except Exception as e:
        print(f"[highlight_pdf_text] Error: {e}")
        return JSONResponse(
            content={"error": "Failed to highlight PDF"}, 
            status_code=500
        )

