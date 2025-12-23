"""
Policy PDF Highlighter Module for GIRA AI
Main coordinator for PDF highlighting operations.
Renamed from MedicalPDFHighlighter to reflect government domain.
"""

import os
import fitz
import uuid
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import requests
import tempfile

from pdf_highlighter.minio_client import MinioClientWrapper
from pdf_highlighter.cleanup import cleanup_manager
from pdf_highlighter.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class PolicyHighlighter:
    """
    Government Policy PDF highlighter with MinIO integration.
    Previously 'MedicalPDFHighlighter'.
    """
    
    def __init__(self, 
                 upload_directory: str = "uploads/highlighted",
                 cleanup_delay: int = 3600,
                 minio_endpoint: str = None,
                 minio_access_key: str = None,
                 minio_secret_key: str = None,
                 minio_bucket: str = "government-policies",
                 minio_secure: bool = False):
        
        self.upload_directory = upload_directory
        os.makedirs(upload_directory, exist_ok=True)
        
        # Initialize components
        self.minio = MinioClientWrapper(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
            documents_bucket=minio_bucket
        )
        
        self.cleanup = cleanup_manager(cleanup_delay)
        self.processor = TextProcessor()
        
        # Government policy highlighting colors (adjusted for domain)
        self.highlight_colors = {
            "policy_core": (1, 1, 0),      # Yellow - Main policy text
            "eligibility": (0, 1, 0),      # Green - Eligibility criteria
            "restrictions": (1, 0, 0),     # Red - Bans/Restrictions
            "procedure": (0, 0, 1),        # Blue - Application steps
            "financial": (1, 0.5, 0),      # Orange - Fees/Subsidies/Fines
            "legal": (0.5, 0, 0.5),        # Purple - Legal acts/clauses
            "default": (1, 1, 0)
        }

    def highlight_text(self, 
                       input_path: str, 
                       highlights: List[Dict[str, str]], 
                       output_filename: str = None) -> Dict[str, str]:
        """
        Highlight text in a PDF document.
        
        Args:
            input_path: Path/URL to source PDF
            highlights: List of dicts with 'text' and optional 'category'
            output_filename: Optional custom output name
            
        Returns:
            Dict with 'local_path' and 'url' (if MinIO enabled)
        """
        doc, original_name, temp_file = self._get_pdf_document(input_path)
        
        try:
            total_matches = 0
            
            for item in highlights:
                text = item.get('text', '')
                category = item.get('category', 'default')
                color = self.highlight_colors.get(category, self.highlight_colors['default'])
                
                # Get variations
                search_texts = self.processor.generate_search_variations(text)
                
                # Search and highlight
                for page in doc:
                    for search_text in search_texts:
                        quads = page.search_for(search_text)
                        if quads:
                            for quad in quads:
                                annot = page.add_highlight_annot(quad)
                                annot.set_colors(stroke=color)
                                annot.update()
                                total_matches += 1
                            break # Found match for this text on this page, move to next var/text? 
                            # (Simplified logic for brevity in refactor)
            
            # Save output
            if not output_filename:
                output_filename = f"highlighted_{uuid.uuid4().hex}_{original_name}"
            
            output_path = os.path.join(self.upload_directory, output_filename)
            doc.save(output_path)
            
            result = {"local_path": output_path}
            
            # Upload to MinIO
            if self.minio.client:
                minio_url = self.minio.upload_file(output_path, output_filename)
                if minio_url:
                    result["url"] = minio_url
                    # Schedule cleanup
                    self.cleanup.schedule_cleanup(output_path, self.minio, output_filename)
            
            return result
            
        finally:
            doc.close()
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def _get_pdf_document(self, input_path: str) -> Tuple[fitz.Document, str, Optional[tempfile.NamedTemporaryFile]]:
        """Resolve input path to PDF document"""
        temp_file = None
        
        if self._is_minio_path(input_path):
            # Extract object name from path like "bucket/obj" or just "obj"
            # Simplified logic for now
            obj_name = input_path.split("/")[-1] 
            temp_file = self.minio.download_file(obj_name)
            doc = fitz.open(temp_file.name)
            original_name = obj_name
            
        elif self._is_url(input_path):
            temp_file = self._download_url_pdf(input_path)
            doc = fitz.open(temp_file.name)
            original_name = os.path.basename(urlparse(input_path).path) or "document.pdf"
            
        else:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")
            doc = fitz.open(input_path)
            original_name = os.path.basename(input_path)
            
        return doc, original_name, temp_file

    def _is_url(self, path: str) -> bool:
        try:
            res = urlparse(path)
            return all([res.scheme, res.netloc])
        except: return False

    def _is_minio_path(self, path: str) -> bool:
        # Simple heuristic: acts like minio path if not local and no scheme
        return not os.path.exists(path) and not self._is_url(path) and not path.startswith("/")

    def _download_url_pdf(self, url: str) -> tempfile.NamedTemporaryFile:
        headers = {'User-Agent': 'GIRA-Agent/1.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp.write(chunk)
        temp.close()
        return temp

# Backward compatibility alias
MedicalPDFHighlighter = PolicyHighlighter
