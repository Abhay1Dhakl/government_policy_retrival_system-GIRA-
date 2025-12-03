import fitz
import json
import os
import tempfile
import uuid
import threading
import time
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import requests
from minio import Minio
from minio.error import S3Error


class MedicalPDFHighlighter:
    """Medical PDF highlighter for GIRA AI system with MinIO integration"""
    
    def __init__(self, 
                 upload_directory: str = "uploads/highlighted",
                 cleanup_delay: int = 3600,  # 1 hour in seconds
                 minio_endpoint: str = None,
                 minio_access_key: str = None,
                 minio_secret_key: str = None,
                 minio_bucket: str = "medical-documents",
                 minio_secure: bool = False):
        """
        Initialize the medical PDF highlighter.
        
        Args:
            upload_directory: Directory to save highlighted PDFs
            cleanup_delay: Time in seconds before auto-deleting highlighted files
            minio_endpoint: MinIO server endpoint
            minio_access_key: MinIO access key
            minio_secret_key: MinIO secret key
            minio_bucket: MinIO bucket name for medical documents
            minio_secure: Whether to use HTTPS (True) or HTTP (False) for MinIO
        """
        self.upload_directory = upload_directory
        self.cleanup_delay = cleanup_delay
        self.temp_files = []  # Track temporary files for cleanup
        self.cleanup_timers = {}  # Track cleanup timers
        
        # Ensure upload directory exists
        os.makedirs(upload_directory, exist_ok=True)
        
        # Initialize MinIO client if credentials provided
        self.minio_client = None
        self.minio_bucket = minio_bucket
        self.highlighted_bucket = "highlighted-pdfs"  # Separate bucket for highlighted files
        
        if minio_endpoint and minio_access_key and minio_secret_key:
            try:
                self.minio_client = Minio(
                    minio_endpoint,
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                    secure=minio_secure  # Configurable HTTPS/HTTP
                )
                print(f"MinIO client initialized for endpoint: {minio_endpoint}")
                
                # Ensure highlighted bucket exists
                self._ensure_bucket_exists(self.highlighted_bucket)
                
            except Exception as e:
                print(f"Failed to initialize MinIO client: {e}")
                self.minio_client = None
        
        # Medical highlighting colors
        self.highlight_colors = {
            "drug_info": (1, 1, 0),        # Yellow - Drug information
            "dosage": (0, 1, 0),           # Green - Dosage information  
            "contraindications": (1, 0, 0), # Red - Warnings/contraindications
            "interactions": (1, 0.5, 0),   # Orange - Drug interactions
            "evidence": (0, 0, 1),         # Blue - Clinical evidence
            "default": (1, 1, 0)           # Yellow - Default highlighting
        }

    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure MinIO bucket exists, create if it doesn't."""
        try:
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                print(f"Created MinIO bucket: {bucket_name}")
            else:
                print(f"MinIO bucket already exists: {bucket_name}")
        except Exception as e:
            print(f"Error checking/creating bucket {bucket_name}: {e}")

    def _upload_to_minio(self, local_file_path: str, object_name: str, bucket: str = None) -> Optional[str]:
        """Upload a file to MinIO and return the object URL."""
        if not self.minio_client:
            print("MinIO client not available, skipping upload")
            return None
        
        bucket_name = bucket or self.highlighted_bucket
        
        try:
            # Upload file to MinIO
            self.minio_client.fput_object(bucket_name, object_name, local_file_path)
            
            # Generate URL for the uploaded file
            minio_url = f"minio://{bucket_name}/{object_name}"
            print(f"Uploaded to MinIO: {minio_url}")
            return minio_url
            
        except Exception as e:
            print(f"Failed to upload to MinIO: {e}")
            return None

    def _delete_from_minio(self, object_name: str, bucket: str = None):
        """Delete a file from MinIO."""
        if not self.minio_client:
            return
        
        bucket_name = bucket or self.highlighted_bucket
        
        try:
            self.minio_client.remove_object(bucket_name, object_name)
            print(f"Deleted from MinIO: {bucket_name}/{object_name}")
        except Exception as e:
            print(f"Failed to delete from MinIO: {e}")

    def _is_url(self, path: str) -> bool:
        """Check if the input path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _is_minio_path(self, path: str) -> bool:
        """Check if the path is a MinIO object path (bucket/object_name format)."""
        return "/" in path and not self._is_url(path) and not os.path.exists(path)

    def _download_from_minio(self, object_name: str) -> tempfile.NamedTemporaryFile:
        """Download PDF from MinIO to a temporary file."""
        if not self.minio_client:
            raise ValueError("MinIO client not initialized")
        
        try:
            # Create temporary file
            temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            
            # Download from MinIO
            self.minio_client.fget_object(self.minio_bucket, object_name, temp.name)
            
            print(f"Downloaded {object_name} from MinIO bucket {self.minio_bucket}")
            return temp
            
        except S3Error as e:
            raise ValueError(f"Failed to download from MinIO: {e}")

    def _download_pdf(self, url: str) -> tempfile.NamedTemporaryFile:
        """Download PDF from URL to a temporary file."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Create temporary file
        temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp.write(chunk)
        temp.close()
        return temp

    def _schedule_file_cleanup(self, file_path: str, minio_object_name: str = None, delay: int = None):
        """Schedule automatic file cleanup after specified delay."""
        if delay is None:
            delay = self.cleanup_delay
        
        def cleanup_file():
            try:
                # Clean up local file
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"Auto-deleted local highlighted PDF: {file_path}")
                
                # Clean up MinIO file
                if minio_object_name:
                    self._delete_from_minio(minio_object_name)
                
                # Remove from cleanup timers
                cleanup_key = f"{file_path}|{minio_object_name}" if minio_object_name else file_path
                if cleanup_key in self.cleanup_timers:
                    del self.cleanup_timers[cleanup_key]
                    
            except Exception as e:
                print(f"Error during auto-cleanup of {file_path}: {e}")
        
        # Use a composite key for cleanup timers
        cleanup_key = f"{file_path}|{minio_object_name}" if minio_object_name else file_path
        
        # Cancel existing timer if any
        if cleanup_key in self.cleanup_timers:
            self.cleanup_timers[cleanup_key].cancel()
        
        # Schedule new cleanup
        timer = threading.Timer(delay, cleanup_file)
        timer.start()
        self.cleanup_timers[cleanup_key] = timer
        
        print(f"Scheduled auto-cleanup for {file_path} and MinIO object in {delay} seconds")

    def cancel_file_cleanup(self, file_path: str, minio_object_name: str = None):
        """Cancel scheduled cleanup for a specific file."""
        cleanup_key = f"{file_path}|{minio_object_name}" if minio_object_name else file_path
        
        if cleanup_key in self.cleanup_timers:
            self.cleanup_timers[cleanup_key].cancel()
            del self.cleanup_timers[cleanup_key]
            print(f"Cancelled auto-cleanup for {file_path}")

    def _get_pdf_document(self, input_path: str) -> Tuple[fitz.Document, str, tempfile.NamedTemporaryFile]:
        """
        Get PDF document from various sources (local file, URL, MinIO).
        
        Returns:
            Tuple of (document, original_filename, temp_file_or_none)
        """
        temp_file = None
        
        if self._is_minio_path(input_path):
            # Download from MinIO
            temp_file = self._download_from_minio(input_path)
            doc = fitz.open(temp_file.name)
            original_filename = os.path.basename(input_path)
            
        elif self._is_url(input_path):
            # Download from URL
            temp_file = self._download_pdf(input_path)
            doc = fitz.open(temp_file.name)
            original_filename = os.path.basename(urlparse(input_path).path)
            
        else:
            # Local file
            doc = fitz.open(input_path)
            original_filename = os.path.basename(input_path)
        
        return doc, original_filename, temp_file

    def find_text_coordinates(self, doc: fitz.Document, text_to_highlight: str, page_num: Optional[int] = None) -> List[Dict]:
        """
        Find coordinates of text to highlight in the PDF.
        
        Args:
            doc: PyMuPDF document object
            text_to_highlight: Text string to find and highlight
            page_num: Specific page number to search (None for all pages)
            
        Returns:
            List of dictionaries with highlighting coordinates
        """
        coordinates = []
        
        # Determine which pages to search
        if page_num is not None:
            pages_to_search = [page_num] if 0 <= page_num < len(doc) else []
        else:
            pages_to_search = range(len(doc))
        
        for page_index in pages_to_search:
            page = doc[page_index]
            
            # Search for text instances on this page
            text_instances = page.search_for(text_to_highlight)
            
            for rect in text_instances:
                coordinates.append({
                    "page": page_index,
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height,
                    "text": text_to_highlight,
                    "rectangles": [
                        {
                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1,
                        }
                    ],
                })
        
        return coordinates

    def normalize_pdf_text(self, text: str) -> str:
        """
        Ultra-comprehensive text normalization to handle ALL PDF formatting and styling issues.
        This function makes text format-agnostic for maximum matching capability.
        
        Args:
            text: Raw text that might have PDF formatting issues
            
        Returns:
            Normalized text that's more likely to match PDF content regardless of styling
        """
        if not text:
            return text
            
        # Step 1: Normalize Unicode characters (handles accents, special chars)
        text = unicodedata.normalize('NFKD', text)
        
        # Step 2: Handle hyphenated line breaks (common in PDFs)
        text = re.sub(r'-\s*\n\s*', '', text)
        text = re.sub(r'-\s*[\r\n]+\s*', '', text)
        
        # Step 3: Normalize ALL types of dashes, hyphens, and minus signs
        dash_chars = ['–', '—', '−', '‐', '‑', '‒', '―', '⁻', '﹣', '－', 
                      '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015',
                      '\u2212', '\uFE58', '\uFE63', '\uFF0D']
        for dash in dash_chars:
            text = text.replace(dash, '-')
        
        # Step 4: Normalize ALL quotation marks and apostrophes
        text = re.sub(r'[""''`´‚„‛‟‹›«»「」『』〝〞〟＂]', '"', text)
        text = re.sub(r'[''‛‚ʻʼʽˈˊˋ`´῾]', "'", text)
        
        # Step 5: Handle ALL common ligatures and special characters
        ligatures = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬀ': 'ff',
            'ﬅ': 'ft', 'ﬆ': 'st', 'æ': 'ae', 'œ': 'oe', 'Æ': 'AE', 'Œ': 'OE',
            'ß': 'ss', 'ij': 'ij', 'IJ': 'IJ', 'ĳ': 'ij', 'Ĳ': 'IJ',
            'ǆ': 'dz', 'ǅ': 'Dz', 'ǲ': 'dz', 'ǳ': 'Dz', 'ǰ': 'j', 'ǉ': 'lj',
            'ǈ': 'Lj', 'ǌ': 'nj', 'ǋ': 'Nj', 'ŉ': 'n', 'ŀ': 'l'
        }
        for ligature, replacement in ligatures.items():
            text = text.replace(ligature, replacement)
        
        # Step 6: Normalize mathematical and scientific symbols
        math_symbols = {
            '×': 'x', '÷': '/', '±': '+/-', '∓': '-/+', '∞': 'infinity',
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=', '≡': '===',
            '°': ' degrees', '℃': 'C', '℉': 'F', 'μ': 'micro', 'α': 'alpha',
            'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
            'π': 'pi', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'ω': 'omega'
        }
        for symbol, replacement in math_symbols.items():
            text = text.replace(symbol, replacement)
        
        # Step 7: Normalize fractions and superscripts
        fractions = {
            '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
            '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
            '⅚': '5/6', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8'
        }
        superscripts = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
            '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁺': '+', '⁻': '-',
            'ⁿ': 'n', 'ⁱ': 'i', 'ˣ': 'x'
        }
        subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
            '₆': '6', '₇': '7', '₈': '8', '₉': '9', '₊': '+', '₋': '-',
            'ₙ': 'n', 'ᵢ': 'i', 'ₓ': 'x'
        }
        
        for char_dict in [fractions, superscripts, subscripts]:
            for char, replacement in char_dict.items():
                text = text.replace(char, replacement)
        
        # Step 8: Handle currency symbols and other special characters
        currency_symbols = {
            '$': 'dollar', '€': 'euro', '£': 'pound', '¥': 'yen', '₹': 'rupee',
            '¢': 'cent', '₽': 'ruble', '₩': 'won', '₪': 'shekel', '₨': 'rupee'
        }
        for symbol, replacement in currency_symbols.items():
            text = text.replace(symbol, ' ' + replacement + ' ')
        
        # Step 9: Replace multiple whitespace/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 10: Remove ALL types of invisible/control characters
        text = re.sub(r'\x00', '', text)  # Remove null bytes
        text = re.sub(r'[\u200B-\u200F]', '', text)  # Remove zero-width spaces
        text = re.sub(r'[\u2028-\u202F]', ' ', text)  # Line/paragraph separators
        text = re.sub(r'[\uFEFF]', '', text)  # Remove BOM
        text = re.sub(r'[\u00AD]', '', text)  # Remove soft hyphens
        text = re.sub(r'[\u0001-\u001F\u007F-\u009F]', '', text)  # Remove control chars
        
        # Step 11: Normalize bullet points and list markers
        bullets = ['•', '◦', '‣', '⁃', '▪', '▫', '▸', '▹', '►', '▻', 
                   '⦾', '⦿', '○', '●', '◯', '◉', '✓', '✔', '☑']
        for bullet in bullets:
            text = text.replace(bullet, '*')
        
        # Step 12: Normalize trademark and copyright symbols
        text = text.replace('®', '(R)').replace('©', '(C)').replace('™', '(TM)')
        text = text.replace('℗', '(P)').replace('℠', '(SM)')
        
        # Step 13: Handle Roman numerals (convert to Arabic)
        roman_numerals = {
            'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
            'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
            'ⅰ': 'i', 'ⅱ': 'ii', 'ⅲ': 'iii', 'ⅳ': 'iv', 'ⅴ': 'v',
            'ⅵ': 'vi', 'ⅶ': 'vii', 'ⅷ': 'viii', 'ⅸ': 'ix', 'ⅹ': 'x'
        }
        for roman, replacement in roman_numerals.items():
            text = text.replace(roman, replacement)
        
        # Step 14: Normalize accented characters to base characters
        accented_chars = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a', 'å': 'a', 'ā': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e', 'ė': 'e', 'ę': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i', 'į': 'i', 'ı': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ō': 'o', 'ø': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u', 'ů': 'u', 'ų': 'u',
            'ý': 'y', 'ỳ': 'y', 'ÿ': 'y', 'ŷ': 'y', 'ȳ': 'y',
            'ñ': 'n', 'ç': 'c', 'ß': 'ss', 'ł': 'l', 'ř': 'r', 'š': 's', 'ť': 't',
            'ž': 'z', 'ď': 'd', 'ğ': 'g', 'ș': 's', 'ț': 't'
        }
        # Apply both lowercase and uppercase versions
        for accented, base in accented_chars.items():
            text = text.replace(accented, base)
            text = text.replace(accented.upper(), base.upper())
        
        # Step 15: Final cleanup - remove extra spaces and trim
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def find_text_with_partial_match(self, doc: fitz.Document, text_to_highlight: str, page_num: Optional[int] = None) -> List[Dict]:
        """
        Find text using partial matching and fuzzy search techniques.
        
        Args:
            doc: PyMuPDF document object
            text_to_highlight: Text string to find and highlight
            page_num: Specific page number to search (None for all pages)
            
        Returns:
            List of dictionaries with highlighting coordinates
        """
        coordinates = []
        
        # Determine which pages to search
        if page_num is not None:
            pages_to_search = [page_num] if 0 <= page_num < len(doc) else []
        else:
            pages_to_search = range(len(doc))
        
        # Create ultra-comprehensive search variations to handle ANY formatting differences
        normalized_text = self.normalize_pdf_text(text_to_highlight)
        
        # Base variations
        search_texts = [
            text_to_highlight,  # Original text
            normalized_text,  # Normalized text
        ]
        
        # Generate ALL possible formatting variations
        base_texts = [text_to_highlight, normalized_text]
        
        for base_text in base_texts:
            # Punctuation variations
            search_texts.extend([
                base_text.replace('-', ''),  # Remove all hyphens
                base_text.replace('-', ' '),  # Replace hyphens with spaces
                base_text.replace('_', ' '),  # Replace underscores with spaces
                base_text.replace('_', ''),   # Remove underscores
                base_text.replace('.', ''),   # Remove periods
                base_text.replace(',', ''),   # Remove commas
                base_text.replace(';', ''),   # Remove semicolons
                base_text.replace(':', ''),   # Remove colons
                base_text.replace('!', ''),   # Remove exclamation marks
                base_text.replace('?', ''),   # Remove question marks
                base_text.replace('(', '').replace(')', ''),  # Remove parentheses
                base_text.replace('[', '').replace(']', ''),  # Remove square brackets
                base_text.replace('{', '').replace('}', ''),  # Remove curly brackets
                base_text.replace('"', '').replace("'", ''),  # Remove quotes
                re.sub(r'[^\w\s]', '', base_text),  # Remove ALL punctuation except spaces
                re.sub(r'[^\w]', '', base_text),    # Remove ALL non-alphanumeric characters
            ])
            
            # Spacing variations
            search_texts.extend([
                base_text.replace('\n', ' '),     # Replace newlines with spaces
                base_text.replace('\r', ' '),     # Replace carriage returns with spaces
                base_text.replace('\t', ' '),     # Replace tabs with spaces
                base_text.replace('  ', ' '),     # Replace double spaces with single
                re.sub(r'\s+', ' ', base_text),   # Normalize all whitespace to single spaces
                base_text.replace(' ', ''),       # Remove ALL spaces
                base_text.replace(' ', '_'),      # Replace spaces with underscores
                base_text.replace(' ', '-'),      # Replace spaces with hyphens
            ])
            
            # Case variations
            search_texts.extend([
                base_text.lower(),                # All lowercase
                base_text.upper(),                # All uppercase  
                base_text.title(),                # Title Case
                base_text.capitalize(),           # Capitalize first letter only
                base_text.swapcase(),            # Swap case of all letters
            ])
            
            # Medical/pharmaceutical term variations
            medical_variations = base_text
            medical_terms = {
                'concentration-dependent': ['concentration dependent', 'concentration_dependent', 
                                          'concentrationdependent', 'concentration-\ndependent'],
                'dose-dependent': ['dose dependent', 'dose_dependent', 'dosedependent', 'dose-\ndependent'],
                'dose-': ['dose ', 'dose_', 'dose\n'],
                'co-administration': ['coadministration', 'co administration', 'co_administration'],
                'anti-': ['anti ', 'anti_'],
                'pre-': ['pre ', 'pre_'],
                'post-': ['post ', 'post_'],
                'non-': ['non ', 'non_'],
                'sub-': ['sub ', 'sub_'],
                'inter-': ['inter ', 'inter_'],
                'intra-': ['intra ', 'intra_'],
                'multi-': ['multi ', 'multi_'],
                'micro-': ['micro ', 'micro_', 'μ'],
                'macro-': ['macro ', 'macro_'],
                're-': ['re ', 're_'],
                'de-': ['de ', 'de_'],
                'over-': ['over ', 'over_'],
                'under-': ['under ', 'under_'],
                'counter-': ['counter ', 'counter_'],
                'cross-': ['cross ', 'cross_'],
                'self-': ['self ', 'self_'],
                'well-': ['well ', 'well_'],
                'long-term': ['long term', 'long_term', 'longterm'],
                'short-term': ['short term', 'short_term', 'shortterm'],
                'real-time': ['real time', 'real_time', 'realtime'],
                'high-dose': ['high dose', 'high_dose', 'highdose'],
                'low-dose': ['low dose', 'low_dose', 'lowdose'],
                'single-dose': ['single dose', 'single_dose', 'singledose'],
                'double-blind': ['double blind', 'double_blind', 'doubleblind'],
                'placebo-controlled': ['placebo controlled', 'placebo_controlled', 'placebocontrolled'],
                'randomized-controlled': ['randomized controlled', 'randomized_controlled', 'randomizedcontrolled'],
            }
            
            for original, variations in medical_terms.items():
                if original.lower() in medical_variations.lower():
                    for variation in variations:
                        search_texts.append(medical_variations.replace(original, variation))
                        search_texts.append(medical_variations.replace(original.upper(), variation.upper()))
                        search_texts.append(medical_variations.replace(original.title(), variation.title()))
            
            # Number and unit variations
            number_variations = medical_variations
            # Handle different ways numbers might be formatted
            number_variations = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', number_variations)  # "5 - 10" -> "5-10"
            number_variations = re.sub(r'(\d+)\s*–\s*(\d+)', r'\1-\2', number_variations)  # "5 – 10" -> "5-10"
            number_variations = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', number_variations)  # "5 / 10" -> "5/10"
            search_texts.append(number_variations)
            
            # Units variations (mg, mcg, etc.)
            unit_variations = {
                'mg/kg': ['mg per kg', 'mg·kg⁻¹', 'mg/kg body weight', 'milligrams per kilogram'],
                'mcg': ['μg', 'micrograms', 'µg'],
                'mg': ['milligrams', 'milligram'],
                'g': ['grams', 'gram'],
                'kg': ['kilograms', 'kilogram'],
                'ml': ['mL', 'milliliters', 'milliliter'],
                'l': ['L', 'liters', 'liter'],
                '%': ['percent', 'per cent'],
                '°C': ['degrees Celsius', 'degrees C', 'deg C'],
                '°F': ['degrees Fahrenheit', 'degrees F', 'deg F'],
            }
            
            for unit, variations in unit_variations.items():
                if unit in medical_variations:
                    for variation in variations:
                        search_texts.append(medical_variations.replace(unit, variation))
            
            # Handle potential line breaks and hyphenation at ANY position
            words = base_text.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    # Create version with potential line break between any two words
                    line_break_version = ' '.join(words[:i+1]) + '-\n' + ' '.join(words[i+1:])
                    search_texts.append(line_break_version)
                    
                    # Create version with hyphen between any two words
                    hyphen_version = ' '.join(words[:i+1]) + '-' + ' '.join(words[i+1:])
                    search_texts.append(hyphen_version)
        
        # Remove duplicates while preserving order and filter empty strings
        unique_search_texts = []
        seen = set()
        for text in search_texts:
            text_cleaned = text.strip()
            if text_cleaned and text_cleaned not in seen and len(text_cleaned) >= 3:  # Must be at least 3 chars
                unique_search_texts.append(text_cleaned)
                seen.add(text_cleaned)
        
        search_texts = unique_search_texts
        print(f"Generated {len(search_texts)} search variations for: '{text_to_highlight[:50]}...'")
        
        for page_index in pages_to_search:
            page = doc[page_index]
            
            # Strategy 1: Try each search variation with exact matching - prioritize longer matches
            best_match = None
            best_match_length = 0
            
            for search_text in search_texts:
                if not search_text.strip():
                    continue
                    
                text_instances = page.search_for(search_text)
                
                if text_instances:
                    # Prioritize longer matches - they're more likely to be the exact text we want
                    if len(search_text) > best_match_length:
                        print(f"Found better match with variation: '{search_text[:50]}...' (length: {len(search_text)})")
                        best_match = (search_text, text_instances)
                        best_match_length = len(search_text)
            
            # If we found exact matches, use the best (longest) one
            if best_match:
                search_text, text_instances = best_match
                print(f"Using best exact match: '{search_text[:50]}...'")
                if text_instances:
                    rects_data = [
                        {
                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1,
                        }
                        for rect in text_instances
                    ]
                    min_x = min(r["x0"] for r in rects_data)
                    min_y = min(r["y0"] for r in rects_data)
                    max_x = max(r["x1"] for r in rects_data)
                    max_y = max(r["y1"] for r in rects_data)

                    coordinates.append({
                        "page": page_index,
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "text": search_text,
                        "original_text": text_to_highlight,
                        "match_type": "exact_variation_best",
                        "rectangles": rects_data,
                    })
                return coordinates  # Return immediately on finding best match
            
            # Strategy 2: Try advanced text reconstruction with smart segmentation
            coordinates = self._find_text_with_smart_segmentation(page, page_index, text_to_highlight)
            if coordinates:
                return coordinates
            
            # Strategy 3: Try progressive word matching - start with full text and work backwards
            words = text_to_highlight.split()
            if len(words) >= 4:  # Only if we have enough words
                # Try longer segments first to get more precise matches
                for word_count in range(len(words), max(len(words) // 2, 4), -1):  # Don't go below half the words
                    partial_text = ' '.join(words[:word_count])
                    
                    # Skip if this partial text is too generic (very short or common words)
                    if len(partial_text.strip()) < 30:  # Must be at least 30 characters
                        continue
                    
                    # Try the partial text with normalization
                    normalized_partial = self.normalize_pdf_text(partial_text)
                    
                    for partial_search in [partial_text, normalized_partial]:
                        if len(partial_search.strip()) < 30:  # Double check after normalization
                            continue
                            
                        text_instances = page.search_for(partial_search)
                        
                        if text_instances:
                            # Additional check: make sure this match is actually part of our target text
                            page_text = page.get_text().lower()
                            target_lower = text_to_highlight.lower()
                            
                            # Find the match in page context and see if it's followed by more of our target text
                            match_start = page_text.find(partial_search.lower())
                            if match_start != -1:
                                # Check if the next few words after this match are from our target text
                                context_after_match = page_text[match_start:match_start + len(target_lower)]
                                similarity = self._calculate_similarity(target_lower, context_after_match)
                                
                                # Only accept if similarity is high (likely the right match)
                                if similarity > 0.6:
                                    print(f"Found validated progressive match ({word_count} words, similarity: {similarity:.2f}): '{partial_search[:50]}...'")
                                    for rect in text_instances:
                                        coordinates.append({
                                            "page": page_index,
                                            "x": rect.x0,
                                            "y": rect.y0,
                                            "width": rect.width,
                                            "height": rect.height,
                                            "text": partial_search,
                                            "original_text": text_to_highlight,
                                            "match_type": f"progressive_validated_{word_count}_words",
                                            "similarity": similarity,
                                            "partial_match": True
                                        })
                                    return coordinates
                                else:
                                    print(f"Rejected progressive match due to low similarity ({similarity:.2f}): '{partial_search[:30]}...'")
                            else:
                                print(f"Found progressive match ({word_count} words): '{partial_search[:50]}...'")
                                for rect in text_instances:
                                    coordinates.append({
                                        "page": page_index,
                                        "x": rect.x0,
                                        "y": rect.y0,
                                        "width": rect.width,
                                        "height": rect.height,
                                        "text": partial_search,
                                        "original_text": text_to_highlight,
                                        "match_type": f"progressive_{word_count}_words",
                                        "partial_match": True
                                    })
                                return coordinates
            
            # Strategy 4: Try fuzzy matching by looking for text patterns in the page content
            coordinates = self._find_text_with_fuzzy_matching(page, page_index, text_to_highlight)
            if coordinates:
                return coordinates
            
            # Strategy 5: Multi-segment approach - try to find multiple parts and combine them
            coordinates = self._find_text_with_multi_segment_approach(page, page_index, text_to_highlight)
            if coordinates:
                return coordinates
            
            # Strategy 6: Look specifically for key ending phrases that might be formatted differently
            key_endings = ["concentration-dependent manner", "dose- and concentration-dependent manner", 
                          "concentration dependent manner", "dose and concentration dependent manner"]
            
            for ending in key_endings:
                if ending.lower() in text_to_highlight.lower():
                    ending_matches = page.search_for(ending)
                    if ending_matches:
                        print(f"Found key ending phrase: '{ending}'")
                        # Try to find text before this ending that matches our beginning
                        beginning_words = text_to_highlight.split()[:8]  # First 8 words
                        beginning_text = ' '.join(beginning_words)
                        
                        beginning_matches = page.search_for(beginning_text)
                        if beginning_matches:
                            # Find the closest beginning to our ending
                            for begin_rect in beginning_matches:
                                for end_rect in ending_matches:
                                    # Check if they're on the same page and reasonably close
                                    if (abs(begin_rect.y0 - end_rect.y0) < 100 and  # Within reasonable vertical distance
                                        begin_rect.x0 <= end_rect.x1):  # Beginning should be before or overlapping ending
                                        
                                        # Create combined bounding box
                                        combined_rect = fitz.Rect(
                                            min(begin_rect.x0, end_rect.x0),
                                            min(begin_rect.y0, end_rect.y0),
                                            max(begin_rect.x1, end_rect.x1),
                                            max(begin_rect.y1, end_rect.y1)
                                        )
                                        
                                        print(f"Found combined match: beginning + ending")
                                        return [{
                                            "page": page_index,
                                            "x": combined_rect.x0,
                                            "y": combined_rect.y0,
                                            "width": combined_rect.width,
                                            "height": combined_rect.height,
                                            "text": text_to_highlight,
                                            "original_text": text_to_highlight,
                                            "match_type": "beginning_plus_ending",
                                            "beginning_phrase": beginning_text,
                                            "ending_phrase": ending
                                        }]
                        
                        # If no beginning found, just highlight the ending we found
                        print(f"Using ending phrase as partial match: '{ending}'")
                        for rect in ending_matches:
                            coordinates.append({
                                "page": page_index,
                                "x": rect.x0,
                                "y": rect.y0,
                                "width": rect.width,
                                "height": rect.height,
                                "text": ending,
                                "original_text": text_to_highlight,
                                "match_type": "key_ending_phrase",
                                "partial_match": True
                            })
                        return coordinates
                
            # Strategy 7: More selective substring matching - only as last resort
            if len(text_to_highlight) >= 50:  # Only for longer texts
                min_length = max(30, len(text_to_highlight) // 2)  # Use larger minimum and take half the text
                
                # Try to find distinctive parts of the text (middle section often has unique content)
                middle_start = len(text_to_highlight) // 4
                substring = text_to_highlight[middle_start:middle_start + min_length]
                normalized_substring = self.normalize_pdf_text(substring)
                
                for search_sub in [substring, normalized_substring]:
                    if len(search_sub.strip()) < 20:  # Skip if too short after processing
                        continue
                        
                    text_instances = page.search_for(search_sub)
                    
                    if text_instances:
                        # Validate that this substring is likely part of our target text
                        page_text = page.get_text().lower()
                        target_lower = text_to_highlight.lower()
                        
                        # Find context around the match
                        match_pos = page_text.find(search_sub.lower())
                        if match_pos != -1:
                            # Get more context around the match
                            context_start = max(0, match_pos - 100)
                            context_end = min(len(page_text), match_pos + len(search_sub) + 100)
                            context = page_text[context_start:context_end]
                            
                            # Check if this context contains a good portion of our target text
                            similarity = self._calculate_similarity(target_lower, context)
                            
                            if similarity > 0.5:  # Higher threshold for substring matches
                                print(f"Found validated substring match (similarity: {similarity:.2f}): '{search_sub[:30]}...'")
                                for rect in text_instances:
                                    coordinates.append({
                                        "page": page_index,
                                        "x": rect.x0,
                                        "y": rect.y0,
                                        "width": rect.width,
                                        "height": rect.height,
                                        "text": search_sub,
                                        "original_text": text_to_highlight,
                                        "match_type": "validated_substring",
                                        "similarity": similarity,
                                        "partial_match": True
                                    })
                                return coordinates
                            else:
                                print(f"Rejected substring match due to low context similarity ({similarity:.2f}): '{search_sub[:30]}...'")
                        else:
                            # Fallback if we can't find the match in raw page text
                            print(f"Found substring match (no validation): '{search_sub[:30]}...'")
                            for rect in text_instances:
                                coordinates.append({
                                    "page": page_index,
                                    "x": rect.x0,
                                    "y": rect.y0,
                                    "width": rect.width,
                                    "height": rect.height,
                                    "text": search_sub,
                                    "original_text": text_to_highlight,
                                    "match_type": "substring_fallback",
                                    "partial_match": True
                                })
                            return coordinates
        
        return coordinates

    def _find_text_with_smart_segmentation(self, page: fitz.Page, page_index: int, text_to_highlight: str) -> List[Dict]:
        """
        Smart segmentation approach to find text that spans across line breaks with hyphens.
        """
        import re
        
        # Get all text with position information
        text_dict = page.get_text("dict")
        
        # Extract all text blocks and their positions
        all_text_segments = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            all_text_segments.append({
                                "text": text,
                                "bbox": bbox,
                                "x": bbox[0],
                                "y": bbox[1],
                                "width": bbox[2] - bbox[0],
                                "height": bbox[3] - bbox[1]
                            })
        
        # Try to reconstruct the text by combining segments
        for i in range(len(all_text_segments)):
            reconstructed_text = ""
            combined_bbox = None
            segments_used = []
            
            # Start from segment i and try to build the target text
            for j in range(i, min(i + 20, len(all_text_segments))):  # Look ahead up to 20 segments
                segment = all_text_segments[j]
                segment_text = segment["text"]
                
                # Handle hyphenated word at end of line
                if reconstructed_text.endswith('-') and not reconstructed_text.endswith('- '):
                    # This might be a hyphenated line break
                    reconstructed_text = reconstructed_text[:-1] + segment_text  # Remove hyphen and join
                else:
                    # Add space if needed
                    if reconstructed_text and not reconstructed_text.endswith(' ') and not segment_text.startswith(' '):
                        reconstructed_text += " "
                    reconstructed_text += segment_text
                
                segments_used.append(segment)
                
                # Update combined bounding box
                if combined_bbox is None:
                    combined_bbox = [segment["x"], segment["y"], 
                                   segment["x"] + segment["width"], 
                                   segment["y"] + segment["height"]]
                else:
                    combined_bbox[0] = min(combined_bbox[0], segment["x"])
                    combined_bbox[1] = min(combined_bbox[1], segment["y"])
                    combined_bbox[2] = max(combined_bbox[2], segment["x"] + segment["width"])
                    combined_bbox[3] = max(combined_bbox[3], segment["y"] + segment["height"])
                
                # Clean up the reconstructed text for comparison
                clean_reconstructed = re.sub(r'\s+', ' ', reconstructed_text.strip())
                clean_target = re.sub(r'\s+', ' ', text_to_highlight.strip())
                
                # Check if we have a match (with more flexible matching)
                if (clean_target.lower() in clean_reconstructed.lower() or 
                    self._calculate_similarity(clean_target.lower(), clean_reconstructed.lower()) > 0.8):
                    print(f"Found text using smart segmentation: '{clean_reconstructed[:100]}...'")
                    rectangles = [
                        {
                            "x0": seg["x"],
                            "y0": seg["y"],
                            "x1": seg["x"] + seg["width"],
                            "y1": seg["y"] + seg["height"],
                        }
                        for seg in segments_used
                    ]
                    return [{
                        "page": page_index,
                        "x": combined_bbox[0],
                        "y": combined_bbox[1],
                        "width": combined_bbox[2] - combined_bbox[0],
                        "height": combined_bbox[3] - combined_bbox[1],
                        "text": clean_reconstructed,
                        "original_text": text_to_highlight,
                        "match_type": "smart_segmentation",
                        "segments_count": len(segments_used),
                        "rectangles": rectangles,
                    }]
                
                # Also try partial matching - if we've built 70% of the target text
                if len(clean_reconstructed) >= len(clean_target) * 0.7:
                    similarity = self._calculate_similarity(clean_target.lower(), clean_reconstructed.lower())
                    if similarity > 0.7:
                        print(f"Found partial segmentation match (similarity: {similarity:.2f}): '{clean_reconstructed[:100]}...'")
                        rectangles = [
                            {
                                "x0": seg["x"],
                                "y0": seg["y"],
                                "x1": seg["x"] + seg["width"],
                                "y1": seg["y"] + seg["height"],
                            }
                            for seg in segments_used
                        ]
                        return [{
                            "page": page_index,
                            "x": combined_bbox[0],
                            "y": combined_bbox[1],
                            "width": combined_bbox[2] - combined_bbox[0],
                            "height": combined_bbox[3] - combined_bbox[1],
                            "text": clean_reconstructed,
                            "original_text": text_to_highlight,
                            "match_type": "partial_segmentation",
                            "segments_count": len(segments_used),
                            "similarity": similarity,
                            "rectangles": rectangles,
                        }]
                
                # Early exit if text is getting too long
                if len(clean_reconstructed) > len(clean_target) * 2:
                    break
        
        return []

    def _try_hyphen_aware_search(self, doc: fitz.Document, text_to_highlight: str, page_num: Optional[int] = None) -> List[Dict]:
        """
        Try to find text by breaking it at hyphens and searching for segments.
        This handles cases like 'gram-negative' where the PDF might have 'gram-' on one line and 'negative' on the next.
        """
        import re
        
        if '-' not in text_to_highlight:
            return []
        
        # Determine which pages to search
        if page_num is not None:
            pages_to_search = [page_num] if 0 <= page_num < len(doc) else []
        else:
            pages_to_search = range(len(doc))
        
        # Split text at hyphens to find problematic segments
        segments = text_to_highlight.split('-')
        if len(segments) < 2:
            return []
        
        print(f"Trying hyphen-aware search with segments: {segments}")
        
        for page_index in pages_to_search:
            page = doc[page_index]
            page_text = page.get_text()
            
            # Strategy 1: Look for first segment ending with hyphen
            first_segment = segments[0]
            second_segment = segments[1] if len(segments) > 1 else ""
            
            # Find instances where first segment appears with a hyphen at end of line
            hyphen_pattern = first_segment + r'-\s*\n\s*' + second_segment
            
            # Check if this pattern exists in the raw text
            if re.search(hyphen_pattern, page_text, re.IGNORECASE):
                print(f"Found hyphenated line break pattern: {first_segment}-{second_segment}")
                
                # Try to find the first segment with hyphen
                first_with_hyphen = page.search_for(first_segment + '-')
                second_alone = page.search_for(second_segment)
                
                if first_with_hyphen and second_alone:
                    # Find the closest second segment to each first segment
                    for first_rect in first_with_hyphen:
                        for second_rect in second_alone:
                            # Check if they're reasonably close (same or next line)
                            if abs(first_rect.y0 - second_rect.y0) < 30:  # Within 30 units vertically
                                print(f"Found matching segments: '{first_segment}-' and '{second_segment}'")
                                
                                # Create combined bounding box
                                combined_rect = fitz.Rect(
                                    min(first_rect.x0, second_rect.x0),
                                    min(first_rect.y0, second_rect.y0),
                                    max(first_rect.x1, second_rect.x1),
                                    max(first_rect.y1, second_rect.y1)
                                )
                                
                                return [{
                                    "page": page_index,
                                    "x": combined_rect.x0,
                                    "y": combined_rect.y0,
                                    "width": combined_rect.width,
                                    "height": combined_rect.height,
                                    "text": text_to_highlight,
                                    "original_text": text_to_highlight,
                                    "hyphen_aware_match": True,
                                    "segments": [first_segment + '-', second_segment],
                                    "rectangles": [
                                        {
                                            "x0": first_rect.x0,
                                            "y0": first_rect.y0,
                                            "x1": first_rect.x1,
                                            "y1": first_rect.y1,
                                        },
                                        {
                                            "x0": second_rect.x0,
                                            "y0": second_rect.y0,
                                            "x1": second_rect.x1,
                                            "y1": second_rect.y1,
                                        },
                                    ],
                                }]
            
            # Strategy 2: If complete pattern search fails, try progressive segment matching
            for i in range(len(segments) - 1):
                segment1 = segments[i]
                segment2 = segments[i + 1]
                
                # Search for segment1 ending with hyphen and segment2 starting new line
                seg1_matches = page.search_for(segment1)
                seg2_matches = page.search_for(segment2)
                
                if seg1_matches and seg2_matches:
                    for seg1_rect in seg1_matches:
                        for seg2_rect in seg2_matches:
                            # Check if seg2 is roughly below seg1 (next line)
                            if (seg2_rect.y0 > seg1_rect.y0 and 
                                seg2_rect.y0 - seg1_rect.y1 < 20 and  # Close vertically
                                abs(seg2_rect.x0 - seg1_rect.x0) < 50):  # Reasonable horizontal alignment
                                
                                print(f"Found line-break segments: '{segment1}' and '{segment2}'")
                                
                                # Create combined bounding box
                                combined_rect = fitz.Rect(
                                    min(seg1_rect.x0, seg2_rect.x0),
                                    min(seg1_rect.y0, seg2_rect.y0),
                                    max(seg1_rect.x1, seg2_rect.x1),
                                    max(seg1_rect.y1, seg2_rect.y1)
                                )
                                
                                return [{
                                    "page": page_index,
                                    "x": combined_rect.x0,
                                    "y": combined_rect.y0,
                                    "width": combined_rect.width,
                                    "height": combined_rect.height,
                                    "text": text_to_highlight,
                                    "original_text": text_to_highlight,
                                    "hyphen_line_break_match": True,
                                    "segments": [segment1, segment2]
                                }]
        
        return []

    def _find_text_with_fuzzy_matching(self, page: fitz.Page, page_index: int, text_to_highlight: str) -> List[Dict]:
        """
        Advanced fuzzy matching to find text that may be fragmented across different formatting elements.
        
        Args:
            page: PyMuPDF page object
            page_index: Page number
            text_to_highlight: Target text to find
            
        Returns:
            List of coordinate dictionaries for matches
        """
        # Get the raw page text for analysis
        page_text = page.get_text()
        
        # Normalize both texts for comparison
        normalized_target = self.normalize_pdf_text(text_to_highlight.lower())
        normalized_page = self.normalize_pdf_text(page_text.lower())
        
        # Strategy 1: Look for the normalized text in the normalized page content
        if normalized_target in normalized_page:
            print(f"Found normalized text match on page {page_index}")
            
            # Find the approximate position by looking for key words
            target_words = normalized_target.split()
            
            # Try to find a sequence of these words in the page
            if len(target_words) >= 3:
                # Look for the first 3 words and last 3 words
                start_phrase = ' '.join(target_words[:3])
                end_phrase = ' '.join(target_words[-3:]) if len(target_words) > 3 else start_phrase
                
                start_matches = page.search_for(start_phrase)
                if not start_matches:
                    # Try each word individually
                    for word in target_words[:3]:
                        if len(word) > 3:  # Skip very short words
                            start_matches = page.search_for(word)
                            if start_matches:
                                break
                
                end_matches = page.search_for(end_phrase)
                if not end_matches:
                    # Try each word individually
                    for word in reversed(target_words[-3:]):
                        if len(word) > 3:  # Skip very short words
                            end_matches = page.search_for(word)
                            if end_matches:
                                break
                
                # If we found both start and end, create a bounding box
                if start_matches and end_matches:
                    start_rect = start_matches[0]
                    end_rect = end_matches[-1] if len(end_matches) > 1 else end_matches[0]
                    
                    # Create combined bounding box
                    combined_rect = fitz.Rect(
                        min(start_rect.x0, end_rect.x0),
                        min(start_rect.y0, end_rect.y0),
                        max(start_rect.x1, end_rect.x1),
                        max(start_rect.y1, end_rect.y1)
                    )
                    
                    print(f"Found fuzzy match using start/end phrases: '{start_phrase}' ... '{end_phrase}'")
                    return [{
                        "page": page_index,
                        "x": combined_rect.x0,
                        "y": combined_rect.y0,
                        "width": combined_rect.width,
                        "height": combined_rect.height,
                        "text": text_to_highlight,
                        "original_text": text_to_highlight,
                        "match_type": "fuzzy_start_end",
                        "start_phrase": start_phrase,
                        "end_phrase": end_phrase
                    }]
        
        # Strategy 2: Character-by-character fuzzy matching with tolerance
        # This handles cases where there are extra spaces, line breaks, etc.
        return self._find_text_with_character_tolerance(page, page_index, text_to_highlight)
    
    def _find_text_with_character_tolerance(self, page: fitz.Page, page_index: int, text_to_highlight: str) -> List[Dict]:
        """
        Find text with character-level tolerance for formatting differences.
        
        Args:
            page: PyMuPDF page object
            page_index: Page number
            text_to_highlight: Target text to find
            
        Returns:
            List of coordinate dictionaries for matches
        """
        # Get text with detailed positioning
        text_dict = page.get_text("dict")
        
        # Extract all characters with their positions
        all_chars = []
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        # Add each character with its approximate position
                        char_width = (bbox[2] - bbox[0]) / max(len(text), 1)
                        for i, char in enumerate(text):
                            char_x = bbox[0] + (i * char_width)
                            all_chars.append({
                                "char": char,
                                "x": char_x,
                                "y": bbox[1],
                                "width": char_width,
                                "height": bbox[3] - bbox[1]
                            })
        
        # Normalize the target text
        normalized_target = re.sub(r'\s+', ' ', text_to_highlight.lower().strip())
        page_chars = ''.join([c["char"] for c in all_chars]).lower()
        normalized_page_chars = re.sub(r'\s+', ' ', page_chars.strip())
        
        # Try to find the target in the page characters
        target_start = normalized_page_chars.find(normalized_target)
        
        if target_start != -1:
            target_end = target_start + len(normalized_target)
            
            # Map back to original character positions
            original_start = self._map_normalized_to_original_position(page_chars, normalized_page_chars, target_start)
            original_end = self._map_normalized_to_original_position(page_chars, normalized_page_chars, target_end)
            
            if original_start != -1 and original_end != -1 and original_start < len(all_chars) and original_end <= len(all_chars):
                # Get bounding box for the matched characters
                start_char = all_chars[original_start]
                end_char = all_chars[min(original_end - 1, len(all_chars) - 1)]
                
                combined_rect = fitz.Rect(
                    start_char["x"],
                    min(start_char["y"], end_char["y"]),
                    end_char["x"] + end_char["width"],
                    max(start_char["y"] + start_char["height"], end_char["y"] + end_char["height"])
                )
                
                print(f"Found character-tolerance match: '{normalized_target[:50]}...'")
                return [{
                    "page": page_index,
                    "x": combined_rect.x0,
                    "y": combined_rect.y0,
                    "width": combined_rect.width,
                    "height": combined_rect.height,
                    "text": text_to_highlight,
                    "original_text": text_to_highlight,
                    "match_type": "character_tolerance",
                    "matched_length": original_end - original_start
                }]
        
        return []
    
    def _map_normalized_to_original_position(self, original_text: str, normalized_text: str, normalized_pos: int) -> int:
        """
        Map a position in normalized text back to the original text.
        
        Args:
            original_text: Original text with all characters
            normalized_text: Normalized text (spaces collapsed, etc.)
            normalized_pos: Position in the normalized text
            
        Returns:
            Corresponding position in original text, or -1 if not found
        """
        if normalized_pos >= len(normalized_text):
            return len(original_text)
        
        orig_pos = 0
        norm_pos = 0
        
        while orig_pos < len(original_text) and norm_pos < normalized_pos:
            orig_char = original_text[orig_pos].lower()
            
            # Skip multiple whitespace in original, count as single space in normalized
            if orig_char.isspace():
                # Skip consecutive whitespace
                while orig_pos < len(original_text) and original_text[orig_pos].isspace():
                    orig_pos += 1
                # This counts as one space in normalized
                if norm_pos < len(normalized_text) and normalized_text[norm_pos] == ' ':
                    norm_pos += 1
            else:
                # Regular character
                if norm_pos < len(normalized_text) and normalized_text[norm_pos] == orig_char:
                    norm_pos += 1
                orig_pos += 1
        
        return orig_pos

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using character-level comparison.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Use simple character overlap ratio
        text1_chars = set(text1.lower().replace(' ', ''))
        text2_chars = set(text2.lower().replace(' ', ''))
        
        if not text1_chars or not text2_chars:
            return 0.0
        
        intersection = len(text1_chars.intersection(text2_chars))
        union = len(text1_chars.union(text2_chars))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also check word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_similarity = word_intersection / word_union if word_union > 0 else 0.0
        
        # Combine both metrics
        return (jaccard_similarity * 0.3 + word_similarity * 0.7)

    def highlight_text_in_pdf(
        self, 
        input_pdf_path: str, 
        texts_to_highlight: List[Dict],
        user_id: str,
        output_filename: Optional[str] = None,
        auto_cleanup: bool = True,
        cleanup_delay: Optional[int] = None
    ) -> Dict:
        """
        Highlight specified texts in a PDF and return the highlighted PDF path.
        
        Args:
            input_pdf_path: Path to input PDF file (local, URL, or MinIO object path)
            texts_to_highlight: List of dictionaries with text and highlighting info
                Format: [{"text": "text to highlight", "type": "drug_info", "page": 1}]
            user_id: User ID for file naming
            output_filename: Optional custom filename for output
            auto_cleanup: Whether to automatically delete the file after delay
            cleanup_delay: Custom cleanup delay in seconds
            
        Returns:
            Dictionary with result information
        """
        temp_file = None
        try:
            # Get PDF document from various sources
            doc, original_filename, temp_file = self._get_pdf_document(input_pdf_path)
            print(f"Processing file: {original_filename}")
            # Generate output filename
            if not output_filename:
                base_name = os.path.splitext(original_filename)[0]
                unique_id = str(uuid.uuid4())[:8]
                output_filename = f"{base_name}_highlighted_{user_id}_{unique_id}.pdf"
            
            output_path = os.path.join(self.upload_directory, output_filename)
            
            highlighted_regions = []
            total_highlights = 0
            
            # Helper: split long text into sentence-like chunks for tighter highlights
            def _split_into_chunks(value: str) -> List[str]:
                if not value:
                    return []
                # Remove the leading 'Original:' if present
                cleaned = re.sub(r'^\s*Original:\s*', '', value, flags=re.IGNORECASE)
                # Normalize spaces
                cleaned = re.sub(r'[\r\n]+', ' ', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                # If very short, return as is
                if len(cleaned) < 180:
                    return [cleaned]

                # Split on sentence boundaries, including cases like '...].[15] Next'
                parts = re.split(r'(?:(?<=\])|(?<=[\.!?]))\s+(?=[A-Z\[])', cleaned)
                # Fallback: if splitting failed, split on period followed by capital
                if len(parts) <= 1:
                    parts = re.split(r'\.(?=\s+[A-Z])', cleaned)

                # Trim and keep reasonable-length parts
                chunks: List[str] = []
                for p in parts:
                    t = p.strip()
                    if len(t) < 20:
                        continue
                    # Avoid overly long parts; further split by semicolon if needed
                    if len(t) > 320:
                        subparts = [sp.strip() for sp in re.split(r';\s+', t) if len(sp.strip()) >= 20]
                        chunks.extend(subparts or [t[:320]])
                    else:
                        chunks.append(t)
                # De-duplicate while preserving order
                seen = set()
                uniq = []
                for c in chunks:
                    key = c.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(c)
                return uniq or [cleaned]

            # Process each text to highlight
            for highlight_info in texts_to_highlight:
                text = highlight_info.get("text", "")
                highlight_type = highlight_info.get("type", "default")
                specific_page = highlight_info.get("page")
                
                if not text.strip():
                    continue
                
                # Decide whether to split into smaller chunks first for precision
                # Conditions: very long strings, multiple bracket refs, or explicit "Original:" prefix
                split_first = (
                    len(text) > 280
                    or len(re.findall(r"\[\d+\]", text)) >= 2
                    or text.strip().lower().startswith("original:")
                )

                coordinates: List[Dict] = []
                if split_first:
                    for segment in _split_into_chunks(text):
                        seg_coords = self.find_text_coordinates(doc, segment, specific_page)
                        if not seg_coords:
                            seg_coords = self.find_text_with_partial_match(doc, segment, specific_page)
                        if seg_coords:
                            coordinates.extend(seg_coords)
                else:
                    # Find coordinates for this text as-is
                    coordinates = self.find_text_coordinates(doc, text, specific_page)
                
                # If not found, try preprocessing the text for PDF formatting issues
                if not coordinates:
                    print(f"Exact match failed for text starting with: '{text[:50]}...'")
                    print(f"Trying advanced text matching...")
                    # Try as-is, and if still too broad later, we will split
                    coordinates = self.find_text_with_partial_match(doc, text, specific_page)

                # If match area looks too large, refine by splitting
                def _looks_too_broad(coord_list: List[Dict]) -> bool:
                    try:
                        if not coord_list:
                            return False
                        # If any coordinate lacks rectangles and has a very tall region, it's likely broad
                        for c in coord_list:
                            rects = c.get('rectangles', [])
                            h = float(c.get('height', 0) or 0)
                            w = float(c.get('width', 0) or 0)
                            if (not rects and (h > 80 or w > 1000)):
                                return True
                            # If one rectangle spans across many lines, treat as broad
                            if rects and len(rects) == 1:
                                r = rects[0]
                                rh = float((getattr(r, 'height', None) or (r.get('y1', 0) - r.get('y0', 0))))
                                if rh and rh > 80:
                                    return True
                        return False
                    except Exception:
                        return False

                if coordinates and _looks_too_broad(coordinates):
                    refined: List[Dict] = []
                    for segment in _split_into_chunks(text):
                        seg_coords = self.find_text_coordinates(doc, segment, specific_page)
                        if not seg_coords:
                            seg_coords = self.find_text_with_partial_match(doc, segment, specific_page)
                        if seg_coords:
                            refined.extend(seg_coords)
                    if refined:
                        print("Refined broad match into sentence-level highlights")
                        coordinates = refined
                
                # If still not found, try breaking text at natural points for hyphenated line breaks
                if not coordinates and '-' in text:
                    print(f"Trying hyphen-aware segmented search...")
                    coordinates = self._try_hyphen_aware_search(doc, text, specific_page)
                
                if not coordinates:
                    print(f"Warning: Text not found in PDF even with all variations")
                    print(f"  - Original text: '{text[:100]}...'")
                    print(f"  - Searching on page: {specific_page if specific_page is not None else 'all pages'}")
                    
                    # Show what's actually on the specified page for debugging
                    if specific_page is not None and 0 <= specific_page < len(doc):
                        page = doc[specific_page]
                        page_text = page.get_text()
                        print(f"  - Page {specific_page} has {len(page_text)} characters")
                        print(f"  - First 200 chars: '{page_text[:200]}...'")
                        
                        # Check if any part of the text appears on the page
                        text_words = text.split()[:5]  # First 5 words
                        for word in text_words:
                            if word.lower() in page_text.lower():
                                print(f"  - Found word '{word}' on page")
                                break
                    continue
                
                # Get highlighting color
                color = self.highlight_colors.get(highlight_type, self.highlight_colors["default"])
                
                # Apply highlights
                for coord in coordinates:
                    page = doc[coord["page"]]
                    rectangles = coord.get("rectangles", [])
                    
                    def _build_rect(rect_data):
                        try:
                            if isinstance(rect_data, fitz.Rect):
                                rect = fitz.Rect(rect_data)
                            elif isinstance(rect_data, fitz.Quad):
                                rect = fitz.Rect(rect_data.rect)
                            elif isinstance(rect_data, (list, tuple)) and len(rect_data) == 4:
                                x0, y0, x1, y1 = map(float, rect_data)
                                rect = fitz.Rect(x0, y0, x1, y1)
                            elif isinstance(rect_data, dict):
                                if all(k in rect_data for k in ("x0", "y0", "x1", "y1")):
                                    x0 = float(rect_data["x0"])
                                    y0 = float(rect_data["y0"])
                                    x1 = float(rect_data["x1"])
                                    y1 = float(rect_data["y1"])
                                elif all(k in rect_data for k in ("x", "y", "width", "height")):
                                    x0 = float(rect_data["x"])
                                    y0 = float(rect_data["y"])
                                    x1 = x0 + float(rect_data["width"])
                                    y1 = y0 + float(rect_data["height"])
                                else:
                                    return None
                                rect = fitz.Rect(x0, y0, x1, y1)
                            else:
                                return None
                        except (TypeError, ValueError, KeyError):
                            return None
                        
                        if rect.width <= 0 or rect.height <= 0:
                            return None
                        return rect
                    
                    highlight = None
                    if rectangles:
                        quads = []
                        for rect_data in rectangles:
                            try:
                                rect = _build_rect(rect_data)
                                if rect is None:
                                    raise ValueError(f"Invalid rect data: {rect_data}")
                                quads.append(fitz.Quad(rect))
                            except Exception as quad_error:
                                print(f"Error building quad for highlight: {quad_error}")
                        if quads:
                            highlight = page.add_highlight_annot(quads)
                    
                    if highlight is None:
                        rect = _build_rect(
                            {
                                "x": coord.get("x"),
                                "y": coord.get("y"),
                                "width": coord.get("width"),
                                "height": coord.get("height"),
                            }
                        )
                        if rect is None:
                            print(f"Skipping highlight due to invalid coordinates: {coord}")
                            continue
                        highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=color)
                    
                    # Add citation info if provided
                    if highlight_info.get("citation"):
                        highlight.set_info(content=highlight_info["citation"])
                    
                    highlight.update()
                    total_highlights += 1
                    
                    # Store highlight info for response
                    highlighted_regions.append({
                        "text": text,
                        "page": coord["page"],
                        "type": highlight_type,
                        "coordinates": coord
                    })
            
            # Save highlighted PDF locally
            doc.save(output_path)
            doc.close()
            
            # Upload to MinIO for persistent storage
            minio_object_name = f"highlighted/{user_id}/{output_filename}"
            minio_url = self._upload_to_minio(output_path, minio_object_name)
            
            # Clean up temporary file if used
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Schedule automatic cleanup if requested
            if auto_cleanup:
                self._schedule_file_cleanup(output_path, minio_object_name, cleanup_delay)
            
            return {
                "success": True,
                "highlighted_pdf_path": output_path,
                "local_file_path": output_path,  # Added for direct file return API
                "highlighted_pdf_url": minio_object_name,
                "minio_url": minio_url,  # Added MinIO URL
                "minio_object_name": minio_object_name,  # Added object name for reference
                "original_filename": original_filename,
                "output_filename": output_filename,
                "total_highlights": total_highlights,
                "highlighted_regions": highlighted_regions,
                "user_id": user_id,
                "auto_cleanup": auto_cleanup,
                "cleanup_delay": cleanup_delay or self.cleanup_delay,
                "source_type": "minio" if self._is_minio_path(input_pdf_path) else "url" if self._is_url(input_pdf_path) else "local",
                "storage": "local_and_minio" if minio_url else "local_only"
            }
            
        except Exception as e:
            # Clean up on error
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    def highlight_passages_from_coordinates(
        self, 
        input_pdf_path: str, 
        highlight_coordinates: List[Dict],
        user_id: str,
        output_filename: Optional[str] = None,
        auto_cleanup: bool = True,
        cleanup_delay: Optional[int] = None
    ) -> Dict:
        """
        Highlight PDF using pre-calculated coordinates.
        
        Args:
            input_pdf_path: Path to input PDF file (local, URL, or MinIO object path)
            highlight_coordinates: List of coordinate dictionaries
            user_id: User ID for file naming
            output_filename: Optional custom filename
            auto_cleanup: Whether to automatically delete the file after delay
            cleanup_delay: Custom cleanup delay in seconds
            
        Returns:
            Dictionary with result information
        """
        temp_file = None
        try:
            # Get PDF document from various sources
            doc, original_filename, temp_file = self._get_pdf_document(input_pdf_path)
            
            # Generate output filename
            if not output_filename:
                base_name = os.path.splitext(original_filename)[0]
                unique_id = str(uuid.uuid4())[:8]
                output_filename = f"{base_name}_highlighted_{user_id}_{unique_id}.pdf"
            
            output_path = os.path.join(self.upload_directory, output_filename)
            
            total_highlights = 0
            
            # Apply highlights from coordinates
            for coord_info in highlight_coordinates:
                page_num = coord_info.get("page", 0)
                if page_num >= len(doc):
                    continue
                
                page = doc[page_num]
                highlight_type = coord_info.get("type", "default")
                color = self.highlight_colors.get(highlight_type, self.highlight_colors["default"])
                
                # Create rectangle from coordinates
                rect = fitz.Rect(
                    coord_info.get("x", 0),
                    coord_info.get("y", 0),
                    coord_info.get("x", 0) + coord_info.get("width", 0),
                    coord_info.get("y", 0) + coord_info.get("height", 0)
                )
                
                # Add highlight
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=color)
                
                if coord_info.get("citation"):
                    highlight.set_info(content=coord_info["citation"])
                
                highlight.update()
                total_highlights += 1
            
            # Save highlighted PDF locally
            doc.save(output_path)
            doc.close()
            
            # Upload to MinIO for persistent storage
            minio_object_name = f"highlighted/{user_id}/{output_filename}"
            minio_url = self._upload_to_minio(output_path, minio_object_name)
            
            # Clean up temporary file
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Schedule automatic cleanup if requested
            if auto_cleanup:
                self._schedule_file_cleanup(output_path, minio_object_name, cleanup_delay)
            
            return {
                "success": True,
                "highlighted_pdf_path": output_path,
                "local_file_path": output_path,  # Added for direct file return API
                "highlighted_pdf_url": f"/downloads/highlighted/{output_filename}",
                "minio_url": minio_url,  # Added MinIO URL
                "minio_object_name": minio_object_name,  # Added object name for reference
                "original_filename": original_filename,
                "output_filename": output_filename,
                "total_highlights": total_highlights,
                "user_id": user_id,
                "auto_cleanup": auto_cleanup,
                "cleanup_delay": cleanup_delay or self.cleanup_delay,
                "source_type": "minio" if self._is_minio_path(input_pdf_path) else "url" if self._is_url(input_pdf_path) else "local",
                "storage": "local_and_minio" if minio_url else "local_only"
            }
            
        except Exception as e:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    def cleanup_all_files(self):
        """Cleanup all scheduled timers and temporary files."""
        for cleanup_key, timer in self.cleanup_timers.items():
            timer.cancel()
            
            # Parse cleanup key to get file path and minio object name
            if "|" in cleanup_key:
                file_path, minio_object_name = cleanup_key.split("|", 1)
                
                # Clean up local file
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        print(f"Cleaned up local file: {file_path}")
                except Exception as e:
                    print(f"Error cleaning up local file {file_path}: {e}")
                
                # Clean up MinIO file
                if minio_object_name and minio_object_name != "None":
                    self._delete_from_minio(minio_object_name)
                    
            else:
                # Legacy cleanup key format (file path only)
                try:
                    if os.path.exists(cleanup_key):
                        os.unlink(cleanup_key)
                        print(f"Cleaned up: {cleanup_key}")
                except Exception as e:
                    print(f"Error cleaning up {cleanup_key}: {e}")
        
        self.cleanup_timers.clear()
        print("All cleanup timers cancelled and files removed from local and MinIO storage")
    
    def _find_text_with_multi_segment_approach(self, page: fitz.Page, page_index: int, text_to_highlight: str) -> List[Dict]:
        """
        Multi-segment approach to find text by breaking it into smaller chunks and finding each chunk,
        then combining them into a single highlight region.
        
        Args:
            page: PyMuPDF page object
            page_index: Page number
            text_to_highlight: Target text to find
            
        Returns:
            List of coordinate dictionaries for matches
        """
        # Break text into meaningful chunks (sentences, phrases)
        text_chunks = []
        
        # Strategy 1: Split by punctuation but keep meaningful phrases
        import re
        
        # First, try splitting by sentences
        sentence_parts = re.split(r'[.!?;]', text_to_highlight)
        if len(sentence_parts) > 1:
            for part in sentence_parts:
                part = part.strip()
                if len(part) > 10:  # Only meaningful chunks
                    text_chunks.append(part)
        
        # Strategy 2: Split by commas for longer phrases
        if not text_chunks:
            comma_parts = text_to_highlight.split(',')
            if len(comma_parts) > 1:
                for part in comma_parts:
                    part = part.strip()
                    if len(part) > 10:  # Only meaningful chunks
                        text_chunks.append(part)
        
        # Strategy 3: Split into word groups if no punctuation
        if not text_chunks:
            words = text_to_highlight.split()
            chunk_size = max(4, len(words) // 4)  # Create 4 chunks or minimum 4 words per chunk
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                text_chunks.append(chunk)
        
        print(f"Multi-segment approach with {len(text_chunks)} chunks: {[c[:30] + '...' for c in text_chunks]}")
        
        # Find each chunk on the page
        found_chunks = []
        for chunk in text_chunks:
            if len(chunk.strip()) < 5:
                continue
                
            # Try multiple variations of the chunk
            chunk_variations = [
                chunk,
                self.normalize_pdf_text(chunk),
                chunk.replace(',', ''),
                chunk.replace('.', ''),
                re.sub(r'[()[\]{}]', '', chunk),
                re.sub(r'\s+', ' ', chunk.strip())
            ]
            
            chunk_found = False
            for variation in chunk_variations:
                if not variation.strip():
                    continue
                    
                text_instances = page.search_for(variation)
                if text_instances:
                    print(f"Found chunk: '{variation[:40]}...'")
                    found_chunks.append({
                        "original_chunk": chunk,
                        "found_variation": variation,
                        "rectangles": text_instances,
                        "rect": text_instances[0]  # Use first match
                    })
                    chunk_found = True
                    break
            
            # If chunk not found, try finding individual important words
            if not chunk_found:
                important_words = [w for w in chunk.split() if len(w) > 3]  # Words longer than 3 chars
                for word in important_words[:3]:  # Try first 3 important words
                    word_instances = page.search_for(word)
                    if word_instances:
                        print(f"Found important word from chunk: '{word}'")
                        found_chunks.append({
                            "original_chunk": chunk,
                            "found_variation": word,
                            "rectangles": word_instances,
                            "rect": word_instances[0],
                            "partial_word_match": True
                        })
                        break
        
        # If we found at least 60% of the chunks, create a combined highlight
        if len(found_chunks) >= max(2, len(text_chunks) * 0.6):
            print(f"Found {len(found_chunks)}/{len(text_chunks)} chunks, creating combined highlight")
            
            # Calculate bounding box that encompasses all found chunks
            all_rects = []
            for chunk_info in found_chunks:
                all_rects.extend(chunk_info["rectangles"])
            
            if all_rects:
                # Create combined bounding box
                min_x = min(rect.x0 for rect in all_rects)
                min_y = min(rect.y0 for rect in all_rects)
                max_x = max(rect.x1 for rect in all_rects)
                max_y = max(rect.y1 for rect in all_rects)
                
                # Expand the box slightly to ensure we capture everything in between
                padding = 5  # pixels
                combined_rect = fitz.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
                rectangles = [
                    {
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                    }
                    for rect in all_rects
                ]
                
                return [{
                    "page": page_index,
                    "x": combined_rect.x0,
                    "y": combined_rect.y0,
                    "width": combined_rect.width,
                    "height": combined_rect.height,
                    "text": text_to_highlight,
                    "original_text": text_to_highlight,
                    "match_type": "multi_segment",
                    "chunks_found": len(found_chunks),
                    "total_chunks": len(text_chunks),
                    "found_chunks": [c["found_variation"][:30] + "..." for c in found_chunks],
                    "rectangles": rectangles,
                }]
        else:
            print(f"Only found {len(found_chunks)}/{len(text_chunks)} chunks, not enough for multi-segment match")
        
        return []
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using character-level comparison.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Use simple character overlap ratio
        text1_chars = set(text1.lower().replace(' ', ''))
        text2_chars = set(text2.lower().replace(' ', ''))
        
        if not text1_chars or not text2_chars:
            return 0.0
        
        intersection = len(text1_chars.intersection(text2_chars))
        union = len(text1_chars.union(text2_chars))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also check word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_similarity = word_intersection / word_union if word_union > 0 else 0.0
        
        # Combine both metrics
        return (jaccard_similarity * 0.3 + word_similarity * 0.7)
