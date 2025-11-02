"""
Professional Healthcare Document Chunking System
Fixed version with proper page-aware chunking
"""
import re
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from logging_config import get_logger

logger = get_logger(__name__)

# Optional dependencies with graceful fallback
try:
    from nltk.tokenize import sent_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
    except LookupError:
        NLTK_AVAILABLE = False
        sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 500          # Decreased from 2000 for more granular chunks
    chunk_overlap: int = 100         # Decreased from 400 (20% overlap)
    min_chunk_size: int = 200        # Decreased from 800
    max_chunk_size: int = 750       # Decreased from 2500
    overlap_sentences: int = 4       # Increased from 3
    preserve_sections: bool = True
    
    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be less than chunk_size")


class TextNormalizer:
    """Handles text normalization"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize whitespace while preserving structure"""
        if not text:
            return ""
        normalized = text.replace('\r\n', '\n')
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        return normalized.strip()
    
    @staticmethod
    def normalize_for_metadata(text: str) -> str:
        """Normalize text for metadata"""
        if not text:
            return ""
        cleaned = re.sub(r"-\s*\n", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()


class SectionExtractor:
    """Extracts document sections using structural patterns"""
    
    NUMBERED_SECTION = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*(.*)$")
    
    @classmethod
    def extract_sections(cls, text: str) -> List[Dict]:
        """Extract document sections"""
        sections = []
        current_section = {
            "title": "DOCUMENT_START",
            "content": "",
            "start_line": 0
        }
        
        lines = text.split('\n')
        
        for idx, line in enumerate(lines):
            stripped = line.strip()
            
            if cls._is_section_header(stripped, lines, idx):
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                current_section = {
                    "title": stripped,
                    "content": "",
                    "start_line": idx + 1
                }
            else:
                current_section["content"] += (stripped + "\n") if stripped else "\n"
        
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    @classmethod
    def _is_section_header(cls, line: str, lines: List[str], index: int) -> bool:
        """Detect if line is a section header"""
        if not line or len(line) > 150:
            return False
        
        # Pattern 1: Numbered sections
        if cls.NUMBERED_SECTION.match(line):
            return True
        
        # Pattern 2: ALL CAPS headers (>= 10 chars)
        if len(line) >= 10:
            alpha_chars = [c for c in line if c.isalpha()]
            if alpha_chars:
                uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if uppercase_ratio >= 0.7:
                    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
                    if not next_line or not next_line[0].islower():
                        return True
        
        # Pattern 3: Title Case (3+ words, 60%+ capitalized)
        words = line.split()
        if len(words) >= 3:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized / len(words) >= 0.6:
                next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
                if not next_line or len(next_line) < 100:
                    return True
        
        # Pattern 4: Bullet points
        if re.match(r"^\s*[•\-\*]\s+[A-Z]", line):
            return True
        
        return False
    
    @staticmethod
    def extract_section_number(section_title: str) -> str:
        """Extract section number from title"""
        if not section_title:
            return ""
        match = re.match(r"^\s*(\d+(?:\.\d+)*)", section_title)
        return match.group(1) if match else ""


class ContentAnalyzer:
    """Analyzes chunk content"""
    
    STOP_WORDS = {
        'with', 'from', 'this', 'that', 'have', 'been', 'their', 'there', 'which',
        'should', 'would', 'could', 'will', 'into', 'about', 'after', 'before',
        'during', 'without', 'between', 'includes', 'including', 'because',
        'where', 'when', 'those', 'these', 'such', 'other', 'also', 'more', 'most'
    }
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 8) -> List[str]:
        """Extract keywords"""
        if not text:
            return []
        
        tokens = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        filtered = [t for t in tokens if t not in ContentAnalyzer.STOP_WORDS]
        
        counts = {}
        for token in filtered:
            counts[token] = counts.get(token, 0) + 1
        
        sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [token for token, _ in sorted_tokens[:top_n]]
    
    @staticmethod
    def has_numeric_data(text: str) -> bool:
        """Check for quantitative data"""
        patterns = [
            r'\b\d+\.?\d*\s*(?:mg|mcg|µg|g|kg|ml|L|mL|%|percent|mmol|IU|units?)\b',
            r'\b\d+\.?\d*\s*(?:times|fold|×|x)\b',
            r'\bp\s*[<>=]\s*0\.\d+\b',
            r'\b\d+\.?\d*\s*[:/]\s*\d+\.?\d*\b',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    @staticmethod
    def has_medical_patterns(text: str) -> bool:
        """Detect medical terminology"""
        patterns = [
            r'\b\w+(?:itis|osis|emia|pathy|trophy|plasia|oma|algia|dynia)\b',
            r'\b(?:cardio|hepato|nephro|neuro|gastro|pulmon|derm|osteo)\w+\b',
            r'\b(?:mg|mcg|g)/(?:kg|m2|day|dose)\b',
            r'\b(?:patient|treatment|therapy|diagnosis|symptom|adverse|effect)\b',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    @staticmethod
    def calculate_text_density(text: str) -> float:
        """Calculate information density"""
        if not text:
            return 0.0
        indicators = len(re.findall(r'\d+', text))
        indicators += sum(1 for c in text if c.isupper())
        indicators += text.count(',') + text.count(':')
        indicators += len([w for w in text.split() if len(w) > 8])
        return min(indicators / max(len(text), 1) * 100, 1.0)


class UniversalHealthcareChunker:
    """Professional healthcare document chunker"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.normalizer = TextNormalizer()
        self.section_extractor = SectionExtractor()
        self.content_analyzer = ContentAnalyzer()
        logger.info(f"Initialized chunker: size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
    
    def healthcare_semantic_chunking(self, text: str, page_number: int = 1) -> List[Dict]:
        """
        Main chunking method - handles text from single page
        
        Args:
            text: Text to chunk (should be from ONE page)
            page_number: The page number this text belongs to
            
        Returns:
            List of chunks with correct page numbers
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return []
        
        print(f"[CHUNKER] Page {page_number}: Starting chunking ({len(text)} chars)")
        
        normalized_text = self.normalizer.normalize(text)
        
        if self.config.preserve_sections and len(normalized_text) > self.config.chunk_size:
            sections = self.section_extractor.extract_sections(normalized_text)
            print(f"[CHUNKER] Page {page_number}: Found {len(sections)} sections")
            
            # If only 1 section found and it's too large, use simple chunking instead
            if len(sections) == 1 and len(sections[0]['content']) > self.config.max_chunk_size:
                print(f"[CHUNKER] Page {page_number}: Single large section detected, using simple chunking")
                chunks = self._chunk_simple(normalized_text, page_number)
            else:
                chunks = []
                for section_idx, section in enumerate(sections):
                    section_chunks = self._chunk_section(
                        section['content'],
                        section['title'],
                        section_idx,
                        page_number  # Pass actual page number
                    )
                    print(f"[CHUNKER] Page {page_number}, Section '{section['title'][:30]}...': {len(section_chunks)} chunks from {len(section['content'])} chars")
                    chunks.extend(section_chunks)
        else:
            chunks = self._chunk_simple(normalized_text, page_number)
        
        # Reindex chunks
        for idx, chunk in enumerate(chunks):
            chunk['chunk_index'] = idx
        
        print(f"[CHUNKER] Page {page_number}: ✅ Created {len(chunks)} final chunks")
        return chunks
    
    def _chunk_section(
        self,
        text: str,
        section_title: str,
        section_idx: int,
        page_number: int
    ) -> List[Dict]:
        """Chunk a section with overlap"""
        chunks = []
        
        try:
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else re.split(r'(?<=[.!?])\s+', text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        
        current_chunk_sentences = []
        current_char_count = 0
        chunk_start_idx = 0
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if we should create a chunk
            should_chunk = (
                current_chunk_sentences and
                current_char_count + sentence_length > self.config.chunk_size and
                current_char_count >= self.config.min_chunk_size
            ) or current_char_count >= self.config.max_chunk_size
            
            if should_chunk:
                chunk_text = " ".join(current_chunk_sentences)
                chunk = self._create_chunk(
                    text=chunk_text,
                    section_title=section_title,
                    section_idx=section_idx,
                    chunk_idx=len(chunks),
                    page_number=page_number,
                    start_sentence_idx=chunk_start_idx,
                    end_sentence_idx=sentence_idx - 1
                )
                chunks.append(chunk)
                
                # Handle overlap
                overlap_count = min(len(current_chunk_sentences), self.config.overlap_sentences)
                if overlap_count > 0:
                    current_chunk_sentences = current_chunk_sentences[-overlap_count:]
                    current_char_count = sum(len(s) for s in current_chunk_sentences)
                    chunk_start_idx = max(sentence_idx - overlap_count, 0)
                else:
                    current_chunk_sentences = []
                    current_char_count = 0
                    chunk_start_idx = sentence_idx
            
            if not current_chunk_sentences:
                chunk_start_idx = sentence_idx
            
            current_chunk_sentences.append(sentence)
            current_char_count += sentence_length
        
        # Final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk = self._create_chunk(
                text=chunk_text,
                section_title=section_title,
                section_idx=section_idx,
                chunk_idx=len(chunks),
                page_number=page_number,
                start_sentence_idx=chunk_start_idx,
                end_sentence_idx=len(sentences) - 1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_simple(self, text: str, page_number: int) -> List[Dict]:
        """Simple fallback chunking - split by sentences if paragraphs don't work"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        # If only 1 "paragraph" (no paragraph breaks), split by sentences instead
        if len(paragraphs) == 1 or (len(paragraphs) == 2 and not paragraphs[1].strip()):
            print(f"[CHUNKER] Page {page_number}: No paragraph breaks detected, using sentence-based chunking")
            try:
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else re.split(r'(?<=[.!?])\s+', text)
            except Exception:
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_text = ""
            chunk_idx = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_text) + len(sentence) > self.config.chunk_size and current_text:
                    chunk = self._create_chunk(
                        text=current_text.strip(),
                        section_title="DOCUMENT_CONTENT",
                        section_idx=0,
                        chunk_idx=chunk_idx,
                        page_number=page_number,
                        start_sentence_idx=0,
                        end_sentence_idx=0
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    current_text = sentence
                else:
                    current_text += (" " + sentence) if current_text else sentence
            
            if current_text:
                chunk = self._create_chunk(
                    text=current_text.strip(),
                    section_title="DOCUMENT_CONTENT",
                    section_idx=0,
                    chunk_idx=chunk_idx,
                    page_number=page_number,
                    start_sentence_idx=0,
                    end_sentence_idx=0
                )
                chunks.append(chunk)
            
            return chunks
        
        # Otherwise use paragraph-based chunking
        current_text = ""
        chunk_idx = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_text) + len(paragraph) > self.config.chunk_size and current_text:
                chunk = self._create_chunk(
                    text=current_text.strip(),
                    section_title="DOCUMENT_CONTENT",
                    section_idx=0,
                    chunk_idx=chunk_idx,
                    page_number=page_number,
                    start_sentence_idx=0,
                    end_sentence_idx=0
                )
                chunks.append(chunk)
                chunk_idx += 1
                current_text = paragraph
            else:
                current_text += ("\n\n" + paragraph) if current_text else paragraph
        
        if current_text:
            chunk = self._create_chunk(
                text=current_text.strip(),
                section_title="DOCUMENT_CONTENT",
                section_idx=0,
                chunk_idx=chunk_idx,
                page_number=page_number,
                start_sentence_idx=0,
                end_sentence_idx=0
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        section_title: str,
        section_idx: int,
        chunk_idx: int,
        page_number: int,
        start_sentence_idx: int,
        end_sentence_idx: int
    ) -> Dict:
        """Create chunk with metadata"""
        clean_text = text.strip()
        
        return {
            "text": clean_text,
            "chunk_index": chunk_idx,
            "page": page_number,  # Correct page number
            "section_title": section_title,
            "section_code": self.section_extractor.extract_section_number(section_title),
            "section_index": section_idx,
            "word_count": len(clean_text.split()),
            "char_count": len(clean_text),
            "sentence_count": max(0, end_sentence_idx - start_sentence_idx + 1),
            "has_overlap": chunk_idx > 0,
            "start_sentence_index": start_sentence_idx,
            "end_sentence_index": end_sentence_idx,
            "keywords": self.content_analyzer.extract_keywords(clean_text),
            "has_numeric_data": self.content_analyzer.has_numeric_data(clean_text),
            "has_medical_patterns": self.content_analyzer.has_medical_patterns(clean_text),
            "text_density": self.content_analyzer.calculate_text_density(clean_text),
            "normalized_text": self.normalizer.normalize_for_metadata(clean_text),
            "metadata": {
                "section": section_title,
                "processing_method": "healthcare_semantic"
            }
        }


# Global instance
_healthcare_chunker = None

def get_healthcare_chunker(config: Optional[ChunkingConfig] = None) -> UniversalHealthcareChunker:
    """Get or create chunker instance"""
    global _healthcare_chunker
    if _healthcare_chunker is None or config is not None:
        _healthcare_chunker = UniversalHealthcareChunker(config)
    return _healthcare_chunker


class HealthcareDocumentStorage:
    """Document storage with proper chunking"""
    
    def __init__(self):
        self.chunker = get_healthcare_chunker()
    
    def process_document_text(self, text: str, document_name: str, page_number: int = 1) -> List[Dict]:
        """
        Process document text - should be called PER PAGE, not for entire document
        
        Args:
            text: Text from ONE page
            document_name: Document filename
            page_number: The actual page number
            
        Returns:
            List of chunks with correct page numbers and citations
        """
        try:
            chunks = self.chunker.healthcare_semantic_chunking(text, page_number)
            
            # Add document-specific metadata
            for chunk in chunks:
                chunk.update({
                    "document_name": document_name,
                    "citation_id": f"{page_number}.{chunk['chunk_index'] + 1}",
                    "full_citation": f"[{page_number}.{chunk['chunk_index'] + 1}]"
                })
                
                # Update metadata
                chunk["metadata"].update({
                    "source": document_name,
                    "page": page_number
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking error: {e}", exc_info=True)
            return self._fallback_processing(text, document_name, page_number)
    
    def _fallback_processing(self, text: str, document_name: str, page_number: int) -> List[Dict]:
        """Fallback processing"""
        # Simple paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_text = ""
        chunk_idx = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_text) + len(paragraph) > 1200 and current_text:
                chunks.append({
                    "text": current_text.strip(),
                    "document_name": document_name,
                    "page": page_number,
                    "chunk_index": chunk_idx,
                    "citation_id": f"{page_number}.{chunk_idx + 1}",
                    "full_citation": f"[{page_number}.{chunk_idx + 1}]",
                    "section_title": "DOCUMENT_CONTENT",
                    "word_count": len(current_text.split()),
                    "char_count": len(current_text),
                    "metadata": {"source": document_name, "page": page_number}
                })
                chunk_idx += 1
                current_text = paragraph
            else:
                current_text += ("\n\n" + paragraph) if current_text else paragraph
        
        if current_text:
            chunks.append({
                "text": current_text.strip(),
                "document_name": document_name,
                "page": page_number,
                "chunk_index": chunk_idx,
                "citation_id": f"{page_number}.{chunk_idx + 1}",
                "full_citation": f"[{page_number}.{chunk_idx + 1}]",
                "section_title": "DOCUMENT_CONTENT",
                "word_count": len(current_text.split()),
                "char_count": len(current_text),
                "metadata": {"source": document_name, "page": page_number}
            })
        
        return chunks


# Global storage instance
healthcare_storage = HealthcareDocumentStorage()
