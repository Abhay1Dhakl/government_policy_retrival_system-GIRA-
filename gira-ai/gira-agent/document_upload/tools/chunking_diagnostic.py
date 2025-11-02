import argparse
import io
import json
import re
from pathlib import Path
from typing import List, Dict

from PyPDF2 import PdfReader
from docx import Document as DocxDocument

from document_upload.app.services.file_ingestor import healthcare_storage, split_text_into_chunks_with_overlap


def load_text_from_file(path: Path) -> Dict[str, List[str]]:
    """Load text per page/section from PDF, DOCX, or TXT.
    Returns dict with keys: 'pages' (list of page texts) and 'full_text' (single string).
    """
    suffix = path.suffix.lower()
    pages: List[str] = []
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
    elif suffix in {".docx"}:
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        # Treat as single page
        pages = [text]
    else:
        pages = [path.read_text(encoding="utf-8", errors="ignore")]
    return {"pages": pages, "full_text": "\n\n".join(pages)}


def simple_sentence_split(text: str) -> List[str]:
    # Conservative sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def analyze_coverage(original_text: str, chunks: List[Dict]) -> Dict:
    sentences = simple_sentence_split(original_text)
    if not sentences:
        total_chars = len(original_text)
        covered_chars = sum(len(c.get("text", "")) for c in chunks)
        return {
            "mode": "char_only",
            "total_chars": total_chars,
            "covered_chars": covered_chars,
            "coverage_ratio": round((covered_chars / total_chars) if total_chars else 1.0, 4),
        }

    covered = [False] * len(sentences)
    for c in chunks:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        # mark any sentence that appears as substring as covered
        for i, s in enumerate(sentences):
            if not covered[i] and s and s in t:
                covered[i] = True

    coverage_ratio = sum(covered) / len(covered)
    missing_examples = [sentences[i] for i, ok in enumerate(covered) if not ok][:10]

    return {
        "mode": "sentence",
        "total_sentences": len(sentences),
        "covered_sentences": int(sum(covered)),
        "coverage_ratio": round(coverage_ratio, 4),
        "missing_examples": missing_examples,
    }


def analyze_chunks(chunks: List[Dict]) -> Dict:
    lengths = [len((c.get("text") or "")) for c in chunks]
    dup_map = {}
    for c in chunks:
        key = (c.get("page"), c.get("section_title"), (c.get("text") or "")[:80])
        dup_map[key] = dup_map.get(key, 0) + 1
    duplicates = sum(1 for k, v in dup_map.items() if v > 1)
    return {
        "chunk_count": len(chunks),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round((sum(lengths) / len(lengths)) if lengths else 0, 1),
        "duplicates_keys": duplicates,
    }


def main():
    ap = argparse.ArgumentParser(description="Chunking diagnostic for medical documents")
    ap.add_argument("file", type=str, help="Path to PDF/DOCX/TXT file")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    path = Path(args.file)
    data = load_text_from_file(path)

    all_chunks: List[Dict] = []
    for i, page_text in enumerate(data["pages"], start=1):
        # Use the same pipeline as ingestion
        page_chunks = healthcare_storage.process_document_text(page_text, path.name, page_number=i)
        all_chunks.extend(page_chunks)

    coverage = analyze_coverage(data["full_text"], all_chunks)
    stats = analyze_chunks(all_chunks)

    report = {
        "file": str(path),
        "coverage": coverage,
        "stats": stats,
        "sample_chunks": [
            {
                "page": c.get("page"),
                "section_title": c.get("section_title"),
                "chunk_index": c.get("chunk_index"),
                "char_count": c.get("char_count"),
                "snippet": (c.get("text") or "")[:180],
            }
            for c in all_chunks[:8]
        ],
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
