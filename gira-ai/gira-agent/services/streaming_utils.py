import asyncio
import json
from typing import AsyncGenerator
import re
async def stream_by_words(text: str) -> AsyncGenerator[str, None]:
    """Stream text word by word with paragraph detection"""
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    
    # Split by paragraphs
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
    
    total_words = sum(len(p.split()) for p in paragraphs)
    word_index = 0
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Stream words
        words = paragraph.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            data = {
                "type": "chunk",
                "content": chunk,
                "index": word_index,
                "total_words": total_words
            }
            yield f"data: {json.dumps(data)}\n\n"
            word_index += 1
            await asyncio.sleep(0.1)
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"

async def stream_by_sentences(text: str) -> AsyncGenerator[str, None]:
    """Stream text sentence by sentence with paragraph detection"""
    import re
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    
    # Split by paragraphs first
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
    
    sentence_index = 0
    total_sentences = sum(len([s for s in re.split(r'[.!?]+', p) if s.strip()]) for p in paragraphs)
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Split paragraph into sentences
        sentences = re.split(r'[.!?]+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Add punctuation back except for last sentence
            if i < len(sentences) - 1:
                sentence += ". "
            
            data = {
                "type": "chunk", 
                "content": sentence,
                "index": sentence_index,
                "total_sentences": total_sentences
            }
            yield f"data: {json.dumps(data)}\n\n"
            sentence_index += 1
            await asyncio.sleep(0.3)
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"

async def stream_by_tokens(text: str) -> AsyncGenerator[str, None]:
    """Stream text token by token with realistic delays and paragraph detection"""
    # CRITICAL FIX: Convert escaped \\n\\n to actual \n\n for paragraph detection
    escaped_pattern = '\\n\\n'
    actual_pattern = '\n\n'
    print(f"[STREAMING DEBUG] Original text contains {escaped_pattern}: {escaped_pattern in text}")
    print(f"[STREAMING DEBUG] Original text first 200 chars: {repr(text[:200])}")
    
    processed_text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    print(f"[STREAMING DEBUG] After conversion, contains actual newlines: {actual_pattern in processed_text}")
    print(f"[STREAMING DEBUG] Processed text first 200 chars: {repr(processed_text[:200])}")
    
    # Split by actual newlines to detect paragraphs
    paragraphs = processed_text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    print(f"[STREAMING DEBUG] Found {len(paragraphs)} paragraphs after splitting")
    for i, para in enumerate(paragraphs):
        print(f"[STREAMING DEBUG] Paragraph {i}: {para[:20]}...")
    
    # If no paragraphs found (old format), treat as single paragraph
    if len(paragraphs) <= 1:
        paragraphs = [processed_text.strip()]
        print(f"[STREAMING DEBUG] Fallback to single paragraph mode")
    
    total_words = sum(len(p.split()) for p in paragraphs)
    word_index = 0
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Detect source type based on content
        source_type = "unknown"
        para_lower = paragraph.lower()
        
        if para_idx == 0:
            source_type = "PIS"
        elif para_idx == 1:
            source_type = "LRD"
        elif para_idx == 2:
            source_type = "OTHER"
        else:
            source_type = "OTHER"
        
        # Enhanced source detection
        if "no source 1" in para_lower or "no pi document" in para_lower:
            source_type = "PIS"
        elif "no source 2" in para_lower or "no lrd document" in para_lower:
            source_type = "LRD"
        elif "no source 3" in para_lower or "past cases" in para_lower:
            source_type = "OTHER"
        
        # Stream words in this paragraph
        words = paragraph.split()
        for i, word in enumerate(words):
            token = word if i == 0 else f" {word}"
            
            data = {
                "type": "chunk",
                "content": token,
                "index": word_index,
                "total_tokens": total_words,
                "is_citation": bool(re.search(r'\[\d+\.\d+\]', token))
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            word_index += 1
            
            # Variable delay to simulate thinking
            if word_index < 5:  # Start slower
                await asyncio.sleep(0.2)
            elif any(punct in word for punct in '.!?'):  # Pause at sentence endings
                await asyncio.sleep(0.3)
            else:
                await asyncio.sleep(0.08)  # Normal speed
        
        # Send simple newline between paragraphs (except after last paragraph)
        if para_idx < len(paragraphs) - 1:
            newline_data = {
                "type": "chunk",
                "content": "\n\n"
            }
            yield f"data: {json.dumps(newline_data)}\n\n"
        
        # Add slight delay between paragraphs
        if para_idx < len(paragraphs) - 1:
            
            await asyncio.sleep(0.5)