"""
Language Detection Utility
Detects the language of text for multilingual document processing
"""

from typing import Optional
import re

def detect_language(text: str) -> str:
    """
    Detect the language of text using character patterns and keywords
    Returns ISO 639-1 language code (en, ar, fr, es, zh, etc.)
    """
    if not text or len(text.strip()) < 10:
        return "en"  # Default to English for short texts
    
    text_sample = text[:500].lower()  # Use first 500 chars for detection
    
    # Arabic detection (highest priority due to distinctive script)
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text_sample))
    if arabic_chars > len(text_sample) * 0.3:
        return "ar"
    
    # Chinese detection
    chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text_sample))
    if chinese_chars > len(text_sample) * 0.3:
        return "zh"
    
    # Japanese detection
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text_sample))
    if japanese_chars > len(text_sample) * 0.2:
        return "ja"
    
    # Korean detection
    korean_chars = len(re.findall(r'[\uAC00-\uD7AF]', text_sample))
    if korean_chars > len(text_sample) * 0.3:
        return "ko"
    
    # Cyrillic detection (Russian, etc.)
    cyrillic_chars = len(re.findall(r'[\u0400-\u04FF]', text_sample))
    if cyrillic_chars > len(text_sample) * 0.3:
        return "ru"
    
    # Spanish keywords
    spanish_keywords = ['el', 'la', 'los', 'las', 'de', 'del', 'que', 'para', 'con', 'por', 'gobierno', 'política']
    spanish_count = sum(1 for word in spanish_keywords if f' {word} ' in f' {text_sample} ')
    
    # French keywords
    french_keywords = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'pour', 'dans', 'avec', 'gouvernement', 'politique']
    french_count = sum(1 for word in french_keywords if f' {word} ' in f' {text_sample} ')
    
    # German keywords
    german_keywords = ['der', 'die', 'das', 'und', 'ist', 'für', 'mit', 'von', 'regierung', 'politik']
    german_count = sum(1 for word in german_keywords if f' {word} ' in f' {text_sample} ')
    
    # Portuguese keywords
    portuguese_keywords = ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'que', 'para', 'com', 'governo', 'política']
    portuguese_count = sum(1 for word in portuguese_keywords if f' {word} ' in f' {text_sample} ')
    
    # Italian keywords
    italian_keywords = ['il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'del', 'che', 'per', 'governo', 'politica']
    italian_count = sum(1 for word in italian_keywords if f' {word} ' in f' {text_sample} ')
    
    # Determine language based on keyword counts
    language_scores = {
        'es': spanish_count,
        'fr': french_count,
        'de': german_count,
        'pt': portuguese_count,
        'it': italian_count
    }
    
    max_score = max(language_scores.values())
    if max_score >= 3:
        return max(language_scores, key=language_scores.get)
    
    # Default to English
    return "en"


def get_language_name(language_code: str) -> str:
    """Convert language code to full name"""
    language_names = {
        'en': 'English',
        'ar': 'Arabic',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'pt': 'Portuguese',
        'it': 'Italian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'pl': 'Polish',
        'uk': 'Ukrainian'
    }
    return language_names.get(language_code, 'Unknown')


if __name__ == "__main__":
    # Test language detection
    test_texts = {
        "en": "This is a government policy document about education reform and healthcare regulations.",
        "ar": "هذه وثيقة سياسة حكومية حول إصلاح التعليم ولوائح الرعاية الصحية.",
        "fr": "Ceci est un document de politique gouvernementale sur la réforme de l'éducation et les réglementations sanitaires.",
        "es": "Este es un documento de política gubernamental sobre la reforma educativa y las regulaciones de salud.",
        "de": "Dies ist ein Regierungsdokument zur Bildungsreform und zu Gesundheitsvorschriften.",
        "zh": "这是一份关于教育改革和医疗保健法规的政府政策文件。"
    }
    
    print("Testing language detection:")
    for expected_lang, text in test_texts.items():
        detected_lang = detect_language(text)
        result = "✅" if detected_lang == expected_lang else "❌"
        print(f"{result} Expected: {expected_lang}, Detected: {detected_lang} ({get_language_name(detected_lang)})")
