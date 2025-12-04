"""
PII Detection Service
Handles detection of personally identifiable information in user queries
"""
import re
from typing import Dict
from logging_config import get_logger

logger = get_logger(__name__)

# PII Detection availability flag
PII_DETECTION_AVAILABLE = False
_analyzer_engine = None

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PII_DETECTION_AVAILABLE = True
    logger.info("Presidio PII detection module loaded successfully")
except ImportError as e:
    logger.warning(f"PII detection not available: {e}")


def get_analyzer_engine():
    """Lazily initialize and return the Presidio AnalyzerEngine."""
    global _analyzer_engine, PII_DETECTION_AVAILABLE
    
    if _analyzer_engine is None and PII_DETECTION_AVAILABLE:
        logger.info("Loading Presidio Analyzer Engine (one-time setup)...")
        try:
            # Configure NLP engine with a SpaCy model
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            })
            _analyzer_engine = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=["en"]
            )
            logger.info("Presidio Analyzer Engine loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio Analyzer Engine: {e}")
            PII_DETECTION_AVAILABLE = False
            _analyzer_engine = None
    
    return _analyzer_engine


def is_likely_english(text: str) -> bool:
    """
    Lightweight language heuristic to detect if text is likely English.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text is likely English
    """
    if not text or not isinstance(text, str):
        return False

    # Try to use langdetect if it's installed for better accuracy
    try:
        from langdetect import detect
        lang = detect(text)
        return lang == "en"
    except Exception:
        pass

    # Heuristic fallback:
    # - If text contains non-Latin scripts, assume non-English
    if re.search(r"[\u0400-\u04FF\u0900-\u097F\u4E00-\u9FFF]", text):
        return False

    # Count tokens that match common English words
    tokens = re.findall(r"[A-Za-z']{2,}", text)
    if not tokens:
        return False

    common_english = {
        "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
        "that", "this", "it", "as", "are", "was", "were", "be", "by",
        "or", "from", "what", "who", "when", "where"
    }
    matches = sum(1 for t in tokens if t.lower() in common_english)
    ratio = matches / len(tokens)

    # If at least 20% of tokens are common English words, consider it English
    return ratio >= 0.20


def detect_pii(text: str) -> Dict:
    """
    Detect personally identifiable information in text using Presidio Analyzer.
    Falls back to simple regex patterns if Presidio is unavailable.
    
    Args:
        text: Text to analyze for PII
        
    Returns:
        dict: Contains 'has_high_risk_pii', 'risk_level', 'warning_message', and 'details'
    """
    if not text:
        return {
            "has_high_risk_pii": False,
            "risk_level": 0,
            "warning_message": None,
            "details": [],
            "flagged_entities": []
        }

    analyzer = get_analyzer_engine()

    # If Presidio is available, use it — but only if the input text is likely English.
    if PII_DETECTION_AVAILABLE and analyzer:
        try:
            if not is_likely_english(text):
                logger.info("Input not detected as English. Skipping Presidio analysis to avoid false positives.")
            else:
                logger.info("Analyzing text with Presidio...")
                analyzer_results = analyzer.analyze(text=text, language='en')

                # Filter results to only include PERSON entities to avoid flagging government policy terms
                original_count = len(analyzer_results)
                allowed_entities = ["PERSON"]
                analyzer_results = [res for res in analyzer_results if res.entity_type in allowed_entities]
                
                if len(analyzer_results) < original_count:
                    logger.info(f"Filtered Presidio results to only include {', '.join(allowed_entities)} entities.")
                
                if analyzer_results:
                    logger.warning(f"Presidio detected {len(analyzer_results)} PII entities after filtering.")
                    detected_types = list(set([res.entity_type for res in analyzer_results]))
                    details = []
                    flagged_entities = []
                    
                    for res in analyzer_results:
                        details.append({
                            'type': res.entity_type,
                            'text': text[res.start:res.end],
                            'risk_level': 5,
                            'score': res.score
                        })
                        flagged_entities.append(f"{res.entity_type}='{text[res.start:res.end]}'")

                    warning_message = (
                        f"PII/PHI detected in your message. Please remove the following before proceeding: {', '.join(detected_types)}. "
                        "For your privacy and HIPAA compliance, we cannot process requests containing personal information."
                    )

                    return {
                        "has_high_risk_pii": True,
                        "risk_level": 5,
                        "warning_message": warning_message,
                        "details": details,
                        "flagged_entities": flagged_entities
                    }
                else:
                    logger.info("Presidio analysis complete - No PII entities of allowed types found.")
                    return {
                        "has_high_risk_pii": False,
                        "risk_level": 0,
                        "warning_message": None,
                        "details": [],
                        "flagged_entities": []
                    }
        except Exception as e:
            logger.error(f"Presidio analysis failed: {e}. Falling back to regex.")
            # Fall through to regex method if Presidio fails at runtime

    # Fallback to simple regex patterns if Presidio is not available or failed
    logger.info("Using fallback regex-based detection.")
    try:
        # Simple regex patterns for common PII types
        patterns = {
            'EMAIL_ADDRESS': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'PERSON': r'(?i)\b(dr\.?|doctor|prof\.?|professor)\s+([a-zA-Z][a-zA-Z\s]{1,30})\b'
        }
        
        detected = []
        flagged_entities = []
        
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    match_text = ' '.join(match).strip() if isinstance(match, tuple) else match
                    detected.append({
                        'type': pii_type,
                        'text': match_text,
                        'risk_level': 4
                    })
                    flagged_entities.append(f"{pii_type.upper()}='{match_text}'")
        
        if detected:
            detected_types = list(set([d['type'] for d in detected]))
            warning_message = (
                f"PII/PHI detected in your message. Please remove the following before proceeding: {', '.join(detected_types)}. "
                "For your privacy and HIPAA compliance, we cannot process requests containing personal information."
            )
            return {
                "has_high_risk_pii": True,
                "risk_level": 4,
                "warning_message": warning_message,
                "details": detected,
                "flagged_entities": flagged_entities
            }
        else:
            return {
                "has_high_risk_pii": False,
                "risk_level": 0,
                "warning_message": None,
                "details": [],
                "flagged_entities": []
            }
            
    except Exception as e:
        logger.error(f"Fallback regex detection failed: {e}")
        return {
            "has_high_risk_pii": True,
            "risk_level": 5,
            "warning_message": "PII detection system error. Request blocked as a safety measure.",
            "details": [],
            "error": str(e),
            "flagged_entities": []
        }


# NOTE: Analyzer initialization is now lazy-loaded on first use
# to improve startup performance. The get_analyzer_engine() function
# will initialize it automatically when detect_pii() is first called.
