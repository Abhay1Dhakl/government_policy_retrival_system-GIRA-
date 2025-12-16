"""GIRS Core Constants

Centralized constants for Government Information Retrieval System including
stopwords, document type synonyms, section priorities, and region aliases.
"""

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "shall",
    "under", "when", "into", "upon", "such", "which", "been", "were", "your", "their",
    "than", "about", "each", "within", "while", "those", "these", "there", "after",
    "before", "during", "because", "other", "where", "citizens", "citizen", "should",
    "could", "would", "might", "effect", "effects", "used", "using", "use", "term",
    "terms", "per", "day", "days", "week", "weeks", "month", "months", "or", "of",
    "via", "into", "onto"
}

DOCUMENT_TYPE_SYNONYMS = {
    "act": {
        "act", "acts", "legislation", "law", "statute",
        "legislative_act", "government_act",
        "government act"
    },
    "regulation": {
        "regulation", "regulations", "regulatory", "rule",
        "rules", "regulatory_document", "regulatory document", "policy"
    },
    "directive": {
        "directive", "directives", "government_directive", "official_directive", "official directive",
        "policy_directive"
    },
    "amendment": {"amendment", "amendments", "legislative_amendment", "policy_amendment"}
}

SECTION_PRIORITY_WEIGHTS = {
    "penalties": 0.25,
    "enforcement": 0.25,
    "compliance": 0.22,
    "requirements": 0.2,
    "provisions": 0.18,
    "amendments": 0.18,
    "authority": 0.15,
    "scope": 0.15,
    "definitions": 0.12
}

REGION_ALIASES = {
    "us": {"US", "USA", "UNITED STATES", "UNITED_STATES"},
    "eu": {"EU", "EUROPE", "EMA"},
    "uk": {"UK", "UNITED KINGDOM", "UNITED_KINGDOM", "GB"},
    "np": {"NP", "NEPAL"},
    "ca": {"CA", "CANADA"},
    "global": {"GLOBAL", "WORLDWIDE"}
}
