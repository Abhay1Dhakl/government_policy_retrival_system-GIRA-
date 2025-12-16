"""Global constants for MCP server."""

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "shall",
    "under", "when", "into", "upon", "such", "which", "been", "were", "your", "their",
    "than", "about", "each", "within", "while", "those", "these", "there", "after",
    "before", "during", "because", "other", "where", "patients", "patient", "should",
    "could", "would", "might", "effect", "effects", "used", "using", "use", "dose",
    "doses", "per", "day", "days", "week", "weeks", "month", "months", "or", "of",
    "via", "into", "onto"
}

DOCUMENT_TYPE_SYNONYMS = {
    "pis": {
        "pis", "pi", "prescribing_information", "prescribing-information",
        "prescribing information", "product_information", "product-information",
        "product information"
    },
    "lrd": {
        "lrd", "label_repository_data", "label repository data",
        "label_repository_document", "label repository document", "label-data"
    },
    "hpl": {
        "hpl", "health_product_label", "health product label", "product label",
        "product_label"
    },
    "past_cases": {"past_cases", "past-cases", "history", "user_history"}
}

SECTION_PRIORITY_WEIGHTS = {
    "warning": 0.25,
    "contraindication": 0.25,
    "safety": 0.22,
    "adverse": 0.2,
    "reaction": 0.18,
    "overdose": 0.18,
    "dosage": 0.15,
    "pediatric": 0.15,
    "geriatric": 0.12
}

REGION_ALIASES = {
    "us": {"US", "USA", "UNITED STATES", "UNITED_STATES"},
    "eu": {"EU", "EUROPE", "EMA"},
    "uk": {"UK", "UNITED KINGDOM", "UNITED_KINGDOM", "GB"},
    "np": {"NP", "NEPAL"},
    "ca": {"CA", "CANADA"},
    "global": {"GLOBAL", "WORLDWIDE"}
}
