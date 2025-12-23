"""
Intent Classification Package for GIRA AI
Provides query intent classification for optimized retrieval
"""

from intent.types import QueryIntent, IntentClassification
from intent.classifier import QueryIntentClassifier
from typing import Dict, Any, Tuple


# Convenience functions
def classify_query_intent(query: str) -> IntentClassification:
    """Classify a query's intent"""
    classifier = QueryIntentClassifier()
    return classifier.classify(query)


def get_query_strategy(query: str) -> Tuple[IntentClassification, Dict[str, Any]]:
    """Get both classification and retrieval strategy"""
    classifier = QueryIntentClassifier()
    classification = classifier.classify(query)
    strategy = classifier.get_retrieval_strategy(classification)
    return classification, strategy


__all__ = [
    "QueryIntent",
    "IntentClassification",
    "QueryIntentClassifier",
    "classify_query_intent",
    "get_query_strategy",
]
