"""
Query Intent Classifier for GIRA AI
Routes government policy queries to appropriate retrieval strategies
"""

import re
from typing import Dict, List, Any, Tuple

from intent.types import QueryIntent, IntentClassification
from intent.patterns import build_intent_patterns, build_intent_keywords, build_contextual_rules


class QueryIntentClassifier:
    """Classifies government queries by intent for optimized retrieval"""

    def __init__(self):
        self.intent_patterns = build_intent_patterns()
        self.intent_keywords = build_intent_keywords()
        self.contextual_rules = build_contextual_rules()

    def classify(self, query: str) -> IntentClassification:
        """
        Classify query intent with confidence scoring

        Args:
            query: The search query to classify

        Returns:
            IntentClassification with primary intent and alternatives
        """
        query_lower = query.lower().strip()
        features = self._extract_features(query_lower)

        # Score each intent
        intent_scores = {}
        for intent in QueryIntent:
            score = self._score_intent(intent, query_lower, features)
            intent_scores[intent] = score

        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        primary_intent, primary_score = sorted_intents[0]
        alternatives = [(intent, score) for intent, score in sorted_intents[1:3]]  # Top 3

        # Normalize confidence to [0, 1]
        total_score = sum(score for _, score in sorted_intents[:3])
        confidence = primary_score / total_score if total_score > 0 else 0.5

        reasoning = self._build_reasoning(primary_intent, features)

        return IntentClassification(
            intent=primary_intent,
            confidence=min(confidence, 1.0),
            features=features,
            reasoning=reasoning,
            alternative_intents=alternatives
        )

    def _extract_features(self, query: str) -> Dict[str, Any]:
        """Extract linguistic and semantic features from query"""
        features = {
            'length': len(query.split()),
            'has_question_mark': query.endswith('?'),
            'starts_with_wh': query.startswith(('what', 'how', 'why', 'when', 'where', 'which')),
            
            # Government domain features
            'has_policy_terms': bool(re.search(r'\b(policy|scheme|act|bill|law|program|initiative|mission)\b', query)),
            'has_procedure_terms': bool(re.search(r'\b(apply|register|process|steps|how to|form|submit)\b', query)),
            'has_eligibility_terms': bool(re.search(r'\b(qualify|eligible|allowed|criteria|limit|income|age)\b', query)),
            'has_document_terms': bool(re.search(r'\b(document|pdf|proof|card|license|certificate|id)\b', query)),
            'has_financial_terms': bool(re.search(r'\b(tax|cost|free|fee|payment|money|rupees|dollar|subsidy|grant|loan)\b', query)),
            'has_compliance_terms': bool(re.search(r'\b(fine|penalty|illegal|ban|mandatory|rule|violation)\b', query)),
            'has_time_terms': bool(re.search(r'\b(date|deadline|when|time|duration|validity|year|period)\b', query)),
            'has_authority_terms': bool(re.search(r'\b(ministry|govt|government|department|official|board|council)\b', query)),
            'has_benefit_terms': bool(re.search(r'\b(benefit|help|aid|support|welfare|pension|scholarship)\b', query)),
            'has_legal_terms': bool(re.search(r'\b(court|legal|rights|article|constitution|justice|lawyer)\b', query))
        }

        return features

    def _score_intent(self, intent: QueryIntent, query: str, features: Dict[str, Any]) -> float:
        """Score how well a query matches a particular intent"""
        score = 0.0

        # Pattern-based scoring
        if intent in self.intent_patterns:
            for pattern, weight in self.intent_patterns[intent]:
                if pattern.search(query):
                    score += weight

        # Keyword-based scoring
        if intent in self.intent_keywords:
            for keyword, weight in self.intent_keywords[intent]:
                if keyword in query:
                    score += weight

        # Feature-based scoring
        score += self._apply_feature_rules(intent, features)

        # Contextual rules
        score += self._apply_contextual_rules(intent, query, features)

        return score

    def _apply_feature_rules(self, intent: QueryIntent, features: Dict[str, Any]) -> float:
        """Apply feature-based scoring rules"""
        score = 0.0

        if intent == QueryIntent.PROCEDURAL_STEPS:
            if features['has_procedure_terms']: score += 2.0
            if features['starts_with_wh']: score += 1.0

        elif intent == QueryIntent.ELIGIBILITY_CRITERIA:
            if features['has_eligibility_terms']: score += 2.5
            if 'who' in features: score += 1.0

        elif intent == QueryIntent.DOCUMENT_REQUIREMENTS:
            if features['has_document_terms']: score += 3.0
            if features['has_procedure_terms']: score += 1.0

        elif intent == QueryIntent.LEGAL_FRAMEWORK:
            if features['has_legal_terms']: score += 2.5
            if features['has_policy_terms']: score += 1.0

        elif intent == QueryIntent.BENEFITS_SUBSIDIES:
            if features['has_benefit_terms']: score += 2.5
            if features['has_financial_terms']: score += 1.5

        elif intent == QueryIntent.TAXATION_FISCAL:
            if features['has_financial_terms']: score += 2.0
            if 'tax' in str(features): score += 2.0

        elif intent == QueryIntent.TIMELINES_DEADLINES:
            if features['has_time_terms']: score += 2.5

        return score

    def _apply_contextual_rules(self, intent: QueryIntent, query: str, features: Dict[str, Any]) -> float:
        """Apply contextual rules for intent boosting"""
        score = 0.0

        for rule_name, rule in self.contextual_rules.items():
            if rule['condition'](features):
                if rule['boost_intent'] == intent:
                    score += rule['boost_amount']

        return score

    def _build_reasoning(self, intent: QueryIntent, features: Dict[str, Any]) -> str:
        """Build human-readable reasoning for classification"""
        reasons = []

        if features['has_procedure_terms'] and intent == QueryIntent.PROCEDURAL_STEPS:
            reasons.append("asking about process/steps")
        if features['has_eligibility_terms'] and intent == QueryIntent.ELIGIBILITY_CRITERIA:
            reasons.append("asking about criteria/qualifications")
        if features['has_document_terms'] and intent == QueryIntent.DOCUMENT_REQUIREMENTS:
            reasons.append("asking about required documents")
        if features['has_legal_terms'] and intent == QueryIntent.LEGAL_FRAMEWORK:
            reasons.append("referencing acts or laws")
        if features['has_financial_terms'] and intent == QueryIntent.TAXATION_FISCAL:
            reasons.append("finance/tax related query")

        if not reasons:
            reasons.append("pattern matching algorithm")

        return f"Classified as {intent.value.replace('_', ' ')} because: {'; '.join(reasons)}"

    def get_retrieval_strategy(self, classification: IntentClassification) -> Dict[str, Any]:
        """
        Get recommended retrieval strategy based on government query intent

        Returns:
            Dictionary with retrieval parameters optimized for the intent
        """
        strategy = {
            'top_k': 10,
            'use_reranking': True,
            'expand_query': True,
            'prioritize_sections': [],
            'filter_adjustments': {}
        }

        if classification.intent == QueryIntent.LEGAL_FRAMEWORK:
            strategy.update({
                'top_k': 12,
                'prioritize_sections': ['act', 'section', 'article', 'clause'],
                'expand_query': False  # Precise legal search
            })

        elif classification.intent == QueryIntent.PROCEDURAL_STEPS:
            strategy.update({
                'top_k': 8,
                'prioritize_sections': ['procedure', 'steps', 'how to', 'application'],
                'expand_query': True
            })

        elif classification.intent == QueryIntent.DOCUMENT_REQUIREMENTS:
            strategy.update({
                'top_k': 6,
                'prioritize_sections': ['documents', 'attachments', 'requirements'],
                'expand_query': True
            })

        elif classification.intent == QueryIntent.BENEFITS_SUBSIDIES:
            strategy.update({
                'top_k': 15,
                'prioritize_sections': ['benefits', 'amount', 'subsidy', 'eligibility'],
                'expand_query': True
            })
            
        elif classification.intent == QueryIntent.COMPLIANCE_ISSUE:
            strategy.update({
                'top_k': 10,
                'prioritize_sections': ['penalty', 'fine', 'offence'],
                'expand_query': False
            })

        return strategy
