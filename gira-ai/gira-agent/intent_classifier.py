"""
Query Intent Classification for GIRA AI
Routes queries to appropriate retrieval strategies based on intent analysis
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json


class QueryIntent(Enum):
    """Medical query intent categories"""
    SAFETY_CONCERNS = "safety_concerns"
    DOSING_INQUIRY = "dosing_inquiry"
    DRUG_INFORMATION = "drug_information"
    CONDITION_TREATMENT = "condition_treatment"
    DRUG_INTERACTION = "drug_interaction"
    ADVERSE_EVENT = "adverse_event"
    PEDIATRIC_CONCERNS = "pediatric_concerns"
    GERIATRIC_CONCERNS = "geriatric_concerns"
    PREGNANCY_LACTATION = "pregnancy_lactation"
    CONTRAINDICATION = "contraindication"
    PHARMACOKINETICS = "pharmacokinetics"
    MONITORING_GUIDANCE = "monitoring_guidance"
    GENERAL_MEDICAL = "general_medical"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


@dataclass
class IntentClassification:
    """Classification result with confidence and features"""
    intent: QueryIntent
    confidence: float
    features: Dict[str, Any]
    reasoning: str
    alternative_intents: List[Tuple[QueryIntent, float]]


class QueryIntentClassifier:
    """Classifies medical queries by intent for optimized retrieval"""

    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.intent_keywords = self._build_intent_keywords()
        self.contextual_rules = self._build_contextual_rules()

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
            'has_drug_names': bool(re.search(r'\b(azithromycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|acyclovir|valacyclovir|oseltamivir|ibuprofen|acetaminophen|aspirin|warfarin|atorvastatin|simvastatin|metformin|lisinopril|amlodipine|prednisone|albuterol|fluticasone|omeprazole|ondansetron|morphine|oxycodone|hydrocodone)\b', query)),
            'has_dosage_terms': bool(re.search(r'\b(dosage|dose|mg|mcg|g|ml|tablet|capsule|frequency|daily|twice|three times|q\d*h|administration|route)\b', query)),
            'has_safety_terms': bool(re.search(r'\b(side effects|adverse|toxicity|contraindication|warning|interaction|allergic|hypersensitivity|overdose|poisoning|risk|danger|caution)\b', query)),
            'has_age_terms': bool(re.search(r'\b(pediatric|children|infant|neonate|adolescent|geriatric|elderly|senior|pregnant|lactating|pregnancy|breastfeeding)\b', query)),
            'has_condition_terms': bool(re.search(r'\b(pneumonia|infection|hypertension|diabetes|asthma|copd|arthritis|depression|anxiety|epilepsy|migraine|osteoporosis|heart failure|stroke|kidney disease|liver disease)\b', query)),
            'has_cardiac_terms': bool(re.search(r'\b(qt|qtc|torsades|arrhythmia|cardiac|heart|rhythm|tachycardia|bradycardia|fibrillation|palpitation|electrocardiogram|ecg|ekg)\b', query)),
            'has_hepatic_terms': bool(re.search(r'\b(liver|hepatic|hepatotoxicity|jaundice|bilirubin|alt|ast|transaminitis|cirrhosis|hepatitis)\b', query)),
            'has_renal_terms': bool(re.search(r'\b(kidney|renal|nephrotoxicity|creatinine|bun|gfr|dialysis|acute kidney injury|chronic kidney disease)\b', query)),
            'has_monitoring_terms': bool(re.search(r'\b(monitor|monitoring|check|test|lab|blood|urine|ecg|ekg|follow|surveillance)\b', query)),
            'has_regulatory_terms': bool(re.search(r'\b(fda|ema|regulatory|compliance|approved|indication|labeling|pi|lrd|hpl)\b', query)),
            'has_pharmacokinetic_terms': bool(re.search(r'\b(absorption|distribution|metabolism|excretion|half.life|clearance|bioavailability|pharmacokinetics|pk)\b', query))
        }

        # Count specific term categories
        features.update({
            'drug_count': len(re.findall(r'\b(azithromycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|acyclovir|valacyclovir|oseltamivir|ibuprofen|acetaminophen|aspirin|warfarin|atorvastatin|simvastatin|metformin|lisinopril|amlodipine|prednisone|albuterol|fluticasone|omeprazole|ondansetron|morphine|oxycodone|hydrocodone)\b', query)),
            'safety_term_count': len(re.findall(r'\b(side effects|adverse|toxicity|contraindication|warning|interaction|allergic|hypersensitivity|overdose|poisoning|risk|danger|caution)\b', query)),
            'age_term_count': len(re.findall(r'\b(pediatric|children|infant|neonate|adolescent|geriatric|elderly|senior|pregnant|lactating|pregnancy|breastfeeding)\b', query))
        })

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

    def _build_intent_patterns(self) -> Dict[QueryIntent, List[Tuple[re.Pattern, float]]]:
        """Build regex patterns for intent classification"""
        return {
            QueryIntent.SAFETY_CONCERNS: [
                (re.compile(r'\b(side effects?|adverse (effects?|reactions?)|toxicity|poisoning)\b', re.IGNORECASE), 3.0),
                (re.compile(r'\b(risks?|dangers?|cautions?|warnings?)\b', re.IGNORECASE), 2.0),
                (re.compile(r'\b(safe|unsafe|dangerous)\b', re.IGNORECASE), 1.5)
            ],
            QueryIntent.DOSING_INQUIRY: [
                (re.compile(r'\b(dosage|dose|mg|mcg|g|ml|tablet|capsule)\b', re.IGNORECASE), 2.5),
                (re.compile(r'\b(frequency|daily|twice|three times|q\d*h)\b', re.IGNORECASE), 2.0),
                (re.compile(r'\b(administration|route|oral|iv|im|sc)\b', re.IGNORECASE), 1.5)
            ],
            QueryIntent.DRUG_INTERACTION: [
                (re.compile(r'\b(interaction|interactions|combined with|together with)\b', re.IGNORECASE), 3.0),
                (re.compile(r'\b(with|and|plus)\s+(another|other)\s+drug\b', re.IGNORECASE), 2.0)
            ],
            QueryIntent.ADVERSE_EVENT: [
                (re.compile(r'\b(nausea|vomiting|diarrhea|rash|dizziness|headache|fatigue|insomnia|constipation|abdominal pain|fever|chills|cough|shortness of breath|chest pain|palpitations|swelling|bruising|bleeding|jaundice|confusion|seizures)\b', re.IGNORECASE), 2.5)
            ],
            QueryIntent.PEDIATRIC_CONCERNS: [
                (re.compile(r'\b(pediatric|children|infant|neonate|adolescent|kids?|baby|child)\b', re.IGNORECASE), 3.0),
                (re.compile(r'\b(under \d+ years?|age \d+|\d+ years? old)\b', re.IGNORECASE), 2.0)
            ],
            QueryIntent.GERIATRIC_CONCERNS: [
                (re.compile(r'\b(geriatric|elderly|senior|old|aging|age \d+|\d+ years? old)\b', re.IGNORECASE), 3.0)
            ],
            QueryIntent.PREGNANCY_LACTATION: [
                (re.compile(r'\b(pregnant|pregnancy|lactating|lactation|breastfeeding|breast milk|breastfed)\b', re.IGNORECASE), 3.0)
            ],
            QueryIntent.CONTRAINDICATION: [
                (re.compile(r'\b(contraindicated|contraindication|avoid|not recommended)\b', re.IGNORECASE), 3.0)
            ],
            QueryIntent.PHARMACOKINETICS: [
                (re.compile(r'\b(absorption|distribution|metabolism|excretion|half.life|clearance|bioavailability|pharmacokinetics|pk)\b', re.IGNORECASE), 2.5)
            ],
            QueryIntent.MONITORING_GUIDANCE: [
                (re.compile(r'\b(monitor|monitoring|check|test|lab|blood|urine|ecg|ekg|follow|surveillance)\b', re.IGNORECASE), 2.5)
            ],
            QueryIntent.REGULATORY_COMPLIANCE: [
                (re.compile(r'\b(fda|ema|regulatory|compliance|approved|indication|labeling|pi|lrd|hpl)\b', re.IGNORECASE), 2.5)
            ]
        }

    def _build_intent_keywords(self) -> Dict[QueryIntent, List[Tuple[str, float]]]:
        """Build keyword lists for intent classification"""
        return {
            QueryIntent.SAFETY_CONCERNS: [
                ("safe", 1.0), ("unsafe", 1.5), ("danger", 1.5), ("caution", 1.0),
                ("warning", 1.0), ("risk", 1.0), ("adverse", 2.0), ("toxicity", 2.0)
            ],
            QueryIntent.DOSING_INQUIRY: [
                ("how much", 2.0), ("how many", 1.5), ("amount", 1.0), ("quantity", 1.0),
                ("take", 1.0), ("give", 1.0), ("administer", 1.5)
            ],
            QueryIntent.DRUG_INFORMATION: [
                ("what is", 1.5), ("information about", 1.5), ("tell me about", 1.5),
                ("uses", 1.0), ("purpose", 1.0), ("indication", 1.5)
            ],
            QueryIntent.CONDITION_TREATMENT: [
                ("treat", 2.0), ("treatment", 2.0), ("therapy", 1.5), ("cure", 1.5),
                ("manage", 1.0), ("control", 1.0)
            ]
        }

    def _build_contextual_rules(self) -> Dict[str, Any]:
        """Build contextual rules for intent classification"""
        return {
            'safety_with_age': {
                'condition': lambda f: f['has_safety_terms'] and f['has_age_terms'],
                'boost_intent': QueryIntent.SAFETY_CONCERNS,
                'boost_amount': 1.5
            },
            'dosing_with_age': {
                'condition': lambda f: f['has_dosage_terms'] and f['has_age_terms'],
                'boost_intent': QueryIntent.PEDIATRIC_CONCERNS,
                'boost_amount': 2.0
            },
            'cardiac_safety': {
                'condition': lambda f: f['has_cardiac_terms'] and f['has_safety_terms'],
                'boost_intent': QueryIntent.SAFETY_CONCERNS,
                'boost_amount': 2.0
            },
            'hepatic_safety': {
                'condition': lambda f: f['has_hepatic_terms'] and f['has_safety_terms'],
                'boost_intent': QueryIntent.SAFETY_CONCERNS,
                'boost_amount': 2.0
            },
            'renal_safety': {
                'condition': lambda f: f['has_renal_terms'] and f['has_safety_terms'],
                'boost_intent': QueryIntent.SAFETY_CONCERNS,
                'boost_amount': 2.0
            }
        }

    def _apply_feature_rules(self, intent: QueryIntent, features: Dict[str, Any]) -> float:
        """Apply feature-based scoring rules"""
        score = 0.0

        # Safety concerns intent
        if intent == QueryIntent.SAFETY_CONCERNS:
            if features['has_safety_terms']:
                score += 3.0
            if features['safety_term_count'] > 1:
                score += 1.0
            if features['has_cardiac_terms'] or features['has_hepatic_terms'] or features['has_renal_terms']:
                score += 1.5

        # Dosing inquiry intent
        elif intent == QueryIntent.DOSING_INQUIRY:
            if features['has_dosage_terms']:
                score += 3.0
            if features['has_age_terms']:
                score += 1.0

        # Drug information intent
        elif intent == QueryIntent.DRUG_INFORMATION:
            if features['has_drug_names']:
                score += 2.0
            if features['starts_with_wh']:
                score += 1.0

        # Age-specific intents
        elif intent == QueryIntent.PEDIATRIC_CONCERNS:
            if features['has_age_terms'] and 'pediatric' in features:
                score += 3.0
            if features['age_term_count'] > 0:
                score += 1.0

        elif intent == QueryIntent.GERIATRIC_CONCERNS:
            if features['has_age_terms'] and any(term in str(features).lower() for term in ['geriatric', 'elderly', 'senior']):
                score += 3.0

        elif intent == QueryIntent.PREGNANCY_LACTATION:
            if features['has_age_terms'] and any(term in str(features).lower() for term in ['pregnant', 'pregnancy', 'lactating']):
                score += 3.0

        # Monitoring guidance
        elif intent == QueryIntent.MONITORING_GUIDANCE:
            if features['has_monitoring_terms']:
                score += 2.5
            if features['has_cardiac_terms'] or features['has_hepatic_terms'] or features['has_renal_terms']:
                score += 1.0

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

        if intent == QueryIntent.SAFETY_CONCERNS:
            if features['has_safety_terms']:
                reasons.append("contains safety-related keywords")
            if features['safety_term_count'] > 1:
                reasons.append("multiple safety terms detected")
            if features['has_cardiac_terms']:
                reasons.append("cardiac safety concerns")

        elif intent == QueryIntent.DOSING_INQUIRY:
            if features['has_dosage_terms']:
                reasons.append("contains dosage-related terms")
            if features['has_age_terms']:
                reasons.append("age-specific dosing query")

        elif intent == QueryIntent.PEDIATRIC_CONCERNS:
            if features['has_age_terms']:
                reasons.append("pediatric/age-related terms")
            if features['age_term_count'] > 0:
                reasons.append("multiple age references")

        elif intent == QueryIntent.MONITORING_GUIDANCE:
            if features['has_monitoring_terms']:
                reasons.append("monitoring/testing terms")
            if features['has_cardiac_terms']:
                reasons.append("cardiac monitoring context")

        if not reasons:
            reasons.append("pattern matching and feature analysis")

        return f"Classified as {intent.value.replace('_', ' ')} because: {'; '.join(reasons)}"

    def get_retrieval_strategy(self, classification: IntentClassification) -> Dict[str, Any]:
        """
        Get recommended retrieval strategy based on intent classification

        Returns:
            Dictionary with retrieval parameters optimized for the intent
        """
        strategy = {
            'top_k': 10,  # Default
            'use_reranking': True,
            'expand_query': True,
            'prioritize_sections': [],
            'filter_adjustments': {}
        }

        if classification.intent == QueryIntent.SAFETY_CONCERNS:
            strategy.update({
                'top_k': 15,  # More results for safety concerns
                'prioritize_sections': ['warning', 'adverse', 'contraindication', 'safety'],
                'expand_query': True
            })

        elif classification.intent == QueryIntent.DOSING_INQUIRY:
            strategy.update({
                'top_k': 8,
                'prioritize_sections': ['dosage', 'administration', 'pediatric'],
                'expand_query': False  # Precise dosing info
            })

        elif classification.intent == QueryIntent.PEDIATRIC_CONCERNS:
            strategy.update({
                'top_k': 12,
                'prioritize_sections': ['pediatric', '8.4', 'children', 'infant'],
                'filter_adjustments': {'pediatric_focus': True}
            })

        elif classification.intent == QueryIntent.MONITORING_GUIDANCE:
            strategy.update({
                'top_k': 10,
                'prioritize_sections': ['monitoring', 'laboratory', 'tests', 'ecg'],
                'expand_query': True
            })

        elif classification.intent == QueryIntent.CONTRAINDICATION:
            strategy.update({
                'top_k': 8,
                'prioritize_sections': ['contraindication', 'warning', 'precaution'],
                'expand_query': True
            })

        return strategy


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


if __name__ == "__main__":
    # Example usage
    classifier = QueryIntentClassifier()

    test_queries = [
        "What are the side effects of azithromycin?",
        "How much amoxicillin should I give my child?",
        "Is azithromycin safe during pregnancy?",
        "What monitoring is needed for cardiac patients on antibiotics?",
        "Can elderly patients take this medication?",
        "What are the contraindications for ciprofloxacin?"
    ]

    for query in test_queries:
        classification = classifier.classify(query)
        strategy = classifier.get_retrieval_strategy(classification)

        print(f"Query: {query}")
        print(f"Intent: {classification.intent.value} (confidence: {classification.confidence:.2f})")
        print(f"Strategy: top_k={strategy['top_k']}, prioritize={strategy['prioritize_sections']}")
        print(f"Reasoning: {classification.reasoning}")
        print("-" * 80)