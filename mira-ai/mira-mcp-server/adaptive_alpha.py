"""
Adaptive Alpha Parameter for Hybrid Search
Dynamically adjusts the balance between dense and sparse retrieval based on query characteristics
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of medical queries for alpha adjustment"""
    SPECIFIC_MEDICAL_TERM = "specific_medical_term"
    GENERAL_MEDICAL_CONCEPT = "general_medical_concept"
    SAFETY_CONCERNS = "safety_concerns"
    DOSING_QUESTIONS = "dosing_questions"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLEX_MULTI_TERM = "complex_multi_term"


@dataclass
class AlphaRecommendation:
    """Recommendation for alpha parameter with reasoning"""
    alpha: float
    query_type: QueryType
    confidence: float
    reasoning: str
    factors: Dict[str, Any]


class AdaptiveAlphaController:
    """Controls adaptive alpha parameter for hybrid search"""

    def __init__(self):
        # Base alpha values for different query types
        self.base_alphas = {
            QueryType.SPECIFIC_MEDICAL_TERM: 0.8,    # Favor dense embeddings for specific terms
            QueryType.GENERAL_MEDICAL_CONCEPT: 0.6,  # Balanced for general concepts
            QueryType.SAFETY_CONCERNS: 0.7,          # Favor dense for safety questions
            QueryType.DOSING_QUESTIONS: 0.5,          # Balanced for dosing
            QueryType.GENERAL_INQUIRY: 0.4,           # Favor sparse for general questions
            QueryType.COMPLEX_MULTI_TERM: 0.6         # Balanced for complex queries
        }

        # Medical term patterns for classification
        self.medical_patterns = {
            'specific_drugs': re.compile(r'\b(?:azithromycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|acyclovir|valacyclovir|oseltamivir|ibuprofen|acetaminophen|aspirin|warfarin|atorvastatin|simvastatin|metformin|lisinopril|amlodipine|prednisone|albuterol|fluticasone|omeprazole|ondansetron|morphine|oxycodone|hydrocodone)\b', re.IGNORECASE),
            'medical_conditions': re.compile(r'\b(?:pneumonia|infection|hypertension|diabetes|asthma|copd|arthritis|depression|anxiety|epilepsy|migraine|osteoporosis|heart failure|stroke|kidney disease|liver disease)\b', re.IGNORECASE),
            'safety_terms': re.compile(r'\b(?:side effects|adverse|toxicity|contraindication|warning|interaction|allergic|hypersensitivity|overdose|poisoning)\b', re.IGNORECASE),
            'dosing_terms': re.compile(r'\b(?:dosage|dose|mg|mcg|g|ml|tablet|capsule|frequency|daily|twice|three times|q\d*h|administration|route)\b', re.IGNORECASE),
            'adverse_events': re.compile(r'\b(?:nausea|vomiting|diarrhea|rash|dizziness|headache|fatigue|insomnia|constipation|abdominal pain|fever|chills|cough|shortness of breath|chest pain|palpitations|swelling|bruising|bleeding|jaundice|confusion|seizures)\b', re.IGNORECASE),
            'cardiac_terms': re.compile(r'\b(?:qt|qtc|torsades|arrhythmia|cardiac|heart|rhythm|tachycardia|bradycardia|fibrillation|palpitation|electrocardiogram|ecg|ekg)\b', re.IGNORECASE),
            'hepatic_terms': re.compile(r'\b(?:liver|hepatic|hepatotoxicity|jaundice|bilirubin|alt|ast|transaminitis|cirrhosis|hepatitis)\b', re.IGNORECASE),
            'renal_terms': re.compile(r'\b(?:kidney|renal|nephrotoxicity|creatinine|bun|gfr|dialysis|acute kidney injury|chronic kidney disease)\b', re.IGNORECASE)
        }

        # Track alpha performance for learning
        self.performance_history = []

    def get_adaptive_alpha(self, query: str, context: Optional[Dict[str, Any]] = None) -> AlphaRecommendation:
        """
        Determine optimal alpha parameter for a given query

        Args:
            query: The search query
            context: Optional context information (previous performance, user preferences, etc.)

        Returns:
            AlphaRecommendation with optimal alpha and reasoning
        """
        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        base_alpha = self.base_alphas[query_type]

        # Apply adjustments based on query characteristics
        adjustments = self._calculate_adjustments(query_lower, query_type)

        # Apply context-based adjustments
        if context:
            context_adjustments = self._apply_context_adjustments(base_alpha, context)
            adjustments.extend(context_adjustments)

        # Calculate final alpha
        final_alpha = self._apply_adjustments(base_alpha, adjustments)

        # Ensure alpha is within valid range [0, 1]
        final_alpha = max(0.0, min(1.0, final_alpha))

        # Calculate confidence based on adjustment factors
        confidence = self._calculate_confidence(adjustments)

        # Build reasoning
        reasoning = self._build_reasoning(query_type, adjustments, final_alpha)

        return AlphaRecommendation(
            alpha=final_alpha,
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            factors={
                'base_alpha': base_alpha,
                'adjustments': adjustments,
                'query_type': query_type.value,
                'query_length': len(query.split()),
                'medical_term_count': self._count_medical_terms(query_lower)
            }
        )

    def _classify_query(self, query: str) -> QueryType:
        """Classify the query type based on content analysis"""
        words = query.split()
        word_count = len(words)

        # Check for specific medical terms
        specific_drug_matches = len(self.medical_patterns['specific_drugs'].findall(query))
        if specific_drug_matches > 0:
            return QueryType.SPECIFIC_MEDICAL_TERM

        # Check for safety concerns
        safety_matches = len(self.medical_patterns['safety_terms'].findall(query))
        adverse_matches = len(self.medical_patterns['adverse_events'].findall(query))
        if safety_matches > 0 or adverse_matches > 0:
            return QueryType.SAFETY_CONCERNS

        # Check for dosing questions
        dosing_matches = len(self.medical_patterns['dosing_terms'].findall(query))
        if dosing_matches > 0:
            return QueryType.DOSING_QUESTIONS

        # Check for general medical concepts
        condition_matches = len(self.medical_patterns['medical_conditions'].findall(query))
        if condition_matches > 0:
            return QueryType.GENERAL_MEDICAL_CONCEPT

        # Check for complex multi-term queries
        medical_term_count = self._count_medical_terms(query)
        if word_count >= 4 and medical_term_count >= 2:
            return QueryType.COMPLEX_MULTI_TERM

        # Default to general inquiry
        return QueryType.GENERAL_INQUIRY

    def _count_medical_terms(self, query: str) -> int:
        """Count total medical terms in query"""
        total_count = 0
        for pattern_name, pattern in self.medical_patterns.items():
            matches = pattern.findall(query)
            total_count += len(matches)
        return total_count

    def _calculate_adjustments(self, query: str, query_type: QueryType) -> List[Dict[str, Any]]:
        """Calculate alpha adjustments based on query characteristics"""
        adjustments = []

        # Length-based adjustment
        word_count = len(query.split())
        if word_count <= 2:
            adjustments.append({
                'type': 'query_length',
                'adjustment': 0.1,  # Favor dense for short queries
                'reason': 'Short query favors semantic matching'
            })
        elif word_count >= 6:
            adjustments.append({
                'type': 'query_length',
                'adjustment': -0.1,  # Favor sparse for long queries
                'reason': 'Long query favors keyword matching'
            })

        # Medical specificity adjustment
        medical_term_count = self._count_medical_terms(query)
        if medical_term_count >= 3:
            adjustments.append({
                'type': 'medical_specificity',
                'adjustment': 0.15,  # Favor dense for highly medical queries
                'reason': 'High medical term density favors semantic embeddings'
            })

        # Cardiac focus adjustment (dense embeddings better for cardiac terms)
        cardiac_matches = len(self.medical_patterns['cardiac_terms'].findall(query))
        if cardiac_matches > 0:
            adjustments.append({
                'type': 'cardiac_focus',
                'adjustment': 0.1,
                'reason': 'Cardiac queries benefit from semantic understanding'
            })

        # Safety-critical adjustment
        if query_type == QueryType.SAFETY_CONCERNS:
            safety_terms = len(self.medical_patterns['safety_terms'].findall(query))
            if safety_terms >= 2:
                adjustments.append({
                    'type': 'safety_critical',
                    'adjustment': 0.05,
                    'reason': 'Safety queries need precise semantic matching'
                })

        # Question format adjustment
        if query.strip().endswith('?') or query.lower().startswith(('what', 'how', 'why', 'when', 'where')):
            adjustments.append({
                'type': 'question_format',
                'adjustment': -0.05,  # Favor sparse for questions
                'reason': 'Question format suggests keyword-based retrieval'
            })

        return adjustments

    def _apply_context_adjustments(self, base_alpha: float, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply adjustments based on context information"""
        adjustments = []

        # Performance-based adjustment
        if 'previous_performance' in context:
            prev_perf = context['previous_performance']
            if prev_perf.get('f1', 0) < 0.5:  # Poor performance
                # Try different alpha
                current_range = 'high' if base_alpha > 0.6 else 'low'
                if current_range == 'high':
                    adjustments.append({
                        'type': 'performance_feedback',
                        'adjustment': -0.2,
                        'reason': 'Previous high alpha performed poorly, trying lower alpha'
                    })
                else:
                    adjustments.append({
                        'type': 'performance_feedback',
                        'adjustment': 0.2,
                        'reason': 'Previous low alpha performed poorly, trying higher alpha'
                    })

        # User preference adjustment
        if 'user_preference' in context:
            pref = context['user_preference']
            if pref == 'precise':
                adjustments.append({
                    'type': 'user_preference',
                    'adjustment': 0.1,
                    'reason': 'User prefers precise results, favoring dense embeddings'
                })
            elif pref == 'broad':
                adjustments.append({
                    'type': 'user_preference',
                    'adjustment': -0.1,
                    'reason': 'User prefers broad results, favoring sparse retrieval'
                })

        # Time-based adjustment (faster retrieval for urgent queries)
        if 'urgency' in context and context['urgency'] == 'high':
            adjustments.append({
                'type': 'urgency',
                'adjustment': -0.05,  # Slightly favor faster sparse retrieval
                'reason': 'High urgency favors faster retrieval methods'
            })

        return adjustments

    def _apply_adjustments(self, base_alpha: float, adjustments: List[Dict[str, Any]]) -> float:
        """Apply all adjustments to base alpha"""
        alpha = base_alpha
        for adjustment in adjustments:
            alpha += adjustment['adjustment']

        return alpha

    def _calculate_confidence(self, adjustments: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the alpha recommendation"""
        if not adjustments:
            return 0.5  # Default confidence

        # Higher confidence with more adjustments and stronger signals
        adjustment_magnitude = sum(abs(adj['adjustment']) for adj in adjustments)
        adjustment_count = len(adjustments)

        # Base confidence from number of factors
        confidence = min(0.9, 0.5 + (adjustment_count * 0.1))

        # Boost confidence for strong adjustments
        if adjustment_magnitude > 0.2:
            confidence += 0.1

        return min(1.0, confidence)

    def _build_reasoning(self, query_type: QueryType, adjustments: List[Dict[str, Any]], final_alpha: float) -> str:
        """Build human-readable reasoning for alpha choice"""
        reasoning_parts = [f"Query classified as: {query_type.value.replace('_', ' ')}"]

        if adjustments:
            reasoning_parts.append("Adjustments applied:")
            for adj in adjustments:
                reasoning_parts.append(f"  - {adj['reason']} ({adj['adjustment']:+.2f})")
        else:
            reasoning_parts.append("No adjustments needed")

        reasoning_parts.append(f"Final alpha: {final_alpha:.2f}")

        # Add interpretation
        if final_alpha > 0.7:
            reasoning_parts.append("High alpha favors semantic/dense retrieval")
        elif final_alpha < 0.3:
            reasoning_parts.append("Low alpha favors keyword/sparse retrieval")
        else:
            reasoning_parts.append("Balanced alpha uses both retrieval methods")

        return " | ".join(reasoning_parts)

    def record_performance(self, query: str, alpha: float, metrics: Dict[str, float]):
        """Record alpha performance for future learning"""
        self.performance_history.append({
            'query': query,
            'alpha': alpha,
            'metrics': metrics,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get statistics on alpha performance"""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        alphas = [entry['alpha'] for entry in self.performance_history]
        f1_scores = [entry['metrics'].get('f1', 0) for entry in self.performance_history]

        return {
            'total_queries': len(self.performance_history),
            'avg_alpha': sum(alphas) / len(alphas) if alphas else 0,
            'alpha_std': __import__('statistics').stdev(alphas) if len(alphas) > 1 else 0,
            'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            'alpha_performance_correlation': self._calculate_correlation(alphas, f1_scores)
        }

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi**2 for xi in x)
            sum_y2 = sum(yi**2 for yi in y)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5

            return numerator / denominator if denominator != 0 else 0.0
        except:
            return 0.0


# Convenience functions
def get_adaptive_alpha(query: str, context: Optional[Dict[str, Any]] = None) -> float:
    """Get adaptive alpha for a query"""
    controller = AdaptiveAlphaController()
    recommendation = controller.get_adaptive_alpha(query, context)
    return recommendation.alpha


def get_alpha_recommendation(query: str, context: Optional[Dict[str, Any]] = None) -> AlphaRecommendation:
    """Get full alpha recommendation with reasoning"""
    controller = AdaptiveAlphaController()
    return controller.get_adaptive_alpha(query, context)


if __name__ == "__main__":
    # Example usage
    controller = AdaptiveAlphaController()

    test_queries = [
        "azithromycin side effects",
        "what are the cardiac risks of antibiotics",
        "dosage for amoxicillin in children",
        "antibiotic resistance",
        "QT prolongation and torsades de pointes"
    ]

    for query in test_queries:
        recommendation = controller.get_adaptive_alpha(query)
        print(f"Query: {query}")
        print(f"Alpha: {recommendation.alpha:.2f} | Type: {recommendation.query_type.value}")
        print(f"Reasoning: {recommendation.reasoning}")
        print("-" * 50)