"""
Relevance Feedback Loop for MIRA AI
Collects user feedback to continuously improve retrieval quality
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from evaluation import MedicalRetrievalEvaluator


class RelevanceFeedbackCollector:
    """Collects and analyzes user relevance feedback"""

    def __init__(self, feedback_dir: str = "relevance_feedback"):
        self.feedback_dir = feedback_dir
        self.feedback_data = []
        self.evaluator = MedicalRetrievalEvaluator()

        # Create feedback directory
        os.makedirs(feedback_dir, exist_ok=True)

        # Load existing feedback
        self._load_feedback()

    def collect_feedback(self, query: str, retrieved_docs: List[Dict],
                        relevant_doc_ids: Set[str], user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> str:
        """
        Collect user feedback on document relevance

        Args:
            query: The search query
            retrieved_docs: Documents shown to user
            relevant_doc_ids: Document IDs marked as relevant by user
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            Feedback ID for tracking
        """
        feedback_id = f"fb_{int(datetime.now().timestamp())}_{hash(query) % 10000}"

        feedback_entry = {
            'feedback_id': feedback_id,
            'query': query,
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'retrieved_docs': [
                {
                    'id': doc.get('id'),
                    'score': doc.get('score', 0) or doc.get('hybrid_score', 0),
                    'rank': i + 1,
                    'relevant': doc.get('id') in relevant_doc_ids
                }
                for i, doc in enumerate(retrieved_docs)
            ],
            'relevant_count': len(relevant_doc_ids),
            'total_retrieved': len(retrieved_docs),
            'query_metadata': self._extract_query_metadata(query)
        }

        self.feedback_data.append(feedback_entry)
        self._save_feedback_entry(feedback_entry)

        return feedback_id

    def _extract_query_metadata(self, query: str) -> Dict[str, Any]:
        """Extract metadata about the query for analysis"""
        return {
            'length': len(query.split()),
            'has_question_mark': query.endswith('?'),
            'word_count': len(query.split()),
            'contains_drug_name': any(drug in query.lower() for drug in [
                'azithromycin', 'amoxicillin', 'ciprofloxacin', 'metronidazole',
                'fluconazole', 'acyclovir', 'ibuprofen', 'acetaminophen'
            ]),
            'contains_safety_terms': any(term in query.lower() for term in [
                'side effects', 'adverse', 'toxicity', 'safe', 'danger'
            ]),
            'contains_dosing_terms': any(term in query.lower() for term in [
                'dosage', 'dose', 'mg', 'tablet', 'frequency'
            ])
        }

    def _save_feedback_entry(self, feedback_entry: Dict[str, Any]):
        """Save individual feedback entry to file"""
        date_str = datetime.now().strftime("%Y%m%d")
        feedback_file = os.path.join(self.feedback_dir, f"feedback_{date_str}.jsonl")

        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')

    def _load_feedback(self):
        """Load existing feedback data"""
        if not os.path.exists(self.feedback_dir):
            return

        for filename in os.listdir(self.feedback_dir):
            if filename.startswith("feedback_") and filename.endswith(".jsonl"):
                filepath = os.path.join(self.feedback_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            if line.strip():
                                self.feedback_data.append(json.loads(line.strip()))
                except Exception as e:
                    print(f"Warning: Could not load feedback file {filename}: {e}")

    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary statistics of collected feedback"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = [
            fb for fb in self.feedback_data
            if datetime.fromisoformat(fb['timestamp']) > cutoff_date
        ]

        if not recent_feedback:
            return {'error': 'No feedback data available'}

        # Calculate metrics
        total_sessions = len(set(fb['session_id'] for fb in recent_feedback if fb.get('session_id')))
        total_queries = len(recent_feedback)
        avg_relevant_per_query = statistics.mean(fb['relevant_count'] for fb in recent_feedback)

        # Precision analysis
        precisions = []
        recalls = []
        for fb in recent_feedback:
            retrieved_ids = {doc['id'] for doc in fb['retrieved_docs']}
            relevant_ids = {doc['id'] for doc in fb['retrieved_docs'] if doc['relevant']}

            if retrieved_ids:
                precision = len(relevant_ids) / len(retrieved_ids)
                precisions.append(precision)

            # Note: True recall would require knowing all relevant docs in corpus
            # This is an approximation based on retrieved docs
            recall = len(relevant_ids) / fb['relevant_count'] if fb['relevant_count'] > 0 else 0
            recalls.append(recall)

        return {
            'total_feedback_entries': len(recent_feedback),
            'total_sessions': total_sessions,
            'total_queries': total_queries,
            'avg_relevant_per_query': round(avg_relevant_per_query, 2),
            'avg_precision': round(statistics.mean(precisions), 3) if precisions else 0,
            'avg_recall': round(statistics.mean(recalls), 3) if recalls else 0,
            'precision_std': round(statistics.stdev(precisions), 3) if len(precisions) > 1 else 0,
            'date_range': f"{cutoff_date.date()} to {datetime.now().date()}"
        }

    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in feedback to identify improvement opportunities"""
        if not self.feedback_data:
            return {'error': 'No feedback data available'}

        # Group by query types
        query_patterns = defaultdict(list)

        for fb in self.feedback_data:
            metadata = fb['query_metadata']

            # Categorize queries
            if metadata['contains_safety_terms']:
                category = 'safety'
            elif metadata['contains_dosing_terms']:
                category = 'dosing'
            elif metadata['contains_drug_name']:
                category = 'drug_info'
            else:
                category = 'general'

            query_patterns[category].append(fb)

        # Analyze each category
        analysis = {}
        for category, feedbacks in query_patterns.items():
            precisions = []
            for fb in feedbacks:
                retrieved_ids = {doc['id'] for doc in fb['retrieved_docs']}
                relevant_ids = {doc['id'] for doc in fb['retrieved_docs'] if doc['relevant']}
                if retrieved_ids:
                    precision = len(relevant_ids) / len(retrieved_ids)
                    precisions.append(precision)

            analysis[category] = {
                'count': len(feedbacks),
                'avg_precision': round(statistics.mean(precisions), 3) if precisions else 0,
                'precision_std': round(statistics.stdev(precisions), 3) if len(precisions) > 1 else 0
            }

        # Find common failure patterns
        failure_patterns = self._identify_failure_patterns()

        return {
            'category_analysis': analysis,
            'failure_patterns': failure_patterns,
            'recommendations': self._generate_recommendations(analysis, failure_patterns)
        }

    def _identify_failure_patterns(self) -> List[Dict[str, Any]]:
        """Identify common patterns where retrieval fails"""
        patterns = []

        # Low precision queries
        low_precision_queries = []
        for fb in self.feedback_data:
            retrieved_ids = {doc['id'] for doc in fb['retrieved_docs']}
            relevant_ids = {doc['id'] for doc in fb['retrieved_docs'] if doc['relevant']}
            if retrieved_ids:
                precision = len(relevant_ids) / len(retrieved_ids)
                if precision < 0.3:  # Low precision threshold
                    low_precision_queries.append({
                        'query': fb['query'],
                        'precision': precision,
                        'relevant_found': len(relevant_ids),
                        'total_retrieved': len(retrieved_ids)
                    })

        if low_precision_queries:
            patterns.append({
                'type': 'low_precision',
                'description': 'Queries with precision below 30%',
                'count': len(low_precision_queries),
                'examples': low_precision_queries[:5]  # Top 5 examples
            })

        # Zero relevant results
        zero_relevant = [fb for fb in self.feedback_data if fb['relevant_count'] == 0]
        if zero_relevant:
            patterns.append({
                'type': 'zero_relevant',
                'description': 'Queries where user found no relevant results',
                'count': len(zero_relevant),
                'examples': [fb['query'] for fb in zero_relevant[:5]]
            })

        return patterns

    def _generate_recommendations(self, analysis: Dict[str, Any],
                                failure_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []

        # Category-specific recommendations
        for category, stats in analysis.items():
            if stats['avg_precision'] < 0.5:
                if category == 'safety':
                    recommendations.append("Improve safety-related query processing - consider boosting adverse event sections")
                elif category == 'dosing':
                    recommendations.append("Enhance dosing query retrieval - focus on dosage and administration sections")
                elif category == 'drug_info':
                    recommendations.append("Improve drug information retrieval - expand drug synonym handling")

        # Failure pattern recommendations
        for pattern in failure_patterns:
            if pattern['type'] == 'low_precision':
                recommendations.append(f"Address low precision issues ({pattern['count']} cases) - consider re-ranking improvements")
            elif pattern['type'] == 'zero_relevant':
                recommendations.append(f"Improve query understanding for {pattern['count']} failed queries - consider query expansion")

        if not recommendations:
            recommendations.append("Overall performance is good - continue monitoring")

        return recommendations

    def export_feedback_for_training(self, output_file: str, min_relevant: int = 1):
        """
        Export feedback data suitable for model training

        Args:
            output_file: Path to save training data
            min_relevant: Minimum number of relevant docs required for inclusion
        """
        training_data = []

        for fb in self.feedback_data:
            relevant_docs = [doc for doc in fb['retrieved_docs'] if doc['relevant']]
            if len(relevant_docs) >= min_relevant:
                training_example = {
                    'query': fb['query'],
                    'positive_docs': [doc['id'] for doc in relevant_docs],
                    'all_retrieved': [doc['id'] for doc in fb['retrieved_docs']],
                    'query_metadata': fb['query_metadata'],
                    'feedback_timestamp': fb['timestamp']
                }
                training_data.append(training_example)

        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Exported {len(training_data)} training examples to {output_file}")


class AdaptiveRetrievalImprover:
    """Uses feedback to adaptively improve retrieval strategies"""

    def __init__(self, feedback_collector: RelevanceFeedbackCollector):
        self.feedback_collector = feedback_collector
        self.improvement_rules = self._load_improvement_rules()

    def _load_improvement_rules(self) -> Dict[str, Any]:
        """Load rules for adapting retrieval based on feedback"""
        return {
            'low_precision_boost': {
                'condition': lambda stats: stats.get('avg_precision', 0) < 0.4,
                'action': 'increase_top_k',
                'parameter': 'top_k',
                'value': lambda current: min(current + 2, 15)
            },
            'safety_improvement': {
                'condition': lambda analysis: analysis.get('category_analysis', {}).get('safety', {}).get('avg_precision', 1) < 0.5,
                'action': 'boost_safety_sections',
                'parameter': 'prioritize_sections',
                'value': ['warning', 'adverse', 'contraindication', 'safety']
            },
            'dosing_improvement': {
                'condition': lambda analysis: analysis.get('category_analysis', {}).get('dosing', {}).get('avg_precision', 1) < 0.5,
                'action': 'boost_dosing_sections',
                'parameter': 'prioritize_sections',
                'value': ['dosage', 'administration', 'pediatric']
            }
        }

    def get_adaptive_parameters(self, query: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get adaptively improved retrieval parameters based on feedback

        Args:
            query: The search query
            base_params: Base retrieval parameters

        Returns:
            Adapted parameters
        """
        adapted_params = base_params.copy()

        # Get recent feedback analysis
        analysis = self.feedback_collector.analyze_feedback_patterns()

        # Apply improvement rules
        for rule_name, rule in self.improvement_rules.items():
            if rule['condition'](analysis):
                param = rule['parameter']
                if callable(rule['value']):
                    adapted_params[param] = rule['value'](adapted_params.get(param, 10))
                else:
                    adapted_params[param] = rule['value']

                print(f"Applied improvement rule '{rule_name}': {rule['action']}")

        return adapted_params

    def should_trigger_retraining(self) -> bool:
        """
        Determine if accumulated feedback warrants model retraining

        Returns:
            True if retraining is recommended
        """
        summary = self.feedback_collector.get_feedback_summary(days=7)

        # Trigger retraining if we have significant feedback and low performance
        min_feedback_threshold = 50
        performance_threshold = 0.6

        feedback_count = summary.get('total_queries', 0)
        avg_precision = summary.get('avg_precision', 1.0)

        return feedback_count >= min_feedback_threshold and avg_precision < performance_threshold


# Convenience functions
def collect_relevance_feedback(query: str, retrieved_docs: List[Dict],
                             relevant_doc_ids: Set[str], user_id: Optional[str] = None) -> str:
    """Convenience function for collecting feedback"""
    collector = RelevanceFeedbackCollector()
    return collector.collect_feedback(query, retrieved_docs, relevant_doc_ids, user_id)


def get_feedback_summary(days: int = 30) -> Dict[str, Any]:
    """Get feedback summary"""
    collector = RelevanceFeedbackCollector()
    return collector.get_feedback_summary(days)


if __name__ == "__main__":
    # Example usage
    collector = RelevanceFeedbackCollector()

    # Simulate some feedback
    retrieved_docs = [
        {'id': 'doc1', 'score': 0.8},
        {'id': 'doc2', 'score': 0.7},
        {'id': 'doc3', 'score': 0.6}
    ]

    # User marks doc1 and doc3 as relevant
    feedback_id = collector.collect_feedback(
        "azithromycin side effects",
        retrieved_docs,
        {'doc1', 'doc3'},
        user_id="user123"
    )

    print(f"Collected feedback with ID: {feedback_id}")

    # Get summary
    summary = collector.get_feedback_summary()
    print("Feedback Summary:", json.dumps(summary, indent=2))

    # Analyze patterns
    analysis = collector.analyze_feedback_patterns()
    print("Pattern Analysis:", json.dumps(analysis, indent=2))