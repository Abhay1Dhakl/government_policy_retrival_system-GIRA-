"""
Evaluation framework for GIRA AI retrieval system
Provides comprehensive metrics for precision, recall, and ranking quality
"""

import numpy as np
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import json
import os
from datetime import datetime


class RetrievalEvaluator:
    """Comprehensive evaluation framework for retrieval systems"""

    def __init__(self, ground_truth_path: Optional[str] = None):
        self.ground_truth = {}  # query -> set of relevant doc IDs
        self.evaluation_history = []
        self.baseline_metrics = {}

        if ground_truth_path and os.path.exists(ground_truth_path):
            self.load_ground_truth(ground_truth_path)

    def load_ground_truth(self, filepath: str):
        """Load ground truth relevance judgments"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            for query, relevant_docs in data.items():
                self.ground_truth[query] = set(relevant_docs)

    def save_ground_truth(self, filepath: str):
        """Save ground truth data"""
        with open(filepath, 'w') as f:
            json.dump({k: list(v) for k, v in self.ground_truth.items()}, f, indent=2)

    def add_ground_truth(self, query: str, relevant_doc_ids: List[str]):
        """Add ground truth for a query"""
        self.ground_truth[query] = set(relevant_doc_ids)

    def evaluate_search(self, query: str, retrieved_docs: List[Dict], ground_truth_ids: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive retrieval metrics

        Args:
            query: The search query
            retrieved_docs: List of retrieved documents with 'id' field
            ground_truth_ids: Set of relevant document IDs (if None, uses stored ground truth)

        Returns:
            Dictionary with precision, recall, F1, NDCG, and other metrics
        """
        if ground_truth_ids is None:
            ground_truth_ids = self.ground_truth.get(query, set())

        if not ground_truth_ids:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ndcg': 0.0,
                'retrieved_count': len(retrieved_docs),
                'relevant_count': 0,
                'warning': 'No ground truth available'
            }

        retrieved_ids = [doc.get('id') for doc in retrieved_docs if doc.get('id')]
        retrieved_set = set(retrieved_ids)

        # Basic metrics
        true_positives = len(retrieved_set & ground_truth_ids)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # NDCG calculation
        ndcg = self._calculate_ndcg(retrieved_ids, ground_truth_ids)

        # Mean Reciprocal Rank
        mrr = self._calculate_mrr(retrieved_ids, ground_truth_ids)

        # Mean Average Precision
        map_score = self._calculate_map(retrieved_ids, ground_truth_ids)

        # Precision at K for different K values
        precision_at_k = {}
        for k in [1, 3, 5, 10]:
            retrieved_at_k = set(retrieved_ids[:k])
            tp_at_k = len(retrieved_at_k & ground_truth_ids)
            precision_at_k[f'P@{k}'] = tp_at_k / k if k > 0 else 0.0

        result = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'mrr': mrr,
            'map': map_score,
            'retrieved_count': len(retrieved_docs),
            'relevant_count': len(ground_truth_ids),
            **precision_at_k
        }

        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'metrics': result
        })

        return result

    def _calculate_ndcg(self, retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not retrieved_ids or not ground_truth_ids:
            return 0.0

        dcg = 0.0
        idcg = 0.0

        # DCG for retrieved results
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = 1 if doc_id in ground_truth_ids else 0
            dcg += relevance / np.log2(i + 2)

        # IDCG (ideal DCG)
        ideal_relevant_count = min(len(ground_truth_ids), k)
        for i in range(ideal_relevant_count):
            idcg += 1 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
        """Calculate Mean Average Precision"""
        if not ground_truth_ids:
            return 0.0

        relevant_found = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(ground_truth_ids) if ground_truth_ids else 0.0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all evaluations"""
        if not self.evaluation_history:
            return {'error': 'No evaluations performed yet'}

        metrics = ['precision', 'recall', 'f1', 'ndcg', 'mrr', 'map']
        summary = {}

        for metric in metrics:
            values = [eval['metrics'][metric] for eval in self.evaluation_history if metric in eval['metrics']]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)

        summary['total_evaluations'] = len(self.evaluation_history)
        summary['unique_queries'] = len(set(eval['query'] for eval in self.evaluation_history))

        return summary

    def save_evaluation_history(self, filepath: str):
        """Save evaluation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)

    def set_baseline_metrics(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics.copy()

    def compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare current metrics to baseline"""
        if not self.baseline_metrics:
            return {'error': 'No baseline metrics set'}

        comparison = {}
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                comparison[f'{metric}_delta'] = current_value - baseline_value
                comparison[f'{metric}_improvement_pct'] = (
                    (current_value - baseline_value) / baseline_value * 100
                    if baseline_value != 0 else 0
                )

        return comparison


class MedicalRetrievalEvaluator(RetrievalEvaluator):
    """Specialized evaluator for medical document retrieval"""

    def __init__(self, ground_truth_path: Optional[str] = None):
        super().__init__(ground_truth_path)
        self.medical_categories = {
            'safety': ['adverse', 'toxicity', 'contraindication', 'warning'],
            'dosing': ['dosage', 'dose', 'mg', 'administration'],
            'indication': ['treatment', 'indication', 'use', 'therapy'],
            'general': []
        }

    def evaluate_medical_search(self, query: str, retrieved_docs: List[Dict],
                              ground_truth_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Evaluate with medical-specific metrics"""
        base_metrics = self.evaluate_search(query, retrieved_docs, ground_truth_ids)

        # Add medical category classification
        category = self._classify_medical_query(query)
        base_metrics['medical_category'] = category

        # Add document type distribution
        doc_types = defaultdict(int)
        for doc in retrieved_docs:
            doc_type = doc.get('metadata', {}).get('document_type', 'unknown')
            doc_types[doc_type] += 1
        base_metrics['doc_type_distribution'] = dict(doc_types)

        # Add section relevance (if available)
        section_relevance = self._evaluate_section_relevance(query, retrieved_docs)
        base_metrics.update(section_relevance)

        return base_metrics

    def _classify_medical_query(self, query: str) -> str:
        """Classify query into medical categories"""
        query_lower = query.lower()
        for category, keywords in self.medical_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'general'

    def _evaluate_section_relevance(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Evaluate relevance based on document sections"""
        section_scores = defaultdict(float)
        total_docs = len(retrieved_docs)

        if total_docs == 0:
            return {'section_relevance_score': 0.0}

        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            section = metadata.get('section_title', '').lower()
            chunk_type = metadata.get('chunk_type', '').lower()

            # Score based on section relevance to query
            score = 0.0
            query_lower = query.lower()

            if 'safety' in query_lower and any(term in section for term in ['warning', 'adverse', 'contraindication']):
                score = 1.0
            elif 'dose' in query_lower and 'dosage' in section:
                score = 1.0
            elif 'pediatric' in query_lower and any(term in section for term in ['pediatric', 'children']):
                score = 1.0

            section_scores[section or 'unknown'] += score

        avg_section_relevance = sum(section_scores.values()) / total_docs
        return {'section_relevance_score': avg_section_relevance}


# Convenience functions for quick evaluation
def evaluate_retrieval_results(query: str, retrieved_docs: List[Dict], relevant_ids: Set[str]) -> Dict[str, float]:
    """Quick evaluation function"""
    evaluator = RetrievalEvaluator()
    return evaluator.evaluate_search(query, retrieved_docs, relevant_ids)


def evaluate_medical_retrieval(query: str, retrieved_docs: List[Dict], relevant_ids: Set[str]) -> Dict[str, Any]:
    """Quick medical evaluation function"""
    evaluator = MedicalRetrievalEvaluator()
    return evaluator.evaluate_medical_search(query, retrieved_docs, relevant_ids)


if __name__ == "__main__":
    # Example usage
    evaluator = MedicalRetrievalEvaluator()

    # Add some ground truth
    evaluator.add_ground_truth("azithromycin side effects", ["doc1", "doc2", "doc3"])
    evaluator.add_ground_truth("amoxicillin dosage", ["doc4", "doc5"])

    # Example evaluation
    retrieved_docs = [
        {"id": "doc1", "metadata": {"document_type": "pis", "section_title": "Adverse Reactions"}},
        {"id": "doc2", "metadata": {"document_type": "lrd", "section_title": "Warnings"}},
        {"id": "doc4", "metadata": {"document_type": "pis", "section_title": "Dosage"}}
    ]

    results = evaluator.evaluate_medical_search("azithromycin side effects", retrieved_docs)
    print("Evaluation Results:", json.dumps(results, indent=2))