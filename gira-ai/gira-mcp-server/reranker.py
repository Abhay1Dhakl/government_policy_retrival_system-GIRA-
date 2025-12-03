"""
Cross-encoder re-ranking for GIRA AI retrieval system
Provides learned re-ranking to improve precision over heuristic scoring
"""

import asyncio
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️ sentence-transformers not available - cross-encoder re-ranking disabled")

from evaluation import MedicalRetrievalEvaluator


class MedicalReranker:
    """Cross-encoder based re-ranker for medical document retrieval"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512, use_gpu: bool = False):
        """
        Initialize the cross-encoder reranker

        Args:
            model_name: HuggingFace model name for cross-encoder
            max_length: Maximum sequence length for model input
            use_gpu: Whether to use GPU (if available)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_gpu = use_gpu
        self.model = None
        self.evaluator = MedicalRetrievalEvaluator()
        self._executor = ThreadPoolExecutor(max_workers=2)

        if CROSS_ENCODER_AVAILABLE:
            try:
                device = "cuda" if use_gpu else "cpu"
                self.model = CrossEncoder(model_name, max_length=max_length, device=device)
                print(f"✅ Cross-encoder initialized: {model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize cross-encoder: {e}")
                self.model = None
        else:
            print("❌ Cross-encoder not available - install sentence-transformers")

    async def rerank(self, query: str, documents: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Re-rank documents using cross-encoder

        Args:
            query: Search query
            documents: List of document dictionaries with 'metadata' containing 'text'
            top_k: Number of top documents to return (None = all)

        Returns:
            Re-ranked list of documents with rerank_score added
        """
        if not self.model or not documents:
            return documents

        start_time = time.time()

        try:
            # Prepare input pairs
            pairs = []
            valid_docs = []

            for doc in documents:
                metadata = doc.get('metadata', {})
                text = metadata.get('text', '').strip()

                if text:
                    pairs.append([query, text])
                    valid_docs.append(doc)

            if not pairs:
                return documents

            # Run cross-encoder in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(self._executor, self._predict_scores, pairs)

            # Add scores to documents
            for doc, score in zip(valid_docs, scores):
                doc['rerank_score'] = float(score)

            # Sort by rerank score (descending)
            reranked_docs = sorted(valid_docs, key=lambda x: x.get('rerank_score', 0), reverse=True)

            # Apply top_k limit
            if top_k:
                reranked_docs = reranked_docs[:top_k]

            rerank_time = time.time() - start_time
            print(f"✅ Re-ranking completed in {rerank_time:.3f}s for {len(reranked_docs)} documents")
            return reranked_docs

        except Exception as e:
            print(f"❌ Error during re-ranking: {e}")
            return documents

    def _predict_scores(self, pairs: List[List[str]]) -> np.ndarray:
        """Run cross-encoder prediction in thread"""
        return self.model.predict(pairs, batch_size=16)

    async def rerank_with_fallback(self, query: str, documents: List[Dict],
                                 top_k: Optional[int] = None) -> Tuple[List[Dict], str]:
        """
        Re-rank with fallback to original scoring if cross-encoder fails

        Returns:
            Tuple of (reranked_docs, method_used)
        """
        if self.model and documents:
            try:
                reranked = await self.rerank(query, documents, top_k)
                return reranked, "cross_encoder"
            except Exception as e:
                print(f"⚠️ Cross-encoder failed, falling back to original scoring: {e}")

        # Fallback: sort by original score
        fallback_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
        if top_k:
            fallback_docs = fallback_docs[:top_k]

        return fallback_docs, "original_score"

    async def evaluate_reranking(self, query: str, original_docs: List[Dict],
                               ground_truth_ids: List[str]) -> Dict[str, Any]:
        """
        Evaluate the impact of re-ranking on retrieval metrics

        Returns:
            Dictionary with before/after metrics comparison
        """
        # Evaluate original ranking
        original_metrics = self.evaluator.evaluate_search(query, original_docs, set(ground_truth_ids))

        # Apply re-ranking
        reranked_docs, method = await self.rerank_with_fallback(query, original_docs)

        # Evaluate reranked results
        reranked_metrics = self.evaluator.evaluate_search(query, reranked_docs, set(ground_truth_ids))

        # Calculate improvements
        improvements = {}
        for metric in ['precision', 'recall', 'f1', 'ndcg']:
            if metric in original_metrics and metric in reranked_metrics:
                improvement = reranked_metrics[metric] - original_metrics[metric]
                improvements[f'{metric}_improvement'] = improvement
                improvements[f'{metric}_improvement_pct'] = (
                    improvement / original_metrics[metric] * 100
                    if original_metrics[metric] != 0 else 0
                )

        return {
            'original_metrics': original_metrics,
            'reranked_metrics': reranked_metrics,
            'improvements': improvements,
            'method_used': method,
            'query': query
        }

    async def batch_rerank(self, query_doc_pairs: List[Tuple[str, List[Dict]]],
                          top_k: Optional[int] = None) -> List[Tuple[List[Dict], str]]:
        """
        Batch rerank multiple query-document sets

        Args:
            query_doc_pairs: List of (query, documents) tuples
            top_k: Top k documents to keep per query

        Returns:
            List of (reranked_docs, method) tuples
        """
        tasks = [self.rerank_with_fallback(query, docs, top_k) for query, docs in query_doc_pairs]
        return await asyncio.gather(*tasks)


class MedicalCrossEncoderReranker(MedicalReranker):
    """Specialized reranker for medical domain with domain-specific features"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 medical_boost: bool = True):
        super().__init__(model_name)
        self.medical_boost = medical_boost

        # Medical domain specific terms for boosting
        self.medical_terms = {
            'safety': ['adverse', 'toxicity', 'contraindication', 'warning', 'side effect'],
            'dosing': ['dosage', 'dose', 'mg/kg', 'administration', 'frequency'],
            'indication': ['treatment', 'indication', 'therapy', 'use', 'condition'],
            'pediatric': ['pediatric', 'children', 'infant', 'adolescent', 'neonate'],
            'cardiac': ['cardiac', 'heart', 'arrhythmia', 'qt', 'torsades']
        }

    async def rerank(self, query: str, documents: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Medical-aware re-ranking with domain-specific boosting"""
        try:
            # Get base cross-encoder scores
            base_reranked = await super().rerank(query, documents, None)  # Don't apply top_k yet

            if not self.medical_boost:
                return base_reranked[:top_k] if top_k else base_reranked

            # Apply medical domain boosting
            query_lower = query.lower()
            boost_factors = self._calculate_medical_boosts(query_lower)

            for doc in base_reranked:
                base_score = doc.get('rerank_score', 0)
                boost = self._apply_medical_boost(doc, boost_factors)
                doc['rerank_score'] = base_score * (1 + boost)

            # Re-sort after boosting
            boosted_reranked = sorted(base_reranked, key=lambda x: x.get('rerank_score', 0), reverse=True)

            return boosted_reranked[:top_k] if top_k else boosted_reranked
        except Exception as e:
            print(f"❌ Error in medical re-ranking: {e}")
            # Fallback to original order
            return documents[:top_k] if top_k else documents

    def _calculate_medical_boosts(self, query: str) -> Dict[str, float]:
        """Calculate boost factors based on query content"""
        boosts = {}

        for category, terms in self.medical_terms.items():
            if any(term in query for term in terms):
                boosts[category] = 0.2  # 20% boost for matching categories

        return boosts

    def _apply_medical_boost(self, doc: Dict, boost_factors: Dict[str, float]) -> float:
        """Apply medical domain boost to document score"""
        if not boost_factors:
            return 0.0

        metadata = doc.get('metadata', {})
        text = metadata.get('text', '').lower()
        section = metadata.get('section_title', '').lower()

        total_boost = 0.0

        for category, boost in boost_factors.items():
            # Boost if category terms appear in text or section
            category_terms = self.medical_terms[category]
            if any(term in text or term in section for term in category_terms):
                total_boost += boost

        return min(total_boost, 0.5)  # Cap at 50% boost


class EnsembleReranker:
    """Ensemble reranker combining multiple re-ranking strategies"""

    def __init__(self, rerankers: List[MedicalReranker], weights: Optional[List[float]] = None):
        self.rerankers = rerankers
        self.weights = weights or [1.0 / len(rerankers)] * len(rerankers)

        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")

    async def rerank(self, query: str, documents: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Ensemble re-ranking using weighted combination"""
        if not self.rerankers:
            return documents

        # Get scores from all rerankers
        all_scores = []
        for reranker in self.rerankers:
            try:
                reranked = await reranker.rerank(query, documents, None)
                scores = [doc.get('rerank_score', 0) for doc in reranked]
                all_scores.append(scores)
            except Exception as e:
                print(f"⚠️ Reranker failed: {e}")
                all_scores.append([0] * len(documents))

        # Combine scores using weights
        combined_scores = []
        for i in range(len(documents)):
            combined_score = sum(
                scores[i] * weight
                for scores, weight in zip(all_scores, self.weights)
            )
            combined_scores.append(combined_score)

        # Apply combined scores
        for doc, score in zip(documents, combined_scores):
            doc['ensemble_score'] = score

        # Sort by ensemble score
        ensemble_reranked = sorted(documents, key=lambda x: x.get('ensemble_score', 0), reverse=True)

        return ensemble_reranked[:top_k] if top_k else ensemble_reranked


# Convenience functions
async def create_medical_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> MedicalCrossEncoderReranker:
    """Create a medical-aware cross-encoder reranker"""
    return MedicalCrossEncoderReranker(model_name)


async def quick_rerank(query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
    """Quick re-ranking with default settings"""
    reranker = MedicalCrossEncoderReranker()
    return await reranker.rerank(query, documents, top_k)


if __name__ == "__main__":
    # Example usage
    async def test_reranker():
        reranker = MedicalCrossEncoderReranker()

        # Sample documents
        documents = [
            {
                "id": "doc1",
                "metadata": {
                    "text": "Azithromycin may cause QT prolongation and torsades de pointes",
                    "section_title": "Warnings"
                },
                "score": 0.8
            },
            {
                "id": "doc2",
                "metadata": {
                    "text": "The recommended dosage is 500mg once daily",
                    "section_title": "Dosage"
                },
                "score": 0.7
            }
        ]

        query = "azithromycin cardiac toxicity"
        reranked = await reranker.rerank(query, documents)

        print("Original order:", [doc['id'] for doc in documents])
        print("Reranked order:", [doc['id'] for doc in reranked])
        print("Rerank scores:", [doc.get('rerank_score', 0) for doc in reranked])

    asyncio.run(test_reranker())