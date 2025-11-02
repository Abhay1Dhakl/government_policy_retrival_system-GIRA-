"""
Hard Negative Mining for MIRA AI
Actively finds hard negatives to improve retrieval model training and evaluation
"""

import asyncio
import random
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
import numpy as np

from evaluation import MedicalRetrievalEvaluator


class HardNegativeMiner:
    """Mines hard negative examples for improved retrieval training"""

    def __init__(self, evaluator: Optional[MedicalRetrievalEvaluator] = None):
        self.evaluator = evaluator or MedicalRetrievalEvaluator()
        self.hard_negatives_cache = {}
        self.mining_stats = defaultdict(int)

    async def mine_hard_negatives(self, query: str, positive_docs: List[Dict],
                                search_function: callable, k: int = 5,
                                strategy: str = "diverse") -> List[Dict]:
        """
        Mine hard negative examples for a query

        Args:
            query: The search query
            positive_docs: Documents that are relevant (ground truth positives)
            search_function: Function to perform search (should return dict with 'matches')
            k: Number of hard negatives to mine
            strategy: Mining strategy ('diverse', 'semantically_similar', 'boundary')

        Returns:
            List of hard negative documents
        """
        if not positive_docs:
            return []

        positive_ids = {doc.get('id') for doc in positive_docs if doc.get('id')}

        # Get candidate documents (highly ranked but not relevant)
        candidates = await self._get_candidate_negatives(query, search_function, positive_ids)

        if not candidates:
            return []

        # Apply mining strategy
        if strategy == "diverse":
            hard_negatives = self._mine_diverse_negatives(candidates, positive_docs, k)
        elif strategy == "semantically_similar":
            hard_negatives = self._mine_semantically_similar_negatives(candidates, positive_docs, k)
        elif strategy == "boundary":
            hard_negatives = self._mine_boundary_negatives(candidates, positive_docs, k)
        else:
            hard_negatives = candidates[:k]

        # Update mining statistics
        self.mining_stats['total_mined'] += len(hard_negatives)
        self.mining_stats[f'strategy_{strategy}'] += len(hard_negatives)

        return hard_negatives

    async def _get_candidate_negatives(self, query: str, search_function: callable,
                                     positive_ids: Set[str], top_k: int = 50) -> List[Dict]:
        """Get candidate negative documents from search results"""
        try:
            # Perform search with higher top_k to get more candidates
            result = await search_function(query, top_k=top_k)
            matches = result.get('matches', [])

            # Filter out positive documents and get high-scoring negatives
            candidates = []
            for doc in matches:
                doc_id = doc.get('id')
                if doc_id and doc_id not in positive_ids:
                    score = doc.get('score', 0) or doc.get('hybrid_score', 0)
                    if score > 0.1:  # Only consider reasonably high-scoring documents
                        candidates.append(doc)

            return candidates[:20]  # Limit candidates for efficiency

        except Exception as e:
            print(f"❌ Error getting candidate negatives: {e}")
            return []

    def _mine_diverse_negatives(self, candidates: List[Dict], positive_docs: List[Dict], k: int) -> List[Dict]:
        """Mine diverse hard negatives from different document types/sections"""
        if len(candidates) <= k:
            return candidates

        # Group candidates by document type and section
        type_groups = defaultdict(list)
        section_groups = defaultdict(list)

        for doc in candidates:
            metadata = doc.get('metadata', {})

            # Group by document type
            doc_type = metadata.get('document_type', 'unknown')
            type_groups[doc_type].append(doc)

            # Group by section
            section = metadata.get('section_title', '').lower()
            # Simplify section names
            if 'warning' in section or 'adverse' in section:
                section_key = 'safety'
            elif 'dosage' in section or 'administration' in section:
                section_key = 'dosing'
            elif 'contraindication' in section:
                section_key = 'contraindication'
            else:
                section_key = 'other'
            section_groups[section_key].append(doc)

        # Select diverse negatives
        selected = []
        remaining = k

        # First, ensure representation from different document types
        for doc_type, docs in type_groups.items():
            if remaining <= 0:
                break
            take = min(remaining, max(1, len(docs) // 2))  # Take up to half from each type
            selected.extend(docs[:take])
            remaining -= take

        # Then fill with different sections if still need more
        if remaining > 0:
            for section, docs in section_groups.items():
                if remaining <= 0:
                    break
                # Filter out already selected
                available = [d for d in docs if d not in selected]
                take = min(remaining, len(available))
                selected.extend(available[:take])
                remaining -= take

        return selected[:k]

    def _mine_semantically_similar_negatives(self, candidates: List[Dict], positive_docs: List[Dict], k: int) -> List[Dict]:
        """Mine negatives that are semantically similar but not relevant"""
        if not candidates or not positive_docs:
            return candidates[:k]

        # Calculate similarity scores between candidates and positives
        similarities = []

        for candidate in candidates:
            candidate_text = self._extract_text(candidate)
            max_similarity = 0

            for positive in positive_docs:
                positive_text = self._extract_text(positive)
                similarity = self._calculate_text_similarity(candidate_text, positive_text)
                max_similarity = max(max_similarity, similarity)

            similarities.append((candidate, max_similarity))

        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        hard_negatives = [doc for doc, _ in similarities[:k]]

        return hard_negatives

    def _mine_boundary_negatives(self, candidates: List[Dict], positive_docs: List[Dict], k: int) -> List[Dict]:
        """Mine negatives that are on the boundary of relevance"""
        if not candidates:
            return []

        # Sort candidates by score (assuming higher score = more similar)
        scored_candidates = []
        for doc in candidates:
            score = doc.get('score', 0) or doc.get('hybrid_score', 0)
            scored_candidates.append((doc, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take candidates from the "boundary" region (middle of ranking)
        total_candidates = len(scored_candidates)
        if total_candidates <= k:
            return [doc for doc, _ in scored_candidates]

        # Select from middle third of rankings (boundary region)
        start_idx = total_candidates // 3
        end_idx = 2 * total_candidates // 3

        boundary_candidates = scored_candidates[start_idx:end_idx]
        selected = [doc for doc, _ in boundary_candidates[:k]]

        return selected

    def _extract_text(self, doc: Dict) -> str:
        """Extract text content from document"""
        metadata = doc.get('metadata', {})
        return metadata.get('text', '').lower()

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity of words)"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def mine_hard_negatives_batch(self, queries_and_positives: List[Tuple[str, List[Dict]]],
                                      search_function: callable, k: int = 5) -> List[List[Dict]]:
        """
        Mine hard negatives for multiple queries in batch

        Args:
            queries_and_positives: List of (query, positive_docs) tuples
            search_function: Search function to use
            k: Number of hard negatives per query

        Returns:
            List of hard negative lists, one per query
        """
        tasks = [
            self.mine_hard_negatives(query, positives, search_function, k)
            for query, positives in queries_and_positives
        ]

        results = await asyncio.gather(*tasks)
        return results

    def evaluate_hard_negatives_quality(self, query: str, positive_docs: List[Dict],
                                      hard_negatives: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the quality of mined hard negatives

        Returns:
            Dictionary with quality metrics
        """
        if not hard_negatives:
            return {'quality_score': 0.0, 'error': 'No hard negatives provided'}

        # Calculate average score of hard negatives (should be reasonably high)
        scores = []
        for doc in hard_negatives:
            score = doc.get('score', 0) or doc.get('hybrid_score', 0)
            scores.append(score)

        avg_score = np.mean(scores) if scores else 0

        # Calculate diversity (different document types)
        doc_types = set()
        sections = set()
        for doc in hard_negatives:
            metadata = doc.get('metadata', {})
            doc_types.add(metadata.get('document_type', 'unknown'))
            sections.add(metadata.get('section_title', '').lower()[:20])  # First 20 chars

        type_diversity = len(doc_types)
        section_diversity = len(sections)

        # Calculate semantic similarity to positives
        similarities = []
        for neg in hard_negatives:
            neg_text = self._extract_text(neg)
            max_sim = 0
            for pos in positive_docs:
                pos_text = self._extract_text(pos)
                sim = self._calculate_text_similarity(neg_text, pos_text)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        # Quality score combines multiple factors
        quality_score = (
            0.3 * min(avg_score / 0.5, 1.0) +  # Score should be reasonably high but not too high
            0.3 * min(type_diversity / 3, 1.0) +  # Diversity in document types
            0.2 * min(section_diversity / 5, 1.0) +  # Diversity in sections
            0.2 * min(avg_similarity, 1.0)  # Should be similar but not identical
        )

        return {
            'quality_score': quality_score,
            'avg_score': avg_score,
            'type_diversity': type_diversity,
            'section_diversity': section_diversity,
            'avg_similarity_to_positives': avg_similarity,
            'num_hard_negatives': len(hard_negatives)
        }

    def get_mining_statistics(self) -> Dict[str, Any]:
        """Get statistics about hard negative mining performance"""
        return dict(self.mining_stats)

    async def adaptive_mining(self, query: str, positive_docs: List[Dict],
                            search_function: callable, target_quality: float = 0.7) -> List[Dict]:
        """
        Adaptively mine hard negatives until target quality is reached

        Args:
            query: Search query
            positive_docs: Positive documents
            search_function: Search function
            target_quality: Target quality score (0-1)

        Returns:
            High-quality hard negatives
        """
        strategies = ['diverse', 'semantically_similar', 'boundary']
        k = 5

        for strategy in strategies:
            hard_negatives = await self.mine_hard_negatives(
                query, positive_docs, search_function, k, strategy
            )

            if hard_negatives:
                quality = self.evaluate_hard_negatives_quality(query, positive_docs, hard_negatives)
                if quality['quality_score'] >= target_quality:
                    return hard_negatives

                # Try with more negatives if quality is low
                if quality['quality_score'] < 0.5:
                    k = min(k + 3, 10)
                    continue

        # Return best effort if target not reached
        return await self.mine_hard_negatives(query, positive_docs, search_function, k, 'diverse')


class TripletDataGenerator:
    """Generates triplet training data (query, positive, hard_negative)"""

    def __init__(self, hard_negative_miner: HardNegativeMiner):
        self.miner = hard_negative_miner

    async def generate_triplets(self, queries_and_positives: List[Tuple[str, List[Dict]]],
                              search_function: callable) -> List[Dict[str, Any]]:
        """
        Generate triplet training data for improved retrieval models

        Args:
            queries_and_positives: List of (query, positive_docs) tuples
            search_function: Search function to use for mining negatives

        Returns:
            List of triplet dictionaries
        """
        triplets = []

        for query, positive_docs in queries_and_positives:
            if not positive_docs:
                continue

            # Mine hard negatives
            hard_negatives = await self.miner.adaptive_mining(
                query, positive_docs, search_function, target_quality=0.6
            )

            # Create triplets
            for positive in positive_docs:
                for hard_negative in hard_negatives[:3]:  # Limit to 3 negatives per positive
                    triplet = {
                        'query': query,
                        'positive': {
                            'id': positive.get('id'),
                            'text': self.miner._extract_text(positive),
                            'metadata': positive.get('metadata', {})
                        },
                        'hard_negative': {
                            'id': hard_negative.get('id'),
                            'text': self.miner._extract_text(hard_negative),
                            'metadata': hard_negative.get('metadata', {})
                        },
                        'query_type': 'medical_retrieval'
                    }
                    triplets.append(triplet)

        return triplets

    def save_triplets(self, triplets: List[Dict[str, Any]], filepath: str):
        """Save triplets to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(triplets, f, indent=2)


# Convenience functions
async def mine_hard_negatives_for_query(query: str, positive_docs: List[Dict],
                                      search_function: callable, k: int = 5) -> List[Dict]:
    """Convenience function for mining hard negatives"""
    miner = HardNegativeMiner()
    return await miner.mine_hard_negatives(query, positive_docs, search_function, k)


async def generate_training_triplets(queries_and_positives: List[Tuple[str, List[Dict]]],
                                   search_function: callable) -> List[Dict[str, Any]]:
    """Convenience function for generating training triplets"""
    miner = HardNegativeMiner()
    generator = TripletDataGenerator(miner)
    return await generator.generate_triplets(queries_and_positives, search_function)


if __name__ == "__main__":
    # Example usage
    async def example_mining():
        miner = HardNegativeMiner()

        # Mock search function
        async def mock_search(query: str, top_k: int = 10) -> Dict[str, Any]:
            # Simulate search results
            return {
                'matches': [
                    {'id': f'doc_{i}', 'score': 0.9 - i * 0.1, 'metadata': {'text': f'Content {i}', 'document_type': 'pis'}}
                    for i in range(top_k)
                ]
            }

        # Example data
        query = "azithromycin side effects"
        positive_docs = [
            {'id': 'doc_1', 'metadata': {'text': 'Azithromycin may cause nausea and vomiting'}},
            {'id': 'doc_2', 'metadata': {'text': 'Side effects include diarrhea and abdominal pain'}}
        ]

        # Mine hard negatives
        hard_negatives = await miner.mine_hard_negatives(query, positive_docs, mock_search, k=3)

        print(f"Mined {len(hard_negatives)} hard negatives for query: {query}")
        for i, neg in enumerate(hard_negatives):
            print(f"  {i+1}. {neg.get('id')} (score: {neg.get('score', 0):.2f})")

        # Evaluate quality
        quality = miner.evaluate_hard_negatives_quality(query, positive_docs, hard_negatives)
        print(f"Quality score: {quality['quality_score']:.2f}")

    asyncio.run(example_mining())