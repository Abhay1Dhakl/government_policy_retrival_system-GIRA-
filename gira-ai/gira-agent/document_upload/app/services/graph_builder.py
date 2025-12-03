"""
Graph RAG Implementation for GIRA AI
Enhanced document retrieval using entity relationships and graph-based expansion.
"""

import asyncio
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import networkx as nx
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityRelationship:
    """Represents a relationship between two entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str
    document_id: str

@dataclass
class KnowledgeGraph:
    """Knowledge graph for entity relationships"""
    graph: nx.DiGraph
    entity_to_chunks: Dict[str, Set[str]]  # entity -> chunk_ids
    chunk_to_entities: Dict[str, Set[str]]  # chunk_id -> entities

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_to_chunks = defaultdict(set)
        self.chunk_to_entities = defaultdict(set)

class GraphBuilder:
    """Builds and manages the knowledge graph for Graph RAG"""

    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def extract_entities_from_chunk(self, chunk_text: str, chunk_id: str) -> Set[str]:
        """Extract medical entities from chunk text"""
        entities = set()

        # Medical entity patterns
        patterns = [
            # Drug names
            r'\b[A-Z][a-z]+(?:cillin|mycin|floxacin|prazole|sartan|statin|ide|ine|ole|ate|ium)\b',
            # Medical conditions
            r'\b\w*(?:itis|osis|emia|pathy|trophy|plasia|sclerosis|stenosis|megaly|algia|dynia)\b',
            # Body systems
            r'\b(?:cardiovascular|respiratory|hepatic|renal|neurological|dermatological|gastrointestinal)\b',
            # Specific medical terms
            r'\b(?:azithromycin|antibiotics?|infection|therapy|treatment|dosage|side effects?|adverse|reaction)\b',
            # Pregnancy related
            r'\b(?:pregnancy|pregnant|fetal|teratogenic|developmental|birth defect|congenital)\b',
            # Cardiac terms
            r'\b(?:cardiac|arrhythmia|QT|torsades|ventricular|atrial|bradycardia|tachycardia)\b'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, chunk_text, re.IGNORECASE)
            entities.update(match.lower() for match in matches)

        # Update mappings
        for entity in entities:
            self.knowledge_graph.entity_to_chunks[entity].add(chunk_id)
            self.knowledge_graph.chunk_to_entities[chunk_id].add(entity)

        return entities

    def build_relationships(self, chunk_text: str, entities: Set[str], chunk_id: str):
        """Build relationships between entities in a chunk"""
        relationships = []

        # Define relationship patterns
        relationship_patterns = [
            ("treats", r"(\w+)\s+(?:treats?|therapy|treatment)\s+(\w+)"),
            ("causes", r"(\w+)\s+(?:causes?|leads? to|results? in)\s+(\w+)"),
            ("prevents", r"(\w+)\s+(?:prevents?|reduces?|decreases?)\s+(\w+)"),
            ("increases", r"(\w+)\s+(?:increases?|raises?|elevates?)\s+(\w+)"),
            ("decreases", r"(\w+)\s+(?:decreases?|lowers?|reduces?)\s+(\w+)"),
            ("associated_with", r"(\w+)\s+(?:associated|related|linked)\s+(?:with|to)\s+(\w+)"),
            ("contraindicated", r"(\w+)\s+(?:contraindicated|not recommended)\s+(?:in|for|with)\s+(\w+)"),
        ]

        for rel_type, pattern in relationship_patterns:
            matches = re.findall(pattern, chunk_text, re.IGNORECASE)
            for match in matches:
                source, target = match
                source = source.lower()
                target = target.lower()

                if source in entities and target in entities:
                    relationship = EntityRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type=rel_type,
                        confidence=0.8,  # Base confidence
                        context=chunk_text[:200],
                        document_id=chunk_id
                    )
                    relationships.append(relationship)

        # Add relationships to graph
        for rel in relationships:
            self.knowledge_graph.graph.add_edge(
                rel.source_entity,
                rel.target_entity,
                relationship=rel.relationship_type,
                confidence=rel.confidence,
                context=rel.context,
                chunk_id=rel.document_id
            )

    def add_chunk_to_graph(self, chunk_text: str, chunk_id: str):
        """Add a chunk to the knowledge graph"""
        entities = self.extract_entities_from_chunk(chunk_text, chunk_id)
        if entities:
            self.build_relationships(chunk_text, entities, chunk_id)

    def find_related_entities(self, query_entities: Set[str], max_hops: int = 2) -> Set[str]:
        """Find entities related to query entities within max_hops"""
        visited = set()
        to_visit = list(query_entities)
        related_entities = set()

        for hop in range(max_hops):
            next_visit = []
            for entity in to_visit:
                if entity in visited:
                    continue
                visited.add(entity)

                # Get neighbors (both incoming and outgoing)
                neighbors = set()
                if entity in self.knowledge_graph.graph:
                    neighbors.update(self.knowledge_graph.graph.successors(entity))
                    neighbors.update(self.knowledge_graph.graph.predecessors(entity))

                for neighbor in neighbors:
                    if neighbor not in visited:
                        related_entities.add(neighbor)
                        next_visit.append(neighbor)

            to_visit = next_visit

        return related_entities

    def get_chunks_for_entities(self, entities: Set[str]) -> Set[str]:
        """Get all chunk IDs that contain the given entities"""
        chunk_ids = set()
        for entity in entities:
            chunk_ids.update(self.knowledge_graph.entity_to_chunks.get(entity, set()))
        return chunk_ids

class GraphRAGExpander:
    """Expands search results using Graph RAG"""

    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    async def expand_candidates(
        self,
        query: str,
        seed_chunks: List[Dict[str, Any]],
        k_hop: int = 2,
        max_neighbors: int = 10,
        graph_boost: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Expand seed chunks using graph relationships"""

        # Extract entities from query
        query_entities = self.graph_builder.extract_entities_from_chunk(query, "query")

        # Find related entities
        related_entities = self.graph_builder.find_related_entities(query_entities, max_hops=k_hop)

        # Get chunks containing related entities
        related_chunk_ids = self.graph_builder.get_chunks_for_entities(related_entities)

        # Create expanded chunks (this would need actual chunk data from database)
        expanded_chunks = []

        # For now, create placeholder chunks with graph relationships
        for i, chunk_id in enumerate(list(related_chunk_ids)[:max_neighbors]):
            expanded_chunk = {
                "id": f"graph_expanded_{chunk_id}",
                "score": 0.0,  # Will be set by fusion logic
                "metadata": {
                    "text": f"Graph-expanded content related to: {', '.join(list(related_entities)[:3])}",
                    "document_type": "graph_expanded",
                    "source": "graph_rag",
                    "graph_boost": graph_boost,
                    "related_entities": list(related_entities),
                    "original_chunk_id": chunk_id
                }
            }
            expanded_chunks.append(expanded_chunk)

        return expanded_chunks

def get_knowledge_graph() -> GraphBuilder:
    """Get or create the global knowledge graph instance"""
    if not hasattr(get_knowledge_graph, '_instance'):
        get_knowledge_graph._instance = GraphBuilder()
    return get_knowledge_graph._instance

def get_entity_extractor():
    """Get entity extractor (placeholder for now)"""
    return None

# Graph RAG expansion functions for MCP server integration
async def graph_expand_candidates(
    query: str,
    seed_chunks: List[Dict[str, Any]],
    k_hop: int = 2,
    max_neighbors: int = 10,
    graph_boost: float = 0.15
) -> List[Dict[str, Any]]:
    """Expand search candidates using Graph RAG"""
    try:
        graph_builder = get_knowledge_graph()
        expander = GraphRAGExpander(graph_builder)

        # Add seed chunks to graph if not already present
        for chunk in seed_chunks:
            chunk_id = chunk.get("id", "")
            chunk_text = chunk.get("metadata", {}).get("text", "")
            if chunk_id and chunk_text:
                graph_builder.add_chunk_to_graph(chunk_text, chunk_id)

        # Expand using graph relationships
        expanded_chunks = await expander.expand_candidates(
            query, seed_chunks, k_hop, max_neighbors, graph_boost
        )

        return expanded_chunks

    except Exception as e:
        logger.error(f"Graph expansion failed: {e}")
        return []

def fuse_with_graph_expansion(
    original_chunks: List[Dict[str, Any]],
    graph_chunks: List[Dict[str, Any]],
    graph_boost: float = 0.15
) -> List[Dict[str, Any]]:
    """Fuse original chunks with graph-expanded chunks"""

    # Create lookup for original chunks
    chunk_lookup = {chunk["id"]: chunk for chunk in original_chunks}

    # Add graph chunks and apply boosting
    for graph_chunk in graph_chunks:
        chunk_id = graph_chunk["id"]
        if chunk_id not in chunk_lookup:
            # Apply graph boost to score
            graph_chunk["score"] = graph_boost
            graph_chunk["graph_score"] = graph_boost
            chunk_lookup[chunk_id] = graph_chunk

    # Return combined results sorted by score
    fused_chunks = list(chunk_lookup.values())
    fused_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return fused_chunks
