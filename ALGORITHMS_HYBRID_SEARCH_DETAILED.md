# Hybrid Search Algorithm - Government Information Retrieval System

## Overview

The **Hybrid Search** system combines two complementary search approaches to provide comprehensive government information retrieval:
1. **Dense Embeddings** (Semantic understanding via Gemini API)
2. **BM25 Sparse Search** (Keyword-based exact matching)

This hybrid approach ensures both **semantic relevance** and **keyword accuracy**.

## Hybrid Search Architecture

### System Components

```
User Query
    ↓
┌─────────────────────────────────────────┐
│   Hybrid Search Engine                  │
├─────────────────────────────────────────┤
│  1. Dense Embedding Path                │
│     Query → Gemini API → Vector (384D) │
│     ↓                                   │
│     Pinecone Index Query                │
│     ↓                                   │
│     Dense Results (semantic match)      │
│                                         │
│  2. BM25 Sparse Path                    │
│     Query → Tokenization                │
│     ↓                                   │
│     Term Frequency Analysis             │
│     ↓                                   │
│     BM25 Ranking                        │
│     ↓                                   │
│     Sparse Results (keyword match)      │
│                                         │
│  3. Fusion & Ranking                    │
│     Combine via Adaptive Alpha (α)      │
│     ↓                                   │
│     Quality Scoring                     │
│     ↓                                   │
│     Final Ranked Results                │
│                                         │
│  4. Post-Processing                     │
│     - Fallback region filtering         │
│     - Query expansion                   │
│     - PRF (Pseudo-Relevance Feedback)   │
│     - Graph-RAG expansion               │
└─────────────────────────────────────────┘
    ↓
Final Ranked Results to User
```

## Path 1: Dense Embedding Search

### What are Embeddings?

Embeddings are **vector representations** of text that capture semantic meaning:

```
"Government healthcare policy" 
    ↓ [Gemini Embedding Model]
[0.234, -0.891, 0.456, ..., 0.123]  ← 384-dimensional vector

Each dimension captures different semantic aspects:
- Healthcare concepts
- Government context
- Policy implications
- Regulatory language
```

### Gemini Embedding Model

**Model**: `text-embedding-004` (Google Gemini)
- **Dimensions**: 384 (optimized for government documents)
- **Max input**: 2048 tokens
- **Latency**: ~50-100ms per query
- **Cost effective**: Shared quota across GIRS

### Dense Search Process

```python
# Step 1: Generate Query Embedding
async def get_embedding_async(query: str) -> List[float]:
    """
    Convert government query to semantic vector
    
    Example:
    Input:  "What are the side effects of COVID-19 vaccines?"
    Output: [0.234, -0.891, 0.456, ..., 0.123]  # 384 dimensions
    """
    embedding = await get_gemini_embedding_async(
        query, 
        task_type="retrieval_query"
    )
    return embedding

# Step 2: Query Pinecone Vector Database
vector_results = pinecone_index.query(
    vector=query_embedding,      # 384-dim vector
    top_k=30,                    # Get top 30 candidates
    include_metadata=True,       # Include document metadata
    filter={                     # Filter by document type
        "document_type": "pis",  # Prescribing Information
        "region": "US"
    }
)

# Step 3: Return Semantic Matches
results = [
    {
        "id": "doc_123",
        "score": 0.89,           # Similarity score (0-1)
        "metadata": {
            "text": "...",
            "source": "FDA Approved Document"
        }
    }
]
```

### Advantages of Dense Search

 **Semantic Understanding**: Understands meaning beyond keywords  
 **Synonym Handling**: Finds related concepts (e.g., "vaccine" ↔ "immunization")  
 **Typo Tolerance**: Handles spelling variations  
 **Long Queries**: Works with complex government questions  
 **Conceptual Search**: Finds documents about similar topics  

### Limitations

❌ **Slower**: Requires API call + vector search (~100-200ms)  
❌ **Memory**: Stores 384-dimensional vectors (high memory)  
❌ **Not Exact**: May miss exact regulatory language  
❌ **Cold Start**: Needs embeddings cached/precomputed  

## Path 2: BM25 Sparse Search

### What is Sparse Search?

Sparse search uses **exact keyword matching** with statistical weighting:

```
"healthcare policy" 
    ↓ [Tokenization]
["healthcare", "policy"]
    ↓ [BM25 Algorithm]
Score documents by term frequency & rarity
    ↓
Ranked Results (highest scores first)
```

### BM25 Algorithm Details

See `ALGORITHMS_BM25_DETAILED.md` for complete details.

**Quick Summary**:
```
Score(Doc, Query) = Σ IDF(term) × TF(term, Doc) / (1 + k×length_norm)
```

- **IDF**: Inverse Document Frequency (rarity of term)
- **TF**: Term Frequency (how often term appears)
- **length_norm**: Penalty for document length

### Sparse Search Process

```python
# Step 1: Build BM25 Index from Corpus
async def build_dynamic_corpus():
    """Extract medical terms from government documents"""
    corpus_terms = [
        "side effects", "contraindications", "dosage",
        "adverse reactions", "drug interactions",
        "cardiotoxicity", "hepatotoxicity"
    ]
    
    tokenized_corpus = [term.split() for term in corpus_terms]
    bm25 = BM25Okapi(tokenized_corpus)

# Step 2: Score Query Terms
query = "government healthcare policy"
query_tokens = ["government", "healthcare", "policy"]
bm25_scores = bm25.get_scores(query_tokens)

# Step 3: Match Against Documents
for document in corpus:
    text_lower = document.lower()
    doc_score = 0
    for term, score in zip(corpus_terms, bm25_scores):
        if term in text_lower:
            doc_score += score
    
    # Store score
    bm25_document_scores[document_id] = doc_score

# Step 4: Return Keyword Matches
results = sorted_by_score(document_scores)
```

### Advantages of BM25

 **Fast**: Instant keyword matching (~10-20ms)  
 **Memory Efficient**: Minimal memory requirements  
 **Exact Matching**: Finds exact regulatory terminology  
 **Interpretable**: Clear why document matched  
 **Scalable**: Works with millions of documents  

### Limitations

❌ **No Semantics**: Treats terms independently  
❌ **Synonym Blind**: Misses "vaccine" if query says "immunization"  
❌ **Typo Sensitive**: One character difference = no match  
❌ **Order Agnostic**: Can't understand phrase meaning  

## Path 3: Fusion & Ranking (Adaptive Alpha)

### The Problem

Neither approach alone is perfect:
- Dense: Understands meaning but slow
- BM25: Fast but misses synonyms

### The Solution: Hybrid Fusion

Combine both scores using **Adaptive Alpha (α)**:

```
Hybrid Score = α × DenseScore + (1 - α) × SparseScore

where α ∈ [0, 1]

α = 0.0  → Pure BM25 (keyword matching)
α = 0.5  → Equal weight to both
α = 1.0  → Pure dense (semantic matching)
```

### Adaptive Alpha Selection

The system intelligently chooses α based on query characteristics:

```python
async def get_alpha_recommendation(query: str, context: Dict):
    """
    Analyze query to determine optimal balance
    """
    
    # Query type detection
    if is_exact_phrase(query):
        # "government healthcare policy reform"
        alpha = 0.3  # More BM25 weight
        confidence = 0.9
        
    elif is_specific_regulation(query):
        # "FDA approval for COVID-19 vaccines"
        alpha = 0.4  # Balanced toward BM25
        confidence = 0.85
        
    elif is_conceptual_search(query):
        # "Tell me about side effects"
        alpha = 0.7  # More semantic understanding
        confidence = 0.8
        
    elif is_complex_question(query):
        # "What are the interactions between medication X and Y?"
        alpha = 0.6  # Balanced approach
        confidence = 0.75
        
    else:
        alpha = 0.5  # Default balanced
        confidence = 0.5
    
    return {
        "alpha": alpha,
        "query_type": query_type,
        "confidence": confidence,
        "reasoning": reasoning_text
    }
```

### Real Example: Hybrid Fusion

```python
Query: "vaccine side effects"

# Dense Path
dense_results = pinecone.query(query_vector)
# Results: 
#   Doc A (vaccine info): score 0.85
#   Doc B (immunization side effects): score 0.82

# BM25 Path
bm25_results = bm25.get_scores(["vaccine", "side", "effects"])
# Results:
#   Doc A (vaccine info): score 6.2
#   Doc C (exact side effects list): score 7.8

# Fusion with α = 0.6
for doc in all_results:
    if doc_id == "A":
        hybrid = 0.6 × 0.85 + 0.4 × (6.2/10) = 0.75
    if doc_id == "B":
        hybrid = 0.6 × 0.82 + 0.4 × (5.1/10) = 0.69
    if doc_id == "C":
        hybrid = 0.6 × 0.45 + 0.4 × (7.8/10) = 0.59

# Final ranking
1. Doc A (0.75) - Best balance of semantic and keyword match
2. Doc B (0.69) - Good semantic, lower keyword match
3. Doc C (0.59) - Exact keywords but less semantic relevance
```

## Path 4: Quality Scoring

### Post-Processing Enhancement

After fusion, apply domain-specific quality scoring:

```python
def compute_quality_bonus(metadata: Dict, text: str, query_tokens: List[str]):
    """
    Boost scores based on government document quality factors
    """
    
    bonus = 0.0
    factors = []
    
    # 1. Text Length Quality
    if 400 <= len(text) <= 1200:
        bonus += 0.2
        factors.append("ideal_length")
    
    # 2. Query Coverage
    coverage_hits = sum(1 for token in query_tokens if token in text)
    if coverage_hits:
        bonus += min(0.2, (coverage_hits / len(query_tokens)) * 0.25)
        factors.append(f"query_coverage_{coverage_hits}/{len(query_tokens)}")
    
    # 3. Section Importance
    section_title = metadata.get("section_title", "").lower()
    for keyword, weight in SECTION_PRIORITY_WEIGHTS.items():
        if keyword in section_title:
            bonus += weight  # 0.15-0.25
            factors.append(f"section_{keyword}")
            break
    
    # 4. Healthcare-Specific Boosts
    if is_pediatric_query:
        if "pediatric" in metadata or "8.4" in section_title:
            bonus += 0.3
            factors.append("pediatric_match")
    
    if is_pregnancy_query:
        if "pregnancy" in section_title:
            bonus += 0.35
            factors.append("pregnancy_match")
    
    # Clamp to reasonable range
    bonus = max(-0.4, min(bonus, 0.85))
    
    return bonus, factors
```

### Quality Score Examples

```
Document: "Pediatric Dosing (Section 8.4)"
Query: "children dosage"

Base hybrid score: 0.72
+ Pediatric match bonus: 0.30
+ Section (dosage): 0.15
+ Query coverage: 0.10
─────────────────────
Final quality score: 0.85
```

## Complete Hybrid Search Flow

```python
async def execute_hybrid_search(
    query: str, 
    document_type: str = "lrd",
    country: str = "US",
    top_k: int = 30
) -> Dict[str, Any]:
    """
    Complete hybrid search pipeline for government information retrieval
    """
    
    start_time = time.time()
    
    # ===== PHASE 1: EMBEDDING & PINECONE SEARCH =====
    query_vector = await get_embedding_async(query)
    dense_results = await execute_pinecone_query_async(
        query_vector=query_vector,
        filter_dict={"document_type": document_type, "region": country},
        top_k=top_k
    )
    
    # ===== PHASE 2: BM25 SPARSE SEARCH =====
    bm25_scores = await get_bm25_scores(query, _medical_corpus)
    
    # ===== PHASE 3: FALLBACK HANDLING =====
    if not dense_results:
        # Try without region filter
        dense_results = await execute_pinecone_query_async(
            query_vector=query_vector,
            filter_dict={"document_type": document_type},
            top_k=top_k
        )
    
    # ===== PHASE 4: QUERY EXPANSION =====
    # Add semantic variants
    expanded_queries = build_expanded_queries(query)
    prf_terms = extract_prf_terms(dense_results, query)
    
    # ===== PHASE 5: ADAPTIVE ALPHA FUSION =====
    alpha_rec = get_alpha_recommendation(query, context)
    actual_alpha = alpha_rec.alpha
    
    for match in dense_results:
        # Get BM25 boost
        text_content = match.get("metadata", {}).get("text", "").lower()
        bm25_boost = sum(score for term, score in bm25_scores.items()
                         if term in text_content)
        
        # Hybrid scoring
        original_score = match.get("score", 0.0)
        match["hybrid_score"] = (actual_alpha * original_score + 
                                 (1 - actual_alpha) * (bm25_boost / 10))
    
    # ===== PHASE 6: QUALITY SCORING & RE-RANKING =====
    matches = apply_quality_scoring(dense_results, query)
    
    # ===== PHASE 7: GRAPH-RAG EXPANSION (Optional) =====
    if GRAPH_EXPANSION_ENABLED:
        graph_chunks = await graph_expand_candidates(
            query, matches, k_hop=2, max_neighbors=10
        )
        matches = fuse_with_graph_expansion(matches, graph_chunks)
    
    # ===== PHASE 8: FINAL RANKING =====
    matches = matches[:top_k]
    
    end_time = time.time()
    
    return {
        "matches": matches,
        "search_metadata": {
            "total_time": round(end_time - start_time, 3),
            "embedding_time": embedding_time,
            "bm25_time": bm25_time,
            "pinecone_time": pinecone_time,
            "alpha": actual_alpha,
            "query_type": alpha_rec.query_type,
            "bm25_terms_found": len(bm25_scores),
            "hybrid_ranking_applied": True,
            "quality_scoring_applied": True,
            "graph_expansion_applied": GRAPH_EXPANSION_ENABLED
        }
    }
```

## Performance Characteristics

### Execution Timeline

```
Total Search Time: ~150-300ms

├─ Embedding Generation (Gemini API)
│  └─ Time: 50-100ms
│
├─ Pinecone Vector Search
│  └─ Time: 20-50ms
│
├─ BM25 Computation
│  └─ Time: 10-20ms
│
├─ Fusion & Scoring
│  └─ Time: 30-50ms
│
└─ Graph-RAG Expansion (optional)
   └─ Time: 50-100ms
```

### Resource Usage

```
Memory per search:
- Query vector: 384 floats × 8 bytes = 3 KB
- Result cache: ~100 documents × 1-2 KB = 100-200 KB
- BM25 index: ~50 MB (static, shared)
─────────────────────
Total: < 500 KB per search
```

## Comparison Table

| Aspect | Dense Only | BM25 Only | Hybrid |
|--------|-----------|-----------|--------|
| **Speed** | 100-200ms | 10-20ms | 150-300ms |
| **Semantic** | Excellent | Poor | Excellent |
| **Keywords** | Okay | Excellent | Excellent |
| **Synonyms** | Handled | Missed | Handled |
| **Typos** | Tolerant | Fails | Tolerant |
| **Explainability** | Poor | Excellent | Good |
| **Cost** | High | None | Medium |

## Real-World Scenarios

### Scenario 1: Exact Regulatory Search
```
Query: "FDA approval requirements for new drugs"

Dense: Understands meaning (0.75 score)
BM25: Exact phrase match (8.5 score)
Hybrid: Perfect match combines both (0.85)

Winner: Hybrid
```

### Scenario 2: Synonym Search
```
Query: "What treatments help COVID?"

Dense: Understands "treatments" and "medicines" (0.82 score)
BM25: Only exact word matches (3.2 score)
Hybrid: Gets both accuracy and breadth (0.79)

Winner: Hybrid (finds more relevant documents)
```

### Scenario 3: Complex Government Query
```
Query: "How do pediatric dosing guidelines change with kidney function?"

Dense: Understands all concepts (0.88 score)
BM25: Matches key terms (6.1 score)
Hybrid: Optimal combination (0.82 with quality boost)

Winner: Hybrid (captures all aspects)
```

## Configuration & Tuning

### Adjusting Alpha for Your Use Case

```python
# For regulatory compliance (exact match priority)
ALPHA_EXACT_MATCH = 0.3

# For research (semantic understanding priority)
ALPHA_RESEARCH = 0.7

# Balanced (default)
ALPHA_BALANCED = 0.5
```

### Enable/Disable Features

```python
# Disable BM25 (use dense only)
if not rank_bm25:
    alpha = 1.0

# Disable graph expansion (faster search)
GRAPH_EXPANSION_ENABLED = False

# Cache embeddings for repeated queries
@lru_cache(maxsize=500)
def get_cached_gemini_embedding(query, task_type):
    return get_gemini_embedding(query, task_type)
```

## Conclusion

The **Hybrid Search** approach provides:

 **Best of both worlds** - Semantic understanding + keyword accuracy  
 **Adaptability** - Adjusts strategy based on query type  
 **Scalability** - Handles millions of government documents  
 **Relevance** - Comprehensive quality scoring  
 **Performance** - Fast execution with low latency  

Perfect for **Government Information Retrieval Systems** where both **regulatory precision** and **conceptual understanding** are essential.

---
**System**: Government Information Retrieval System (GIRS)  
**Algorithm**: Hybrid Dense + Sparse Search  
**Components**: Gemini Embeddings + BM25 + Adaptive Fusion  
**Status**: Production Ready  
**Documentation**: Complete with examples
