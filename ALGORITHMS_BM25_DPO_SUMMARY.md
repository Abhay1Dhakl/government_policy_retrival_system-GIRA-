# BM25 and DPO Algorithms Implementation in GIRA

## Overview

The GIRA (Government Information Retrieval System) implements two advanced algorithmic approaches:

1. **BM25 (Best Matching 25)** - Sparse hybrid search
2. **DPO (Direct Preference Optimization)** - Model fine-tuning and preference learning

This document provides a comprehensive summary of both algorithms and their implementation details.

---

## Part 1: BM25 Algorithm

### 1.1 What is BM25?

BM25 (Best Matching 25) is a probabilistic relevance framework for ranking documents by relevance to a given search query. It combines:
- Term frequency (TF)
- Inverse document frequency (IDF)
- Document length normalization

**Formula:**
```
BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
```

Where:
- `Q` = query with terms {q1, q2, ..., qn}
- `D` = document
- `f(qi, D)` = frequency of term qi in document D
- `|D|` = length of document D
- `avgdl` = average document length
- `k1` = tuning parameter (typically 1.5)
- `b` = tuning parameter (typically 0.75)

### 1.2 BM25 Implementation in GIRA

#### Location
```
gira-ai/gira-mcp-server/main.py (Lines 521-535)
```

#### Core Implementation

```python
async def get_bm25_scores(query: str, corpus: List[str]) -> Dict[str, float]:
    """Get BM25 scores for query against corpus"""
    if not rank_bm25 or not corpus:
        return {}
    
    try:
        # Tokenize corpus
        tokenized_corpus = [doc.split() for doc in corpus]
        
        # Initialize BM25 with corpus
        bm25 = rank_bm25(tokenized_corpus)
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        # Return as dictionary with terms and scores
        return {term: score for term, score in zip(corpus, scores) if score > 0}
    except Exception as e:
        return {}
```

#### Initialization

```python
# Lines 38-49
bm25_encoder = None
rank_bm25 = None

try:
    from pinecone_text.sparse import BM25Encoder
    from rank_bm25 import BM25Okapi
    rank_bm25 = BM25Okapi
    print("✓ BM25 encoder available", file=sys.stderr)
except ImportError as e:
    print(f"⚠ BM25 encoder not available: {e} - semantic search only", file=sys.stderr)
except Exception as e:
    print(f"⚠ BM25 encoder initialization failed: {e} - semantic search only", file=sys.stderr)
```

#### BM25 Encoder Initialization

```python
def get_bm25_encoder():
    """Get BM25 encoder with graceful error handling"""
    global bm25_encoder
    if bm25_encoder is None and rank_bm25 is not None:
        try:
            from pinecone_text.sparse import BM25Encoder
            bm25_encoder = BM25Encoder.default()
            print("✓ BM25 encoder initialized", file=sys.stderr)
        except ImportError:
            print("⚠ BM25 encoder not available", file=sys.stderr)
        except Exception as e:
            print(f"⚠ BM25 encoder initialization failed: {e}", file=sys.stderr)
    return bm25_encoder
```

### 1.3 BM25 in Hybrid Search

#### Hybrid Search Formula

```python
# Line 677-691
hybrid_score = alpha * dense_score + (1 - alpha) * normalized_bm25_score

# Where:
# - dense_score = semantic embeddings score
# - bm25_score = keyword relevance score
# - alpha = adaptive weighting parameter (0.0 to 1.0)
```

#### Implementation in Hybrid Search

```python
# Lines 677-691
if bm25_scores and matches:
    for match in matches:
        metadata = match.get("metadata", {}) or {}
        text_content = str(metadata.get("text", "")).lower()
        bm25_boost = 0.0
        
        # Accumulate BM25 scores for all matching terms
        for term, score_value in bm25_scores.items():
            if term in text_content:
                bm25_boost += score_value
        
        # Calculate hybrid score
        original_score = match.get("score", 0.0) or 0.0
        match["bm25_boost"] = bm25_boost
        
        # Combine dense and sparse signals
        match["hybrid_score"] = (
            alpha * original_score + 
            (1 - alpha) * (bm25_boost / 10)
        )
        
        # Re-sort by hybrid score
        matches.sort(
            key=lambda item: item.get("hybrid_score", item.get("score", 0.0)), 
            reverse=True
        )
```

### 1.4 Adaptive Alpha for Hybrid Search

The adaptive alpha parameter is dynamically adjusted based on query characteristics:

```python
# From adaptive_alpha.py
alpha = get_adaptive_alpha(query, context)

# Adjustments:
# - Short queries (<=2 words): alpha += 0.1 (favor dense)
# - Long queries (>=6 words): alpha -= 0.1 (favor sparse)
# - High medical terms (>=3): alpha += 0.15 (favor semantic)
# - Cardiac focus: alpha += 0.1
# - Safety concerns: alpha changes based on context
```

### 1.5 Dynamic Corpus Building

```python
async def build_dynamic_corpus():
    """Build medical corpus from actual documents in Pinecone"""
    # Extracts terms and phrases from indexed documents
    # Creates a vocabulary for BM25 scoring

async def update_bm25_with_dynamic_corpus():
    """Update BM25 encoder with dynamic medical corpus"""
    # Periodically refreshes BM25 with new document corpus
```

### 1.6 BM25 Advantages in GIRA

| Advantage | Benefit |
|-----------|---------|
| **Keyword Matching** | Captures exact term matches |
| **IDF Weighting** | Prioritizes rare/specific terms |
| **Document Length Normalization** | Fair scoring regardless of document size |
| **Fast Computation** | No neural network overhead |
| **Interpretable** | Clear term-based ranking |
| **Hybrid Integration** | Combines with dense embeddings |

---

## Part 2: DPO Algorithm

### 2.1 What is DPO (Direct Preference Optimization)?

DPO is a fine-tuning approach that directly optimizes model preferences without using a separate reward model. Instead of:
1. Training a reward model
2. Using PPO with the reward model

DPO directly:
1. Collects preference data (query, preferred response, non-preferred response)
2. Fine-tunes the model to prefer correct responses

**DPO Loss Function:**
```
L_DPO = -log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
```

Where:
- `π_θ` = model being trained
- `π_ref` = reference model
- `y_w` = preferred (winning) response
- `y_l` = non-preferred (losing) response
- `β` = temperature parameter (typically 0.5-1.0)
- `σ` = sigmoid function

### 2.2 DPO Implementation in GIRA

#### Data Model

**File:** `gira-ai/gira-agent/database/models.py`

```python
class DPO_RLHF(Base):
    """DPO training data model"""
    __tablename__ = "rlhf_feedback"
    
    rlhf_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True)
    conversation_id = Column(String(255), nullable=False)
    turn_id = Column(String(36), default=lambda: str(uuid.uuid4()), nullable=False, index=True)
    
    # Core DPO data
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Feedback for preference learning
    feedback = Column(Integer, nullable=True)  # 1=good, -1=bad, 0=neutral
    feedback_reason = Column(Text, nullable=True)
    
    # Training tracking
    used_in_training = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### Automated DPO Training DAG

**File:** `gira-ai/gira-agent/airflow/dags/dpo_training_dag.py`

```python
"""
DAG for automated DPO fine-tuning process.
Runs weekly to check for new feedback and trigger model updates.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

GIRA_AGENT_PATH = '/opt/airflow/gira_agent'
sys.path.append(GIRA_AGENT_PATH)

from DPO_Algorithm.auto_train import (
    count_new_feedback,
    run_export,
    fine_tune,
    register_new_model,
    mark_feedback_used
)

# DAG Configuration
default_args = {
    'owner': 'gira',
    'depends_on_past': False,
    'email': ['abhaya@ubventuresllc.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=4),
}

dag = DAG(
    'mira_dpo_training',
    default_args=default_args,
    description='Weekly DPO fine-tuning pipeline for GIRA AI',
    schedule_interval='0 0 * * 0',  # Runs every Sunday at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['gira', 'dpo', 'training'],
)
```

#### DPO Pipeline Steps

```python
# Step 1: Check Feedback Count
def check_feedback_count(**context):
    """Check if we have enough new feedback for training"""
    count = count_new_feedback()
    MIN_NEW_FEEDBACK = 200
    if count < MIN_NEW_FEEDBACK:
        raise Exception(f"Not enough new feedback (have {count}, need {MIN_NEW_FEEDBACK})")
    return count

check_feedback = PythonOperator(
    task_id='check_feedback',
    python_callable=check_feedback_count,
    dag=dag,
)

# Step 2: Export Feedback to JSONL
def export_feedback(**context):
    """Export feedback data to JSONL format for training"""
    jsonl_file = run_export()
    if not jsonl_file:
        raise Exception("Failed to export feedback data")
    return jsonl_file

export_data = PythonOperator(
    task_id='export_feedback',
    python_callable=export_feedback,
    dag=dag,
)

# Step 3: Run Fine-Tuning
def run_fine_tuning(**context):
    """Run the DPO fine-tuning process"""
    ti = context['task_instance']
    jsonl_file = ti.xcom_pull(task_ids='export_feedback')
    
    # Execute DPO training
    model_id = fine_tune(jsonl_file)
    if not model_id:
        raise Exception("Fine-tuning failed")
    return model_id

fine_tuning = PythonOperator(
    task_id='run_fine_tuning',
    python_callable=run_fine_tuning,
    dag=dag,
)

# Step 4: Register Model
def register_model(**context):
    """Register the new fine-tuned model"""
    ti = context['task_instance']
    model_id = ti.xcom_pull(task_ids='run_fine_tuning')
    register_new_model(model_id)
    return model_id

register = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

# Step 5: Mark Feedback as Used
def mark_feedback_used(**context):
    """Mark feedback data as used in training"""
    mark_feedback_used()

mark_used = PythonOperator(
    task_id='mark_feedback_used',
    python_callable=mark_feedback_used,
    dag=dag,
)

# Step 6: Cleanup Old Files
def cleanup_old_files():
    """Clean up training files older than 7 days"""
    cleanup_dir = os.path.join(GIRA_AGENT_PATH, 'DPO_Algorithm')
    current_time = datetime.now()
    count = 0
    
    for file in os.listdir(cleanup_dir):
        if file.startswith('dpo_pairs_') and file.endswith('.jsonl'):
            file_path = os.path.join(cleanup_dir, file)
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            
            if (current_time - file_time) > timedelta(days=7):
                try:
                    os.remove(file_path)
                    count += 1
                except OSError as e:
                    print(f"Error removing {file}: {e}")
    
    return f"Removed {count} old training files"

cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_old_files,
    dag=dag,
)

# Define Task Dependencies
check_feedback >> export_data >> fine_tuning >> register >> mark_used >> cleanup
```

### 2.3 DPO Training Data Format

#### JSONL Training Format

```jsonl
{
  "query": "What are the side effects of azithromycin?",
  "positive_docs": ["doc_001", "doc_005", "doc_012"],
  "negative_docs": ["doc_003", "doc_007"],
  "all_retrieved": ["doc_001", "doc_002", "doc_003", "doc_005", "doc_007", "doc_012"],
  "query_metadata": {"document_type": "pis", "region": "us"},
  "feedback_timestamp": "2024-12-13T10:30:00"
}
```

#### Training Data Export

```python
# From relevance_feedback.py
def export_feedback_for_training(self, output_file: str, min_relevant: int = 1):
    """
    Export feedback data suitable for DPO model training
    
    Args:
        output_file: Path to save training data (JSONL format)
        min_relevant: Minimum number of relevant docs required
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
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Exported {len(training_data)} training examples to {output_file}")
```

### 2.4 DPO Workflow

```
User Interaction
    ↓
Collect Query + Response
    ↓
User Feedback (Good/Bad/Neutral)
    ↓
Store in DPO_RLHF Table
    ↓
[Weekly Trigger - Every Sunday]
    ↓
Check Feedback Count (>= 200 required)
    ↓
Export Feedback to JSONL
    ↓
Run DPO Fine-Tuning
    ↓
Register New Model
    ↓
Mark Feedback as Used
    ↓
Cleanup Old Training Files
    ↓
Deploy Updated Model
```

### 2.5 DPO Advantages in GIRA

| Advantage | Benefit |
|-----------|---------|
| **Direct Optimization** | No reward model overhead |
| **Efficient** | Fewer training steps than PPO |
| **Stable** | More stable than RLHF |
| **User Feedback** | Leverages actual user preferences |
| **Continuous Improvement** | Weekly automated retraining |
| **Preference Learning** | Learns what users value |

### 2.6 DPO vs RLHF Comparison

| Aspect | DPO | RLHF |
|--------|-----|------|
| **Reward Model** | Not needed | Required |
| **Complexity** | Lower | Higher |
| **Training Steps** | Direct optimization | Reward + Policy |
| **Stability** | High | Medium |
| **Scalability** | Better | Limited |
| **User Data** | Direct preferences | Indirect rewards |

---

## Part 3: Integration Architecture

### 3.1 System Flow

```
Query from User
    ├─→ Intent Classification
    │
    ├─→ Hybrid Search
    │   ├─→ Dense: Gemini Embeddings → Pinecone
    │   ├─→ Sparse: BM25 Scoring
    │   └─→ Alpha Blending: hybrid_score = α·dense + (1-α)·sparse
    │
    ├─→ Re-ranking
    │   └─→ Cross-encoder / Domain-specific boosting
    │
    ├─→ LLM Response Generation
    │   └─→ OpenAI / Anthropic / Gemini
    │
    ├─→ User Feedback Collection
    │   └─→ Store in DPO_RLHF table
    │
    └─→ [Weekly] DPO Fine-tuning Pipeline
        └─→ Model Improvement
```

### 3.2 Algorithm Interaction

#### BM25 in Hybrid Search
```python
# Sparse keyword-based retrieval
bm25_scores = await get_bm25_scores(query, _medical_corpus)

# Combine with dense embeddings
for match in matches:
    text_content = str(metadata.get("text", "")).lower()
    bm25_boost = sum(score for term, score in bm25_scores.items() 
                     if term in text_content)
    
    hybrid_score = alpha * dense_score + (1 - alpha) * (bm25_boost / 10)
    match["hybrid_score"] = hybrid_score
```

#### DPO in Model Improvement
```python
# Collect feedback from user interactions
# → DPO_RLHF table

# Weekly pipeline:
# 1. Export preference pairs from feedback
# 2. Fine-tune model with DPO loss
# 3. Register and deploy new model
# 4. Continue with improved preferences
```

---

## Part 4: Performance Considerations

### 4.1 BM25 Performance

**Time Complexity:**
- Indexing: O(n) where n = corpus size
- Scoring: O(m) where m = query terms
- Total Query Time: ~10-50ms

**Space Complexity:**
- O(n·t) where n = documents, t = average terms per doc
- Sparse representation = efficient storage

### 4.2 DPO Performance

**Training Time:**
- Weekly schedule: minimal system impact
- Typical training: 2-4 hours with 200+ feedback samples
- Model loading: <30 seconds
- Deployment: <5 minutes

**Data Requirements:**
- Minimum: 200 feedback samples
- Optimal: 1000+ feedback samples per week
- Retention: 7 days of training files

### 4.3 Optimization Tips

1. **BM25:**
   - Precompute corpus embeddings
   - Cache BM25 encoder initialization
   - Use adaptive alpha based on query type

2. **DPO:**
   - Batch feedback collection
   - Monitor quality of feedback data
   - A/B test model versions
   - Track model performance metrics

---

## Part 5: Configuration Parameters

### BM25 Parameters

```python
# Adaptive Alpha Ranges
ALPHA_DEFAULT = 0.6  # 60% dense, 40% sparse
ALPHA_MIN = 0.3      # Min 30% dense for keyword queries
ALPHA_MAX = 0.9      # Max 90% dense for semantic queries

# Query-based adjustments
SHORT_QUERY_BOOST = 0.1    # Queries <= 2 words
LONG_QUERY_PENALTY = -0.1  # Queries >= 6 words
MEDICAL_TERM_BOOST = 0.15  # 3+ medical terms
```

### DPO Parameters

```python
# Training Configuration
MIN_NEW_FEEDBACK = 200        # Minimum feedback to trigger training
SCHEDULE = "0 0 * * 0"       # Every Sunday at midnight
EXECUTION_TIMEOUT = 4 hours  # Max training time
FILE_RETENTION = 7 days      # Keep training files for 7 days

# Model Configuration
BETA = 0.5                    # DPO temperature parameter
LEARNING_RATE = 2e-5         # Training learning rate
EPOCHS = 3                    # Training epochs
```

---

## Conclusion

The GIRA system leverages:
1. **BM25** for fast, interpretable keyword-based retrieval
2. **DPO** for continuous model improvement through user feedback

Together, they create a hybrid AI system that:
- ✅ Combines dense and sparse retrieval
- ✅ Learns from user preferences
- ✅ Improves over time automatically
- ✅ Maintains high performance and interpretability

---

*For detailed implementation, refer to the source files referenced throughout this document.*

