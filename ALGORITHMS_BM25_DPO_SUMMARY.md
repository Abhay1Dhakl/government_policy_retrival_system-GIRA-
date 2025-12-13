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

---

# Part 6: Technology Stack & Programming Languages

## Frontend Technologies

### **TypeScript**
- **Purpose:** Type-safe programming language for building the frontend application
- **Usage in GIRA:** Used extensively in Next.js components and React UI development
- **Benefits:**
  - Compile-time type checking prevents runtime errors
  - Better IDE support and autocomplete
  - Enhanced code readability and maintainability
  - Interfaces and type definitions for component props
- **Files:** `gira_frontend/src/**/*.tsx`, `gira_frontend/src/**/*.ts`
- **Key Features in GIRA:**
  ```typescript
  // Type-safe component props
  interface ChatProps {
    query: string;
    documentType: string;
    onSubmit: (query: string) => Promise<void>;
  }
  
  // Typed API responses
  interface SearchResult {
    id: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
  }
  ```

### **Next.js**
- **Purpose:** React framework for building production-ready web applications with Server-Side Rendering (SSR) and Static Generation
- **Usage in GIRA:** Primary frontend framework for building the user interface
- **Benefits:**
  - Built-in API routes for backend integration
  - Automatic code splitting and optimization
  - Image optimization and performance enhancements
  - File-based routing system for simplicity
  - Support for both SSR and SSG
- **Files:** `gira_frontend/`
- **Key Components:**
  - App Router (`src/app/`) - Modern Next.js 13+ routing
  - API Routes (`src/app/api/`) - Serverless functions
  - Pages (`src/app/login/`, `src/app/chat/`, `src/app/register/`)
- **Usage Example:**
  ```typescript
  // App Router - Server Component
  export default async function ChatPage() {
    return (
      <div>
        <ChatInterface />
      </div>
    );
  }
  
  // API Route - Handles backend communication
  export async function POST(request: Request) {
    const data = await request.json();
    return Response.json({ success: true });
  }
  ```

### **React.js**
- **Purpose:** JavaScript library for building dynamic, interactive user interfaces with reusable components
- **Usage in GIRA:** Component-based UI development for all frontend pages
- **Benefits:**
  - Component reusability reduces code duplication
  - Virtual DOM for efficient rendering
  - One-way data binding simplifies state management
  - Extensive ecosystem and community support
  - Context API for state management
- **Files:** `gira_frontend/src/components/`, `gira_frontend/src/app/`
- **Key Components:**
  - **ChatInterface** - Main chat component for policy queries
  - **DocumentUploader** - File upload functionality
  - **AuthForms** - Login and registration forms
  - **AdminDashboard** - User management interface
- **Example:**
  ```typescript
  // Functional Component with Hooks
  export function ChatInterface() {
    const [messages, setMessages] = React.useState([]);
    const [loading, setLoading] = React.useState(false);
    
    const handleSubmit = async (query: string) => {
      setLoading(true);
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        body: JSON.stringify({ query })
      });
      setLoading(false);
    };
    
    return (
      <div className="chat-container">
        {/* Component JSX */}
      </div>
    );
  }
  ```

### **Tailwind CSS**
- **Purpose:** Utility-first CSS framework for rapidly building custom designs without leaving HTML
- **Usage in GIRA:** Styling all frontend components with responsive design
- **Benefits:**
  - Utility classes for rapid UI development
  - Built-in responsive design (mobile-first approach)
  - Dark mode support integrated
  - Reduced CSS file size compared to traditional CSS
  - Consistent design system across the application
- **Files:** `gira_frontend/src/app/globals.css`, component className attributes
- **Configuration:** `gira_frontend/tailwind.config.js`
- **Usage Example:**
  ```typescript
  // Responsive Chat Interface
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6">
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
      {/* Content */}
    </div>
  </div>
  ```

### **HTML & CSS**
- **Purpose:** HTML provides structure and semantic markup; CSS provides styling and layout
- **Usage in GIRA:**
  - HTML: Document structure, semantic elements, accessibility
  - CSS: Layout, colors, fonts, responsive design
- **Benefits:**
  - Clean separation of content and presentation
  - SEO-friendly semantic HTML
  - Accessibility features (ARIA attributes)
  - Performance optimization through CSS optimization
- **Files:** Next.js templates, component JSX structure

---

## Backend Technologies

### **Python**
- **Purpose:** High-level, interpreted programming language used for backend development and AI/ML services
- **Usage in GIRA:** Primary language for backend API, AI agent, and data processing
- **Benefits:**
  - Rich ecosystem for AI/ML libraries
  - Fast development and prototyping
  - Excellent for data science and NLP tasks
  - Large community and extensive documentation
  - Easy to learn and maintain
- **Files:** `gira-backend/`, `gira-ai/`
- **Key Uses:**
  - **Django** for REST API backend
  - **FastAPI** for AI agent service
  - **Celery** for async task processing
  - **SQLAlchemy** for ORM

### **Django**
- **Purpose:** High-level Python web framework for building robust, scalable web applications
- **Usage in GIRA:** Primary framework for backend REST API
- **Benefits:**
  - Model-View-Template (MVT) architecture
  - Built-in ORM for database operations
  - Automatic admin panel generation
  - Comprehensive authentication and authorization
  - Middleware system for request/response processing
  - Security features (CSRF, SQL injection protection)
- **Files:** `gira-backend/src/`
- **Key Components:**
  - **Models** (`users/models.py`, `documents/models.py`) - Database schema
  - **Views** - Request handlers and business logic
  - **Serializers** - DRF serializers for API responses
  - **Authentication** - JWT, OAuth 2.0 integration
- **Example:**
  ```python
  # Django Model
  class User(AbstractBaseUser, PermissionsMixin):
      email = models.EmailField(unique=True)
      first_name = models.CharField(max_length=255)
      is_active = models.BooleanField(default=False)
      created_at = models.DateTimeField(auto_now_add=True)
  
  # Django REST Framework View
  class UserProfileView(generics.RetrieveUpdateAPIView):
      serializer_class = UserProfileSerializer
      permission_classes = [IsAuthenticated]
      
      def get_object(self):
          return self.request.user
  ```

### **FastAPI**
- **Purpose:** Modern, fast Python web framework for building APIs with automatic documentation
- **Usage in GIRA:** AI agent microservice for query processing and response generation
- **Benefits:**
  - Fast performance (comparable to Node.js and Go)
  - Automatic OpenAPI documentation generation
  - Built-in request validation using Pydantic
  - Async/await support for concurrent requests
  - Easy integration with background tasks
  - Type hints for better code quality
- **Files:** `gira-ai/gira-mcp-server/main.py`, `gira-ai/gira-agent/main.py`
- **Key Endpoints:**
  - `/api/v1/chat/query` - Process policy queries
  - `/api/v1/documents/upload` - Document ingestion
  - `/api/v1/search/semantic` - Semantic search
  - `/health` - Health check endpoint
- **Example:**
  ```python
  from fastapi import FastAPI, Depends
  from pydantic import BaseModel
  
  app = FastAPI()
  
  class QueryRequest(BaseModel):
      query: str
      document_type: str
      top_k: int = 10
  
  @app.post("/api/v1/chat/query")
  async def process_query(request: QueryRequest):
      """Process government policy query"""
      results = await execute_hybrid_search(
          query=request.query,
          document_type=request.document_type,
          top_k=request.top_k
      )
      return results
  ```

### **Django REST Framework (DRF)**
- **Purpose:** Powerful and flexible toolkit for building REST APIs in Django
- **Usage in GIRA:** Building RESTful API endpoints with serialization and validation
- **Benefits:**
  - Built-in authentication backends (JWT, OAuth)
  - Request/response serialization
  - Pagination and filtering
  - API viewsets and routers
  - Interactive API browsable interface
  - Permission and throttling classes
- **Files:** `gira-backend/src/users/serializers.py`, views, permissions
- **Example:**
  ```python
  from rest_framework import serializers, viewsets
  
  class UserSerializer(serializers.ModelSerializer):
      class Meta:
          model = User
          fields = ['id', 'email', 'first_name', 'last_name']
  
  class UserViewSet(viewsets.ModelViewSet):
      queryset = User.objects.all()
      serializer_class = UserSerializer
      permission_classes = [IsAuthenticated]
  ```

---

## Database & Data Layer

### **PostgreSQL**
- **Purpose:** Powerful, open-source relational database for persistent data storage
- **Usage in GIRA:** Primary database for user data, documents, and DPO feedback
- **Benefits:**
  - ACID compliance ensures data integrity
  - Advanced data types (JSON, Arrays, Full-text search)
  - Excellent for complex queries and transactions
  - Strong security features
  - Highly scalable and performant
- **Container:** `postgres:16-alpine` in Docker Compose
- **Databases:**
  - `gira_db` - Main application database
  - `airflow` - Apache Airflow metadata
- **Tables:**
  - `auth_user` - User authentication
  - `documents_document` - Document metadata
  - `rlhf_feedback` - DPO training data

### **SQLAlchemy**
- **Purpose:** Python SQL toolkit and Object-Relational Mapping (ORM) library
- **Usage in GIRA:** ORM for database operations in FastAPI services
- **Benefits:**
  - Database-agnostic ORM
  - Query builder with type safety
  - Relationship management
  - Session management for transactions
  - Lazy loading and eager loading optimization
- **Files:** `gira-ai/gira-agent/database/models.py`
- **Example:**
  ```python
  from sqlalchemy import Column, String, Integer
  from sqlalchemy.orm import Session
  
  class DPO_RLHF(Base):
      __tablename__ = "rlhf_feedback"
      rlhf_id = Column(Integer, primary_key=True)
      user_query = Column(String(Text), nullable=False)
      assistant_response = Column(Text, nullable=False)
  
  # Usage
  def get_feedback(db: Session, user_id: str):
      return db.query(DPO_RLHF).filter(
          DPO_RLHF.user_id == user_id
      ).all()
  ```

---

## AI & Machine Learning

### **Google Generative AI (Gemini)**
- **Purpose:** Google's generative AI API for embeddings and language model capabilities
- **Usage in GIRA:**
  - Text embeddings for semantic search (768-dimensional vectors)
  - LLM responses for policy queries
  - Natural language understanding
- **Benefits:**
  - High-quality embeddings (superior to open-source alternatives)
  - Multilingual support
  - Fast API response times
  - Cost-effective for production
- **Implementation:**
  ```python
  import google.generativeai as genai
  
  async def get_embedding_async(text: str, task_type: str = "retrieval_document"):
      """Get Google Gemini embeddings"""
      model = "models/embedding-001"
      embedding = await genai.embed_content(
          model=model,
          content=text,
          task_type=task_type
      )
      return embedding["embedding"]
  ```

### **Pinecone**
- **Purpose:** Vector database for storing and searching embeddings at scale
- **Usage in GIRA:** Semantic search over government policy documents
- **Benefits:**
  - Hybrid search (dense + sparse vectors)
  - Fast similarity search with filtering
  - Metadata filtering capabilities
  - Scalable to millions of vectors
  - Built-in reranking support
- **Configuration:**
  - Index: `policy-embeddings`
  - Dimension: 768 (Gemini embeddings)
  - Metric: Cosine similarity
- **Implementation:**
  ```python
  from pinecone import Pinecone
  
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(PINECONE_INDEX_NAME)
  
  # Upsert embeddings
  index.upsert(vectors=[
      ("doc_001", embedding_vector, {"text": "...", "source": "..."})
  ])
  
  # Query
  results = index.query(vector=query_embedding, top_k=10, filter={"region": "us"})
  ```

### **Rank-BM25**
- **Purpose:** Python implementation of BM25 algorithm for sparse, keyword-based search
- **Usage in GIRA:** Hybrid search component for combining keyword and semantic signals
- **Benefits:**
  - Fast keyword matching
  - IDF-weighted term importance
  - Document length normalization
  - Interpretable ranking
- **Implementation:**
  ```python
  from rank_bm25 import BM25Okapi
  
  corpus = ["document one", "document two", "document three"]
  tokenized_corpus = [doc.split() for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  
  query_tokens = "query terms".split()
  scores = bm25.get_scores(query_tokens)
  ```

### **spaCy**
- **Purpose:** Industrial-strength Natural Language Processing (NLP) library
- **Usage in GIRA:** Named entity recognition, text preprocessing, and linguistic analysis
- **Benefits:**
  - Fast and accurate NLP operations
  - Pre-trained language models
  - Efficient text processing pipeline
  - Entity recognition for policy terms
- **Implementation:**
  ```python
  import spacy
  
  nlp = spacy.load("en_core_web_sm")
  doc = nlp("Government policy on healthcare programs")
  
  for ent in doc.ents:
      print(f"{ent.text} - {ent.label_}")
  ```

### **Presidio**
- **Purpose:** PII (Personally Identifiable Information) detection and masking
- **Usage in GIRA:** Privacy preservation by filtering sensitive user information
- **Benefits:**
  - Detects names, emails, phone numbers, etc.
  - Pattern-based and ML-based detection
  - Customizable recognizers
  - Anonymization support
- **Implementation:**
  ```python
  from presidio_analyzer import AnalyzerEngine
  
  analyzer = AnalyzerEngine()
  results = analyzer.analyze(
      text="My name is John Doe, email is john@example.com",
      language="en"
  )
  ```

### **PyMuPDF (fitz)**
- **Purpose:** Library for reading and manipulating PDF documents
- **Usage in GIRA:** PDF parsing, text extraction, and document processing
- **Benefits:**
  - Fast PDF processing
  - Support for annotations and highlights
  - Text extraction with layout preservation
  - Page-level operations
- **Implementation:**
  ```python
  import fitz
  
  doc = fitz.open("policy.pdf")
  for page_num, page in enumerate(doc):
      text = page.get_text()
      # Process text content
  ```

### **OpenAI & Anthropic APIs**
- **Purpose:** Alternative LLM providers for generating responses and embeddings
- **Usage in GIRA:** Multiple LLM options for response generation
- **Benefits:**
  - High-quality language models
  - Flexible API for different tasks
  - Token-based pricing
  - Support for different model sizes
- **Configuration:**
  ```python
  OPENAI_API_KEY = "sk-..."
  ANTHROPIC_API_KEY = "sk-ant-..."
  ```

---

## Task Queue & Orchestration

### **Celery**
- **Purpose:** Distributed task queue for asynchronous job processing
- **Usage in GIRA:** Background document processing, email notifications, batch operations
- **Benefits:**
  - Distributed task execution
  - Task scheduling with Celery Beat
  - Retry mechanisms
  - Task monitoring and reporting
  - Redis integration for message broker
- **Configuration:**
  ```python
  from celery import Celery
  
  app = Celery('gira')
  app.conf.broker_url = 'redis://redis:6379/0'
  app.conf.result_backend = 'redis://redis:6379/0'
  
  @app.task
  def process_document(document_id):
      # Long-running task
      pass
  ```

### **Apache Airflow**
- **Purpose:** Workflow orchestration platform for managing complex data pipelines
- **Usage in GIRA:** DPO training pipeline automation, document processing workflows
- **Benefits:**
  - DAG-based workflow definition
  - Scheduled execution
  - Dependency management
  - Monitoring and alerting
  - Data lineage tracking
- **Files:** `gira-ai/gira-agent/airflow/dags/`
- **DAGs:**
  - `dpo_training_dag.py` - Weekly model fine-tuning

---

## Infrastructure & DevOps

### **Docker**
- **Purpose:** Containerization platform for packaging applications
- **Usage in GIRA:** Containerizing all services for consistent deployment
- **Benefits:**
  - Environment consistency
  - Easy scaling
  - Microservices architecture
  - Simplified dependency management
- **Files:**
  - `gira-backend/Dockerfile`
  - `gira-ai/gira-agent/Dockerfile`
  - `gira-ai/gira-mcp-server/Dockerfile`

### **Docker Compose**
- **Purpose:** Multi-container orchestration for local development
- **Usage in GIRA:** Managing all services (PostgreSQL, Redis, MinIO, API services)
- **Services:**
  - PostgreSQL database
  - Redis cache/broker
  - MinIO object storage
  - Django backend
  - FastAPI agent
  - Celery workers
  - Airflow scheduler
  - Nginx reverse proxy
- **File:** `docker-compose.yml`

### **Redis**
- **Purpose:** In-memory data structure store for caching and message brokering
- **Usage in GIRA:**
  - Session caching
  - Celery message broker
  - Query result caching
- **Benefits:**
  - Extremely fast operations
  - Supports complex data structures
  - Atomic operations
  - TTL support for cache expiration

### **MinIO**
- **Purpose:** S3-compatible object storage for file management
- **Usage in GIRA:** Storing uploaded PDF documents and processed files
- **Benefits:**
  - S3 API compatibility
  - Self-hosted alternative to AWS S3
  - Scalable storage
  - Access control and security

---

## Summary Table

| Technology | Category | Purpose | Location |
|-----------|----------|---------|----------|
| **TypeScript** | Frontend | Type-safe JavaScript | `gira_frontend/src/` |
| **Next.js** | Frontend | React framework | `gira_frontend/` |
| **React** | Frontend | UI components | `gira_frontend/src/components/` |
| **Tailwind CSS** | Frontend | Styling framework | `gira_frontend/tailwind.config.js` |
| **Python** | Backend | Server language | `gira-backend/`, `gira-ai/` |
| **Django** | Backend | Web framework | `gira-backend/` |
| **FastAPI** | Backend | API framework | `gira-ai/gira-mcp-server/`, `gira-ai/gira-agent/` |
| **PostgreSQL** | Database | Relational DB | Docker container |
| **SQLAlchemy** | Database | ORM | `gira-ai/gira-agent/database/` |
| **Google Gemini** | AI/ML | Embeddings & LLM | Vector embeddings |
| **Pinecone** | AI/ML | Vector DB | Cloud service |
| **Rank-BM25** | AI/ML | Keyword search | `gira-ai/gira-mcp-server/main.py` |
| **spaCy** | NLP | Text processing | AI agent services |
| **Presidio** | Security | PII detection | `gira-ai/gira-agent/pii_service.py` |
| **Celery** | Queue | Task processing | `gira-backend/` |
| **Airflow** | Orchestration | Workflow management | `gira-ai/gira-agent/airflow/` |
| **Docker** | DevOps | Containerization | All services |
| **Redis** | Cache | In-memory store | Docker container |
| **MinIO** | Storage | Object storage | Docker container |

---



