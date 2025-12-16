# GIRA Codebase Refactoring & Professional Organization Plan

## Executive Summary
Comprehensive analysis of `gira-agent` (59 files) and `gira-mcp-server` (21+ files) codebases with refactoring recommendations to remove unused code, consolidate large files (>300 lines), and establish professional folder structure.

---

## Phase 1: gira-agent Refactoring

### 1.1 Files to Delete (Unused/Redundant)

| File | Reason | Location |
|------|--------|----------|
| `Dockerfile.backup` | Backup file, not used | Root |
| `prompt_service_old.py` | Old version, replaced | services/ |
| `prompt_service_simple.py` | Simplified version, unused | services/ |
| `config.py` (if duplicate) | Check for duplication | Root |

### 1.2 Large Files Requiring Refactoring (>300 lines)

| File | Lines | Strategy |
|------|-------|----------|
| `pdf_highlighter.py` | 1951 | **Split into modules**: extract_highlights, render_pdf, coordinate_mapping |
| `document_upload/app/processor.py` | 1174 | **Extract submodules**: pdf_extraction, embedding_generation, validation |
| `document_upload/tools/pdf_extraction.py` | 614 | **Extract utilities**: text_extraction, chunking, metadata |
| `services/streaming_service.py` | 420 | **Extract logic**: streaming_handlers, chunk_processors |
| `llm_options/llm_choose.py` | 381 | **Extract logic**: provider_selector, model_config |
| `services/mcp_service.py` | 393 | **Extract logic**: mcp_client, query_handler |
| `database/services.py` | 469 | **Extract services**: crud_operations, analytics |
| `intent_classifier.py` | 356 | **Extract logic**: classification_rules, pattern_matching |

### 1.3 Proposed New Folder Structure - gira-agent

```
gira-agent/
├── core/                          # Core business logic
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── logging_config.py           # Logging setup
│   └── constants.py               # Application constants
│
├── models/                         # Pydantic models & schemas
│   ├── __init__.py
│   ├── request_models.py          # Request DTOs
│   ├── response_models.py         # Response DTOs
│   └── database_models.py         # SQLAlchemy models (from database/)
│
├── api/                           # API endpoints
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py
│           ├── query.py           # Chat/Query endpoints
│           ├── feedback.py        # Feedback endpoints
│           ├── pdf.py             # PDF processing endpoints
│           ├── pages.py           # Page-related endpoints
│           └── documents.py       # Document management endpoints
│
├── services/                      # Business logic services
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_provider.py      # Abstract LLM provider
│   │   ├── gemini_provider.py    # Google Gemini
│   │   ├── openai_provider.py    # OpenAI
│   │   └── anthropic_provider.py # Anthropic
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base_retriever.py     # Abstract retriever
│   │   ├── hybrid_search.py      # BM25 + dense search
│   │   └── reranker.py           # Result reranking
│   │
│   ├── document/
│   │   ├── __init__.py
│   │   ├── processor.py          # Document processing (refactored)
│   │   ├── pdf_handler.py        # PDF operations (refactored from pdf_highlighter)
│   │   ├── chunking.py           # Text chunking strategies
│   │   └── validator.py          # File validation
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── gemini_embeddings.py  # Google Gemini embeddings
│   │   └── embedding_manager.py  # Embedding orchestration
│   │
│   ├── response/
│   │   ├── __init__.py
│   │   ├── generator.py          # Response generation
│   │   ├── prompt_service.py     # Prompt engineering
│   │   └── streaming.py          # Streaming responses
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── client.py             # MCP client integration
│   │   └── handlers.py           # MCP request handlers
│   │
│   └── intent/
│       ├── __init__.py
│       ├── classifier.py         # Intent classification
│       └── patterns.py           # Classification patterns
│
├── database/                      # Database & persistence
│   ├── __init__.py
│   ├── config.py                 # Database configuration
│   ├── models.py                 # Data models (kept from database/)
│   └── services.py               # Database operations (refactored)
│
├── middleware/                    # Custom middleware
│   ├── __init__.py
│   ├── cors.py
│   ├── error_handler.py
│   ├── logging.py
│   └── auth.py
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── validators.py             # Input validation
│   ├── helpers.py                # General helpers
│   ├── exceptions.py             # Custom exceptions
│   └── decorators.py             # Useful decorators
│
├── document_upload/               # Document upload module (kept for compatibility)
│   ├── __init__.py
│   └── legacy_routes.py          # Legacy API routes
│
├── llm_options/                   # LLM selection (legacy - to deprecate)
│   └── llm_choose.py
│
├── airflow/                       # Workflow orchestration
│   ├── dags/
│   │   ├── dpo_training_dag.py
│   │   ├── document_processing_dag.py
│   │   └── embedding_generation_dag.py
│   └── Dockerfile
│
├── main.py                        # FastAPI application entry
├── config.py                      # Root config (if not moved to core/)
├── requirements.txt
├── Dockerfile
└── pytest.ini                     # Testing configuration
```

---

## Phase 2: gira-mcp-server Refactoring

### 2.1 Files to Delete/Archive

| File | Reason | Action |
|------|--------|--------|
| `=2.2.0` | Invalid file name | Delete |
| Check for duplicate `gemini_embeddings.py` | May be duplicated | Consolidate |
| Backup JSON files | If unused | Archive |

### 2.2 Large Files Requiring Refactoring (>300 lines)

| File | Lines | Strategy |
|------|-------|----------|
| `main.py` | 1314 | **Extract modules**: search_engine, tools_registry, performance_monitor |
| `performance_monitoring.py` | 357 | **Extract into monitoring package** |
| `relevance_feedback.py` | 345 | **Extract into feedback package** |
| `intent_classifier.py` | 356 | **Consolidate with gira-agent or create shared module** |
| `ab_testing.py` | 311 | **Extract into testing package** |
| `adaptive_alpha.py` | 319 | **Extract into search_optimization package** |
| `hard_negative_mining.py` | 349 | **Extract into training package** |

### 2.3 Proposed New Folder Structure - gira-mcp-server

```
gira-mcp-server/
├── core/                          # Core MCP server
│   ├── __init__.py
│   ├── server.py                  # Main MCP server (refactored from main.py)
│   ├── tools_registry.py          # Tool definitions
│   └── config.py                  # Configuration
│
├── search/                        # Search & retrieval module
│   ├── __init__.py
│   ├── engine.py                  # Hybrid search engine (refactored from main.py)
│   ├── bm25_search.py            # BM25 sparse search
│   ├── dense_search.py           # Dense vector search
│   ├── reranker.py               # Result reranking
│   └── query_processor.py        # Query preprocessing
│
├── optimization/                  # Search optimization
│   ├── __init__.py
│   ├── adaptive_alpha.py          # Adaptive weighting
│   ├── concept_expander.py        # Query expansion
│   └── query_optimization.py      # Query tuning
│
├── embeddings/                    # Embedding management
│   ├── __init__.py
│   ├── gemini_embeddings.py      # Google Gemini embeddings
│   ├── embedding_cache.py        # Caching layer
│   └── embedding_manager.py      # Manager service
│
├── feedback/                      # RLHF & Feedback
│   ├── __init__.py
│   ├── feedback_handler.py       # Feedback processing (refactored)
│   ├── relevance_feedback.py     # Relevance feedback (refactored)
│   └── preference_learning.py    # Learning from preferences
│
├── training/                      # Model training & optimization
│   ├── __init__.py
│   ├── dpo_trainer.py            # DPO training (refactored)
│   ├── hard_negative_mining.py   # Hard negative mining (refactored)
│   ├── data_preparation.py       # Training data prep
│   └── model_registry.py         # Model management
│
├── monitoring/                    # Performance & monitoring
│   ├── __init__.py
│   ├── performance_monitor.py    # Performance tracking (refactored)
│   ├── metrics_collector.py      # Metrics collection
│   └── analytics.py              # Analytics & reporting
│
├── evaluation/                    # Evaluation & testing
│   ├── __init__.py
│   ├── evaluator.py              # Evaluation framework
│   ├── evaluation.py             # Evaluation metrics (refactored)
│   ├── ab_testing.py             # A/B testing (refactored)
│   └── benchmark.py              # Benchmarking
│
├── ontology/                      # Ontology management
│   ├── __init__.py
│   ├── loader.py                 # Ontology loader (refactored from ontology_loader.py)
│   ├── structures.py             # Ontology data structures
│   └── files/                    # Ontology files
│       └── medical_ontology.json
│
├── intent/                        # Intent & NLP
│   ├── __init__.py
│   ├── classifier.py             # Intent classification
│   ├── patterns.py               # Classification patterns
│   └── entity_extractor.py       # NER & entity extraction
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── validators.py             # Input validation
│   ├── helpers.py                # General helpers
│   ├── exceptions.py             # Custom exceptions
│   └── decorators.py             # Performance decorators
│
├── config/                        # Configuration files
│   ├── config_development.json
│   ├── config_production.json
│   └── settings.py               # Python config
│
├── tests/                         # Testing directory
│   ├── __init__.py
│   ├── test_search.py
│   ├── test_embeddings.py
│   └── test_feedback.py
│
├── main.py                        # Entry point (refactored)
├── requirements.txt
├── Dockerfile
├── ground_truth_template.json
└── pytest.ini
```

---

## Phase 3: Shared Code Between Modules

### Files to Consolidate/Share
```
gira-ai/
├── shared/                        # Shared utilities
│   ├── __init__.py
│   ├── embeddings.py             # Abstract embedding interface
│   ├── intent_classifier.py      # Shared intent classification
│   ├── models.py                 # Common data models
│   └── exceptions.py             # Shared exception classes
│
├── gira-agent/                    # Updated structure
│   └── [refactored as above]
│
└── gira-mcp-server/               # Updated structure
    └── [refactored as above]
```

---

## Phase 4: Unused Imports & Cleanup

### Common Issues Found

**gira-agent/main.py:**
- Commented MCP imports (lines 13-14)
- Unused imports that can be removed

**gira-mcp-server/main.py:**
- Multiple initialization paths (Pinecone with/without secure)
- Debug/test code mixed with production code

### Action Items
- Remove commented code sections
- Consolidate initialization logic
- Extract magic strings to constants
- Remove duplicate imports

---

## Phase 5: Implementation Roadmap

### Week 1: Analysis & Preparation
- ✅ Document current structure (COMPLETED)
- Document all imports and dependencies
- Create migration mapping

### Week 2: gira-agent Refactoring
- Delete unused files (`Dockerfile.backup`, `*_old.py`, `*_simple.py`)
- Extract pdf_highlighter.py into modules
- Refactor document_upload processor
- Create new folder structure

### Week 3: gira-mcp-server Refactoring
- Delete invalid files (`=2.2.0`)
- Refactor main.py into modular components
- Extract large files into packages
- Create new folder structure

### Week 4: Integration & Testing
- Create shared/ module for common code
- Update imports across both projects
- Run comprehensive tests
- Documentation & cleanup

---

## Detailed Refactoring: Large Files

### pdf_highlighter.py (1951 lines)

**Extract into:**

```python
# services/document/pdf/extractor.py - Extract PDF content
# services/document/pdf/highlighter.py - Text highlighting logic
# services/document/pdf/coordinate_mapper.py - Coordinate mapping
# services/document/pdf/renderer.py - PDF rendering
```

### gira-mcp-server/main.py (1314 lines)

**Extract into:**

```python
# core/server.py - FastMCP server setup
# search/engine.py - Search execution
# core/tools_registry.py - MCP tools definition
# monitoring/performance_monitor.py - Metrics collection
```

---

## Benefits of This Refactoring

✅ **Maintainability**: Clear separation of concerns
✅ **Scalability**: Easy to add new features
✅ **Testing**: Modular code is easier to test
✅ **Collaboration**: Clear structure for team development
✅ **Readability**: Files under 300 lines, focused purpose
✅ **Reusability**: Shared code in common locations
✅ **Professional**: Industry-standard project structure

---

## Checkpoints

- [ ] Phase 1: gira-agent deletion (unused files)
- [ ] Phase 2: gira-agent refactoring (large files split)
- [ ] Phase 3: gira-agent folder reorganization
- [ ] Phase 4: gira-mcp-server deletion (unused files)
- [ ] Phase 5: gira-mcp-server refactoring (large files split)
- [ ] Phase 6: gira-mcp-server folder reorganization
- [ ] Phase 7: Shared module creation
- [ ] Phase 8: Import updates across projects
- [ ] Phase 9: Testing & validation
- [ ] Phase 10: Documentation update

---

