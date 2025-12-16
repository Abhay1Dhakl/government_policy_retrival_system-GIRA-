# Professional Folder Structure Guide

## GIRA-AI Project Professional Organization

### Current Directory Tree (After Cleanup)

```
gira-ai/
│
├── shared/                              # ✅ NEW: Shared utilities & code
│   ├── __init__.py                      # Module initialization
│   ├── exceptions.py                    # Custom exceptions (39 lines)
│   ├── logging.py                       # Logging utilities (47 lines)
│   ├── utils.py                         # Helper functions (102 lines)
│   └── constants.py                     # Enums & constants (66 lines)
│
├── gira-agent/                          # FastAPI AI Agent
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                    # Configuration management
│   │   └── constants.py                 # Application constants
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── routes/
│   │           ├── __init__.py
│   │           ├── query.py             # Chat/Query endpoints
│   │           ├── feedback.py          # Feedback endpoints
│   │           ├── pdf.py               # PDF processing
│   │           ├── pages.py             # Page endpoints
│   │           └── documents.py         # Document management
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py              # LLM abstraction
│   │   ├── mcp_service.py              # MCP integration
│   │   ├── prompt_service.py           # Prompt engineering
│   │   ├── response_service.py         # Response generation
│   │   ├── streaming_service.py        # Streaming responses
│   │   ├── title_service.py            # Title generation
│   │   └── embedding_service.py        # Embedding management
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── config.py                    # DB configuration
│   │   ├── models.py                    # SQLAlchemy models
│   │   └── services.py                 # Database operations
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── cors.py
│   │   ├── error_handler.py
│   │   ├── logging.py
│   │   └── auth.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── validators.py               # Input validation
│   │
│   ├── document_upload/                # Document upload module
│   │   ├── __init__.py
│   │   └── app/
│   │       └── api/
│   │           └── v1/
│   │               └── routes_ingestion.py
│   │
│   ├── llm_options/                    # Legacy: To be deprecated
│   │   └── llm_choose.py
│   │
│   ├── airflow/                        # Workflow orchestration
│   │   ├── __init__.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── dags/
│   │   │   ├── dpo_training_dag.py
│   │   │   ├── document_processing_dag.py
│   │   │   └── embedding_generation_dag.py
│   │   ├── logs/
│   │   └── plugins/
│   │
│   ├── main.py                         # FastAPI entry point
│   ├── config.py                       # Root configuration
│   ├── logging_config.py              # Logging setup
│   ├── intent_classifier.py           # Intent classification
│   ├── gemini_embeddings.py           # Gemini embeddings
│   ├── pdf_highlighter.py             # PDF highlighting (1951 lines - NEEDS REFACTORING)
│   ├── min_ontology.json              # Minimal ontology
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pytest.ini
│
├── gira-mcp-server/                    # MCP Server for Advanced Search
│   ├── core/
│   │   ├── __init__.py
│   │   ├── server.py                   # MCP server setup
│   │   ├── tools_registry.py          # Tool definitions
│   │   └── config.py                   # Configuration
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── engine.py                   # Main search engine
│   │   ├── bm25_search.py             # BM25 implementation
│   │   ├── dense_search.py            # Dense vector search
│   │   ├── reranker.py                # Result reranking
│   │   └── query_processor.py         # Query preprocessing
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── adaptive_alpha.py          # Adaptive weighting (319 lines)
│   │   ├── concept_expander.py        # Query expansion
│   │   └── query_optimization.py      # Query tuning
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── gemini_embeddings.py      # Gemini API
│   │   ├── embedding_cache.py        # Caching layer
│   │   └── embedding_manager.py      # Manager service
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── feedback_handler.py       # Feedback processing
│   │   ├── relevance_feedback.py     # Relevance feedback (345 lines)
│   │   └── preference_learning.py    # Preference learning
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dpo_trainer.py            # DPO training
│   │   ├── hard_negative_mining.py   # Hard negative mining (349 lines)
│   │   ├── data_preparation.py       # Training data prep
│   │   └── model_registry.py         # Model management
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── performance_monitor.py    # Performance tracking (357 lines)
│   │   ├── metrics_collector.py      # Metrics collection
│   │   └── analytics.py              # Analytics & reporting
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Evaluation framework
│   │   ├── evaluation.py             # Evaluation metrics
│   │   ├── ab_testing.py             # A/B testing (311 lines)
│   │   └── benchmark.py              # Benchmarking
│   │
│   ├── ontology/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Ontology loader
│   │   ├── structures.py             # Ontology structures
│   │   └── files/
│   │       └── medical_ontology.json
│   │
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── classifier.py             # Intent classification (356 lines)
│   │   └── patterns.py               # Classification patterns
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── validators.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_development.json
│   │   ├── config_production.json
│   │   └── settings.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_search.py
│   │   ├── test_embeddings.py
│   │   └── test_feedback.py
│   │
│   ├── main.py                        # Entry point (1314 lines - NEEDS REFACTORING)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── ground_truth_template.json
│   └── pytest.ini
│
├── CODEBASE_REFACTORING_PLAN.md        # Detailed refactoring guide
├── CLEANUP_SUMMARY.md                   # Cleanup progress
└── README.md                            # Project documentation
```

---

## File Organization Principles

### 1. **Separation of Concerns**
Each module handles one specific responsibility:
- `api/` - HTTP endpoint definitions
- `services/` - Business logic
- `database/` - Data persistence
- `middleware/` - HTTP middleware
- `utils/` - Reusable utilities

### 2. **Modular Service Architecture**
Services are organized by domain:
- `services/llm/` - Language model operations
- `services/document/` - Document handling
- `services/embedding/` - Embeddings
- `services/retrieval/` - Search operations

### 3. **Maximum File Size**
- **Target:** < 300 lines per file
- **Acceptable:** 300-400 lines (refactor soon)
- **Critical:** > 400 lines (must refactor)

### 4. **Clear Imports**
```python
# Bad: Importing large modules
from services import pdf_highlighter  # 1951 lines!

# Good: Importing specific functions
from services.document.pdf import extract_text, highlight_regions
```

### 5. **Consistent Naming**
- `*_service.py` - Service classes
- `*_handler.py` - Request/event handlers
- `*_processor.py` - Data processors
- `*_manager.py` - Manager/coordinator classes
- `*_model.py` - Data models
- `*_schema.py` - Pydantic schemas

---

## Current Cleanup Status

### ✅ Completed
- [x] Deleted 4 unused files
- [x] Created shared module (254 lines)
- [x] Documented refactoring plan
- [x] Generated cleanup summary
- [x] Professional folder structure guide

### 🔄 In Progress
- [ ] Refactor large files > 300 lines
- [ ] Create new folder structures
- [ ] Update imports to use shared module
- [ ] Add unit tests

### ⏳ Upcoming
- [ ] Performance testing
- [ ] Documentation updates
- [ ] Git commit & push
- [ ] Code review

---

## Dependencies Between Modules

```
shared/
  ├── exceptions
  ├── logging
  ├── utils
  └── constants

gira-agent/
  ├── core/ (depends on shared)
  ├── api/ (depends on core, services)
  ├── services/ (depends on shared, database)
  ├── database/ (depends on shared)
  └── middleware/ (depends on shared)

gira-mcp-server/
  ├── core/ (depends on shared)
  ├── search/ (depends on shared, optimization)
  ├── training/ (depends on shared, feedback)
  └── monitoring/ (depends on shared)
```

---

## Migration Checklist for Large Files

### pdf_highlighter.py (1951 lines)
**Target Split:**
```
services/document/pdf/
  ├── __init__.py
  ├── extractor.py      (Extract PDF content)
  ├── highlighter.py    (Text highlighting)
  ├── mapper.py         (Coordinate mapping)
  └── renderer.py       (PDF rendering)
```

### gira-mcp-server/main.py (1314 lines)
**Target Split:**
```
core/
  ├── server.py         (MCP server setup)
  └── tools_registry.py (Tool definitions)

search/
  └── engine.py         (Search logic)

monitoring/
  └── performance.py    (Metrics)
```

---

## Best Practices

### Import Organization
```python
# 1. Standard library
import os
import sys
from typing import Dict, List

# 2. Third-party
import numpy as np
from fastapi import FastAPI

# 3. Local absolute imports
from gira.shared.exceptions import SearchError
from gira.shared.utils import clean_text

# 4. Local relative imports
from .models import User
from ..services import search_service
```

### Module __init__.py Pattern
```python
"""Module description"""

from .main_class import MainClass
from .util_function import util_function

__all__ = ["MainClass", "util_function"]
```

### Error Handling
```python
from gira.shared.exceptions import SearchError

try:
    results = search(query)
except Exception as e:
    raise SearchError(f"Search failed: {str(e)}") from e
```

---

## Monitoring Progress

Track refactoring with these metrics:

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Unused Files | 0 | 0 | ✅ |
| Files > 400 lines | 0 | 7 | 🔄 |
| Shared Utilities | 4 modules | 4 modules | ✅ |
| Code Duplication | Minimal | Reduced | ✅ |
| Test Coverage | 80%+ | TBD | ⏳ |

---

## Support & References

See also:
- `CODEBASE_REFACTORING_PLAN.md` - Detailed refactoring strategies
- `CLEANUP_SUMMARY.md` - Cleanup progress report
- `CHAPTER_4_IMPLEMENTATION_AND_TESTING.md` - Implementation details
- `DATABASE_PLATFORMS_SUMMARY.md` - Database information

---

Generated: December 15, 2025
Last Updated: December 15, 2025
Status: 🟢 Phase 1 Complete, Phase 2 Starting
