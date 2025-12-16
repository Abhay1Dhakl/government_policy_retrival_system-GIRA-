# Database Platform Information for GIRA Project

## Overview

The GIRA (Government Information Retrieval System) uses a comprehensive database architecture combining multiple database platforms and technologies to handle different aspects of the system:

1. **PostgreSQL** - Relational database for core application data
2. **Pinecone** - Vector database for semantic search
3. **Redis** - In-memory cache and message broker
4. **MinIO** - Object storage for documents

This document provides detailed information about each database platform used in the project.

---

## Part 1: PostgreSQL - Primary Relational Database

### 1.1 Overview

**PostgreSQL** is a powerful, open-source relational database management system (RDBMS) that serves as the primary data store for GIRA.

- **Purpose:** Store user data, documents, feedback, and application metadata
- **Version in GIRA:** PostgreSQL 16-alpine (latest stable lightweight version)
- **Container Image:** `postgres:16-alpine`

### 1.2 Why PostgreSQL?

#### Advantages for GIRA:

| Feature | Benefit |
|---------|---------|
| **ACID Compliance** | Data integrity and reliability |
| **Advanced Data Types** | JSON, Arrays, Full-text search support |
| **Scalability** | Handle millions of records efficiently |
| **Security** | Row-level security, encryption support |
| **Performance** | Indexing, query optimization |
| **Extensions** | PostGIS, pgvector for AI/ML |
| **Open Source** | No licensing costs, community support |
| **Reliability** | Used in production at scale |

### 1.3 Docker Container Configuration

**File:** `docker-compose.yml`

```yaml
postgres:
  image: postgres:16-alpine
  container_name: gira-postgres
  environment:
    POSTGRES_DB: ${POSTGRES_DB}
    POSTGRES_USER: ${POSTGRES_USER}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --auth-host=scram-sha-256"
    POSTGRES_HOST_AUTH_METHOD: scram-sha-256
  ports:
    # SECURITY: Only bind to localhost
    - "127.0.0.1:${POSTGRES_EXTERNAL_PORT}:${POSTGRES_INTERNAL_PORT}"
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./init-db:/docker-entrypoint-initdb.d
  command:
    [
      "postgres",
      "-c", "ssl=off",
      "-c", "password_encryption=scram-sha-256",
      "-c", "logging_collector=on",
      "-c", "log_statement=all",
      "-c", "log_connections=on",
    ]
  networks:
    - gira-network
  restart: unless-stopped
  deploy:
    resources:
      limits:
        cpus: "2"
        memory: 2GB
      reservations:
        cpus: "0.5"
        memory: 512MB
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
    interval: "10s"
    timeout: "5s"
    retries: 3
    start_period: "30s"
```

### 1.4 Connection Configuration

**File:** `gira-backend/src/gira/settings.py`

```python
# Database Configuration
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DATABASE_NAME"),
        "USER": os.getenv("DATABASE_USER"),
        "PASSWORD": os.getenv("DATABASE_PASSWORD"),
        "HOST": os.getenv("DATABASE_HOST"),
        "PORT": os.getenv("DATABASE_PORT"),
    }
}
```

#### Environment Variables

```env
# Database Configuration (Backend)
DATABASE_NAME=gira_db
DATABASE_USER=gira_user
DATABASE_PASSWORD=secure_password_here
DATABASE_HOST=postgres
DATABASE_PORT=5432

# PostgreSQL Container
POSTGRES_DB=gira_db
POSTGRES_USER=gira_user
POSTGRES_PASSWORD=secure_password_here
POSTGRES_EXTERNAL_PORT=5432
POSTGRES_INTERNAL_PORT=5432
```

### 1.5 FastAPI Database Configuration

**File:** `gira-ai/gira-agent/database/config.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    """
    Get database URL based on environment configuration.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Try to get environment-specific database URL first
    if environment == "production":
        db_url = os.getenv("DATABASE_URL_PROD")
    else:
        db_url = os.getenv("DATABASE_URL_DEV")
    
    # If environment-specific URL not found, try general DATABASE_URL
    if not db_url:
        db_url = os.getenv("DATABASE_URL")
    
    # If still no URL, construct from individual components
    if not db_url:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "gira_db")
        username = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        
        db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    return db_url

# Create SQLAlchemy engine
DATABASE_URL = get_database_url()

def get_engine_config():
    """Get SQLAlchemy engine configuration based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    config = {
        "pool_pre_ping": True,
        "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true"
    }
    
    if environment == "production":
        config.update({
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "20")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "30")),
            "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
        })
    else:
        config.update({
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "20")),
        })
    
    return config

engine = create_engine(DATABASE_URL, **get_engine_config())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
```

### 1.6 Connection Pooling Configuration

**Development Environment:**
- Pool Size: 10 connections
- Max Overflow: 20 additional connections
- Timeout: 20 seconds
- Connection Recycling: Not configured

**Production Environment:**
- Pool Size: 20 connections
- Max Overflow: 30 additional connections
- Timeout: 30 seconds
- Connection Recycling: 3600 seconds (1 hour)

### 1.7 Security Features

#### Authentication

- **Method:** SCRAM-SHA-256 (Salted Challenge Response Authentication Mechanism)
- **Password Encryption:** SCRAM-SHA-256
- **Host Authentication:** Limited to specific networks

```yaml
POSTGRES_HOST_AUTH_METHOD: scram-sha-256
POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --auth-host=scram-sha-256"
```

#### Network Security

- **Port Binding:** Only localhost (127.0.0.1)
- **Container Network:** Isolated Docker network (`gira-network`)
- **SSL:** Disabled for development, should be enabled in production

```yaml
ports:
  - "127.0.0.1:${POSTGRES_EXTERNAL_PORT}:${POSTGRES_INTERNAL_PORT}"
```

#### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: "2"
      memory: 2GB
    reservations:
      cpus: "0.5"
      memory: 512MB
```

### 1.8 Database Schema

#### Core Tables

**1. Users Table (`auth_user`)**
```sql
CREATE TABLE auth_user (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    phone_number VARCHAR(20),
    country VARCHAR(100),
    city VARCHAR(100),
    address VARCHAR(255),
    zip_code VARCHAR(10),
    institution VARCHAR(255),
    role VARCHAR(20),
    is_active BOOLEAN DEFAULT false,
    is_staff BOOLEAN DEFAULT false,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**2. Documents Table (`documents_document`)**
```sql
CREATE TABLE documents_document (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255),
    file_path VARCHAR(1000),
    content_hash VARCHAR(255),
    status VARCHAR(50),
    user_id BIGINT REFERENCES auth_user(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    page_count INTEGER
);
```

**3. DPO Feedback Table (`rlhf_feedback`)**
```sql
CREATE TABLE rlhf_feedback (
    rlhf_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    conversation_id VARCHAR(255) NOT NULL,
    turn_id VARCHAR(36) UNIQUE NOT NULL,
    user_query TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    feedback INTEGER,
    feedback_reason TEXT,
    used_in_training BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 1.9 Health Checks

PostgreSQL health check configuration:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
  interval: "10s"
  timeout: "5s"
  retries: 3
  start_period: "30s"
```

**Parameters:**
- **Interval:** Check every 10 seconds
- **Timeout:** Wait 5 seconds for response
- **Retries:** Mark as unhealthy after 3 failed checks
- **Start Period:** Wait 30 seconds before first check

### 1.10 Data Persistence

**Volume Configuration:**

```yaml
volumes:
  - postgres_data:/var/lib/postgresql/data
  - ./init-db:/docker-entrypoint-initdb.d
```

- **postgres_data:** Persists database data between container restarts
- **init-db:** Scripts for database initialization

**File:** `init-db/01-init.sql` - Database initialization scripts

---

## Part 2: Driver & ORM Libraries

### 2.1 psycopg2

**Purpose:** PostgreSQL adapter for Python (used by Django)

**Version:** `psycopg2-binary==2.9.10`

**Features:**
- Native PostgreSQL protocol support
- Connection pooling
- Type casting
- Transaction support

```python
# Django backend declaration
"ENGINE": "django.db.backends.postgresql"
```

### 2.2 SQLAlchemy

**Purpose:** Python SQL toolkit and Object-Relational Mapping (ORM)

**Usage in GIRA:**
- FastAPI database operations
- AI agent database access
- Type-safe query building

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
```

### 2.3 Django ORM

**Purpose:** Object-Relational Mapping built into Django

**Features:**
- Model definition
- Query API
- Migrations
- Signal system

```python
# Model definition
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
```

---

## Part 3: Pinecone - Vector Database

### 3.1 Overview

**Pinecone** is a cloud-based vector database designed for semantic search and similarity matching.

- **Purpose:** Store and search document embeddings
- **Index Name:** `policy-embeddings`
- **Embedding Dimension:** 768 (Google Gemini embeddings)
- **Search Metric:** Cosine similarity

### 3.2 Configuration

```python
# From gira-ai/gira-agent/config.py
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "aped-4627-b74a")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "government-policy-retrival-system")
```

### 3.3 Vector Storage Structure

```python
Vector Entry Format:
{
    "id": "doc_001_page_5",
    "values": [0.123, 0.456, ..., 0.789],  # 768-dimensional
    "metadata": {
        "document_id": "001",
        "page_number": 5,
        "title": "Government Policy Act",
        "section": "Eligibility",
        "content_snippet": "...",
        "document_type": "pis",
        "region": "us"
    }
}
```

### 3.4 Hybrid Search with BM25

Pinecone is combined with BM25 for hybrid search:

```python
# Dense search (semantic)
dense_results = pinecone_index.query(
    vector=query_embedding,
    top_k=30,
    filter={"region": "us"}
)

# Sparse search (keyword-based)
bm25_scores = await get_bm25_scores(query, corpus)

# Hybrid combining
hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score
```

---

## Part 4: Redis - Cache & Message Broker

### 4.1 Overview

**Redis** is an in-memory data structure store used for:
- Session caching
- Celery message broker
- Query result caching

**Container Image:** Latest stable Redis

### 4.2 Configuration

```yaml
redis:
  image: redis:latest
  container_name: gira-redis
  ports:
    - "127.0.0.1:6379:6379"
  networks:
    - gira-network
  restart: unless-stopped
```

### 4.3 Celery Integration

```python
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# From django settings
CELERY_ACCEPT_CONTENT = ["application/json"]
CELERY_RESULT_SERIALIZER = "json"
CELERY_TASK_SERIALIZER = "json"
```

### 4.4 Use Cases

1. **Background Task Queue:**
   - Document processing
   - Email notifications
   - Batch operations

2. **Caching:**
   - Session data
   - Query results
   - Vector embeddings

3. **Real-time Features:**
   - WebSocket connections
   - Live notifications
   - Rate limiting

---

## Part 5: MinIO - Object Storage

### 5.1 Overview

**MinIO** is an S3-compatible object storage system for storing files.

- **Purpose:** Store PDF documents and processed files
- **Bucket:** `government-policy-documents`
- **API Compatibility:** AWS S3

### 5.2 Configuration

```yaml
minio:
  image: minio/minio:latest
  container_name: gira-minio
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    MINIO_DOMAIN: ${MINIO_DOMAIN:-localhost}
  ports:
    - "127.0.0.1:9000:9000"  # API
    - "127.0.0.1:9001:9001"  # Console
  volumes:
    - minio_data:/data
  networks:
    - gira-network
```

### 5.3 File Types Stored

```
├── Documents/
│   ├── PDFs (raw uploads)
│   ├── Processed documents
│   └── Indexed chunks
├── Highlights/
│   └── Annotated PDFs
└── Backups/
    └── Database backups
```

---

## Part 6: Database Performance Optimization

### 6.1 Indexing Strategy

**PostgreSQL Indexes:**

```sql
-- User lookups
CREATE INDEX idx_auth_user_email ON auth_user(email);

-- Document queries
CREATE INDEX idx_documents_user_id ON documents_document(user_id);
CREATE INDEX idx_documents_status ON documents_document(status);

-- DPO feedback
CREATE INDEX idx_rlhf_feedback_user_id ON rlhf_feedback(user_id);
CREATE INDEX idx_rlhf_feedback_conversation ON rlhf_feedback(conversation_id);
```

### 6.2 Query Optimization

**Connection Pool Management:**
```python
# Development
pool_size=10, max_overflow=20

# Production
pool_size=20, max_overflow=30
```

**Connection Recycling:**
```python
# Production: Recycle connections every hour
pool_recycle=3600
```

### 6.3 Pinecone Optimization

- **Batch Upserts:** Update multiple embeddings at once
- **Filtering:** Use metadata filters to reduce search space
- **Sparse-Dense Hybrid:** Combine keyword and semantic search

---

## Part 7: Backup & Recovery

### 7.1 PostgreSQL Backup

**Automatic Backups:**
```bash
# Weekly full backups
pg_dump gira_db > gira_db_$(date +%Y%m%d).sql

# Continuous WAL archiving
wal_level = replica
```

### 7.2 Volume Persistence

```yaml
volumes:
  postgres_data:/var/lib/postgresql/data
  minio_data:/data
```

### 7.3 Recovery Procedures

**Point-in-Time Recovery:**
```bash
# Restore from backup
psql gira_db < gira_db_backup.sql

# Restore specific point
pg_restore -d gira_db gira_db.dump
```

---

## Part 8: Monitoring & Logging

### 8.1 PostgreSQL Logging

Configuration:
```yaml
command:
  - "postgres"
  - "-c"
  - "logging_collector=on"
  - "-c"
  - "log_statement=all"
  - "-c"
  - "log_connections=on"
```

**Logs Include:**
- All SQL statements
- Connection attempts
- Errors and warnings

### 8.2 Health Monitoring

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
  interval: "10s"
  timeout: "5s"
  retries: 3
```

### 8.3 Performance Monitoring

**Useful Queries:**
```sql
-- Active connections
SELECT * FROM pg_stat_activity;

-- Index usage
SELECT * FROM pg_stat_user_indexes;

-- Table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_stat_user_tables;
```

---

## Part 9: Database Scalability

### 9.1 Horizontal Scaling

**Read Replicas:**
```sql
-- PostgreSQL streaming replication
-- Primary: accepts writes
-- Replica: accepts reads only
```

**Sharding Strategies:**
- Shard by user_id
- Shard by document_id
- Geographic sharding by region

### 9.2 Vertical Scaling

**Current Resources:**
```yaml
limits:
  cpus: "2"
  memory: 2GB
reservations:
  cpus: "0.5"
  memory: 512MB
```

**Upgrade Path:**
```yaml
# Scale to 4 CPUs and 4GB for high traffic
limits:
  cpus: "4"
  memory: 4GB
reservations:
  cpus: "2"
  memory: 2GB
```

### 9.3 Caching Layer

**Redis for:**
- Session data
- Frequently accessed queries
- Vector cache
- Rate limiting

---

## Part 10: Database Security

### 10.1 Authentication

```yaml
POSTGRES_HOST_AUTH_METHOD: scram-sha-256
POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --auth-host=scram-sha-256"
```

### 10.2 Network Security

- Localhost-only binding
- Isolated Docker network
- No public internet exposure

### 10.3 Data Protection

```python
# Password hashing
from django.contrib.auth.hashers import make_password
hashed_password = make_password(password)

# PII filtering
from presidio_analyzer import AnalyzerEngine
```

### 10.4 Encryption

**At Rest:**
- Use encrypted volumes in production
- PostgreSQL native encryption

**In Transit:**
- SSL/TLS for database connections
- HTTPS for API endpoints

---

## Summary Table

| Database | Purpose | Version | Scale | Status |
|----------|---------|---------|-------|--------|
| **PostgreSQL** | Primary data store | 16-alpine | Multi-GB | Production-ready |
| **Pinecone** | Vector search | Cloud | Millions of vectors | Cloud service |
| **Redis** | Cache & broker | Latest | Session data | Essential |
| **MinIO** | Object storage | Latest | Large files | Self-hosted |

---

## Conclusion

GIRA uses a sophisticated multi-database architecture:

✅ **PostgreSQL** for structured, transactional data
✅ **Pinecone** for semantic search at scale
✅ **Redis** for fast caching and messaging
✅ **MinIO** for scalable document storage

This combination provides:
- High performance and reliability
- Scalability for enterprise use
- Security and data protection
- Cost-effectiveness through open-source solutions

---

*For production deployment, ensure all databases are properly configured with SSL, backups, monitoring, and replication.*

