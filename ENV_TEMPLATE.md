# Docker Environment Configuration Template

# Copy this file to .env and fill in the missing values
# Run this command to generate FERNET_KEY:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# =================================================================
# DATABASE CONFIGURATION
# =================================================================
POSTGRES_DB=gira_db
POSTGRES_USER=gira_user
POSTGRES_PASSWORD=<your_secure_password>
POSTGRES_HOST=postgres
POSTGRES_DOCKER_HOST=postgres
POSTGRES_INTERNAL_PORT=5432
POSTGRES_EXTERNAL_PORT=5432
DATABASE_URL_PROD=postgresql://gira_user:<your_secure_password>@postgres:5432/gira_db

# =================================================================
# SECURITY
# =================================================================
# Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FERNET_KEY=<generate_this>

# =================================================================
# GIRA AGENT SERVICE
# =================================================================
GIRA_AGENT_INTERNAL_PORT=8081
GIRA_AGENT_EXTERNAL_PORT=8081
GIRA_AGENT_DOCKER_URL=http://gira-agent:8081

# =================================================================
# GIRA BACKEND SERVICE
# =================================================================
GIRA_BACKEND_INTERNAL_PORT=8082
GIRA_BACKEND_EXTERNAL_PORT=8082
DJANGO_PORT=8082

# =================================================================
# GIRA MCP SERVER
# =================================================================
GIRA_MCP_SERVER_INTERNAL_PORT=8085
GIRA_MCP_SERVER_EXTERNAL_PORT=8085
MCP_SERVER_URL_DEV=http://localhost:8085
MCP_SERVER_URL_PROD=http://gira-mcp-server:8085
MCP_SERVER_TRANSPORT=http

# =================================================================
# GIRA FRONTEND
# =================================================================
GIRA_FRONTEND_INTERNAL_PORT=3000
GIRA_FRONTEND_EXTERNAL_PORT=3000
GIRA_FRONTEND_DOCKER_URL=http://gira-frontend:3000
NODE_ENV=development
NEXT_TELEMETRY_DISABLED=1

# =================================================================
# API CONFIGURATION
# =================================================================
GIRA_API_BASE_URL=http://gira-backend:8082/api/v1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8082/api/v1
NEXT_PUBLIC_CHAT_API_BASE_URL=http://localhost:8081

# =================================================================
# EXTERNAL APIS
# =================================================================
OPENAI_API_KEY=<your_openai_api_key>
OPENAI_BASE_URL=https://api.openai.com/v1
PINECONE_API_KEY=<your_pinecone_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
NEXT_PUBLIC_GOOGLE_CLIENT_ID=<your_google_client_id>

# =================================================================
# MINIO (Object Storage)
# =================================================================
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT=minio:9000
MINIO_DOMAIN=localhost
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_SECURE=false

# =================================================================
# REDIS & CELERY
# =================================================================
REDIS_HOST=redis
REDIS_PORT=6379
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# =================================================================
# AIRFLOW
# =================================================================
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# =================================================================
# APPLICATION SETTINGS
# =================================================================
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
TOKENIZERS_PARALLELISM=false
PDF_CLEANUP_DELAY=3600

# Email Configuration (Optional)
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=<your_email>
EMAIL_HOST_PASSWORD=<your_app_password>
DEFAULT_FROM_EMAIL=<your_email>

# JWT Token Configuration
ACCESS_TOKEN_LIFETIME=7
REFRESH_TOKEN_LIFETIME=30

# Django Settings
DJANGO_SETTINGS_MODULE=gira.settings

# CORS Origins
CORS_ORIGINS_DEV=http://localhost:3000,http://localhost:8081,http://localhost:8082
CORS_ORIGINS_PROD=https://gira.govinfo.com

# Timeout settings
timeout=120
