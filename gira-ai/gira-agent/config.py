"""
Configuration settings for GIRA AI Agent
Centralizes all environment variables and constants for better maintainability
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration"""
    
    # ========== API Keys ==========
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "aped-4627-b74a")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "quickstart-py")
    
    # ========== MinIO Configuration ==========
    MINIO_ENDPOINT: Optional[str] = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: Optional[str] = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: Optional[str] = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "medical-documents")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    # ========== Database Configuration ==========
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # ========== Performance Settings ==========
    MAX_MATCHES_TO_PROCESS: int = int(os.getenv("MAX_MATCHES_TO_PROCESS", "15"))
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "25000"))
    PAST_TEXT_PREVIEW_LENGTH: int = int(os.getenv("PAST_TEXT_PREVIEW_LENGTH", "200"))
    
    # ========== LLM Configuration ==========
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    
    # ========== CORS Configuration ==========
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    CORS_ORIGINS_PROD: str = os.getenv("CORS_ORIGINS_PROD", "")
    CORS_ORIGINS_DEV: str = "http://localhost:8000,http://localhost:3535,http://localhost:3000"
    
    # ========== PDF Settings ==========
    PDF_CLEANUP_DELAY: int = int(os.getenv("PDF_CLEANUP_DELAY", "3600"))
    
    # ========== Embedding Model ==========
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # ========== Citation Validation ==========
    STRICT_CITATION_VALIDATION: bool = os.getenv("STRICT_CITATION_VALIDATION", "true").lower() == "true"
    
    # ========== MCP Configuration ==========
    MCP_CONFIG_DEV: str = "mcp_server_config/config_development.json"
    MCP_CONFIG_PROD: str = "mcp_server_config/config_production.json"
    
    @property
    def mcp_config_file(self) -> str:
        """Get MCP config file based on environment"""
        return self.MCP_CONFIG_PROD if self.ENVIRONMENT == "production" else self.MCP_CONFIG_DEV
    
    @property
    def cors_origins(self) -> list[str]:
        """Get CORS origins based on environment"""
        if self.ENVIRONMENT == "production":
            origins_str = self.CORS_ORIGINS_PROD
            origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
            
            # Fallback for production
            if not origins:
                origins = ["https://gira-backend.medgentics.com", "https://gira.medgentics.com"]
            
            # Add internal Docker network origins
            internal_origins = [
                "http://gira-backend:8082",
                "http://172.21.0.6:8082",
                "http://gira-backend",
                "https://gira-backend.medgentics.com",
                "http://gira-backend.medgentics.com",
            ]
            origins.extend(internal_origins)
        else:
            origins = [origin.strip() for origin in self.CORS_ORIGINS_DEV.split(",") if origin.strip()]
        
        return origins
    
    def validate(self) -> list[str]:
        """Validate critical settings and return list of errors"""
        errors = []
        
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        if not self.OPENAI_BASE_URL:
            errors.append("OPENAI_BASE_URL is not set")
        if not self.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL is not set")
            
        return errors


# Global settings instance
settings = Settings()


# Validate settings on import
validation_errors = settings.validate()
if validation_errors:
    print(f"⚠️  Configuration warnings:\n" + "\n".join(f"  - {err}" for err in validation_errors))
