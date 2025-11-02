from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    """
    Get database URL based on environment configuration.
    
    Returns:
        str: Database connection URL
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
        database = os.getenv("POSTGRES_DB", "mira_db")
        username = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        
        db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    print(f"[Database] Environment: {environment}")
    print(f"[Database] Using database URL: {db_url}")
    
    return db_url

# Database configuration
DATABASE_URL = get_database_url()

# Create engine with environment-aware configuration
def get_engine_config():
    """Get SQLAlchemy engine configuration based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Base configuration
    config = {
        "pool_pre_ping": True,
        "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true"
    }
    
    # Environment-specific configurations
    if environment == "production":
        config.update({
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "20")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "30")),
            "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),  # 1 hour
        })
    else:
        config.update({
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "20")),
        })
    
    return config

# Create engine
engine_config = get_engine_config()
engine = create_engine(DATABASE_URL, **engine_config)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get database session
def get_db():
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

def create_tables():
    """Create all tables"""
    # Import Base AFTER it's been defined in models.py
    from .models import Base
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print(" Tables created successfully!")

def drop_tables():
    """Drop all tables"""
    from .models import Base
    Base.metadata.drop_all(bind=engine)
