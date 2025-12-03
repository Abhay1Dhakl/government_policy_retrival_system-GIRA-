#!/usr/bin/env python3
"""
Database setup script for GIRA AI system
"""
import sys
import os
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import create_tables, drop_tables, engine
from sqlalchemy import text

def create_database():
    """Create the database if it doesn't exist"""
    load_dotenv()
    
    # Import the database URL function from config
    from database.config import get_database_url
    database_url = get_database_url()
    
    # Extract database name and connection details
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    
    db_name = parsed.path[1:]  # Remove leading slash
    server_url = f"{parsed.scheme}://{parsed.netloc}/postgres"
    
    print(f"Creating database: {db_name}")
    
    try:
        # Connect to postgres database to create our database
        from sqlalchemy import create_engine
        temp_engine = create_engine(server_url)
        
        with temp_engine.connect() as conn:
            conn.execute(text("COMMIT"))  # End any existing transaction
            
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name}
            ).fetchone()
            
            if not result:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"Database '{db_name}' created successfully")
            else:
                print(f"Database '{db_name}' already exists")
                
    except Exception as e:
        print(f"Error creating database: {e}")
        print("Make sure PostgreSQL is running and credentials are correct")
        return False
    
    return True

def setup_tables():
    """Create all tables"""
    try:
        print("Creating database tables...")
        create_tables()
        print("Database tables created successfully")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def reset_database():
    """Drop and recreate all tables"""
    print("WARNING: This will delete all data!")
    response = input("Are you sure you want to reset the database? (yes/no): ")
    
    if response.lower() == 'yes':
        try:
            print("Dropping existing tables...")
            drop_tables()
            print("Creating new tables...")
            create_tables()
            print("Database reset successfully")
            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
    else:
        print("Database reset cancelled")
        return False

def main():
    load_dotenv()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            success = create_database()
            if success:
                setup_tables()
        elif command == "setup":
            setup_tables()
        elif command == "reset":
            reset_database()
        else:
            print("Unknown command. Use: create, setup, or reset")
    else:
        print("GIRA Database Setup")
        print("Commands:")
        print("  python setup_db.py create  - Create database and tables")
        print("  python setup_db.py setup   - Create tables only")
        print("  python setup_db.py reset   - Reset all data (WARNING: destructive)")

if __name__ == "__main__":
    main()
