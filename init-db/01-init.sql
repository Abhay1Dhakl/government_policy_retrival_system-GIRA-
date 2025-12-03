-- Create Airflow database and user
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
ALTER USER airflow CREATEDB;

-- Connect to the airflow database and grant schema permissions
\c airflow;

-- Grant all permissions on public schema to airflow user
GRANT ALL ON SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO airflow;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO airflow;

-- Make airflow the owner of the public schema
ALTER SCHEMA public OWNER TO airflow;

-- You can add any additional initialization SQL here
SELECT 'Databases gira_db and airflow are ready for use!' as message;
