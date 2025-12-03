"""
DAG for automated DPO fine-tuning process.
Runs weekly to check for new feedback and trigger model updates.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

# Calculate paths relative to the Airflow container structure
# The GIRA_AGENT directory is mounted at /opt/airflow/gira_agent
GIRA_AGENT_PATH = '/opt/airflow/gira_agent'

# Add MIRA agent to Python path
sys.path.append(GIRA_AGENT_PATH)

# Set environment variables for database and other configurations
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres:postgres@postgres/gira_db')
os.environ.setdefault('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY', ''))

from DPO_Algorithm.auto_train import (
    count_new_feedback,
    run_export,
    fine_tune,
    register_new_model,
    mark_feedback_used
)

# DAG configuration
default_args = {
    'owner': 'gira',
    'depends_on_past': False,
    'email': ['abhaya@ubventuresllc.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,  # Increased retries
    'retry_delay': timedelta(minutes=10),  # Increased retry delay
    'execution_timeout': timedelta(hours=4),  # Max execution time
    'on_failure_callback': None,  # Can add a failure notification function here
}

dag = DAG(
    'mira_dpo_training',
    default_args=default_args,
    description='Weekly DPO fine-tuning pipeline for GIRA AI',
    schedule_interval='0 0 * * 0',  # Run at midnight every Sunday
    start_date=days_ago(1),
    catchup=False,
    tags=['gira', 'dpo', 'training'],
)

def check_feedback_count(**context):
    """Check if we have enough new feedback for training"""
    count = count_new_feedback()
    if count < 200:  # MIN_NEW_FEEDBACK
        raise Exception(f"Not enough new feedback (have {count}, need 200)")
    return count

def export_feedback(**context):
    """Export feedback data to JSONL"""
    jsonl_file = run_export()
    if not jsonl_file:
        raise Exception("Failed to export feedback data")
    return jsonl_file

def run_fine_tuning(**context):
    """Run the fine-tuning process"""
    ti = context['task_instance']
    jsonl_file = ti.xcom_pull(task_ids='export_feedback')
    
    model_id = fine_tune(jsonl_file)
    if not model_id:
        raise Exception("Fine-tuning failed")
    return model_id

def register_model(**context):
    """Register the new model"""
    ti = context['task_instance']
    model_id = ti.xcom_pull(task_ids='run_fine_tuning')
    register_new_model(model_id)
    return model_id

# Define tasks
check_feedback = PythonOperator(
    task_id='check_feedback',
    python_callable=check_feedback_count,
    dag=dag,
)

export_data = PythonOperator(
    task_id='export_feedback',
    python_callable=export_feedback,
    dag=dag,
)

fine_tuning = PythonOperator(
    task_id='run_fine_tuning',
    python_callable=run_fine_tuning,
    dag=dag,
)

register = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

mark_used = PythonOperator(
    task_id='mark_feedback_used',
    python_callable=mark_feedback_used,
    dag=dag,
)

# Function to clean up old files
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

# Clean up old files
cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_old_files,
    dag=dag,
)

# Define task dependencies
check_feedback >> export_data >> fine_tuning >> register >> mark_used >> cleanup
