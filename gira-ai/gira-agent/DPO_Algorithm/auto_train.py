"""
Auto-train module for DPO (Direct Preference Optimization)
Includes functions for feedback counting, data export, fine-tuning, and model registration.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Optional
import sys

# Add parent directory to path to access database models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import DPO_RLHF
from sqlalchemy.orm import Session
from database.config import get_db

# Configuration
MIN_NEW_FEEDBACK = 200
DPO_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def count_new_feedback() -> int:
    """
    Count the number of new feedback entries available for training.
    
    Returns:
        int: Number of unused feedback entries
    """
    try:
        db_gen = get_db()
        db: Session = next(db_gen)
        
        count = db.query(DPO_RLHF).filter(
            DPO_RLHF.used_in_training == False,
            DPO_RLHF.feedback != 0  # Only positive/negative feedback
        ).count()
        
        print(f"Found {count} new feedback entries")
        return count
    except Exception as e:
        print(f"Error counting feedback: {e}")
        return 0


def run_export() -> Optional[str]:
    """
    Export unused feedback data to JSONL format for DPO training.
    
    Returns:
        str: Path to the exported JSONL file, or None if failed
    """
    try:
        db_gen = get_db()
        db: Session = next(db_gen)
        
        # Get unused feedback
        feedback_entries = db.query(DPO_RLHF).filter(
            DPO_RLHF.used_in_training == False,
            DPO_RLHF.feedback != 0
        ).all()
        
        if not feedback_entries:
            print("No feedback to export")
            return None
        
        # Create export directory
        os.makedirs(DPO_DATA_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dpo_pairs_{timestamp}.jsonl"
        filepath = os.path.join(DPO_DATA_DIR, filename)
        
        training_data = []
        for entry in feedback_entries:
            # We need pairs of (prompt, chosen, rejected)
            # For this simplified implementation, we assume we can construct pairs
            # This is a placeholder logic - real DPO needs pairs of responses
            
            # If feedback is positive, treat current response as chosen
            if entry.feedback > 0:
                item = {
                    "prompt": entry.user_query,
                    "chosen": entry.assistant_response,
                    "rejected": "I apologize, but I cannot provide a helpful answer at this time." # Placeholder rejected
                }
            # If feedback is negative, treat current response as rejected
            else:
                item = {
                    "prompt": entry.user_query,
                    "chosen": "Please consult official government sources for this information.", # Placeholder chosen
                    "rejected": entry.assistant_response
                }
                
            training_data.append(item)
            
        # Write to JSONL
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
                
        print(f"Exported {len(training_data)} training pairs to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error exporting feedback: {e}")
        return None


def fine_tune(jsonl_file: str) -> Optional[str]:
    """
    Run DPO fine-tuning using the exported data.
    
    Args:
        jsonl_file: Path to the training data file
        
    Returns:
        str: ID of the new model, or None if failed
    """
    print(f"Starting fine-tuning with data from {jsonl_file}...")
    
    # In a real scenario, this would trigger a training job (e.g., on SageMaker, OpenAI, etc.)
    # For GIRA simulation, we'll mock the training process
    
    try:
        # Simulate training time
        time.sleep(5) 
        
        # Generate a new model ID
        new_model_id = f"gira-dpo-{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        print(f"Fine-tuning complete. New model ID: {new_model_id}")
        return new_model_id
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return None


def register_new_model(model_id: str) -> bool:
    """
    Register the newly trained model in the system registry.
    
    Args:
        model_id: ID of the model to register
        
    Returns:
        bool: True if successful
    """
    print(f"Registering new model: {model_id}...")
    
    # Update config/settings or database with new model ID
    # This is a placeholder for the actual registration logic
    
    print(f"Model {model_id} successfully registered and ready for deployment")
    return True


def mark_feedback_used() -> int:
    """
    Mark all currently unused feedback as used in training.
    
    Returns:
        int: Number of updated records
    """
    try:
        db_gen = get_db()
        db: Session = next(db_gen)
        
        # Update records
        result = db.query(DPO_RLHF).filter(
            DPO_RLHF.used_in_training == False,
            DPO_RLHF.feedback != 0
        ).update({DPO_RLHF.used_in_training: True})
        
        db.commit()
        print(f"Marked {result} feedback entries as used")
        return result
        
    except Exception as e:
        print(f"Error marking feedback as used: {e}")
        db.rollback()
        return 0
