"""
Title Generation Service
Generates concise titles for medical queries using LLM
"""
import os
import requests
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def generate_title(user_query: str) -> str:
    """
    Generate a concise title for the user's query using OpenAI's GPT model.
    
    Args:
        user_query: The user's medical question
        
    Returns:
        Generated title or "Untitled Query" on error
    """
    try:
        if not settings.OPENAI_API_KEY or not settings.OPENAI_BASE_URL:
            raise ValueError("OpenAI API key or base URL is not set in environment variables.")
        
        logger.debug(f"Generating title for query: {user_query[:50]}...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
        }
        
        prompt = f"Generate a concise title (max 10 words) for the following medical question:\n\n{user_query}\n\nTitle:"
        
        payload = {
            "model": settings.DEFAULT_LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates concise titles."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 20,
            "temperature": 0.3,
            "n": 1,
            "stop": ["\n"]
        }
        
        response = requests.post(
            f"{settings.OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        logger.debug(f"OpenAI response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        title = data['choices'][0]['message']['content'].strip()
        
        logger.info(f"Generated title: {title}")
        return title
    
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return "Untitled Query"
