# Use lightweight SimpleLLM instead of heavy langchain packages
from llm_options.llm_choose import SimpleLLM
import os
from typing import Union, Dict, List
from dotenv import load_dotenv

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Configuration
LLM_CONFIGS = {
    # OpenAI Models
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "OpenAI GPT-4o - Most capable model"
    },
    "gpt-4-turbo": {
        "provider": "openai", 
        "model": "gpt-4-turbo",
        "description": "OpenAI GPT-4 Turbo - Fast and capable"
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo", 
        "description": "OpenAI GPT-3.5 Turbo - Fast and cost-effective"
    },
    "gemini-flash": {
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "description": "Google Gemini Flash - Fast policy assistant fallback"
    },
    
    # Anthropic Models
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "description": "Anthropic Claude 3.5 Sonnet - Excellent reasoning"
    },
    "claude-3-opus": {
        "provider": "anthropic", 
        "model": "claude-3-opus-20240229",
        "description": "Anthropic Claude 3 Opus - Most capable Claude model"
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307", 
        "description": "Anthropic Claude 3 Haiku - Fast and efficient"
    },
    
    # Mistral Models (via OpenAI-compatible API)
    "mistral-large": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "description": "Mistral Large - High performance model"
    },
    "mistral-medium": {
        "provider": "mistral",
        "model": "mistral-medium-latest", 
        "description": "Mistral Medium - Balanced performance"
    },
    "mistral-small": {
        "provider": "mistral",
        "model": "mistral-small-latest",
        "description": "Mistral Small - Fast and efficient"
    },
    
    # Groq Models (via OpenAI-compatible API)
    "groq-llama-70b": {
        "provider": "groq",
        "model": "llama2-70b-4096",
        "description": "Groq Llama 2 70B - Ultra-fast inference"
    },
    "groq-mixtral": {
        "provider": "groq", 
        "model": "mixtral-8x7b-32768",
        "description": "Groq Mixtral 8x7B - Fast mixture of experts"
    },
    
    # DeepSeek Models (via OpenAI-compatible API)
    "deepseek-coder": {
        "provider": "deepseek",
        "model": "deepseek-coder",
        "description": "DeepSeek Coder - Specialized for coding tasks"
    },
    "deepseek-chat": {
        "provider": "deepseek",
        "model": "deepseek-chat", 
        "description": "DeepSeek Chat - General conversation model"
    },
    
    # Local Ollama Models
    "llama3.1-8b": {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "description": "Llama 3.1 8B - Local deployment"
    },
    "llama3.1-70b": {
        "provider": "ollama",
        "model": "llama3.1:70b", 
        "description": "Llama 3.1 70B - Local deployment (requires GPU)"
    },
    "mistral-7b": {
        "provider": "ollama",
        "model": "mistral:7b",
        "description": "Mistral 7B - Local deployment"
    },
    "codellama": {
        "provider": "ollama",
        "model": "codellama:13b",
        "description": "Code Llama 13B - Local coding assistant"
    }
}

def get_available_llms() -> Dict[str, Dict]:
    """Get all available LLM options with descriptions"""
    return LLM_CONFIGS

def get_llms_by_provider(provider: str) -> List[str]:
    """Get LLM options for a specific provider"""
    return [llm_id for llm_id, config in LLM_CONFIGS.items() 
            if config["provider"] == provider]

def choose_llm(llm_option: str, temperature: float = 0.01):
    """
    Choose the LLM based on the provided option - Lightweight without langchain
    
    Args:
        llm_option (str): The LLM provider (e.g., 'openai', 'anthropic')
        temperature (float): Temperature for the model (default: 0.01)
        
    Returns:
        SimpleLLM instance ready for use
        
    Raises:
        ValueError: If unsupported LLM option or missing API keys
    """
    normalized_option = {
        "chatgpt": "gpt-4o",
        "openai": "gpt-4o",
        "gemini": "gemini-flash",
        "claude": "claude-3.5-sonnet",
        "anthropic": "claude-3.5-sonnet",
        "llama": "llama3.1-8b",
        "grok": "gpt-4o",
    }.get(llm_option, llm_option)

    llm_config = LLM_CONFIGS.get(normalized_option)
    if llm_config:
        provider = llm_config["provider"]
        model = llm_config["model"]
    else:
        # Backward compatibility for callers that still pass a provider name.
        provider = normalized_option
        model = "gpt-4o"

    print(f"LLM Provider: {provider}")
    
    try:
        if provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")

            return SimpleLLM(
                provider="gemini",
                model=model,
                api_key=GEMINI_API_KEY,
                temperature=temperature,
            )

        if provider == "openai":
            if not OPENAI_API_KEY:
                if GEMINI_API_KEY:
                    return SimpleLLM(
                        provider="gemini",
                        model=GEMINI_MODEL,
                        api_key=GEMINI_API_KEY,
                        temperature=temperature,
                    )
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            
            return SimpleLLM(
                provider="openai",
                model=model,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=temperature
            )
            
        # elif provider == "anthropic":
        #     if not ANTHROPIC_API_KEY:
        #         raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic models")
            
        #     return ChatAnthropic(
        #         model=model,
        #         temperature=temperature,
        #         anthropic_api_key=ANTHROPIC_API_KEY,
        #         timeout=30,
        #         max_retries=2
        #     )
            
        # elif provider == "mistral":
        #     if not MISTRAL_API_KEY:
        #         raise ValueError("MISTRAL_API_KEY environment variable is required for Mistral models")
            
        #     return ChatOpenAI(
        #         model=model,
        #         temperature=temperature,
        #         openai_api_key=MISTRAL_API_KEY,
        #         openai_api_base="https://api.mistral.ai/v1",
        #         timeout=30,
        #         max_retries=2
        #     )
            
        # elif provider == "groq":
        #     if not GROQ_API_KEY:
        #         raise ValueError("GROQ_API_KEY environment variable is required for Groq models")
            
        #     return ChatOpenAI(
        #         model=model,
        #         temperature=temperature,
        #         openai_api_key=GROQ_API_KEY,
        #         openai_api_base="https://api.groq.com/openai/v1",
        #         timeout=30,
        #         max_retries=2
        #     )
            
        # elif provider == "deepseek":
        #     if not DEEPSEEK_API_KEY:
        #         raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek models")
            
        #     return ChatOpenAI(
        #         model=model,
        #         temperature=temperature,
        #         openai_api_key=DEEPSEEK_API_KEY,
        #         openai_api_base="https://api.deepseek.com/v1",
        #         timeout=30,
        #         max_retries=2
        #     )
            
        # elif provider == "ollama":
        #     return ChatOllama(
        #         model=model,
        #         temperature=temperature,
        #         base_url=OLLAMA_BASE_URL,
        #         timeout=60  # Local models might need more time
        #     )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        raise ValueError(f"Failed to initialize {normalized_option}: {str(e)}")

def validate_llm_setup(llm_option: str) -> Dict[str, Union[bool, str]]:
    """
    Validate if the LLM can be properly initialized
    
    Returns:
        Dict with validation status and message
    """
    try:
        llm = choose_llm(llm_option)
        return {
            "valid": True,
            "message": f" {llm_option} is properly configured"
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"❌ {llm_option} configuration error: {str(e)}"
        }
