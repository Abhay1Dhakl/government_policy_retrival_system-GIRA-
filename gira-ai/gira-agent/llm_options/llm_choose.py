from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import httpx
import json
from typing import Union, Dict, List, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()

class SimpleLLM:
    """Lightweight LLM wrapper without langchain"""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.01):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        
        if provider in ["openai", "mistral", "groq", "deepseek"]:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        elif provider == "anthropic":
            self.client = AsyncAnthropic(api_key=api_key)
        elif provider == "ollama":
            self.client = None  # Use httpx directly
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def ainvoke(self, messages: List[Dict], tools: Optional[List] = None, **kwargs):
        """Async invoke method"""
        if self.provider in ["openai", "mistral", "groq", "deepseek"]:
            return await self._invoke_openai(messages, tools)
        elif self.provider == "anthropic":
            return await self._invoke_anthropic(messages, tools)
        elif self.provider == "ollama":
            return await self._invoke_ollama(messages, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _invoke_openai(self, messages, tools):
        """Invoke OpenAI-compatible API"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            params["tools"] = tools
        
        response = await self.client.chat.completions.create(**params)
        return response
    
    async def _invoke_anthropic(self, messages, tools):
        """Invoke Anthropic API"""
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(msg)
        
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        if system_message:
            params["system"] = system_message
        if tools:
            params["tools"] = tools
        
        response = await self.client.messages.create(**params)
        
        # Convert to OpenAI-like format
        tool_calls = []
        if response.stop_reason == "tool_use":
            for content in response.content:
                if content.type == "tool_use":
                    tool_calls.append({
                        "id": content.id,
                        "type": "function",
                        "function": {
                            "name": content.name,
                            "arguments": json.dumps(content.input)
                        }
                    })
        
        content = ""
        for c in response.content:
            if c.type == "text":
                content += c.text
        
        # Create a mock OpenAI response
        class MockMessage:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
        
        return MockResponse([MockChoice(MockMessage(content, tool_calls))])
    
    async def _invoke_ollama(self, messages, tools):
        """Invoke Ollama API"""
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        if tools:
            data["tools"] = tools
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
            result = response.json()
        
        message = result["message"]
        
        # Convert to OpenAI-like format
        tool_calls = message.get("tool_calls", [])
        
        class MockMessage:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
        
        return MockResponse([MockChoice(MockMessage(message.get("content", ""), tool_calls))])

# Environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# # LLM Configuration
# LLM_CONFIGS = {
#     # OpenAI Models
#     "gpt-4o": {
#         "provider": "openai",
#         "model": "gpt-4o",
#         "description": "OpenAI GPT-4o - Most capable model"
#     },
#     "gpt-4-turbo": {
#         "provider": "openai", 
#         "model": "gpt-4-turbo",
#         "description": "OpenAI GPT-4 Turbo - Fast and capable"
#     },
#     "gpt-3.5-turbo": {
#         "provider": "openai",
#         "model": "gpt-3.5-turbo", 
#         "description": "OpenAI GPT-3.5 Turbo - Fast and cost-effective"
#     },
    
#     # Anthropic Models
#     "claude-3.5-sonnet": {
#         "provider": "anthropic",
#         "model": "claude-3-5-sonnet-20241022",
#         "description": "Anthropic Claude 3.5 Sonnet - Excellent reasoning"
#     },
#     "claude-3-opus": {
#         "provider": "anthropic", 
#         "model": "claude-3-opus-20240229",
#         "description": "Anthropic Claude 3 Opus - Most capable Claude model"
#     },
#     "claude-3-haiku": {
#         "provider": "anthropic",
#         "model": "claude-3-haiku-20240307", 
#         "description": "Anthropic Claude 3 Haiku - Fast and efficient"
#     },
    
#     # Mistral Models (via OpenAI-compatible API)
#     "mistral-large": {
#         "provider": "mistral",
#         "model": "mistral-large-latest",
#         "description": "Mistral Large - High performance model"
#     },
#     "mistral-medium": {
#         "provider": "mistral",
#         "model": "mistral-medium-latest", 
#         "description": "Mistral Medium - Balanced performance"
#     },
#     "mistral-small": {
#         "provider": "mistral",
#         "model": "mistral-small-latest",
#         "description": "Mistral Small - Fast and efficient"
#     },
    
#     # Groq Models (via OpenAI-compatible API)
#     "groq-llama-70b": {
#         "provider": "groq",
#         "model": "llama2-70b-4096",
#         "description": "Groq Llama 2 70B - Ultra-fast inference"
#     },
#     "groq-mixtral": {
#         "provider": "groq", 
#         "model": "mixtral-8x7b-32768",
#         "description": "Groq Mixtral 8x7B - Fast mixture of experts"
#     },
    
#     # DeepSeek Models (via OpenAI-compatible API)
#     "deepseek-coder": {
#         "provider": "deepseek",
#         "model": "deepseek-coder",
#         "description": "DeepSeek Coder - Specialized for coding tasks"
#     },
#     "deepseek-chat": {
#         "provider": "deepseek",
#         "model": "deepseek-chat", 
#         "description": "DeepSeek Chat - General conversation model"
#     },
    
#     # Local Ollama Models
#     "llama3.1-8b": {
#         "provider": "ollama",
#         "model": "llama3.1:8b",
#         "description": "Llama 3.1 8B - Local deployment"
#     },
#     "llama3.1-70b": {
#         "provider": "ollama",
#         "model": "llama3.1:70b", 
#         "description": "Llama 3.1 70B - Local deployment (requires GPU)"
#     },
#     "mistral-7b": {
#         "provider": "ollama",
#         "model": "mistral:7b",
#         "description": "Mistral 7B - Local deployment"
#     },
#     "codellama": {
#         "provider": "ollama",
#         "model": "codellama:13b",
#         "description": "Code Llama 13B - Local coding assistant"
#     }
# }

# def get_available_llms() -> Dict[str, Dict]:
#     """Get all available LLM options with descriptions"""
#     return LLM_CONFIGS

# def get_llms_by_provider(provider: str) -> List[str]:
#     """Get LLM options for a specific provider"""
#     return [llm_id for llm_id, config in LLM_CONFIGS.items() 
#             if config["provider"] == provider]

# def choose_llm(llm_option: str, temperature: float = 0.01) -> Union[ChatOpenAI, ChatAnthropic, ChatOllama]:
#     """
#     Choose the LLM based on the provided option
    
#     Args:
#         llm_option (str): The LLM identifier (e.g., 'gpt-4o', 'claude-3.5-sonnet')
#         temperature (float): Temperature for the model (default: 0.01)
        
#     Returns:
#         LLM instance ready for use
        
#     Raises:
#         ValueError: If unsupported LLM option or missing API keys
#     """
    
#     # if llm_option not in LLM_CONFIGS:
#     #     raise ValueError(f"Unsupported LLM option: {llm_option}. Available options: {list(LLM_CONFIGS.keys())}")
    
#     # config = LLM_CONFIGS[llm_option]
#     provider = llm_option
#     print(f"LLM Provider: {provider}")
#     model = "gpt-4o"
    
#     try:
#         if provider == "openai":
#             if not OPENAI_API_KEY:
#                 raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            
#             return ChatOpenAI(
#                 model=model,
#                 temperature=temperature,
#                 openai_api_key=OPENAI_API_KEY,
#                 openai_api_base=OPENAI_BASE_URL,
#                 timeout=30,
#                 max_retries=2
#             )
            
#         elif provider == "anthropic":
#             if not ANTHROPIC_API_KEY:
#                 raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic models")
            
#             return ChatAnthropic(
#                 model=model,
#                 temperature=temperature,
#                 anthropic_api_key=ANTHROPIC_API_KEY,
#                 timeout=30,
#                 max_retries=2
#             )
            
#         elif provider == "mistral":
#             if not MISTRAL_API_KEY:
#                 raise ValueError("MISTRAL_API_KEY environment variable is required for Mistral models")
            
#             return ChatOpenAI(
#                 model=model,
#                 temperature=temperature,
#                 openai_api_key=MISTRAL_API_KEY,
#                 openai_api_base="https://api.mistral.ai/v1",
#                 timeout=30,
#                 max_retries=2
#             )
            
#         elif provider == "groq":
#             if not GROQ_API_KEY:
#                 raise ValueError("GROQ_API_KEY environment variable is required for Groq models")
            
#             return ChatOpenAI(
#                 model=model,
#                 temperature=temperature,
#                 openai_api_key=GROQ_API_KEY,
#                 openai_api_base="https://api.groq.com/openai/v1",
#                 timeout=30,
#                 max_retries=2
#             )
            
#         elif provider == "deepseek":
#             if not DEEPSEEK_API_KEY:
#                 raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek models")
            
#             return ChatOpenAI(
#                 model=model,
#                 temperature=temperature,
#                 openai_api_key=DEEPSEEK_API_KEY,
#                 openai_api_base="https://api.deepseek.com/v1",
#                 timeout=30,
#                 max_retries=2
#             )
            
#         elif provider == "ollama":
#             return ChatOllama(
#                 model=model,
#                 temperature=temperature,
#                 base_url=OLLAMA_BASE_URL,
#                 timeout=60  # Local models might need more time
#             )
            
#         else:
#             raise ValueError(f"Unsupported provider: {provider}")
            
#     except Exception as e:
#         raise ValueError(f"Failed to initialize {llm_option}: {str(e)}")

# def validate_llm_setup(llm_option: str) -> Dict[str, Union[bool, str]]:
#     """
#     Validate if the LLM can be properly initialized
    
#     Returns:
#         Dict with validation status and message
#     """
#     try:
#         llm = choose_llm(llm_option)
#         return {
#             "valid": True,
#             "message": f"✅ {llm_option} is properly configured"
#         }
#     except Exception as e:
#         return {
#             "valid": False,
#             "message": f"❌ {llm_option} configuration error: {str(e)}"
#         }