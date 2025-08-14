"""Model interfaces and implementations for different AI providers."""

from .model_interface import ModelInterface
from .openrouter_model import OpenRouterModel
from .model_registry import model_registry, get_model_config, list_available_models, get_recommended_models

# Legacy model imports (kept for backward compatibility)
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel

__all__ = [
    "ModelInterface", 
    "OpenRouterModel", 
    "model_registry",
    "get_model_config",
    "list_available_models", 
    "get_recommended_models",
    # Legacy exports
    "OpenAIModel", 
    "AnthropicModel"
]