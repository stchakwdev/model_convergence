"""
Model registry for OpenRouter integration.

This module provides centralized configuration and management for all available
models through OpenRouter's unified API, including the latest 2024-2025 models
like GLM-4.5, Kimi-K2, Qwen-3, and GPT-OSS.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ModelFamily(Enum):
    """Model family categories for organization and analysis."""
    GPT_OSS = "gpt-oss"
    GLM = "glm"
    KIMI = "kimi" 
    QWEN = "qwen"
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    OTHER = "other"


class ModelTier(Enum):
    """Cost tiers for model selection."""
    FREE = "free"
    LOW_COST = "low_cost"
    MEDIUM_COST = "medium_cost"
    HIGH_COST = "high_cost"
    PREMIUM = "premium"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    id: str                          # OpenRouter model ID
    name: str                       # Display name
    family: ModelFamily             # Model family
    provider: str                   # Provider company
    architecture: str               # Architecture type
    parameters: str                 # Parameter count description
    context_length: int             # Maximum context length
    capabilities: List[str]         # Key capabilities
    tier: ModelTier                # Cost tier
    description: str               # Brief description
    strengths: List[str]           # Model strengths
    use_cases: List[str]           # Recommended use cases
    released: str                  # Release date/period
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.id:
            raise ValueError("Model ID cannot be empty")
        if not self.name:
            raise ValueError("Model name cannot be empty")


class ModelRegistry:
    """
    Central registry for all OpenRouter models with metadata and selection helpers.
    
    Provides model discovery, filtering, and configuration management for the
    Universal Alignment Patterns research system.
    """
    
    def __init__(self):
        """Initialize the model registry with all supported models."""
        self._models: Dict[str, ModelConfig] = {}
        self._load_models()
    
    def _load_models(self):
        """Load all model configurations."""
        
        # GPT-OSS Models (OpenAI's new open-source models)
        self.register_model(ModelConfig(
            id="openai/gpt-oss-120b",
            name="GPT-OSS 120B",
            family=ModelFamily.GPT_OSS,
            provider="OpenAI",
            architecture="moe-transformer",
            parameters="120B (5.1B active)",
            context_length=8192,
            capabilities=["reasoning", "coding", "chain-of-thought", "structured-output"],
            tier=ModelTier.FREE,
            description="OpenAI's open-source reasoning model with MoE architecture",
            strengths=["Math reasoning (96.6% AIME)", "Efficient inference", "Open source"],
            use_cases=["Mathematical reasoning", "Code generation", "Research"],
            released="2025-08"
        ))
        
        self.register_model(ModelConfig(
            id="openai/gpt-oss-20b",
            name="GPT-OSS 20B",
            family=ModelFamily.GPT_OSS,
            provider="OpenAI",
            architecture="dense-transformer",
            parameters="20B",
            context_length=8192,
            capabilities=["reasoning", "coding", "efficiency"],
            tier=ModelTier.FREE,
            description="Smaller GPT-OSS variant for efficient deployment",
            strengths=["Fast inference", "Lower resource usage", "Open source"],
            use_cases=["Edge deployment", "Quick reasoning", "Cost optimization"],
            released="2025-08"
        ))
        
        # GLM Models (Zhipu AI)
        self.register_model(ModelConfig(
            id="zhipu/glm-4.5",
            name="GLM-4.5",
            family=ModelFamily.GLM,
            provider="Zhipu AI",
            architecture="moe-transformer",
            parameters="355B (32B active)",
            context_length=128000,
            capabilities=["agentic", "tool-use", "reasoning", "coding", "function-calling"],
            tier=ModelTier.LOW_COST,
            description="Best open-source agentic model with superior tool calling",
            strengths=["Tool calling (90.6% success)", "Agentic capabilities", "Deep architecture"],
            use_cases=["Agentic workflows", "Tool integration", "Complex reasoning"],
            released="2025-07"
        ))
        
        self.register_model(ModelConfig(
            id="zhipu/glm-4.5-air",
            name="GLM-4.5 Air",
            family=ModelFamily.GLM,
            provider="Zhipu AI",
            architecture="moe-transformer",
            parameters="106B (12B active)",
            context_length=128000,
            capabilities=["efficiency", "tool-use", "reasoning"],
            tier=ModelTier.LOW_COST,
            description="Lightweight version of GLM-4.5 for cost-effective deployment",
            strengths=["Cost efficiency", "Fast inference", "Good tool use"],
            use_cases=["Production deployment", "Cost optimization", "High-volume tasks"],
            released="2025-07"
        ))
        
        # Kimi Models (Moonshot AI)
        self.register_model(ModelConfig(
            id="moonshot/kimi-k2",
            name="Kimi K2",
            family=ModelFamily.KIMI,
            provider="Moonshot AI",
            architecture="moe-transformer",
            parameters="1T (32B active)",
            context_length=256000,
            capabilities=["agentic", "long-context", "coding", "mcp-support", "swe-bench"],
            tier=ModelTier.MEDIUM_COST,
            description="1T parameter agentic model with native MCP support",
            strengths=["65% SWE-bench", "2.5x faster than Qwen3", "Long context"],
            use_cases=["Code generation", "Large codebase analysis", "Agentic tasks"],
            released="2025-08"
        ))
        
        # Qwen Models (Alibaba)
        self.register_model(ModelConfig(
            id="alibaba/qwen3-coder-480b",
            name="Qwen 3 Coder",
            family=ModelFamily.QWEN,
            provider="Alibaba",
            architecture="moe-transformer",
            parameters="480B (35B active)",
            context_length=256000,
            capabilities=["coding", "agentic", "browser-use", "long-context"],
            tier=ModelTier.MEDIUM_COST,
            description="Leading coding model with 67% SWE-bench performance",
            strengths=["67% SWE-bench", "1M context expandable", "Repository analysis"],
            use_cases=["Code generation", "Large repository analysis", "Software engineering"],
            released="2025-07"
        ))
        
        self.register_model(ModelConfig(
            id="alibaba/qwen3-235b-thinking",
            name="Qwen 3 Thinking",
            family=ModelFamily.QWEN,
            provider="Alibaba",
            architecture="moe-transformer",
            parameters="235B (22B active)",
            context_length=256000,
            capabilities=["thinking", "reasoning", "chain-of-thought", "depth"],
            tier=ModelTier.MEDIUM_COST,
            description="Thinking-optimized model with enhanced reasoning depth",
            strengths=["Deep reasoning", "Quality thinking", "Long context"],
            use_cases=["Complex problem solving", "Research analysis", "Deep reasoning"],
            released="2025-07"
        ))
        
        # Legacy/Comparison Models
        self.register_model(ModelConfig(
            id="openai/gpt-4-turbo",
            name="GPT-4 Turbo",
            family=ModelFamily.GPT,
            provider="OpenAI",
            architecture="transformer",
            parameters="Unknown",
            context_length=128000,
            capabilities=["reasoning", "coding", "multimodal", "function-calling"],
            tier=ModelTier.HIGH_COST,
            description="OpenAI's flagship model with strong general capabilities",
            strengths=["General intelligence", "Multimodal", "Reliability"],
            use_cases=["General tasks", "Baseline comparison", "High-quality output"],
            released="2024"
        ))
        
        self.register_model(ModelConfig(
            id="anthropic/claude-3.5-sonnet",
            name="Claude 3.5 Sonnet",
            family=ModelFamily.CLAUDE,
            provider="Anthropic",
            architecture="transformer",
            parameters="Unknown",
            context_length=200000,
            capabilities=["reasoning", "coding", "analysis", "safety"],
            tier=ModelTier.HIGH_COST,
            description="Anthropic's advanced reasoning model with strong safety",
            strengths=["Code analysis", "Safety", "Long context"],
            use_cases=["Code review", "Analysis", "Safety-critical tasks"],
            released="2024"
        ))
        
        # Free/Low-cost options for testing
        self.register_model(ModelConfig(
            id="meta/llama-3.1-8b-instruct:free",
            name="Llama 3.1 8B (Free)",
            family=ModelFamily.LLAMA,
            provider="Meta",
            architecture="transformer",
            parameters="8B",
            context_length=128000,
            capabilities=["general", "coding", "reasoning"],
            tier=ModelTier.FREE,
            description="Free tier access to Llama 3.1 for testing",
            strengths=["Free usage", "Good performance", "Open source"],
            use_cases=["Testing", "Development", "Cost optimization"],
            released="2024"
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration."""
        self._models[config.id] = config
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self._models.get(model_id)
    
    def list_models(self, 
                   family: Optional[ModelFamily] = None,
                   tier: Optional[ModelTier] = None,
                   capabilities: Optional[List[str]] = None) -> List[ModelConfig]:
        """
        List models with optional filtering.
        
        Args:
            family: Filter by model family
            tier: Filter by cost tier
            capabilities: Filter by required capabilities
            
        Returns:
            List of matching model configurations
        """
        models = list(self._models.values())
        
        if family:
            models = [m for m in models if m.family == family]
        
        if tier:
            models = [m for m in models if m.tier == tier]
        
        if capabilities:
            models = [m for m in models 
                     if all(cap in m.capabilities for cap in capabilities)]
        
        return sorted(models, key=lambda m: (m.tier.value, m.name))
    
    def get_recommended_models(self, use_case: str = "general") -> List[ModelConfig]:
        """
        Get recommended models for specific use cases.
        
        Args:
            use_case: The intended use case
            
        Returns:
            List of recommended models
        """
        recommendations = {
            "agentic": [
                "zhipu/glm-4.5",
                "moonshot/kimi-k2", 
                "alibaba/qwen3-coder-480b"
            ],
            "coding": [
                "alibaba/qwen3-coder-480b",
                "moonshot/kimi-k2",
                "zhipu/glm-4.5"
            ],
            "reasoning": [
                "openai/gpt-oss-120b",
                "alibaba/qwen3-235b-thinking",
                "anthropic/claude-3.5-sonnet"
            ],
            "cost_optimized": [
                "openai/gpt-oss-120b",
                "zhipu/glm-4.5-air",
                "meta/llama-3.1-8b-instruct:free"
            ],
            "research": [
                "openai/gpt-oss-120b",
                "zhipu/glm-4.5",
                "alibaba/qwen3-235b-thinking",
                "anthropic/claude-3.5-sonnet"
            ],
            "general": [
                "openai/gpt-oss-120b",
                "zhipu/glm-4.5",
                "openai/gpt-4-turbo"
            ]
        }
        
        model_ids = recommendations.get(use_case, recommendations["general"])
        return [self._models[mid] for mid in model_ids if mid in self._models]
    
    def get_convergence_test_set(self) -> List[ModelConfig]:
        """
        Get the optimal set of models for convergence testing.
        
        Returns models representing different:
        - Architectures (dense vs MoE)
        - Providers (Western vs Chinese)
        - Training approaches
        - Cost tiers
        """
        test_models = [
            "openai/gpt-oss-120b",      # Open-source Western MoE
            "zhipu/glm-4.5",            # Chinese agentic MoE
            "moonshot/kimi-k2",         # Large Chinese MoE
            "alibaba/qwen3-coder-480b", # Specialized coding MoE
            "anthropic/claude-3.5-sonnet", # Commercial Western dense
        ]
        
        return [self._models[mid] for mid in test_models if mid in self._models]
    
    def get_families(self) -> List[ModelFamily]:
        """Get all available model families."""
        return list(set(model.family for model in self._models.values()))
    
    def get_model_count(self) -> int:
        """Get total number of registered models."""
        return len(self._models)
    
    def print_registry_summary(self):
        """Print a summary of the model registry."""
        print("📊 OpenRouter Model Registry Summary")
        print(f"Total models: {self.get_model_count()}")
        print()
        
        # Group by family
        for family in self.get_families():
            family_models = self.list_models(family=family)
            print(f"{family.value.upper()}: {len(family_models)} models")
            for model in family_models:
                tier_emoji = {"free": "🆓", "low_cost": "💰", "medium_cost": "💳", "high_cost": "💎", "premium": "👑"}
                print(f"  {tier_emoji.get(model.tier.value, '💰')} {model.name} ({model.parameters})")
        print()
        
        # Cost tier summary
        print("Cost Tiers:")
        for tier in ModelTier:
            tier_models = self.list_models(tier=tier)
            if tier_models:
                print(f"  {tier.value}: {len(tier_models)} models")


# Global registry instance
model_registry = ModelRegistry()


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Convenience function to get model configuration."""
    return model_registry.get_model(model_id)


def list_available_models(**kwargs) -> List[ModelConfig]:
    """Convenience function to list available models."""
    return model_registry.list_models(**kwargs)


def get_recommended_models(use_case: str = "general") -> List[ModelConfig]:
    """Convenience function to get recommended models."""
    return model_registry.get_recommended_models(use_case)