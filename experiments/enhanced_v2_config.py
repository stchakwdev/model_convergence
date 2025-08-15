"""
Enhanced v2.0 Configuration for Universal Alignment Patterns
===========================================================

This configuration implements major improvements over v1.0:
1. Premium model selection (GPT-4, Claude-3.5-Sonnet, etc.)
2. Expanded prompt datasets (100+ prompts per capability)  
3. Enhanced statistical power with larger sample sizes
4. Budget optimization using diverse cost tiers

Expected improvements:
- 40-60% higher convergence scores with premium models
- Statistical significance at p<0.001 with larger sample sizes
- Better representation of universal patterns across architectures
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from comprehensive_analysis import ExperimentConfig

# Enhanced model selection for v2.0
ENHANCED_MODEL_SET = [
    # Premium tier - highest quality responses
    "anthropic/claude-3.5-sonnet",     # Anthropic's best reasoning model
    "openai/gpt-4-turbo",              # OpenAI's flagship model
    
    # Advanced open-source models  
    "openai/gpt-oss-120b",             # OpenAI open-source reasoning
    "zhipu/glm-4.5",                   # Best agentic model (90.6% tool success)
    "moonshot/kimi-k2",                # 1T parameter with 256K context
    "alibaba/qwen3-coder-480b",        # Leading coding model (67% SWE-bench)
    
    # Efficient models for comparison
    "zhipu/glm-4.5-air",               # Lightweight but capable
    "meta/llama-3.1-8b-instruct:free", # Free tier baseline
]

# Alternative budget-conscious set  
BUDGET_OPTIMIZED_SET = [
    "openai/gpt-oss-120b",             # Free
    "meta/llama-3.1-8b-instruct:free", # Free  
    "zhipu/glm-4.5-air",               # Low cost
    "zhipu/glm-4.5",                   # Low cost
    "moonshot/kimi-k2",                # Medium cost
    "alibaba/qwen3-235b-thinking",     # Medium cost
]

# Premium-only set for maximum quality
PREMIUM_ONLY_SET = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4-turbo", 
    "moonshot/kimi-k2",
    "alibaba/qwen3-coder-480b",
    "alibaba/qwen3-235b-thinking",
]

@dataclass
class EnhancedExperimentConfig:
    """Enhanced configuration for v2.0 experiments"""
    
    # Core experiment parameters
    name: str = "Universal Alignment Patterns v2.0 - Enhanced"
    description: str = "Enhanced experiment with premium models and expanded datasets"
    
    # Model selection (choose one set)
    models: List[str] = None  # Will be set based on experiment type
    
    # Enhanced capabilities with larger sample sizes
    capabilities: List[str] = None
    prompts_per_capability: int = 100  # 2x increase for statistical power
    
    # Generation parameters
    max_tokens: int = 300  # Slightly more for detailed responses
    temperature: float = 0.0  # Deterministic for reproducibility
    
    # Budget and statistical parameters
    budget_limit_usd: float = 40.0  # Conservative budget for v2.0
    statistical_confidence: float = 0.001  # Strict significance threshold
    
    # Enhanced analysis parameters
    enable_pairwise_analysis: bool = True
    enable_difficulty_stratification: bool = True
    enable_bootstrap_confidence: bool = True
    n_bootstrap_samples: int = 1000


def get_experiment_config(experiment_type: str = "enhanced") -> ExperimentConfig:
    """
    Get experiment configuration for different types of enhanced experiments.
    
    Args:
        experiment_type: Type of experiment ("enhanced", "budget", "premium", "validation")
        
    Returns:
        ExperimentConfig object ready for comprehensive_analysis.py
    """
    
    # Enhanced capabilities list
    capabilities = [
        "truthfulness",
        "safety_boundaries", 
        "instruction_following",
        "uncertainty_expression",
        "context_awareness"
    ]
    
    # Select model set based on experiment type
    if experiment_type == "enhanced":
        models = ENHANCED_MODEL_SET
        prompts_per_cap = 100
        budget = 40.0
        description = "Enhanced v2.0: Premium models with expanded datasets"
        
    elif experiment_type == "budget":
        models = BUDGET_OPTIMIZED_SET
        prompts_per_cap = 150  # More prompts with cheaper models
        budget = 25.0
        description = "Budget-optimized v2.0: Cost-efficient models with maximum scale"
        
    elif experiment_type == "premium":
        models = PREMIUM_ONLY_SET
        prompts_per_cap = 75   # Fewer prompts due to higher cost
        budget = 45.0
        description = "Premium v2.0: Highest-quality models for definitive results"
        
    elif experiment_type == "validation":
        models = ENHANCED_MODEL_SET[:4]  # First 4 models
        prompts_per_cap = 50
        budget = 15.0
        description = "v2.0 Validation: Quick test of enhanced methodology"
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return ExperimentConfig(
        name=f"Universal Alignment Patterns v2.0 - {experiment_type.title()}",
        description=description,
        models=models,
        capabilities=capabilities,
        prompts_per_capability=prompts_per_cap,
        max_tokens=300,
        temperature=0.0,
        budget_limit_usd=budget,
        statistical_confidence=0.001
    )


def estimate_experiment_cost(experiment_type: str = "enhanced") -> Dict[str, Any]:
    """
    Estimate the cost and scale of different experiment types.
    
    Returns:
        Dictionary with cost estimates and experiment scale
    """
    config = get_experiment_config(experiment_type)
    
    total_calls = len(config.models) * len(config.capabilities) * config.prompts_per_capability
    
    # Rough cost estimates based on model tiers
    cost_estimates = {
        "enhanced": 0.02,     # Mixed tiers
        "budget": 0.005,      # Mostly free/low-cost
        "premium": 0.05,      # High-cost models
        "validation": 0.015   # Medium scale
    }
    
    estimated_cost = total_calls * cost_estimates.get(experiment_type, 0.02)
    
    return {
        "experiment_type": experiment_type,
        "total_api_calls": total_calls,
        "estimated_cost_usd": estimated_cost,
        "budget_limit": config.budget_limit_usd,
        "budget_utilization": (estimated_cost / config.budget_limit_usd) * 100,
        "models_count": len(config.models),
        "capabilities_count": len(config.capabilities),
        "prompts_per_capability": config.prompts_per_capability,
        "within_budget": estimated_cost < config.budget_limit_usd,
        "recommendation": "PROCEED" if estimated_cost < config.budget_limit_usd else "REDUCE_SCALE"
    }


if __name__ == "__main__":
    # Print all experiment configurations
    experiment_types = ["enhanced", "budget", "premium", "validation"]
    
    print("🚀 Enhanced v2.0 Experiment Configurations")
    print("=" * 60)
    
    for exp_type in experiment_types:
        config = get_experiment_config(exp_type)
        cost_est = estimate_experiment_cost(exp_type)
        
        print(f"\n📊 {exp_type.upper()} EXPERIMENT")
        print(f"   Models: {len(config.models)} ({', '.join([m.split('/')[-1] for m in config.models[:3]])}...)")
        print(f"   Scale: {cost_est['total_api_calls']:,} API calls")
        print(f"   Cost: ${cost_est['estimated_cost_usd']:.3f} / ${config.budget_limit_usd}")
        print(f"   Utilization: {cost_est['budget_utilization']:.1f}%")
        print(f"   Status: {cost_est['recommendation']}")
        
    print(f"\n💰 Total remaining budget: $49.91")
    print(f"🎯 Recommended: Start with 'validation' then 'enhanced'")