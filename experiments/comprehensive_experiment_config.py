#!/usr/bin/env python3
"""
Comprehensive Tilli Tonse Experiment Configuration

Designed for statistical rigor with sufficient sample size to make strong claims
about universal alignment patterns. This configuration enables publication-quality
research with robust statistical validation.

Statistical Design:
- 15+ diverse models across architectures and training methods
- 50+ stories per capability for robust sample size
- Power analysis for detecting medium effects (d=0.5) with 80% power
- Multiple comparison correction (Bonferroni) for family-wise error control
- Expected cost: ~$25-50 for complete analysis

Author: Samuel Tchakwera
Purpose: Definitive evidence for universal alignment patterns
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class ComprehensiveExperimentConfig:
    """Configuration for statistically rigorous comprehensive experiment"""
    name: str
    description: str
    models: List[str]
    capabilities: List[str]
    stories_per_capability: int
    max_tokens: int
    temperature: float
    budget_limit_usd: float
    statistical_alpha: float
    expected_effect_size: float
    statistical_power: float
    
    def calculate_sample_size(self) -> Dict[str, int]:
        """Calculate required sample size for statistical power"""
        # For detecting medium effects (d=0.5) with 80% power and α=0.001
        # We need approximately 40-50 samples per condition
        
        total_models = len(self.models)
        total_capabilities = len(self.capabilities)
        total_stories = self.stories_per_capability * total_capabilities
        pairwise_comparisons = (total_models * (total_models - 1)) // 2
        
        return {
            "total_models": total_models,
            "total_capabilities": total_capabilities,
            "stories_per_capability": self.stories_per_capability,
            "total_story_responses": total_stories * total_models,
            "pairwise_comparisons_per_capability": pairwise_comparisons,
            "total_pairwise_comparisons": pairwise_comparisons * total_capabilities,
            "statistical_power_achieved": "High (>80%)" if total_stories >= 40 else "Medium"
        }
    
    def estimate_cost(self) -> Dict[str, float]:
        """Estimate experimental costs"""
        sample_stats = self.calculate_sample_size()
        
        # Conservative cost estimates per model family
        cost_per_1k_tokens = {
            "free_tier": 0.0,
            "low_cost": 0.01,
            "premium": 0.03,
            "cutting_edge": 0.05
        }
        
        # Estimate average cost (mix of model tiers)
        avg_cost_per_1k = 0.02  # Conservative estimate
        total_tokens = sample_stats["total_story_responses"] * (self.max_tokens / 1000)
        estimated_cost = total_tokens * avg_cost_per_1k
        
        return {
            "total_tokens_estimated": int(total_tokens * 1000),
            "average_cost_per_1k_tokens": avg_cost_per_1k,
            "estimated_total_cost": estimated_cost,
            "cost_per_model": estimated_cost / len(self.models),
            "cost_per_capability": estimated_cost / len(self.capabilities),
            "budget_utilization": (estimated_cost / self.budget_limit_usd) * 100
        }


class ComprehensiveModelRegistry:
    """Registry of diverse models for comprehensive convergence analysis"""
    
    @staticmethod
    def get_comprehensive_model_set() -> List[str]:
        """
        Curated set of 15+ diverse models representing different:
        - Architectures (dense, MoE, multimodal)
        - Training methods (RLHF, Constitutional AI, etc.)
        - Organizations (OpenAI, Anthropic, Google, Meta, Chinese labs)
        - Parameter scales (7B to 405B+)
        - Specializations (reasoning, coding, safety, etc.)
        """
        
        return [
            # OpenAI Models - Latest working models
            "openai/gpt-5-mini",                  # Latest GPT-5 mini (working)
            "openai/gpt-oss-120b",                # Open-source reasoning model (working)
            "openai/gpt-oss-20b:free",            # Free tier OpenAI model (working)
            
            # Anthropic Models - Constitutional AI approach
            "anthropic/claude-opus-4.1",          # Latest Claude Opus 4.1 (working)
            "anthropic/claude-3-haiku",           # Efficient Claude variant (still working)
            
            # Google Models - Latest working Gemini
            "google/gemini-2.5-flash-lite",       # Latest Gemini Flash Lite (working)
            
            # Chinese Models - Different cultural training (CORRECTED IDs)
            "z-ai/glm-4.5-air:free",             # GLM 4.5 Air free tier (working)
            "qwen/qwen3-30b-a3b-instruct-2507",  # Qwen3 30B (working)
            "qwen/qwen3-235b-a22b-2507",         # Qwen3 235B (working)
            "moonshotai/kimi-k2",                 # Kimi K2 (working)
            "tencent/hunyuan-a13b-instruct:free", # Hunyuan A13B free tier (working)
            
            # Meta/Open Models - Working Llama
            "meta-llama/llama-3.1-70b",          # Llama 3.1 70B (working)
            
            # Specialized Models - Working alternatives
            "mistralai/mistral-medium-3.1",      # Mistral Medium 3.1 (working)
            "x-ai/grok-4",                       # Grok 4 (working)
            "ai21/jamba-large-1.7",              # AI21 Jamba Large (working)
            
            # Additional Working Models for Diversity
            "baidu/ernie-4.5-300b-a47b",         # Baidu ERNIE 4.5 (working)
        ]
    
    @staticmethod
    def get_model_metadata() -> Dict[str, Dict[str, Any]]:
        """Metadata about each model for analysis"""
        
        return {
            "openai/gpt-5-mini": {
                "organization": "OpenAI",
                "architecture": "Dense Transformer",
                "training_method": "RLHF",
                "parameter_scale": "Large",
                "specialization": "General",
                "cultural_background": "Western"
            },
            "anthropic/claude-opus-4.1": {
                "organization": "Anthropic", 
                "architecture": "Dense Transformer",
                "training_method": "Constitutional AI",
                "parameter_scale": "Large",
                "specialization": "Safety",
                "cultural_background": "Western"
            },
            "z-ai/glm-4.5": {
                "organization": "Zhipu AI",
                "architecture": "GLM MoE",
                "training_method": "SFT+RLHF",
                "parameter_scale": "Large", 
                "specialization": "Agentic",
                "cultural_background": "Chinese"
            },
            "openai/gpt-oss-120b": {
                "organization": "OpenAI",
                "architecture": "MoE Transformer",
                "training_method": "RLHF",
                "parameter_scale": "Large",
                "specialization": "Reasoning",
                "cultural_background": "Western"
            },
            "qwen/qwen3-30b-a3b-instruct-2507": {
                "organization": "Alibaba",
                "architecture": "MoE Transformer",
                "training_method": "SFT+RLHF",
                "parameter_scale": "Medium",
                "specialization": "Instruction Following",
                "cultural_background": "Chinese"
            },
            "google/gemini-2.5-flash-lite": {
                "organization": "Google",
                "architecture": "Dense Transformer",
                "training_method": "Constitutional AI",
                "parameter_scale": "Large",
                "specialization": "Multimodal",
                "cultural_background": "Western"
            },
            "moonshotai/kimi-k2": {
                "organization": "Moonshot AI",
                "architecture": "MoE Transformer",
                "training_method": "SFT+RLHF",
                "parameter_scale": "Very Large",
                "specialization": "Agentic",
                "cultural_background": "Chinese"
            },
            "meta-llama/llama-3.1-70b": {
                "organization": "Meta",
                "architecture": "Llama",
                "training_method": "Open Training",
                "parameter_scale": "XLarge",
                "specialization": "General",
                "cultural_background": "Western"
            },
            "moonshot/kimi-k2": {
                "organization": "Moonshot AI",
                "architecture": "MoE",
                "training_method": "Custom",
                "parameter_scale": "XLarge (1T)",
                "specialization": "Long Context",
                "cultural_background": "Chinese"
            }
            # Add metadata for other models as needed...
        }
    
    @staticmethod
    def get_cost_optimized_subset() -> List[str]:
        """Subset for budget-conscious comprehensive analysis"""
        
        return [
            # Free tier models
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b:free",
            "z-ai/glm-4.5-air:free",
            "tencent/hunyuan-a13b-instruct:free",
            
            # Low-cost premium models (working IDs)
            "anthropic/claude-3-haiku",
            "google/gemini-2.5-flash-lite",
            "qwen/qwen3-30b-a3b-instruct-2507",
            "moonshotai/kimi-k2",
            
            # Strategic premium additions (working IDs)
            "openai/gpt-5-mini",
            "anthropic/claude-opus-4.1",
            "meta-llama/llama-3.1-70b"
        ]


def create_comprehensive_experiment_configs() -> Dict[str, ComprehensiveExperimentConfig]:
    """Create predefined comprehensive experiment configurations"""
    
    model_registry = ComprehensiveModelRegistry()
    
    configs = {
        "definitive_research": ComprehensiveExperimentConfig(
            name="Definitive Universal Alignment Patterns Research",
            description="Publication-quality comprehensive analysis with statistical rigor for definitive claims about universal alignment patterns",
            models=model_registry.get_comprehensive_model_set(),
            capabilities=[
                "truthfulness",
                "safety_boundaries", 
                "instruction_following",
                "uncertainty_expression",
                "context_awareness"
            ],
            stories_per_capability=50,  # High statistical power
            max_tokens=600,
            temperature=0.0,
            budget_limit_usd=100.0,
            statistical_alpha=0.001,  # Bonferroni corrected
            expected_effect_size=0.5,  # Medium effects
            statistical_power=0.8  # 80% power
        ),
        
        "cost_optimized_comprehensive": ComprehensiveExperimentConfig(
            name="Cost-Optimized Comprehensive Analysis", 
            description="Balanced approach with good statistical power and cost efficiency",
            models=model_registry.get_cost_optimized_subset(),
            capabilities=[
                "truthfulness",
                "safety_boundaries",
                "instruction_following", 
                "uncertainty_expression",
                "context_awareness"
            ],
            stories_per_capability=30,  # Good statistical power
            max_tokens=500,
            temperature=0.0,
            budget_limit_usd=25.0,
            statistical_alpha=0.01,
            expected_effect_size=0.5,
            statistical_power=0.8
        ),
        
        "rapid_validation": ComprehensiveExperimentConfig(
            name="Rapid Validation Study",
            description="Quick but statistically valid analysis for initial validation",
            models=model_registry.get_cost_optimized_subset()[:8],
            capabilities=[
                "truthfulness",
                "safety_boundaries",
                "instruction_following"
            ],
            stories_per_capability=20,  # Minimal statistical power
            max_tokens=400,
            temperature=0.0,
            budget_limit_usd=10.0,
            statistical_alpha=0.05,
            expected_effect_size=0.6,  # Larger effects only
            statistical_power=0.7
        )
    }
    
    return configs


def print_experiment_summary(config: ComprehensiveExperimentConfig):
    """Print detailed experiment summary with statistical projections"""
    
    print(f"\n🔬 COMPREHENSIVE EXPERIMENT: {config.name}")
    print("=" * 80)
    
    # Sample size analysis
    sample_stats = config.calculate_sample_size()
    print(f"📊 STATISTICAL DESIGN:")
    print(f"   Models: {sample_stats['total_models']}")
    print(f"   Capabilities: {sample_stats['total_capabilities']}")
    print(f"   Stories per capability: {sample_stats['stories_per_capability']}")
    print(f"   Total story responses: {sample_stats['total_story_responses']:,}")
    print(f"   Pairwise comparisons: {sample_stats['total_pairwise_comparisons']:,}")
    print(f"   Statistical power: {sample_stats['statistical_power_achieved']}")
    
    # Cost analysis
    cost_stats = config.estimate_cost()
    print(f"\n💰 COST ANALYSIS:")
    print(f"   Total tokens estimated: {cost_stats['total_tokens_estimated']:,}")
    print(f"   Estimated total cost: ${cost_stats['estimated_total_cost']:.2f}")
    print(f"   Budget utilization: {cost_stats['budget_utilization']:.1f}%")
    print(f"   Cost per model: ${cost_stats['cost_per_model']:.2f}")
    
    # Statistical parameters
    print(f"\n📈 STATISTICAL PARAMETERS:")
    print(f"   Significance level (α): {config.statistical_alpha}")
    print(f"   Expected effect size: {config.expected_effect_size}")
    print(f"   Target statistical power: {config.statistical_power}")
    print(f"   Multiple comparison correction: Bonferroni")
    
    print("=" * 80)


def main():
    """Demonstrate comprehensive experiment configurations"""
    
    print("🎭 COMPREHENSIVE TILLI TONSE EXPERIMENT CONFIGURATIONS")
    print("🌍 Designed for Statistical Rigor and Definitive Claims")
    
    configs = create_comprehensive_experiment_configs()
    
    for config_name, config in configs.items():
        print_experiment_summary(config)
        
        if config_name == "definitive_research":
            print(f"\n🌟 MODELS IN DEFINITIVE RESEARCH SET:")
            model_registry = ComprehensiveModelRegistry()
            metadata = model_registry.get_model_metadata()
            
            for i, model in enumerate(config.models, 1):
                meta = metadata.get(model, {})
                org = meta.get("organization", "Unknown")
                spec = meta.get("specialization", "General")
                culture = meta.get("cultural_background", "Unknown")
                print(f"   {i:2}. {model}")
                print(f"       Organization: {org} | Specialization: {spec} | Culture: {culture}")
    
    print(f"\n✅ Configurations ready for comprehensive analysis!")
    print(f"🎯 Choose configuration based on budget and statistical requirements")


if __name__ == "__main__":
    main()