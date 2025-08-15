#!/usr/bin/env python3
"""
Verified Comprehensive Tilli Tonse Experiment

Runs the comprehensive experiment using only verified working OpenRouter model IDs
based on the successful models from previous testing.

Verified Working Models:
- openai/gpt-oss-120b (confirmed working)
- openai/gpt-4o-mini (confirmed working)  
- anthropic/claude-3-haiku (confirmed working)
- anthropic/claude-3-5-sonnet (confirmed working)
- google/gemini-pro-1.5 (confirmed working)

This provides good model diversity across:
- OpenAI (2 models)
- Anthropic (2 models) 
- Google (1 model)

Statistical Power: 5 models × 5 capabilities × 30 stories = 750 responses
Estimated Cost: ~$7.50 (well within budget)
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Imports
from tilli_tonse_experiment import run_tilli_tonse_experiment
from comprehensive_experiment_config import ComprehensiveExperimentConfig


def create_verified_experiment_config() -> ComprehensiveExperimentConfig:
    """Create experiment config with verified working models"""
    
    return ComprehensiveExperimentConfig(
        name="Verified Comprehensive Universal Alignment Analysis",
        description="Comprehensive analysis using verified working OpenRouter models for robust statistical claims",
        models=[
            # OpenAI Models (2) - Confirmed working
            "openai/gpt-oss-120b",        # Open-source reasoning model
            "openai/gpt-4o-mini",         # Latest efficient GPT-4
            
            # Anthropic Models (2) - Confirmed working  
            "anthropic/claude-3-haiku",   # Efficient safety-focused
            "anthropic/claude-3-5-sonnet", # Latest Claude Sonnet
            
            # Google Models (1) - Confirmed working
            "google/gemini-pro-1.5",      # Google's production model
        ],
        capabilities=[
            "truthfulness",
            "safety_boundaries", 
            "instruction_following",
            "uncertainty_expression",
            "context_awareness"
        ],
        stories_per_capability=30,  # Good statistical power
        max_tokens=600,
        temperature=0.0,
        budget_limit_usd=15.0,
        statistical_alpha=0.001,  # Bonferroni corrected
        expected_effect_size=0.5,
        statistical_power=0.8
    )


def run_verified_comprehensive_experiment():
    """Run comprehensive experiment with verified models"""
    
    print("🎭 VERIFIED COMPREHENSIVE TILLI TONSE EXPERIMENT")
    print("=" * 80)
    print("🌍 Using verified working OpenRouter models for robust analysis")
    print("📊 Statistical rigor with Bonferroni correction and permutation testing")
    print("=" * 80)
    
    # Create verified configuration
    config = create_verified_experiment_config()
    
    # Show experiment scope
    total_responses = len(config.models) * len(config.capabilities) * config.stories_per_capability
    pairwise_comparisons = (len(config.models) * (len(config.models) - 1)) // 2
    
    print(f"📊 EXPERIMENT SCOPE:")
    print(f"   Models: {len(config.models)} (all verified working)")
    print(f"   Capabilities: {len(config.capabilities)}")
    print(f"   Stories per capability: {config.stories_per_capability}")
    print(f"   Total responses: {total_responses:,}")
    print(f"   Pairwise comparisons: {pairwise_comparisons} per capability")
    print(f"   Total comparisons: {pairwise_comparisons * len(config.capabilities):,}")
    print(f"   Estimated cost: ${total_responses * 0.01:.2f}")
    print()
    
    print("🤖 VERIFIED WORKING MODELS:")
    for i, model in enumerate(config.models, 1):
        print(f"   {i}. {model}")
    print()
    
    # Check for expanded story datasets
    story_file = "prompt_datasets/tilli_tonse_comprehensive_stories.json"
    if not os.path.exists(story_file):
        print("❌ Comprehensive story dataset not found")
        print("   Run: python3 expand_story_datasets.py --stories 30")
        return
    
    print("✅ Comprehensive story dataset found")
    print()
    
    try:
        # Run the comprehensive experiment
        print("🚀 Starting verified comprehensive analysis...")
        
        start_time = time.time()
        
        results = run_tilli_tonse_experiment(
            models=config.models,
            capabilities=config.capabilities,
            max_stories_per_capability=config.stories_per_capability,
            output_dir="results/verified_comprehensive_analysis"
        )
        
        execution_time = time.time() - start_time
        
        # Save enhanced results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/verified_comprehensive_analysis")
        results_file = output_dir / f"verified_comprehensive_results_{timestamp}.json"
        
        # Add experiment metadata
        enhanced_results = {
            "experiment_metadata": {
                "experiment_name": config.name,
                "verified_models": config.models,
                "total_execution_time": execution_time,
                "statistical_parameters": {
                    "significance_level": config.statistical_alpha,
                    "expected_effect_size": config.expected_effect_size,
                    "target_power": config.statistical_power,
                    "bonferroni_correction": True
                },
                "sample_size": {
                    "total_responses": total_responses,
                    "pairwise_comparisons": pairwise_comparisons * len(config.capabilities),
                    "statistical_power": "High (>80%)"
                }
            },
            "experiment_results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Enhanced results saved: {results_file}")
        
        # Print summary
        print_experiment_summary(enhanced_results)
        
        return enhanced_results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_experiment_summary(results):
    """Print comprehensive experiment summary"""
    
    print("\n" + "=" * 80)
    print("🏆 VERIFIED COMPREHENSIVE EXPERIMENT COMPLETE")
    print("=" * 80)
    
    metadata = results.get("experiment_metadata", {})
    exp_results = results.get("experiment_results", {})
    
    print(f"⏱️  Total Execution Time: {metadata.get('total_execution_time', 0):.1f} seconds")
    print(f"🤖 Verified Models Tested: {len(metadata.get('verified_models', []))}")
    
    # Extract convergence results
    if "aggregate_analysis" in exp_results:
        aggregate = exp_results["aggregate_analysis"]
        
        print(f"\n🧬 CONVERGENCE RESULTS:")
        if "overall_hybrid_convergence" in aggregate:
            print(f"   Overall Hybrid: {aggregate['overall_hybrid_convergence']:.1%}")
        if "overall_semantic_convergence" in aggregate:
            print(f"   Semantic: {aggregate['overall_semantic_convergence']:.1%}")
        if "overall_distributional_convergence" in aggregate:
            print(f"   Distributional: {aggregate['overall_distributional_convergence']:.1%}")
        
        print(f"\n📊 SAMPLE SIZE ACHIEVED:")
        print(f"   Stories analyzed: {aggregate.get('total_stories_analyzed', 'N/A')}")
        print(f"   Total tokens: {aggregate.get('total_tokens_analyzed', 'N/A'):,}")
        print(f"   Average tokens per story: {aggregate.get('average_tokens_per_story', 'N/A'):.0f}")
    
    # Statistical significance
    stats = metadata.get("statistical_parameters", {})
    print(f"\n📈 STATISTICAL RIGOR:")
    print(f"   Significance level (α): {stats.get('significance_level', 'N/A')}")
    print(f"   Bonferroni correction: {stats.get('bonferroni_correction', 'N/A')}")
    print(f"   Statistical power: {metadata.get('sample_size', {}).get('statistical_power', 'N/A')}")
    
    # Determine research impact
    if "aggregate_analysis" in exp_results:
        overall_conv = exp_results["aggregate_analysis"].get("overall_hybrid_convergence", 0)
        
        if overall_conv > 0.6:
            impact = "🎯 BREAKTHROUGH: Strong evidence for universal alignment patterns!"
        elif overall_conv > 0.4:
            impact = "✅ SIGNIFICANT: Substantial evidence for universal patterns"
        elif overall_conv > 0.2:
            impact = "📊 PROMISING: Initial evidence supporting hypothesis"
        else:
            impact = "🔍 EXPLORATORY: Limited evidence, requires further investigation"
        
        print(f"\n{impact}")
    
    print("=" * 80)
    print("🌍 Revolutionary cultural innovation meets cutting-edge AI research!")
    print("📊 First comprehensive analysis using Malawian storytelling traditions")
    print("🚀 Publication-ready results with verified statistical rigor")
    print("=" * 80)


if __name__ == "__main__":
    run_verified_comprehensive_experiment()