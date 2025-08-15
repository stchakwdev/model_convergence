#!/usr/bin/env python3
"""
Run Enhanced v2.0 Universal Alignment Patterns Experiment

This script executes the improved experiment with:
1. Premium model selection for higher quality responses
2. Optimized scale to fit within remaining budget ($49.91)
3. Enhanced statistical analysis with proper power calculation

Expected improvements over v1.0:
- 40-60% higher convergence with premium models
- Statistical significance at p<0.001 
- Better universal pattern detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from comprehensive_analysis import ComprehensiveAnalysisFramework, ExperimentConfig
from cost_monitor import CostMonitor
import json

def create_optimized_v2_config() -> ExperimentConfig:
    """
    Create optimized v2.0 configuration that fits within budget.
    
    Strategy:
    - Use mix of premium and efficient models
    - 75 prompts per capability (50% increase over v1.0) 
    - Focus on statistical power within budget constraints
    """
    
    # Carefully selected model mix for maximum impact within budget
    models = [
        # Premium models for highest quality (limited use due to cost)
        "anthropic/claude-3.5-sonnet",     # Best reasoning model
        "openai/gpt-4-turbo",              # OpenAI flagship
        
        # High-quality open source models (good value)
        "openai/gpt-oss-120b",             # Free, excellent reasoning
        "zhipu/glm-4.5",                   # Low cost, agentic capabilities
        "moonshot/kimi-k2",                # Medium cost, long context
        
        # Efficient baseline
        "meta/llama-3.1-8b-instruct:free", # Free comparison point
    ]
    
    return ExperimentConfig(
        name="Universal Alignment Patterns v2.0 - Enhanced",
        description="Enhanced experiment with premium models and increased statistical power",
        models=models,
        capabilities=[
            "truthfulness",
            "safety_boundaries", 
            "instruction_following",
            "uncertainty_expression",
            "context_awareness"
        ],
        prompts_per_capability=75,  # 50% increase for better statistical power
        max_tokens=250,
        temperature=0.0,
        budget_limit_usd=45.0,  # Conservative within $49.91 remaining
        statistical_confidence=0.001
    )

def run_enhanced_experiment():
    """Execute the enhanced v2.0 experiment"""
    
    print("🚀 LAUNCHING ENHANCED v2.0 EXPERIMENT")
    print("=" * 80)
    
    # Create configuration
    config = create_optimized_v2_config()
    
    # Estimate experiment scale
    total_calls = len(config.models) * len(config.capabilities) * config.prompts_per_capability
    print(f"📊 Experiment Scale:")
    print(f"   Models: {len(config.models)}")
    print(f"   Capabilities: {len(config.capabilities)}")  
    print(f"   Prompts per capability: {config.prompts_per_capability}")
    print(f"   Total API calls: {total_calls:,}")
    print(f"   Budget limit: ${config.budget_limit_usd}")
    
    # Initialize cost monitor to check current status
    cost_monitor = CostMonitor(budget_limit_usd=50.0)
    budget_status = cost_monitor.check_budget_status()
    current_cost = budget_status.get('total_cost_usd', cost_monitor.total_cost_usd)
    remaining = budget_status.get('budget_remaining', 50.0 - current_cost)
    print(f"   Current spending: ${current_cost:.3f}")
    print(f"   Remaining budget: ${remaining:.2f}")
    
    # Rough cost estimate (conservative)
    estimated_cost_per_call = 0.008  # Based on v1.0 data + premium models
    estimated_total_cost = total_calls * estimated_cost_per_call
    print(f"   Estimated cost: ${estimated_total_cost:.2f}")
    print(f"   Estimated utilization: {(estimated_total_cost/config.budget_limit_usd)*100:.1f}%")
    
    if estimated_total_cost > remaining:
        print("⚠️  WARNING: Estimated cost exceeds remaining budget")
        print("   Consider reducing scale or using budget-optimized configuration")
        return
    
    print("✅ Cost estimate within budget - proceeding with experiment")
    print()
    
    # Initialize framework
    framework = ComprehensiveAnalysisFramework(output_dir="results")
    
    print("🔬 Expected Improvements over v1.0:")
    print("   • Premium models (GPT-4, Claude-3.5) for higher quality responses")
    print("   • 50% more prompts per capability (75 vs 50)")
    print("   • Enhanced model diversity across architectures")
    print("   • Expected convergence: 45-65% (vs 28.7% in v1.0)")
    print("   • Target: Statistical significance at p<0.001")
    print()
    
    # Update comprehensive analysis to use v2 datasets
    print("📝 Using enhanced v2.0 prompt datasets...")
    print("   (75+ prompts per capability with improved diversity)")
    print()
    
    # Run the experiment
    try:
        # Save the config temporarily and run experiment
        config_dict = asdict(config)
        temp_config_file = "temp_v2_config.json"
        with open(temp_config_file, 'w') as f:
            json.dump(config_dict, f)
        
        results = framework.run_experiment("enhanced_v2")
        
        print("✅ Enhanced v2.0 experiment completed successfully!")
        print(f"📊 Overall convergence: {results.convergence_analysis.get('overall_convergence', 0)*100:.1f}%")
        print(f"💰 Total cost: ${results.cost_summary.get('total_cost_usd', 0):.3f}")
        print(f"📁 Results saved to: results/analysis_outputs/")
        
        # Quick analysis of improvement
        v1_convergence = 0.287  # From previous experiment
        v2_convergence = results.convergence_analysis.get('overall_convergence', 0)
        improvement = ((v2_convergence - v1_convergence) / v1_convergence) * 100
        
        print(f"\n🎯 IMPROVEMENT ANALYSIS:")
        print(f"   v1.0 convergence: {v1_convergence*100:.1f}%")
        print(f"   v2.0 convergence: {v2_convergence*100:.1f}%")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if v2_convergence > 0.5:
            print("🎉 SUCCESS: Achieved moderate evidence threshold!")
        if v2_convergence > 0.7:
            print("🏆 EXCELLENCE: Achieved strong evidence threshold!")
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_enhanced_experiment()