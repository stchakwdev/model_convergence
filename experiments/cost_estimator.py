#!/usr/bin/env python3
"""
Cost Estimation Utility for Universal Alignment Patterns Research

Quick tool to estimate costs before running experiments and validate
budget planning for the Anthropic Fellowship application.

Usage:
    python cost_estimator.py --experiment fellowship_research
    python cost_estimator.py --models 5 --capabilities 5 --prompts 50
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from cost_monitor import CostMonitor
from comprehensive_analysis import ComprehensiveAnalysisFramework


def estimate_predefined_experiment(experiment_name: str):
    """Estimate cost for a predefined experiment configuration."""
    
    framework = ComprehensiveAnalysisFramework()
    
    if experiment_name not in framework.experiment_configs:
        print(f"❌ Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(framework.experiment_configs.keys())}")
        return
    
    config = framework.experiment_configs[experiment_name]
    cost_monitor = CostMonitor(budget_limit_usd=config.budget_limit_usd)
    
    print(f"\n🔬 Cost Estimation for: {config.name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")
    print(f"Models: {len(config.models)} ({', '.join([m.split('/')[-1] for m in config.models])})")
    print(f"Capabilities: {len(config.capabilities)} ({', '.join(config.capabilities)})")
    print(f"Prompts per capability: {config.prompts_per_capability}")
    print(f"Budget limit: ${config.budget_limit_usd}")
    
    # Estimate costs
    estimation = cost_monitor.estimate_experiment_cost(
        n_models=len(config.models),
        n_capabilities=len(config.capabilities),
        prompts_per_capability=config.prompts_per_capability,
        avg_prompt_length=50,  # Conservative estimate
        avg_response_length=config.max_tokens
    )
    
    print(f"\n💰 Cost Estimation Results:")
    print(f"{'='*60}")
    print(f"Total API calls: {estimation['total_api_calls']:,}")
    print(f"Estimated cost: ${estimation['estimated_cost_usd']:.4f}")
    print(f"Cost with 20% buffer: ${estimation['cost_with_buffer']:.4f}")
    print(f"Budget utilization: {estimation['budget_utilization']:.1f}%")
    print(f"Within budget: {'✅ YES' if estimation['within_budget'] else '❌ NO'}")
    print(f"Recommendation: {estimation['recommendation']}")
    
    if estimation['model_breakdown']:
        print(f"\n📊 Cost Breakdown by Model:")
        print(f"{'='*60}")
        for model_id, breakdown in estimation['model_breakdown'].items():
            if breakdown['total_cost'] > 0:
                print(f"{model_id}:")
                print(f"  Calls: {breakdown['calls']}")
                print(f"  Total cost: ${breakdown['total_cost']:.4f}")
                print(f"  Cost per call: ${breakdown['cost_per_call']:.6f}")
    
    # Risk assessment
    print(f"\n⚠️  Risk Assessment:")
    print(f"{'='*60}")
    if estimation['budget_utilization'] < 50:
        print("🟢 LOW RISK: Well within budget, safe to proceed")
    elif estimation['budget_utilization'] < 75:
        print("🟡 MEDIUM RISK: Approaching budget limits, monitor carefully")
    elif estimation['budget_utilization'] < 90:
        print("🟠 HIGH RISK: Close to budget limit, consider reducing scope")
    else:
        print("🔴 CRITICAL RISK: Exceeds or very close to budget limit")
        print("   Recommendations:")
        print("   - Reduce number of prompts per capability")
        print("   - Use more free-tier models")
        print("   - Run validation test first")
        print("   - Consider increasing budget")


def estimate_custom_experiment(n_models: int, n_capabilities: int, prompts_per_capability: int, budget: float):
    """Estimate cost for custom experiment parameters."""
    
    cost_monitor = CostMonitor(budget_limit_usd=budget)
    
    print(f"\n🔬 Custom Experiment Cost Estimation")
    print(f"{'='*60}")
    print(f"Models: {n_models}")
    print(f"Capabilities: {n_capabilities}")
    print(f"Prompts per capability: {prompts_per_capability}")
    print(f"Budget limit: ${budget}")
    
    estimation = cost_monitor.estimate_experiment_cost(
        n_models=n_models,
        n_capabilities=n_capabilities,
        prompts_per_capability=prompts_per_capability
    )
    
    print(f"\n💰 Cost Estimation Results:")
    print(f"{'='*60}")
    print(f"Total API calls: {estimation['total_api_calls']:,}")
    print(f"Estimated cost: ${estimation['estimated_cost_usd']:.4f}")
    print(f"Cost with buffer: ${estimation['cost_with_buffer']:.4f}")
    print(f"Budget utilization: {estimation['budget_utilization']:.1f}%")
    print(f"Within budget: {'✅ YES' if estimation['within_budget'] else '❌ NO'}")
    print(f"Recommendation: {estimation['recommendation']}")


def main():
    parser = argparse.ArgumentParser(description="Estimate costs for Universal Alignment Patterns experiments")
    
    # Predefined experiment
    parser.add_argument("--experiment", type=str, help="Predefined experiment name (fellowship_research, validation_test)")
    
    # Custom experiment parameters
    parser.add_argument("--models", type=int, help="Number of models to test")
    parser.add_argument("--capabilities", type=int, help="Number of capabilities to test") 
    parser.add_argument("--prompts", type=int, help="Number of prompts per capability")
    parser.add_argument("--budget", type=float, default=50.0, help="Budget limit in USD")
    
    # List available experiments
    parser.add_argument("--list", action="store_true", help="List available predefined experiments")
    
    args = parser.parse_args()
    
    if args.list:
        framework = ComprehensiveAnalysisFramework()
        print("\n📋 Available Predefined Experiments:")
        print("="*60)
        for name, config in framework.experiment_configs.items():
            print(f"{name}:")
            print(f"  Description: {config.description}")
            print(f"  Models: {len(config.models)}")
            print(f"  Capabilities: {len(config.capabilities)}")
            print(f"  Prompts/capability: {config.prompts_per_capability}")
            print(f"  Budget: ${config.budget_limit_usd}")
            print()
        return
    
    if args.experiment:
        estimate_predefined_experiment(args.experiment)
    elif args.models and args.capabilities and args.prompts:
        estimate_custom_experiment(args.models, args.capabilities, args.prompts, args.budget)
    else:
        print("❌ Error: Either specify --experiment name or provide --models, --capabilities, and --prompts")
        print("Use --help for usage information")
        print("Use --list to see available predefined experiments")


if __name__ == "__main__":
    main()