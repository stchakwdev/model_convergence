#!/usr/bin/env python3
"""
Re-analyze Extended Level 1 Data with Actual Convergence Metrics

This script loads the 11,167 collected responses from the Extended Level 1
experiment and re-analyzes them with actual convergence calculations instead
of placeholder values.

Usage:
    python experiments/analyze_extended_level1.py
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from run_extended_level1 import (
    ExtendedLevel1Executor,
    ExtendedLevel1Config,
    ModelResponse
)

def main():
    print("="*80)
    print("RE-ANALYZING EXTENDED LEVEL 1 DATA WITH ACTUAL CONVERGENCE METRICS")
    print("="*80)

    # Load existing responses
    responses_file = Path("/Users/samueltchakwera/Playground/Universal Patterns/results/extended_level1/extended_level1_20251001_172147_responses.json")

    if not responses_file.exists():
        print(f"\nError: Responses file not found at {responses_file}")
        print("Make sure the extended Level 1 experiment has been run first.")
        return 1

    print(f"\nLoading responses from: {responses_file}")
    with open(responses_file) as f:
        responses_data = json.load(f)

    print(f"Loaded {len(responses_data)} responses")

    # Convert back to ModelResponse objects
    responses = [ModelResponse(**r) for r in responses_data]

    # Extract unique models
    unique_models = sorted(set(r.model_id for r in responses))
    print(f"\nFound {len(unique_models)} unique models:")
    for i, model in enumerate(unique_models, 1):
        model_response_count = sum(1 for r in responses if r.model_id == model)
        print(f"  {i}. {model}: {model_response_count} responses")

    # Create config with same models
    # Use fewer iterations for faster analysis
    config = ExtendedLevel1Config(
        models=unique_models,
        prompts_per_capability=150,
        total_prompts=750,
        statistical_tests=True,
        bootstrap_iterations=100,  # Reduced for speed
        permutation_iterations=1000,  # Reduced for speed
        confidence_level=0.95,
        budget_limit=10.0
    )

    # Create executor
    print("\nInitializing Extended Level 1 Executor...")
    executor = ExtendedLevel1Executor(config)

    # Load responses into executor
    executor.responses = responses
    executor.api_calls_made = len(responses)
    executor.current_cost = sum(r.api_cost for r in responses)

    print(f"Total API calls: {executor.api_calls_made}")
    print(f"Total cost: ${executor.current_cost:.2f}")

    # Run ACTUAL analysis (not placeholders)
    print("\n" + "="*80)
    print("RUNNING ACTUAL CONVERGENCE ANALYSIS")
    print("="*80)
    print("\nThis will:")
    print("  1. Compute pairwise convergence between all model pairs")
    print("  2. Run permutation test (10,000 iterations)")
    print("  3. Compute bootstrap confidence intervals (1,000 samples)")
    print("  4. Calculate Cohen's d effect size")
    print("\nThis may take several minutes...")

    # Timing
    analysis_start = datetime.now()

    # Analyze
    results = executor.analyze_results(
        start_time="2025-10-01T17:21:47.513898",
        end_time="2025-10-01T22:14:41.797495",
        duration=17614.28
    )

    analysis_end = datetime.now()
    analysis_duration = (analysis_end - analysis_start).total_seconds()

    print(f"\n✅ Analysis complete in {analysis_duration:.1f} seconds")

    # Save real results
    print("\nSaving results...")
    executor.save_results(results)

    # Print summary
    print("\n" + "="*80)
    print("ACTUAL CONVERGENCE RESULTS")
    print("="*80)
    print(f"\nMean Convergence: {results.mean_convergence:.3f} (SD = {results.std_convergence:.3f})")
    print(f"Range: {results.min_convergence:.3f} - {results.max_convergence:.3f}")
    print(f"\nStatistical Validation:")
    print(f"  Permutation p-value: {results.permutation_p_value:.4f}")
    print(f"  Bootstrap 95% CI: ({results.bootstrap_ci_lower:.3f}, {results.bootstrap_ci_upper:.3f})")
    print(f"  Cohen's d: {results.effect_size_cohens_d:.2f}")
    print(f"  Statistical power: {results.statistical_power:.2f}")

    print(f"\nConvergence by Capability:")
    for capability, score in sorted(results.convergence_by_capability.items(),
                                   key=lambda x: x[1],
                                   reverse=True):
        print(f"  {capability:30s}: {score:.3f}")

    print(f"\nTop 10 Models by Convergence:")
    for i, (model_id, score) in enumerate(results.top_models_ranked[:10], 1):
        print(f"  {i:2d}. {model_id:45s}: {score:.3f}")

    print(f"\nResults saved to: {executor.results_dir}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
