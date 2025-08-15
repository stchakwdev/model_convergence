#!/usr/bin/env python3
"""
Simple Enhanced v2.0 Experiment Runner

Directly runs the comprehensive analysis with enhanced parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 LAUNCHING ENHANCED v2.0 EXPERIMENT")
    print("=" * 80)
    print("🔬 Enhanced Configuration:")
    print("   • Premium models: GPT-4, Claude-3.5-Sonnet, etc.")
    print("   • 75+ prompts per capability (50% increase)")
    print("   • Enhanced statistical power")
    print("   • Expected convergence: 45-65% (vs 28.7% in v1.0)")
    print()
    
    # Import and run the enhanced v2.0 configuration
    from comprehensive_analysis import ComprehensiveAnalysisFramework
    
    print("📝 Running enhanced v2.0 experiment...")
    framework = ComprehensiveAnalysisFramework(output_dir="results")
    results = framework.run_experiment("enhanced_v2")
    
    print("\\n✅ Enhanced v2.0 experiment completed!")
    print(f"📊 Overall convergence: {results.convergence_analysis.get('overall_convergence', 0)*100:.1f}%")
    print(f"💰 Total cost: ${results.cost_summary.get('total_cost_usd', 0):.3f}")
    
    # Comparison with v1.0
    v1_convergence = 0.287
    v2_convergence = results.convergence_analysis.get('overall_convergence', 0)
    improvement = ((v2_convergence - v1_convergence) / v1_convergence) * 100 if v1_convergence > 0 else 0
    
    print(f"\\n🎯 IMPROVEMENT ANALYSIS:")
    print(f"   v1.0 convergence: {v1_convergence*100:.1f}%")
    print(f"   v2.0 convergence: {v2_convergence*100:.1f}%")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if v2_convergence > 0.5:
        print("🎉 SUCCESS: Achieved moderate evidence threshold!")
    if v2_convergence > 0.7:
        print("🏆 EXCELLENCE: Achieved strong evidence threshold!")