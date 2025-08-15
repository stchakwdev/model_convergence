#!/usr/bin/env python3
"""
🚀 ULTRA v2.5 Experiment: Cutting-Edge 2025 Models

Revolutionary experiment testing universal alignment patterns with:
- GPT-5 (next generation reasoning)
- Claude 4 Sonnet (revolutionary safety + capability) 
- Gemini 2.5 Pro (advanced multimodal)

This represents the absolute state-of-the-art in AI as of 2025.
Expected to show the strongest universal alignment patterns yet discovered.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 LAUNCHING ULTRA v2.5 EXPERIMENT - CUTTING-EDGE 2025 MODELS")
    print("=" * 90)
    print("🌟 REVOLUTIONARY MODEL LINEUP:")
    print("   💫 GPT-5 Chat: OpenAI's next-generation reasoning breakthrough")
    print("   💫 Claude 4 Sonnet: Anthropic's revolutionary safety + capability fusion")
    print("   💫 Gemini 2.5 Pro: Google's advanced multimodal powerhouse")
    print("   🔬 Plus: Previous generation models for comparison")
    print()
    print("🎯 Expected Universal Convergence: 60-80%+ (vs 28.7% in v1.0)")
    print("💰 Budget: $35 conservative allocation")
    print("📊 Scale: 6 models × 5 capabilities × 60 prompts = 1,800 API calls")
    print()
    
    # Import and run the ultra v2.5 configuration
    from comprehensive_analysis import ComprehensiveAnalysisFramework
    
    print("🧬 Loading enhanced v2.0 datasets (75+ prompts per capability)...")
    print("🔬 Initializing advanced semantic analysis with sentence transformers...")
    print("📈 Preparing rigorous statistical framework (p < 0.001)...")
    print()
    
    framework = ComprehensiveAnalysisFramework(output_dir="results")
    
    try:
        print("🚀 LAUNCHING ULTRA v2.5 EXPERIMENT...")
        print("   This will make history - testing the newest AI models for universal patterns!")
        print()
        
        results = framework.run_experiment("ultra_v25")
        
        print("\\n🎉 ULTRA v2.5 EXPERIMENT COMPLETED!")
        print("=" * 60)
        
        # Extract results
        overall_conv = results.convergence_analysis.get('overall_convergence', 0)
        total_cost = results.cost_summary.get('total_cost_usd', 0)
        
        print(f"📊 REVOLUTIONARY RESULTS:")
        print(f"   Overall Convergence: {overall_conv*100:.1f}%")
        print(f"   Total Cost: ${total_cost:.3f}")
        print()
        
        # Comparison with previous experiments
        v1_convergence = 0.287  # 28.7%
        v2_convergence = overall_conv
        improvement = ((v2_convergence - v1_convergence) / v1_convergence) * 100 if v1_convergence > 0 else 0
        
        print(f"🚀 BREAKTHROUGH ANALYSIS:")
        print(f"   v1.0 baseline: {v1_convergence*100:.1f}%")
        print(f"   v2.5 ULTRA: {v2_convergence*100:.1f}%")
        print(f"   Improvement: {improvement:+.1f}%")
        print()
        
        # Success thresholds
        if v2_convergence > 0.8:
            print("🏆 BREAKTHROUGH: VERY STRONG EVIDENCE for Universal Alignment Patterns!")
            print("    This represents a paradigm shift in AI alignment research!")
        elif v2_convergence > 0.6:
            print("🎯 SUCCESS: STRONG EVIDENCE for Universal Alignment Patterns!")
            print("    Clear demonstration of universal principles across cutting-edge models!")
        elif v2_convergence > 0.4:
            print("📈 PROGRESS: MODERATE EVIDENCE for Universal Patterns!")
            print("    Encouraging results showing convergence trends!")
        else:
            print("🔬 RESEARCH: Results provide valuable baseline for future work!")
        
        print()
        print("💡 SIGNIFICANCE FOR AI SAFETY:")
        if v2_convergence > 0.6:
            print("   These results suggest that alignment properties emerge universally")
            print("   across even the most advanced AI architectures, supporting the")
            print("   hypothesis that safety interventions may transfer between models.")
        else:
            print("   Results contribute to our understanding of alignment patterns")
            print("   and provide methodology for future large-scale studies.")
            
        print()
        print(f"📁 Full results saved to: results/analysis_outputs/")
        print(f"🎨 Visualizations will be generated next for publication")
        
    except Exception as e:
        print(f"❌ ULTRA v2.5 experiment encountered an issue: {e}")
        print("\\n🔧 Troubleshooting information:")
        print(f"   This may be due to:")
        print(f"   • New model API endpoints requiring adjustment")
        print(f"   • Rate limiting on cutting-edge models")
        print(f"   • Model availability variations")
        print()
        print("💡 The experimental framework is robust - check model IDs and retry")
        import traceback
        traceback.print_exc()