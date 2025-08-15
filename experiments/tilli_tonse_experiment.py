"""
Tilli Tonse Experiment Runner: Multi-Turn Story-Based Convergence Analysis

This script runs the enhanced convergence analysis using story-based prompts
that generate 200-500 token responses for robust KL divergence analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
from datetime import datetime
from typing import List, Dict, Any

from patterns.tilli_tonse_framework import TilliTonseFramework, TilliTonseAnalyzer
from patterns.kl_enhanced_analyzer import HybridConvergenceAnalyzer
from patterns.semantic_analyzer import EnhancedSemanticAnalyzer
from models.openrouter_model import OpenRouterModel
from models.model_registry import model_registry


def load_tilli_tonse_stories(filepath: str) -> TilliTonseFramework:
    """Load stories from JSON file into framework"""
    
    framework = TilliTonseFramework()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    
    # Reconstruct stories from JSON data
    for capability, stories in story_data.items():
        for story_dict in stories:
            # Reconstruct segments
            narrative_parts = []
            checkpoints = []
            
            for i, segment in enumerate(story_dict["segments"]):
                if segment["is_checkpoint"]:
                    # This is a checkpoint
                    from patterns.tilli_tonse_framework import StoryCheckpointType
                    checkpoint_type = StoryCheckpointType(segment["checkpoint_type"])
                    checkpoints.append((len(narrative_parts), checkpoint_type, segment["checkpoint_prompt"]))
                else:
                    # This is narrative content
                    narrative_parts.append(segment["content"])
            
            # Create the story
            framework.create_story_sequence(
                story_id=story_dict["story_id"],
                capability=story_dict["capability"],
                title=story_dict["title"],
                narrative_parts=narrative_parts,
                checkpoints=checkpoints,
                cultural_context=story_dict["cultural_context"]
            )
    
    print(f"📚 Loaded {sum(len(stories) for stories in framework.stories.values())} stories")
    return framework


def run_tilli_tonse_experiment(models: List[str], 
                             capabilities: List[str],
                             max_stories_per_capability: int = 5,
                             output_dir: str = "results/tilli_tonse_analysis") -> Dict[str, Any]:
    """
    Run the complete Tilli Tonse experiment.
    
    This compares the new story-based approach against the original simple Q&A.
    """
    
    print("🎭 STARTING TILLI TONSE EXPERIMENT")
    print("=" * 60)
    print("🌍 Using Malawian oral storytelling tradition for AI alignment research")
    print(f"📊 Testing {len(models)} models × {len(capabilities)} capabilities")
    print(f"📚 Maximum {max_stories_per_capability} stories per capability")
    
    # Load stories
    script_dir = os.path.dirname(__file__)
    story_filepath = os.path.join(script_dir, "prompt_datasets", "tilli_tonse_stories.json")
    if not os.path.exists(story_filepath):
        raise FileNotFoundError(f"Story file not found: {story_filepath}")
    
    framework = load_tilli_tonse_stories(story_filepath)
    
    # Initialize models
    model_objects = []
    for model_name in models:
        try:
            model = OpenRouterModel(model_name)
            model_objects.append(model)
            print(f"✅ Initialized {model.name}")
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
    
    if not model_objects:
        raise ValueError("No models successfully initialized")
    
    # Initialize analyzers
    semantic_analyzer = EnhancedSemanticAnalyzer()
    hybrid_analyzer = HybridConvergenceAnalyzer(semantic_analyzer=semantic_analyzer)
    tilli_tonse_analyzer = TilliTonseAnalyzer(hybrid_analyzer=hybrid_analyzer)
    
    # Pass the loaded framework to the analyzer
    tilli_tonse_analyzer.framework = framework
    
    # Run experiments for each capability
    experiment_results = {}
    total_cost = 0.0
    
    for capability in capabilities:
        print(f"\n🧠 ANALYZING CAPABILITY: {capability.upper()}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run story-based experiment
            capability_results = tilli_tonse_analyzer.run_story_experiment(
                models=model_objects,
                capability=capability,
                max_stories=max_stories_per_capability
            )
            
            # Calculate costs (rough estimate)
            if "story_experiment_results" in capability_results:
                total_tokens = capability_results["story_experiment_results"]["total_tokens_analyzed"]
                estimated_cost = total_tokens * 0.00002  # Rough estimate $0.02 per 1k tokens
                total_cost += estimated_cost
                capability_results["estimated_cost_usd"] = estimated_cost
            
            experiment_results[capability] = capability_results
            
            elapsed = time.time() - start_time
            print(f"⏱️ Completed {capability} in {elapsed:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Error analyzing {capability}: {e}")
            experiment_results[capability] = {"error": str(e)}
    
    # Aggregate overall results
    overall_results = {
        "experiment_metadata": {
            "framework": "Tilli Tonse - Malawian Storytelling Tradition",
            "timestamp": datetime.now().isoformat(),
            "models_tested": [model.name for model in model_objects],
            "capabilities_tested": capabilities,
            "max_stories_per_capability": max_stories_per_capability,
            "cultural_inspiration": "Malawian 'tilli tonse' oral storytelling checkpoints",
            "methodology": "Multi-turn story-based convergence analysis"
        },
        "capability_results": experiment_results,
        "aggregate_analysis": _aggregate_tilli_tonse_results(experiment_results),
        "cost_summary": {
            "total_estimated_cost_usd": total_cost,
            "cost_per_model": total_cost / len(model_objects),
            "cost_per_capability": total_cost / len(capabilities)
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"tilli_tonse_analysis_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    return overall_results


def _aggregate_tilli_tonse_results(capability_results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results across all capabilities"""
    
    valid_results = [result for result in capability_results.values() 
                    if "story_experiment_results" in result]
    
    if not valid_results:
        return {"error": "No valid results to aggregate"}
    
    # Aggregate hybrid convergence scores
    all_hybrid_scores = []
    all_semantic_scores = []
    all_distributional_scores = []
    total_stories = 0
    total_tokens = 0
    
    for result in valid_results:
        exp_results = result["story_experiment_results"]
        
        if "aggregate_hybrid_convergence" in exp_results:
            hybrid_data = exp_results["aggregate_hybrid_convergence"]
            if "individual_scores" in hybrid_data:
                all_hybrid_scores.extend(hybrid_data["individual_scores"])
            
            if "mean_semantic_convergence" in hybrid_data:
                all_semantic_scores.append(hybrid_data["mean_semantic_convergence"])
            
            if "mean_distributional_convergence" in hybrid_data:
                all_distributional_scores.append(hybrid_data["mean_distributional_convergence"])
        
        total_stories += exp_results.get("story_count", 0)
        total_tokens += exp_results.get("total_tokens_analyzed", 0)
    
    # Calculate overall metrics
    import numpy as np
    
    overall_metrics = {
        "total_stories_analyzed": total_stories,
        "total_tokens_analyzed": total_tokens,
        "average_tokens_per_story": total_tokens / total_stories if total_stories > 0 else 0,
        "capabilities_tested": len(valid_results)
    }
    
    if all_hybrid_scores:
        overall_metrics.update({
            "overall_hybrid_convergence": np.mean(all_hybrid_scores),
            "hybrid_convergence_std": np.std(all_hybrid_scores),
            "hybrid_convergence_range": [np.min(all_hybrid_scores), np.max(all_hybrid_scores)]
        })
    
    if all_semantic_scores:
        overall_metrics.update({
            "overall_semantic_convergence": np.mean(all_semantic_scores),
            "semantic_convergence_std": np.std(all_semantic_scores)
        })
    
    if all_distributional_scores:
        overall_metrics.update({
            "overall_distributional_convergence": np.mean(all_distributional_scores),
            "distributional_convergence_std": np.std(all_distributional_scores)
        })
    
    return overall_metrics


def main():
    """Run Tilli Tonse experiment with default configuration"""
    
    # Default configuration
    models = [
        "openai/gpt-oss-120b",        # Open-source reasoning
        "anthropic/claude-3-haiku",   # Safety-focused
        "zhipu/glm-4.5"              # Agentic leader
    ]
    
    capabilities = [
        "truthfulness",
        "safety_boundaries", 
        "instruction_following",
        "uncertainty_expression",
        "context_awareness"
    ]
    
    try:
        results = run_tilli_tonse_experiment(
            models=models,
            capabilities=capabilities,
            max_stories_per_capability=3  # Start with fewer stories for testing
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 TILLI TONSE EXPERIMENT SUMMARY")
        print("=" * 60)
        
        agg = results["aggregate_analysis"]
        
        print(f"📚 Stories analyzed: {agg.get('total_stories_analyzed', 'N/A')}")
        print(f"📝 Total tokens: {agg.get('total_tokens_analyzed', 'N/A'):,}")
        print(f"📊 Average tokens per story: {agg.get('average_tokens_per_story', 'N/A'):.0f}")
        
        if "overall_hybrid_convergence" in agg:
            print(f"\n🧬 CONVERGENCE RESULTS:")
            print(f"   Overall Hybrid: {agg['overall_hybrid_convergence']:.1%}")
            print(f"   Semantic: {agg.get('overall_semantic_convergence', 0):.1%}")
            print(f"   Distributional: {agg.get('overall_distributional_convergence', 0):.1%}")
        
        print(f"\n💰 Cost: ${results['cost_summary']['total_estimated_cost_usd']:.3f}")
        print(f"🌍 Cultural Innovation: Malawian 'tilli tonse' storytelling tradition")
        print(f"📈 Expected Improvement: 10-20x richer token distributions")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()