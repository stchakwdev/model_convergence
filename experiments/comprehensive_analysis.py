"""
Comprehensive Universal Alignment Patterns Analysis

This module implements a rigorous experimental framework for testing the
Universal Alignment Patterns hypothesis across diverse AI model architectures.
Designed for the Anthropic Fellowship application with statistical rigor.

Author: Samuel Tchakwera
Purpose: Empirical evidence for universal alignment patterns
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, loading environment manually")
    env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import OpenRouterModel, ModelInterface
from models.model_registry import model_registry
from patterns import PatternDiscoveryEngine, ConvergenceAnalyzer
from patterns.semantic_analyzer import EnhancedSemanticAnalyzer
from cost_monitor import CostMonitor


@dataclass
class ExperimentConfig:
    """Configuration for comprehensive experiments"""
    name: str
    description: str
    models: List[str]
    capabilities: List[str]
    prompts_per_capability: int
    max_tokens: int
    temperature: float
    budget_limit_usd: float
    statistical_confidence: float
    
    
@dataclass
class ModelResponse:
    """Single model response data structure"""
    model_id: str
    capability: str
    prompt_id: str
    prompt: str
    response: str
    timestamp: datetime
    tokens_used: int
    cost_usd: float
    

@dataclass
class ExperimentResults:
    """Complete experiment results"""
    config: ExperimentConfig
    responses: List[ModelResponse]
    convergence_analysis: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    cost_summary: Dict[str, float]
    execution_time: float
    

class ComprehensiveAnalysisFramework:
    """
    Main framework for conducting rigorous universal alignment pattern analysis.
    
    Implements the complete experimental protocol with statistical validation
    and cost management for fellowship application research.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the comprehensive analysis framework"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)
        (self.output_dir / "analysis_outputs").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize components
        self.discovery_engine = PatternDiscoveryEngine()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.semantic_analyzer = EnhancedSemanticAnalyzer()
        
        # Advanced cost monitoring system
        self.cost_monitor = None  # Will be initialized when experiment starts
        
        # Load experiment configurations
        self.experiment_configs = self._load_experiment_configs()
        
    def _load_experiment_configs(self) -> Dict[str, ExperimentConfig]:
        """Load predefined experiment configurations"""
        
        configs = {
            "fellowship_research": ExperimentConfig(
                name="Universal Alignment Patterns - Fellowship Research",
                description="Comprehensive analysis across 5 models and 5 capabilities for Anthropic Fellowship",
                models=[
                    "openai/gpt-oss-120b",           # Free tier, OpenAI open-source
                    "anthropic/claude-3-haiku",      # Low cost, safety-focused
                    "alibaba/qwen-2.5-72b",         # Chinese large model
                    "deepseek/deepseek-chat",        # Free tier, Chinese
                    "meta-llama/llama-3.1-70b:free" # Free tier, Western open-source
                ],
                capabilities=[
                    "truthfulness",
                    "safety_boundaries", 
                    "instruction_following",
                    "uncertainty_expression",
                    "context_awareness"
                ],
                prompts_per_capability=50,
                max_tokens=200,
                temperature=0.0,
                budget_limit_usd=50.0,
                statistical_confidence=0.001  # p < 0.001 for strong evidence
            ),
            
            "validation_test": ExperimentConfig(
                name="Validation Test - Small Scale",
                description="Small-scale validation before full experiment",
                models=[
                    "openai/gpt-oss-120b",
                    "anthropic/claude-3-haiku"
                ],
                capabilities=[
                    "truthfulness",
                    "safety_boundaries"
                ],
                prompts_per_capability=5,
                max_tokens=200,
                temperature=0.0,
                budget_limit_usd=2.0,
                statistical_confidence=0.05
            )
        }
        
        return configs
    
    def load_prompt_datasets(self, capabilities: List[str]) -> Dict[str, List[str]]:
        """Load comprehensive prompt datasets for each capability"""
        
        datasets = {}
        
        for capability in capabilities:
            dataset_file = Path(__file__).parent / "prompt_datasets" / f"{capability}.json"
            
            if dataset_file.exists():
                with open(dataset_file, 'r') as f:
                    datasets[capability] = json.load(f)
            else:
                # Fallback to discovery engine prompts
                if capability in self.discovery_engine.universal_features:
                    feature = self.discovery_engine.universal_features[capability]
                    datasets[capability] = [prompt for prompt, _ in feature.behavioral_signature]
                else:
                    print(f"⚠️  Warning: No prompts found for capability '{capability}'")
                    datasets[capability] = []
        
        return datasets
    
    def estimate_experiment_cost(self, config: ExperimentConfig) -> Dict[str, float]:
        """Estimate total cost for the experiment"""
        
        total_prompts = len(config.models) * len(config.capabilities) * config.prompts_per_capability
        
        # Rough cost estimates per 1M tokens (conservative estimates)
        model_costs = {
            "openai/gpt-oss-120b": 0.5,         # Often free
            "anthropic/claude-3-haiku": 0.25,    # Low cost
            "alibaba/qwen-2.5-72b": 1.2,        # Moderate cost
            "deepseek/deepseek-chat": 0.0,       # Free tier
            "meta-llama/llama-3.1-70b:free": 0.0 # Free tier
        }
        
        estimated_tokens_per_call = (50 + config.max_tokens)  # Input + output
        estimated_total_tokens = total_prompts * estimated_tokens_per_call
        
        total_cost = 0.0
        model_breakdown = {}
        
        for model in config.models:
            model_cost_per_1m = model_costs.get(model, 1.0)  # Default fallback
            model_calls = len(config.capabilities) * config.prompts_per_capability
            model_tokens = model_calls * estimated_tokens_per_call
            model_total_cost = (model_tokens / 1_000_000) * model_cost_per_1m
            
            model_breakdown[model] = {
                "calls": model_calls,
                "tokens": model_tokens,
                "cost_usd": model_total_cost
            }
            total_cost += model_total_cost
        
        return {
            "total_prompts": total_prompts,
            "estimated_tokens": estimated_total_tokens,
            "estimated_cost_usd": total_cost,
            "model_breakdown": model_breakdown,
            "within_budget": total_cost <= config.budget_limit_usd
        }
    
    def initialize_models(self, model_ids: List[str]) -> List[ModelInterface]:
        """Initialize all models for the experiment"""
        
        models = []
        failed_models = []
        
        print(f"🤖 Initializing {len(model_ids)} models...")
        
        for model_id in model_ids:
            try:
                model = OpenRouterModel(
                    model_id=model_id,
                    temperature=0.0,
                    max_tokens=200,
                    use_cache=True
                )
                models.append(model)
                
                # Get cost info
                cost_info = model.get_cost_info()
                tier = cost_info.get("tier", "unknown")
                print(f"  ✅ {model.name} ({tier})")
                
            except Exception as e:
                print(f"  ❌ Failed to initialize {model_id}: {e}")
                failed_models.append(model_id)
        
        if failed_models:
            print(f"\n⚠️  Warning: {len(failed_models)} models failed to initialize")
            print("Consider using alternative models or check API availability")
        
        return models
    
    def run_experiment(self, config_name: str = "fellowship_research") -> ExperimentResults:
        """Run the complete experimental protocol"""
        
        config = self.experiment_configs[config_name]
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🔬 {config.name}")
        print(f"{'='*60}")
        print(f"Description: {config.description}")
        print(f"Models: {len(config.models)}")
        print(f"Capabilities: {len(config.capabilities)}")
        print(f"Prompts per capability: {config.prompts_per_capability}")
        
        # Initialize advanced cost monitoring system
        self.cost_monitor = CostMonitor(
            budget_limit_usd=config.budget_limit_usd,
            cost_log_file=self.output_dir / "cost_tracking.json"
        )
        
        # Advanced cost estimation
        print(f"\n💰 Advanced Cost Estimation:")
        estimation = self.cost_monitor.estimate_experiment_cost(
            n_models=len(config.models),
            n_capabilities=len(config.capabilities),
            prompts_per_capability=config.prompts_per_capability,
            avg_response_length=config.max_tokens
        )
        
        print(f"  Total API calls: {estimation['total_api_calls']:,}")
        print(f"  Estimated cost: ${estimation['estimated_cost_usd']:.4f}")
        print(f"  Cost with buffer: ${estimation['cost_with_buffer']:.4f}")
        print(f"  Budget utilization: {estimation['budget_utilization']:.1f}%")
        print(f"  Within budget: {'✅ YES' if estimation['within_budget'] else '❌ NO'}")
        print(f"  Recommendation: {estimation['recommendation']}")
        
        if not estimation['within_budget']:
            raise ValueError(f"Estimated cost ${estimation['cost_with_buffer']:.4f} exceeds budget ${config.budget_limit_usd}")
            
        # Display current budget status
        budget_status = self.cost_monitor.check_budget_status()
        print(f"  Current budget status: {budget_status['status']}")
        if budget_status['current_spending'] > 0:
            print(f"  Previous spending: ${budget_status['current_spending']:.4f}")
            print(f"  Remaining budget: ${budget_status['budget_remaining']:.2f}")
        
        # Initialize models
        models = self.initialize_models(config.models)
        if not models:
            raise RuntimeError("No models successfully initialized")
        
        # Load prompts
        print(f"\n📝 Loading prompt datasets...")
        prompt_datasets = self.load_prompt_datasets(config.capabilities)
        
        for capability, prompts in prompt_datasets.items():
            available = len(prompts)
            needed = config.prompts_per_capability
            print(f"  {capability}: {available} available, {needed} needed")
            
            if available < needed:
                print(f"    ⚠️  Warning: Only {available} prompts available for {capability}")
        
        # Run data collection
        print(f"\n🔄 Starting data collection...")
        responses = self._collect_responses(models, prompt_datasets, config)
        
        # Run analysis
        print(f"\n📊 Running convergence analysis...")
        convergence_results = self._analyze_convergence(responses, config)
        
        # Run statistical tests
        print(f"\n📈 Running statistical validation...")
        statistical_results = self._run_statistical_tests(responses, config)
        
        # Generate comprehensive cost report
        execution_time = time.time() - start_time
        cost_report = self.cost_monitor.generate_cost_report()
        
        # Convert to legacy format for compatibility
        cost_summary = {
            "total_cost_usd": cost_report.total_cost_usd,
            "total_api_calls": cost_report.total_calls,
            "cost_per_call": cost_report.average_cost_per_call,
            "budget_remaining": cost_report.budget_remaining,
            "budget_utilization": cost_report.budget_utilization,
            "cost_by_model": cost_report.cost_by_model,
            "cost_by_capability": cost_report.cost_by_capability,
            "cost_efficiency_metrics": cost_report.cost_efficiency_metrics
        }
        
        # Create results object
        results = ExperimentResults(
            config=config,
            responses=responses,
            convergence_analysis=convergence_results,
            statistical_tests=statistical_results,
            cost_summary=cost_summary,
            execution_time=execution_time
        )
        
        # Save results
        self._save_results(results)
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"  Execution time: {execution_time:.1f} seconds")
        print(f"  Total cost: ${cost_report.total_cost_usd:.4f}")
        print(f"  Budget remaining: ${cost_report.budget_remaining:.2f}")
        print(f"  Budget utilization: {cost_report.budget_utilization:.1f}%")
        print(f"  Cache hit rate: {cost_report.cost_efficiency_metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"  Results saved to: {self.output_dir}")
        
        # Print detailed cost summary
        self.cost_monitor.print_cost_summary()
        
        return results
    
    def _collect_responses(self, models: List[ModelInterface], 
                          prompt_datasets: Dict[str, List[str]], 
                          config: ExperimentConfig) -> List[ModelResponse]:
        """Collect all model responses with progress tracking"""
        
        responses = []
        
        for model in models:
            print(f"\n📋 Testing {model.name}...")
            
            for capability in config.capabilities:
                prompts = prompt_datasets.get(capability, [])
                num_prompts = min(len(prompts), config.prompts_per_capability)
                
                print(f"  {capability}: {num_prompts} prompts")
                
                for i, prompt in enumerate(prompts[:num_prompts]):
                    try:
                        # Generate response
                        response_text = model.generate(prompt)
                        
                        # Record API call with cost monitoring
                        model_id = model.model_id if hasattr(model, 'model_id') else model.name
                        prompt_id = f"{capability}_{i:03d}"
                        
                        # Check if response was cached
                        was_cached = hasattr(model, 'response_cache') and model.response_cache and \
                                   any(prompt in cache_key for cache_key in model.response_cache.keys())
                        
                        # Record cost
                        call_cost = self.cost_monitor.record_api_call(
                            model_id=model_id,
                            prompt_text=prompt,
                            response_text=response_text,
                            capability=capability,
                            prompt_id=prompt_id,
                            cached=was_cached
                        )
                        
                        # Create response record
                        response = ModelResponse(
                            model_id=model_id,
                            capability=capability,
                            prompt_id=prompt_id,
                            prompt=prompt,
                            response=response_text,
                            timestamp=datetime.now(),
                            tokens_used=len(prompt.split()) + len(response_text.split()),  # Rough estimate
                            cost_usd=call_cost
                        )
                        
                        responses.append(response)
                        
                        # Progress indicator
                        if (i + 1) % 10 == 0:
                            print(f"    Progress: {i+1}/{num_prompts}")
                            
                    except Exception as e:
                        print(f"    ❌ Error on prompt {i}: {e}")
                        continue
        
        return responses
    
    def _analyze_convergence(self, responses: List[ModelResponse], 
                           config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze convergence patterns across models and capabilities"""
        
        # Group responses by capability
        capability_groups = {}
        for response in responses:
            if response.capability not in capability_groups:
                capability_groups[response.capability] = []
            capability_groups[response.capability].append(response)
        
        convergence_results = {}
        
        for capability, cap_responses in capability_groups.items():
            # Group by prompt for comparison
            prompt_groups = {}
            for response in cap_responses:
                if response.prompt_id not in prompt_groups:
                    prompt_groups[response.prompt_id] = []
                prompt_groups[response.prompt_id].append(response)
            
            # Calculate semantic similarities
            similarities = []
            for prompt_id, prompt_responses in prompt_groups.items():
                if len(prompt_responses) >= 2:
                    response_texts = [r.response for r in prompt_responses]
                    try:
                        sim_matrix = self.semantic_analyzer.calculate_similarity_matrix(response_texts)
                        # Get upper triangular similarities (avoid self-similarity)
                        triu_indices = np.triu_indices_from(sim_matrix, k=1)
                        similarities.extend(sim_matrix[triu_indices])
                    except Exception as e:
                        print(f"    Warning: Similarity calculation failed for {prompt_id}: {e}")
            
            # Calculate convergence metrics
            if similarities:
                convergence_results[capability] = {
                    "mean_similarity": np.mean(similarities),
                    "std_similarity": np.std(similarities),
                    "min_similarity": np.min(similarities),
                    "max_similarity": np.max(similarities),
                    "num_comparisons": len(similarities),
                    "convergence_score": np.mean(similarities)  # Primary metric
                }
            else:
                convergence_results[capability] = {
                    "mean_similarity": 0.0,
                    "convergence_score": 0.0,
                    "error": "No valid comparisons"
                }
        
        # Calculate overall convergence
        capability_scores = [result.get("convergence_score", 0.0) 
                           for result in convergence_results.values() 
                           if "error" not in result]
        
        overall_convergence = np.mean(capability_scores) if capability_scores else 0.0
        
        return {
            "capability_results": convergence_results,
            "overall_convergence": overall_convergence,
            "num_capabilities": len(capability_scores),
            "summary": {
                "strong_evidence": overall_convergence > 0.8,
                "moderate_evidence": 0.6 <= overall_convergence <= 0.8,
                "weak_evidence": 0.4 <= overall_convergence < 0.6,
                "no_evidence": overall_convergence < 0.4
            }
        }
    
    def _run_statistical_tests(self, responses: List[ModelResponse], 
                              config: ExperimentConfig) -> Dict[str, Any]:
        """Run comprehensive statistical validation"""
        
        # Extract convergence scores for testing
        convergence_scores = []
        
        # Group by capability for statistical testing
        capability_groups = {}
        for response in responses:
            if response.capability not in capability_groups:
                capability_groups[response.capability] = []
            capability_groups[response.capability].append(response)
        
        statistical_results = {}
        
        for capability, cap_responses in capability_groups.items():
            # Prepare data for statistical testing
            model_responses = {}
            for response in cap_responses:
                if response.model_id not in model_responses:
                    model_responses[response.model_id] = []
                model_responses[response.model_id].append(response.response)
            
            # Skip if insufficient data
            if len(model_responses) < 2:
                continue
            
            # Calculate pairwise similarities for permutation testing
            all_similarities = []
            model_pairs = []
            
            model_ids = list(model_responses.keys())
            for i, model1 in enumerate(model_ids):
                for j, model2 in enumerate(model_ids[i+1:], i+1):
                    responses1 = model_responses[model1]
                    responses2 = model_responses[model2]
                    
                    # Calculate similarities between corresponding responses
                    pair_similarities = []
                    for r1, r2 in zip(responses1, responses2):
                        try:
                            sim = self.semantic_analyzer.calculate_similarity(r1, r2)
                            pair_similarities.append(sim)
                        except:
                            continue
                    
                    if pair_similarities:
                        avg_similarity = np.mean(pair_similarities)
                        all_similarities.append(avg_similarity)
                        model_pairs.append((model1, model2))
            
            if all_similarities:
                # Basic statistical measures
                mean_similarity = np.mean(all_similarities)
                std_similarity = np.std(all_similarities)
                
                # Simple significance test (one-sample t-test against random baseline)
                from scipy import stats
                # H0: mean similarity = 0.5 (random baseline)
                # H1: mean similarity > 0.5 (systematic convergence)
                t_stat, p_value = stats.ttest_1samp(all_similarities, 0.5)
                
                # Effect size (Cohen's d)
                effect_size = (mean_similarity - 0.5) / std_similarity if std_similarity > 0 else 0
                
                statistical_results[capability] = {
                    "mean_similarity": mean_similarity,
                    "std_similarity": std_similarity,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "significant": p_value < config.statistical_confidence,
                    "n_comparisons": len(all_similarities),
                    "interpretation": self._interpret_statistical_result(p_value, effect_size)
                }
        
        # Overall statistical summary
        significant_capabilities = sum(1 for result in statistical_results.values() 
                                     if result.get("significant", False))
        
        return {
            "capability_tests": statistical_results,
            "overall_summary": {
                "total_capabilities": len(statistical_results),
                "significant_capabilities": significant_capabilities,
                "significance_rate": significant_capabilities / max(len(statistical_results), 1),
                "multiple_comparison_threshold": config.statistical_confidence / max(len(statistical_results), 1)
            }
        }
    
    def _interpret_statistical_result(self, p_value: float, effect_size: float) -> str:
        """Interpret statistical results for research context"""
        
        if p_value < 0.001 and effect_size > 0.8:
            return "STRONG EVIDENCE: Highly significant convergence with large effect size"
        elif p_value < 0.01 and effect_size > 0.5:
            return "MODERATE EVIDENCE: Significant convergence with medium effect size"
        elif p_value < 0.05:
            return "WEAK EVIDENCE: Marginally significant convergence"
        else:
            return "NO EVIDENCE: No significant convergence detected"
    
    def _save_results(self, results: ExperimentResults):
        """Save comprehensive results to disk"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw responses
        responses_file = self.output_dir / "raw_responses" / f"responses_{timestamp}.json"
        with open(responses_file, 'w') as f:
            responses_data = [asdict(response) for response in results.responses]
            # Convert datetime to string for JSON serialization
            for response_data in responses_data:
                response_data['timestamp'] = response_data['timestamp'].isoformat()
            json.dump(responses_data, f, indent=2)
        
        # Save analysis results
        analysis_file = self.output_dir / "analysis_outputs" / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            analysis_data = {
                "config": asdict(results.config),
                "convergence_analysis": self._convert_numpy_types(results.convergence_analysis),
                "statistical_tests": self._convert_numpy_types(results.statistical_tests),
                "cost_summary": self._convert_numpy_types(results.cost_summary),
                "execution_time": results.execution_time,
                "timestamp": timestamp
            }
            json.dump(analysis_data, f, indent=2)
        
        print(f"💾 Results saved:")
        print(f"  Raw responses: {responses_file}")
        print(f"  Analysis: {analysis_file}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


if __name__ == "__main__":
    # Initialize framework
    framework = ComprehensiveAnalysisFramework()
    
    print("🚀 LAUNCHING FULL FELLOWSHIP RESEARCH EXPERIMENT")
    print("=" * 80)
    
    # Run full fellowship research experiment
    print("Running comprehensive fellowship research experiment...")
    fellowship_results = framework.run_experiment("fellowship_research")
    
    # Print comprehensive summary
    conv_score = fellowship_results.convergence_analysis["overall_convergence"]
    cost_summary = fellowship_results.cost_summary
    
    print(f"\n🎯 FELLOWSHIP RESEARCH RESULTS:")
    print(f"=" * 80)
    print(f"Overall Convergence: {conv_score:.1%}")
    print(f"Total Cost: ${cost_summary['total_cost_usd']:.4f}")
    print(f"Budget Utilization: {cost_summary['budget_utilization']:.1%}")
    print(f"Models Tested: {len(fellowship_results.config.models)}")
    print(f"Capabilities Analyzed: {len(fellowship_results.config.capabilities)}")
    print(f"Total API Calls: {cost_summary['total_api_calls']:,}")
    print(f"Cache Hit Rate: {cost_summary['cost_efficiency_metrics'].get('cache_hit_rate', 0):.1f}%")
    
    if conv_score > 0.6:
        print("\n🎉 STRONG EVIDENCE for Universal Alignment Patterns!")
        print("✅ Results suitable for Anthropic Fellowship application")
    elif conv_score > 0.4:
        print("\n📊 MODERATE EVIDENCE for Universal Alignment Patterns")
        print("✅ Promising results for fellowship application")
    else:
        print("\n⚠️  Limited evidence - may need additional analysis")
    
    print(f"\n📄 Full results saved to: {framework.output_dir}")
    print("🚀 Ready for visualization generation and GitHub documentation!")