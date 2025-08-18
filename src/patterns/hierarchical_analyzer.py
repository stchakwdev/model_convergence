"""
Hierarchical Testing Infrastructure for Universal Alignment Pattern Analysis

This module implements a 3-level testing protocol that progressively applies more
sophisticated (and expensive) analysis techniques to identify models showing
strong convergence patterns.

Testing Levels:
- Level 1: Behavioral Screening (cheap, all models) - Basic similarity metrics
- Level 2: Computational Analysis (medium cost, selected models) - Advanced metrics  
- Level 3: Mechanistic Probing (expensive, top models) - Deep analysis

Authors: Samuel Chakwera
Date: 2025-08-18
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
import time
import logging
from pathlib import Path
import json

# Import our analysis modules
from .semantic_analyzer import EnhancedSemanticAnalyzer
from .kl_enhanced_analyzer import HybridConvergenceAnalyzer
from .advanced_metrics import AdvancedConvergenceAnalyzer, AdvancedConvergenceResult


@dataclass
class ModelPerformance:
    """Tracks model performance across hierarchy levels"""
    model_id: str
    level_1_score: Optional[float] = None
    level_2_score: Optional[float] = None
    level_3_score: Optional[float] = None
    level_1_details: Optional[Dict[str, Any]] = None
    level_2_details: Optional[Dict[str, Any]] = None
    level_3_details: Optional[Dict[str, Any]] = None
    cost_usd: float = 0.0
    api_calls: int = 0
    total_tokens: int = 0
    processing_time: float = 0.0
    failed_tests: List[str] = field(default_factory=list)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical testing"""
    # Level thresholds for advancement
    level_1_threshold: float = 0.3  # Behavioral screening cutoff
    level_2_threshold: float = 0.5  # Computational analysis cutoff
    
    # Sample sizes per level
    level_1_samples: int = 30  # Quick screening
    level_2_samples: int = 75  # Medium analysis
    level_3_samples: int = 150  # Deep analysis
    
    # Budget constraints
    max_budget_usd: float = 50.0
    level_1_budget_fraction: float = 0.2  # 20% for screening
    level_2_budget_fraction: float = 0.3  # 30% for computational
    level_3_budget_fraction: float = 0.5  # 50% for mechanistic
    
    # Performance criteria
    min_models_for_level_2: int = 10  # Minimum models advancing to level 2
    min_models_for_level_3: int = 5   # Minimum models advancing to level 3
    max_models_per_level: Dict[int, int] = field(default_factory=lambda: {1: 50, 2: 20, 3: 10})
    
    # Failure handling
    max_retries: int = 3
    timeout_seconds: int = 300  # 5 minutes per model test


class Level1BehavioralScreener:
    """
    Level 1: Fast behavioral screening using semantic similarity
    Goal: Quickly identify models with basic convergence patterns
    """
    
    def __init__(self):
        self.semantic_analyzer = EnhancedSemanticAnalyzer()
    
    def screen_models(self, 
                     model_responses: Dict[str, List[str]], 
                     capability: str,
                     samples: int = 30) -> Dict[str, ModelPerformance]:
        """
        Screen models using basic semantic similarity.
        
        Args:
            model_responses: {model_id: [responses]}
            capability: Capability being tested
            samples: Number of samples to use
            
        Returns:
            {model_id: ModelPerformance} with level_1_score filled
        """
        results = {}
        model_ids = list(model_responses.keys())
        
        # Limit samples for efficiency
        limited_responses = {
            model_id: responses[:samples] 
            for model_id, responses in model_responses.items()
        }
        
        print(f"🔍 Level 1 Screening: {len(model_ids)} models for {capability}")
        
        # Pairwise comparisons
        convergence_scores = {}
        for i, model_a in enumerate(model_ids):
            convergence_scores[model_a] = []
            
            for j, model_b in enumerate(model_ids):
                if i != j:
                    try:
                        start_time = time.time()
                        
                        # Quick semantic similarity
                        similarities = []
                        responses_a = limited_responses[model_a]
                        responses_b = limited_responses[model_b]
                        
                        for resp_a, resp_b in zip(responses_a, responses_b):
                            sim = self.semantic_analyzer.calculate_similarity(resp_a, resp_b)
                            similarities.append(sim)
                        
                        avg_similarity = np.mean(similarities)
                        convergence_scores[model_a].append(avg_similarity)
                        
                        processing_time = time.time() - start_time
                        
                    except Exception as e:
                        logging.warning(f"Level 1 failed for {model_a} vs {model_b}: {e}")
                        convergence_scores[model_a].append(0.0)
        
        # Compute average convergence for each model
        for model_id in model_ids:
            scores = convergence_scores.get(model_id, [0.0])
            avg_score = np.mean(scores) if scores else 0.0
            
            results[model_id] = ModelPerformance(
                model_id=model_id,
                level_1_score=avg_score,
                level_1_details={
                    'pairwise_scores': scores,
                    'capability': capability,
                    'n_comparisons': len(scores),
                    'method': 'semantic_similarity'
                },
                api_calls=samples,  # Approximate
                total_tokens=samples * 100,  # Rough estimate
                cost_usd=samples * 0.001,  # Rough estimate
            )
        
        return results


class Level2ComputationalAnalyzer:
    """
    Level 2: Computational analysis using hybrid semantic + distributional metrics
    Goal: Identify models with robust computational convergence patterns
    """
    
    def __init__(self):
        self.hybrid_analyzer = HybridConvergenceAnalyzer()
    
    def analyze_models(self, 
                      model_responses: Dict[str, List[str]], 
                      capability: str,
                      samples: int = 75) -> Dict[str, ModelPerformance]:
        """
        Analyze models using hybrid semantic + distributional convergence.
        
        Args:
            model_responses: {model_id: [responses]}
            capability: Capability being tested
            samples: Number of samples to use
            
        Returns:
            {model_id: ModelPerformance} with level_2_score filled
        """
        results = {}
        model_ids = list(model_responses.keys())
        
        # Limit samples
        limited_responses = {
            model_id: responses[:samples] 
            for model_id, responses in model_responses.items()
        }
        
        print(f"🧮 Level 2 Analysis: {len(model_ids)} models for {capability}")
        
        # Pairwise hybrid analysis
        convergence_matrix = {}
        for i, model_a in enumerate(model_ids):
            convergence_matrix[model_a] = {}
            
            for j, model_b in enumerate(model_ids):
                if i != j:
                    try:
                        start_time = time.time()
                        
                        responses_a = limited_responses[model_a]
                        responses_b = limited_responses[model_b]
                        
                        # Hybrid analysis
                        result = self.hybrid_analyzer.analyze_convergence(
                            {model_a: responses_a, model_b: responses_b},
                            capability
                        )
                        
                        # Extract hybrid convergence score
                        hybrid_score = result.get('hybrid_convergence', 0.0)
                        convergence_matrix[model_a][model_b] = {
                            'hybrid_score': hybrid_score,
                            'semantic_score': result.get('semantic_convergence', 0.0),
                            'distributional_score': result.get('distributional_convergence', 0.0),
                            'processing_time': time.time() - start_time
                        }
                        
                    except Exception as e:
                        logging.warning(f"Level 2 failed for {model_a} vs {model_b}: {e}")
                        convergence_matrix[model_a][model_b] = {
                            'hybrid_score': 0.0,
                            'semantic_score': 0.0,
                            'distributional_score': 0.0,
                            'processing_time': 0.0
                        }
        
        # Compute average convergence for each model
        for model_id in model_ids:
            comparisons = convergence_matrix.get(model_id, {})
            hybrid_scores = [comp['hybrid_score'] for comp in comparisons.values()]
            
            avg_hybrid = np.mean(hybrid_scores) if hybrid_scores else 0.0
            
            results[model_id] = ModelPerformance(
                model_id=model_id,
                level_2_score=avg_hybrid,
                level_2_details={
                    'pairwise_comparisons': comparisons,
                    'capability': capability,
                    'n_comparisons': len(hybrid_scores),
                    'method': 'hybrid_semantic_distributional',
                    'avg_semantic': np.mean([comp['semantic_score'] for comp in comparisons.values()]),
                    'avg_distributional': np.mean([comp['distributional_score'] for comp in comparisons.values()])
                },
                api_calls=samples,
                total_tokens=samples * 150,  # Larger estimate
                cost_usd=samples * 0.0015,  # Higher cost estimate
            )
        
        return results


class Level3MechanisticProber:
    """
    Level 3: Deep mechanistic probing using advanced mathematical techniques
    Goal: Identify fundamental convergence patterns using sophisticated metrics
    """
    
    def __init__(self):
        self.advanced_analyzer = AdvancedConvergenceAnalyzer()
    
    def probe_models(self, 
                    model_responses: Dict[str, List[str]], 
                    capability: str,
                    samples: int = 150) -> Dict[str, ModelPerformance]:
        """
        Deep probe models using advanced convergence metrics.
        
        Args:
            model_responses: {model_id: [responses]}
            capability: Capability being tested
            samples: Number of samples to use
            
        Returns:
            {model_id: ModelPerformance} with level_3_score filled
        """
        results = {}
        model_ids = list(model_responses.keys())
        
        # Use full sample for deep analysis
        limited_responses = {
            model_id: responses[:samples] 
            for model_id, responses in model_responses.items()
        }
        
        print(f"🔬 Level 3 Probing: {len(model_ids)} models for {capability}")
        
        # Advanced pairwise analysis
        analysis_results = {}
        for i, model_a in enumerate(model_ids):
            analysis_results[model_a] = {}
            
            for j, model_b in enumerate(model_ids):
                if i != j:
                    try:
                        start_time = time.time()
                        
                        responses_a = limited_responses[model_a]
                        responses_b = limited_responses[model_b]
                        
                        # Advanced convergence analysis
                        result = self.advanced_analyzer.analyze_convergence(
                            responses_a, responses_b, model_a, model_b
                        )
                        
                        analysis_results[model_a][model_b] = {
                            'result': result,
                            'processing_time': time.time() - start_time
                        }
                        
                    except Exception as e:
                        logging.warning(f"Level 3 failed for {model_a} vs {model_b}: {e}")
                        # Create empty result
                        empty_result = AdvancedConvergenceResult(
                            mutual_information=0.0,
                            wasserstein_distance=float('inf'),
                            canonical_correlation=0.0,
                            topological_similarity=0.0,
                            combined_score=0.0,
                            confidence_interval=(0.0, 0.0),
                            n_samples=0,
                            statistical_significance={'significant': False, 'p_value': 1.0}
                        )
                        analysis_results[model_a][model_b] = {
                            'result': empty_result,
                            'processing_time': 0.0
                        }
        
        # Compute comprehensive scores for each model
        for model_id in model_ids:
            comparisons = analysis_results.get(model_id, {})
            
            # Extract scores from all comparisons
            combined_scores = []
            mutual_infos = []
            canonical_corrs = []
            topo_sims = []
            significance_flags = []
            
            for comp_data in comparisons.values():
                result = comp_data['result']
                combined_scores.append(result.combined_score)
                mutual_infos.append(result.mutual_information)
                canonical_corrs.append(result.canonical_correlation)
                topo_sims.append(result.topological_similarity)
                significance_flags.append(result.statistical_significance.get('significant', False))
            
            # Aggregate metrics
            avg_combined = np.mean(combined_scores) if combined_scores else 0.0
            significance_rate = np.mean(significance_flags) if significance_flags else 0.0
            
            results[model_id] = ModelPerformance(
                model_id=model_id,
                level_3_score=avg_combined,
                level_3_details={
                    'pairwise_analyses': comparisons,
                    'capability': capability,
                    'n_comparisons': len(combined_scores),
                    'method': 'advanced_multi_metric',
                    'avg_mutual_information': np.mean(mutual_infos) if mutual_infos else 0.0,
                    'avg_canonical_correlation': np.mean(canonical_corrs) if canonical_corrs else 0.0,
                    'avg_topological_similarity': np.mean(topo_sims) if topo_sims else 0.0,
                    'significance_rate': significance_rate,
                    'combined_score_std': np.std(combined_scores) if combined_scores else 0.0
                },
                api_calls=samples * len(model_ids),  # Account for pairwise comparisons
                total_tokens=samples * len(model_ids) * 200,  # High estimate
                cost_usd=samples * len(model_ids) * 0.002,  # Premium cost
            )
        
        return results


class HierarchicalConvergenceAnalyzer:
    """
    Main orchestrator for hierarchical testing of universal alignment patterns.
    Manages progression through 3 levels of increasingly sophisticated analysis.
    """
    
    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """
        Args:
            config: Configuration for hierarchical testing
        """
        self.config = config or HierarchicalConfig()
        self.level_1_screener = Level1BehavioralScreener()
        self.level_2_analyzer = Level2ComputationalAnalyzer()
        self.level_3_prober = Level3MechanisticProber()
        
        # Tracking
        self.total_cost = 0.0
        self.total_time = 0.0
        self.results_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def analyze_capability(self, 
                          model_responses: Dict[str, List[str]], 
                          capability: str,
                          reference_model: Optional[str] = None) -> Dict[str, ModelPerformance]:
        """
        Run hierarchical analysis for a specific capability.
        
        Args:
            model_responses: {model_id: [responses]}
            capability: Name of capability being tested
            reference_model: Optional reference model for comparisons
            
        Returns:
            {model_id: ModelPerformance} with all levels completed
        """
        start_time = time.time()
        
        print(f"\n🚀 Starting Hierarchical Analysis for '{capability}'")
        print(f"   Models: {len(model_responses)}")
        print(f"   Config: L1≥{self.config.level_1_threshold}, L2≥{self.config.level_2_threshold}")
        print(f"   Budget: ${self.config.max_budget_usd:.2f}")
        
        # LEVEL 1: Behavioral Screening
        print(f"\n📊 LEVEL 1: Behavioral Screening")
        level_1_results = self.level_1_screener.screen_models(
            model_responses, capability, self.config.level_1_samples
        )
        
        # Select models for Level 2
        level_1_scores = [(model_id, perf.level_1_score) for model_id, perf in level_1_results.items()]
        level_1_scores.sort(key=lambda x: x[1], reverse=True)
        
        level_2_candidates = [
            model_id for model_id, score in level_1_scores 
            if score >= self.config.level_1_threshold
        ][:self.config.max_models_per_level[2]]
        
        print(f"   ✅ Level 1 Complete: {len(level_1_results)} models tested")
        print(f"   📈 Top 5 scores: {[(m, f'{s:.3f}') for m, s in level_1_scores[:5]]}")
        print(f"   ⬆️  Advancing to Level 2: {len(level_2_candidates)} models")
        
        if len(level_2_candidates) < self.config.min_models_for_level_2:
            # Lower threshold if too few models advance
            print(f"   ⚠️  Too few models advancing, lowering threshold...")
            level_2_candidates = [model_id for model_id, _ in level_1_scores[:self.config.min_models_for_level_2]]
        
        # LEVEL 2: Computational Analysis
        level_2_results = {}
        if level_2_candidates:
            print(f"\n🧮 LEVEL 2: Computational Analysis")
            level_2_responses = {
                model_id: model_responses[model_id] 
                for model_id in level_2_candidates
            }
            
            level_2_results = self.level_2_analyzer.analyze_models(
                level_2_responses, capability, self.config.level_2_samples
            )
            
            # Select models for Level 3
            level_2_scores = [(model_id, perf.level_2_score) for model_id, perf in level_2_results.items()]
            level_2_scores.sort(key=lambda x: x[1], reverse=True)
            
            level_3_candidates = [
                model_id for model_id, score in level_2_scores 
                if score >= self.config.level_2_threshold
            ][:self.config.max_models_per_level[3]]
            
            print(f"   ✅ Level 2 Complete: {len(level_2_results)} models tested")
            print(f"   📈 Top 5 scores: {[(m, f'{s:.3f}') for m, s in level_2_scores[:5]]}")
            print(f"   ⬆️  Advancing to Level 3: {len(level_3_candidates)} models")
            
            if len(level_3_candidates) < self.config.min_models_for_level_3:
                print(f"   ⚠️  Few models advancing, selecting top performers...")
                level_3_candidates = [model_id for model_id, _ in level_2_scores[:self.config.min_models_for_level_3]]
        
        # LEVEL 3: Mechanistic Probing
        level_3_results = {}
        if level_2_candidates and len(level_3_candidates) > 0:
            print(f"\n🔬 LEVEL 3: Mechanistic Probing")
            level_3_responses = {
                model_id: model_responses[model_id] 
                for model_id in level_3_candidates
            }
            
            level_3_results = self.level_3_prober.probe_models(
                level_3_responses, capability, self.config.level_3_samples
            )
            
            level_3_scores = [(model_id, perf.level_3_score) for model_id, perf in level_3_results.items()]
            level_3_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   ✅ Level 3 Complete: {len(level_3_results)} models tested")
            print(f"   📈 Final rankings: {[(m, f'{s:.3f}') for m, s in level_3_scores]}")
        
        # Combine all results
        final_results = {}
        for model_id in model_responses.keys():
            performance = level_1_results.get(model_id, ModelPerformance(model_id=model_id))
            
            # Update with Level 2 results if available
            if model_id in level_2_results:
                level_2_perf = level_2_results[model_id]
                performance.level_2_score = level_2_perf.level_2_score
                performance.level_2_details = level_2_perf.level_2_details
                performance.api_calls += level_2_perf.api_calls
                performance.total_tokens += level_2_perf.total_tokens
                performance.cost_usd += level_2_perf.cost_usd
            
            # Update with Level 3 results if available
            if model_id in level_3_results:
                level_3_perf = level_3_results[model_id]
                performance.level_3_score = level_3_perf.level_3_score
                performance.level_3_details = level_3_perf.level_3_details
                performance.api_calls += level_3_perf.api_calls
                performance.total_tokens += level_3_perf.total_tokens
                performance.cost_usd += level_3_perf.cost_usd
            
            final_results[model_id] = performance
        
        # Summary
        total_time = time.time() - start_time
        total_cost = sum(perf.cost_usd for perf in final_results.values())
        
        print(f"\n🎯 HIERARCHICAL ANALYSIS COMPLETE")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Total Cost: ${total_cost:.4f}")
        print(f"   Models Analyzed: L1={len(level_1_results)}, L2={len(level_2_results)}, L3={len(level_3_results)}")
        
        # Store for later analysis
        self.results_history.append({
            'capability': capability,
            'timestamp': time.time(),
            'results': final_results,
            'total_cost': total_cost,
            'total_time': total_time
        })
        
        return final_results
    
    def get_top_converging_models(self, 
                                 results: Dict[str, ModelPerformance], 
                                 level: int = 3, 
                                 top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top converging models from hierarchical analysis results.
        
        Args:
            results: Results from analyze_capability
            level: Which level to rank by (1, 2, or 3)
            top_k: Number of top models to return
            
        Returns:
            List of (model_id, score) tuples sorted by convergence score
        """
        score_attr = f'level_{level}_score'
        
        model_scores = []
        for model_id, performance in results.items():
            score = getattr(performance, score_attr)
            if score is not None:
                model_scores.append((model_id, score))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return model_scores[:top_k]
    
    def save_results(self, filepath: str):
        """Save analysis results to JSON file"""
        # Convert ModelPerformance objects to dicts for JSON serialization
        serializable_history = []
        
        for entry in self.results_history:
            serializable_entry = entry.copy()
            serializable_entry['results'] = {
                model_id: {
                    'model_id': perf.model_id,
                    'level_1_score': perf.level_1_score,
                    'level_2_score': perf.level_2_score,
                    'level_3_score': perf.level_3_score,
                    'cost_usd': perf.cost_usd,
                    'api_calls': perf.api_calls,
                    'total_tokens': perf.total_tokens,
                    'failed_tests': perf.failed_tests,
                    # Skip details for file size
                }
                for model_id, perf in entry['results'].items()
            }
            serializable_history.append(serializable_entry)
        
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'results_history': serializable_history,
                'total_experiments': len(self.results_history)
            }, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("🏗️ Hierarchical Testing Infrastructure")
    print("Ready for progressive convergence analysis!")
    
    # This would be used with real model responses:
    # analyzer = HierarchicalConvergenceAnalyzer()
    # results = analyzer.analyze_capability(model_responses, "truthfulness")
    # top_models = analyzer.get_top_converging_models(results, level=3)