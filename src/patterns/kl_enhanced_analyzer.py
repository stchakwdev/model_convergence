"""
Enhanced Universal Convergence Analysis with KL Divergence

This module combines semantic similarity analysis with information-theoretic
KL divergence measurement for the most rigorous convergence analysis possible.

Key Innovation: Measures both semantic convergence (what models say) and 
distributional convergence (how they say it) for comprehensive evidence 
of universal alignment patterns.

Author: Samuel Chakwera
Purpose: Enhanced convergence measurement for Anthropic Fellowship research
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

@dataclass
class HybridConvergenceResults:
    """Results combining semantic and distributional convergence analysis"""
    # Semantic similarity results
    semantic_similarities: Dict[str, float]
    semantic_convergence_score: float
    
    # KL divergence results
    kl_divergences: Dict[str, float]
    jensen_shannon_distances: Dict[str, float]
    distributional_convergence_score: float
    
    # Combined analysis
    hybrid_convergence_score: float
    statistical_significance: Dict[str, Any]
    interpretation: str
    confidence_level: float


class EnhancedDistributionExtractor:
    """
    Extract probability distributions from model responses for KL analysis.
    Handles the complexity of estimating distributions from API models.
    """
    
    def __init__(self, common_vocab_size: int = 1000):
        self.common_vocab_size = common_vocab_size
        self.vocabulary = None
        self.token_to_id = {}
        
    def extract_distributions_from_responses(self, 
                                           grouped_responses: Dict[str, List[str]],
                                           temperature_samples: int = 3) -> Dict[str, torch.Tensor]:
        """
        Extract probability distributions from collected model responses.
        
        Args:
            grouped_responses: {model_name: [response1, response2, ...]}
            temperature_samples: Number of variations to estimate per response
            
        Returns:
            Dict mapping model names to distribution tensors
        """
        
        print("🔬 Extracting probability distributions for KL analysis...")
        
        # Build unified vocabulary from all responses
        all_tokens = []
        for model_responses in grouped_responses.values():
            for response in model_responses:
                tokens = self._tokenize_response(response)
                all_tokens.extend(tokens)
        
        # Create vocabulary from most common tokens
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.common_vocab_size)
        self.vocabulary = [token for token, _ in most_common]
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}
        
        print(f"  📚 Built vocabulary of {len(self.vocabulary)} tokens")
        
        # Extract distributions for each model
        model_distributions = {}
        
        for model_name, responses in grouped_responses.items():
            print(f"  🤖 Processing {model_name}: {len(responses)} responses")
            
            distributions = []
            for response in responses:
                # Estimate distribution from this response
                dist = self._estimate_distribution_from_response(response)
                distributions.append(dist)
            
            # Stack into tensor: (n_responses, vocab_size)
            model_distributions[model_name] = torch.stack(distributions)
        
        return model_distributions
    
    def _tokenize_response(self, response: str) -> List[str]:
        """Simple but effective tokenization for distribution estimation"""
        # Clean and tokenize
        response = response.lower().strip()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', response)
        
        # Filter out very short tokens and add special tokens
        filtered_tokens = []
        
        # Add special start token
        filtered_tokens.append('<start>')
        
        for token in tokens:
            if len(token) >= 1:
                filtered_tokens.append(token)
        
        # Add special end token
        filtered_tokens.append('<end>')
        
        return filtered_tokens
    
    def _estimate_distribution_from_response(self, response: str) -> torch.Tensor:
        """
        Estimate probability distribution from a single response.
        Uses n-gram frequency analysis as proxy for model's internal distribution.
        """
        
        tokens = self._tokenize_response(response)
        
        # Count token frequencies
        token_counts = np.zeros(self.common_vocab_size)
        total_tokens = 0
        
        for token in tokens:
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
                token_counts[token_id] += 1
                total_tokens += 1
        
        # Handle case of no recognized tokens
        if total_tokens == 0:
            # Uniform distribution as fallback
            token_counts = np.ones(self.common_vocab_size)
            total_tokens = self.common_vocab_size
        
        # Convert to probability distribution
        probabilities = token_counts / total_tokens
        
        # Add small epsilon to avoid zero probabilities
        epsilon = 1e-8
        probabilities = probabilities + epsilon
        probabilities = probabilities / np.sum(probabilities)
        
        return torch.from_numpy(probabilities).float()


class KLDivergenceAnalyzer:
    """
    Measures model convergence using information-theoretic metrics.
    Enhanced version for the universal alignment patterns experiment.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.convergence_threshold = 0.1  # Low KL = high convergence
        
    def calculate_distributional_convergence(self, 
                                           model_distributions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Calculate convergence using KL divergence between model distributions.
        
        Args:
            model_distributions: {model_name: tensor of shape (n_responses, vocab_size)}
        """
        
        models = list(model_distributions.keys())
        n_models = len(models)
        
        if n_models < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        print(f"🔬 Calculating KL divergence between {n_models} models...")
        
        # Calculate pairwise KL divergences and JS distances
        kl_divergences = {}
        js_distances = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                # Calculate KL divergence (asymmetric)
                kl_div = self._calculate_kl_divergence(
                    model_distributions[model1],
                    model_distributions[model2]
                )
                
                # Calculate Jensen-Shannon distance (symmetric)
                js_dist = self._calculate_jensen_shannon_distance(
                    model_distributions[model1],
                    model_distributions[model2]
                )
                
                pair_key = f"{model1}_vs_{model2}"
                kl_divergences[pair_key] = kl_div
                js_distances[pair_key] = js_dist
                
                print(f"  📊 {model1} vs {model2}: KL={kl_div:.4f}, JS={js_dist:.4f}")
        
        # Calculate overall convergence metrics
        mean_kl = np.mean(list(kl_divergences.values()))
        mean_js = np.mean(list(js_distances.values()))
        
        # Convert to convergence score (lower divergence = higher convergence)
        kl_convergence_score = np.exp(-mean_kl)  # Maps [0,∞] → [0,1]
        js_convergence_score = 1 - mean_js  # JS distance is bounded [0,1]
        
        # Combine KL and JS for robust estimate
        distributional_convergence = 0.6 * kl_convergence_score + 0.4 * js_convergence_score
        
        # Statistical significance testing
        significance_results = self._test_distributional_significance(
            model_distributions, mean_kl, mean_js
        )
        
        print(f"  🎯 Distributional convergence: {distributional_convergence:.3f}")
        
        return {
            "kl_divergences": kl_divergences,
            "jensen_shannon_distances": js_distances,
            "mean_kl_divergence": mean_kl,
            "mean_js_distance": mean_js,
            "kl_convergence_score": kl_convergence_score,
            "js_convergence_score": js_convergence_score,
            "distributional_convergence": distributional_convergence,
            "statistical_significance": significance_results,
            "interpretation": self._interpret_distributional_convergence(
                mean_kl, distributional_convergence
            )
        }
    
    def _calculate_kl_divergence(self, P: torch.Tensor, Q: torch.Tensor) -> float:
        """Calculate KL divergence D(P||Q) between two model distributions"""
        
        # Average distributions across responses
        P_avg = torch.mean(P, dim=0)
        Q_avg = torch.mean(Q, dim=0)
        
        # Ensure valid probability distributions
        P_avg = F.normalize(P_avg, p=1, dim=0)
        Q_avg = F.normalize(Q_avg, p=1, dim=0)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-8
        P_avg = P_avg + epsilon
        Q_avg = Q_avg + epsilon
        
        # Re-normalize after epsilon addition
        P_avg = P_avg / torch.sum(P_avg)
        Q_avg = Q_avg / torch.sum(Q_avg)
        
        # Calculate KL divergence
        kl_div = torch.sum(P_avg * torch.log(P_avg / Q_avg))
        
        return kl_div.item()
    
    def _calculate_jensen_shannon_distance(self, P: torch.Tensor, Q: torch.Tensor) -> float:
        """Calculate Jensen-Shannon distance (symmetric version of KL divergence)"""
        
        P_avg = torch.mean(P, dim=0).numpy()
        Q_avg = torch.mean(Q, dim=0).numpy()
        
        # Normalize to valid probability distributions
        P_avg = P_avg / np.sum(P_avg)
        Q_avg = Q_avg / np.sum(Q_avg)
        
        # Jensen-Shannon distance
        js_distance = jensenshannon(P_avg, Q_avg)
        
        return js_distance
    
    def _test_distributional_significance(self, 
                                        model_distributions: Dict[str, torch.Tensor],
                                        observed_kl: float,
                                        observed_js: float,
                                        n_permutations: int = 1000) -> Dict[str, Any]:
        """Test statistical significance of distributional convergence"""
        
        print(f"  🧪 Running {n_permutations} permutation tests for significance...")
        
        models = list(model_distributions.keys())
        
        # Generate null distribution by permuting
        null_kl_values = []
        null_js_values = []
        
        for _ in range(n_permutations):
            # Create null hypothesis by randomly shuffling distributions
            shuffled_distributions = {}
            
            for model in models:
                original_dist = model_distributions[model]
                n_responses, vocab_size = original_dist.shape
                
                # Randomly permute each response's distribution
                shuffled = torch.zeros_like(original_dist)
                for resp_idx in range(n_responses):
                    perm_idx = torch.randperm(vocab_size)
                    shuffled[resp_idx] = original_dist[resp_idx][perm_idx]
                
                shuffled_distributions[model] = shuffled
            
            # Calculate convergence under null hypothesis
            null_kl_sum = 0
            null_js_sum = 0
            n_pairs = 0
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models[i+1:], i+1):
                    null_kl = self._calculate_kl_divergence(
                        shuffled_distributions[model1],
                        shuffled_distributions[model2]
                    )
                    null_js = self._calculate_jensen_shannon_distance(
                        shuffled_distributions[model1],
                        shuffled_distributions[model2]
                    )
                    
                    null_kl_sum += null_kl
                    null_js_sum += null_js
                    n_pairs += 1
            
            null_kl_values.append(null_kl_sum / n_pairs)
            null_js_values.append(null_js_sum / n_pairs)
        
        null_kl_values = np.array(null_kl_values)
        null_js_values = np.array(null_js_values)
        
        # Calculate p-values (lower divergence = better convergence)
        kl_p_value = np.mean(null_kl_values <= observed_kl)
        js_p_value = np.mean(null_js_values >= observed_js)  # Higher JS = worse convergence
        
        # Effect sizes
        kl_effect_size = (np.mean(null_kl_values) - observed_kl) / np.std(null_kl_values)
        js_effect_size = (observed_js - np.mean(null_js_values)) / np.std(null_js_values)
        
        return {
            "kl_p_value": kl_p_value,
            "js_p_value": js_p_value,
            "combined_p_value": min(kl_p_value, js_p_value),  # Conservative estimate
            "kl_effect_size": kl_effect_size,
            "js_effect_size": js_effect_size,
            "null_kl_mean": np.mean(null_kl_values),
            "null_js_mean": np.mean(null_js_values),
            "significant": min(kl_p_value, js_p_value) < 0.05
        }
    
    def _interpret_distributional_convergence(self, mean_kl: float, convergence_score: float) -> str:
        """Interpret distributional convergence results"""
        
        if mean_kl < 0.1 and convergence_score > 0.9:
            return ("VERY STRONG DISTRIBUTIONAL CONVERGENCE: Models show nearly "
                   "identical probability distributions, providing compelling evidence "
                   "for universal alignment patterns at the distributional level.")
        
        elif mean_kl < 0.3 and convergence_score > 0.7:
            return ("STRONG DISTRIBUTIONAL CONVERGENCE: Models show similar "
                   "distributional patterns, consistent with universal features "
                   "emerging across architectures.")
        
        elif mean_kl < 0.7 and convergence_score > 0.5:
            return ("MODERATE DISTRIBUTIONAL CONVERGENCE: Some distributional "
                   "similarity detected, suggesting partial convergence to common "
                   "patterns.")
        
        else:
            return ("LIMITED DISTRIBUTIONAL CONVERGENCE: Models show distinct "
                   "distributions, indicating architecture-specific rather than "
                   "universal patterns dominate.")


class HybridConvergenceAnalyzer:
    """
    Combines semantic similarity and KL divergence for comprehensive convergence analysis.
    This is the main class for the enhanced universal patterns experiment.
    """
    
    def __init__(self, semantic_analyzer=None):
        self.semantic_analyzer = semantic_analyzer
        self.kl_analyzer = KLDivergenceAnalyzer()
        self.distribution_extractor = EnhancedDistributionExtractor()
        
    def analyze_hybrid_convergence(self, 
                                 model_responses: Dict[str, List[str]],
                                 capability: str = "unknown") -> HybridConvergenceResults:
        """
        Perform comprehensive convergence analysis combining semantic and distributional measures.
        
        Args:
            model_responses: {model_name: [response1, response2, ...]}
            capability: Name of capability being analyzed
            
        Returns:
            HybridConvergenceResults with complete analysis
        """
        
        print(f"\n🔬 Hybrid Convergence Analysis for {capability}")
        print("=" * 60)
        
        models = list(model_responses.keys())
        n_models = len(models)
        
        if n_models < 2:
            raise ValueError("Need at least 2 models for convergence analysis")
        
        # Phase 1: Semantic similarity analysis
        print("📝 Phase 1: Semantic Similarity Analysis")
        semantic_similarities = {}
        
        if self.semantic_analyzer:
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models[i+1:], i+1):
                    # Calculate semantic similarity between response sets
                    responses1 = model_responses[model1]
                    responses2 = model_responses[model2]
                    
                    pair_similarities = []
                    for r1, r2 in zip(responses1, responses2):
                        try:
                            sim = self.semantic_analyzer.calculate_similarity(r1, r2)
                            pair_similarities.append(sim)
                        except Exception as e:
                            print(f"  ⚠️  Semantic similarity error: {e}")
                            continue
                    
                    if pair_similarities:
                        avg_similarity = np.mean(pair_similarities)
                        semantic_similarities[f"{model1}_vs_{model2}"] = avg_similarity
                        print(f"  📊 {model1} vs {model2}: {avg_similarity:.3f}")
        
        semantic_convergence_score = np.mean(list(semantic_similarities.values())) if semantic_similarities else 0.0
        print(f"  🎯 Semantic convergence: {semantic_convergence_score:.3f}")
        
        # Phase 2: Distributional convergence analysis
        print("\n🔢 Phase 2: Distributional Convergence Analysis")
        
        # Extract probability distributions
        model_distributions = self.distribution_extractor.extract_distributions_from_responses(
            model_responses
        )
        
        # Analyze KL divergence
        distributional_results = self.kl_analyzer.calculate_distributional_convergence(
            model_distributions
        )
        
        # Phase 3: Hybrid convergence calculation
        print("\n🎯 Phase 3: Hybrid Convergence Synthesis")
        
        # Weight semantic and distributional convergence
        # Semantic convergence weight: 0.4 (what models say)
        # Distributional convergence weight: 0.6 (how they say it)
        semantic_weight = 0.4
        distributional_weight = 0.6
        
        hybrid_convergence_score = (
            semantic_weight * semantic_convergence_score + 
            distributional_weight * distributional_results["distributional_convergence"]
        )
        
        print(f"  📊 Semantic: {semantic_convergence_score:.3f} (weight: {semantic_weight})")
        print(f"  📊 Distributional: {distributional_results['distributional_convergence']:.3f} (weight: {distributional_weight})")
        print(f"  🎯 Hybrid convergence: {hybrid_convergence_score:.3f}")
        
        # Phase 4: Statistical significance and interpretation
        significance_results = self._combined_significance_analysis(
            semantic_convergence_score,
            distributional_results,
            n_models
        )
        
        interpretation = self._interpret_hybrid_results(
            semantic_convergence_score,
            distributional_results["distributional_convergence"],
            hybrid_convergence_score,
            significance_results
        )
        
        confidence_level = self._calculate_confidence_level(
            semantic_convergence_score,
            distributional_results,
            significance_results
        )
        
        print(f"  📈 Confidence level: {confidence_level:.3f}")
        print(f"  🔍 {interpretation}")
        
        return HybridConvergenceResults(
            semantic_similarities=semantic_similarities,
            semantic_convergence_score=semantic_convergence_score,
            kl_divergences=distributional_results["kl_divergences"],
            jensen_shannon_distances=distributional_results["jensen_shannon_distances"],
            distributional_convergence_score=distributional_results["distributional_convergence"],
            hybrid_convergence_score=hybrid_convergence_score,
            statistical_significance=significance_results,
            interpretation=interpretation,
            confidence_level=confidence_level
        )
    
    def _combined_significance_analysis(self, 
                                      semantic_convergence: float,
                                      distributional_results: Dict[str, Any],
                                      n_models: int) -> Dict[str, Any]:
        """Combine significance testing from both semantic and distributional analysis"""
        
        dist_significance = distributional_results.get("statistical_significance", {})
        
        # Conservative approach: require both methods to show significance
        semantic_significant = semantic_convergence > 0.6  # Threshold for semantic similarity
        distributional_significant = dist_significance.get("significant", False)
        
        combined_significant = semantic_significant and distributional_significant
        
        # Calculate combined p-value using Fisher's method
        semantic_p = max(0.001, 1 - semantic_convergence)  # Convert similarity to p-value estimate
        distributional_p = max(0.001, dist_significance.get("combined_p_value", 0.5))
        
        # Fisher's combined p-value (with safe log calculation)
        chi_square = -2 * (np.log(max(semantic_p, 1e-10)) + np.log(max(distributional_p, 1e-10)))
        combined_p_value = 1 - stats.chi2.cdf(chi_square, df=4)
        
        return {
            "semantic_significant": semantic_significant,
            "distributional_significant": distributional_significant,
            "combined_significant": combined_significant,
            "semantic_p_estimate": semantic_p,
            "distributional_p_value": distributional_p,
            "combined_p_value": combined_p_value,
            "fisher_chi_square": chi_square,
            "n_models": n_models
        }
    
    def _interpret_hybrid_results(self, 
                                semantic_conv: float,
                                distributional_conv: float,
                                hybrid_conv: float,
                                significance: Dict[str, Any]) -> str:
        """Provide comprehensive interpretation of hybrid results"""
        
        if hybrid_conv > 0.8 and significance["combined_significant"]:
            return ("COMPELLING EVIDENCE: Both semantic and distributional analysis "
                   "show strong convergence. This provides robust evidence for "
                   "universal alignment patterns across model architectures.")
        
        elif hybrid_conv > 0.6 and significance["combined_significant"]:
            return ("STRONG EVIDENCE: Significant convergence detected in both "
                   "semantic content and probability distributions, supporting "
                   "the universal patterns hypothesis.")
        
        elif hybrid_conv > 0.5:
            if semantic_conv > distributional_conv:
                return ("MODERATE EVIDENCE: Strong semantic convergence but weaker "
                       "distributional alignment. Models agree on content but "
                       "differ in output patterns.")
            else:
                return ("MODERATE EVIDENCE: Strong distributional convergence but "
                       "weaker semantic alignment. Models show similar output "
                       "patterns but vary in content.")
        
        else:
            return ("LIMITED EVIDENCE: Both semantic and distributional analysis "
                   "show weak convergence, suggesting architecture-specific "
                   "rather than universal patterns dominate.")
    
    def _calculate_confidence_level(self, 
                                  semantic_conv: float,
                                  distributional_results: Dict[str, Any],
                                  significance: Dict[str, Any]) -> float:
        """Calculate overall confidence in universal patterns hypothesis"""
        
        # Weight different evidence sources
        semantic_confidence = min(semantic_conv, 1.0)
        distributional_confidence = distributional_results["distributional_convergence"]
        statistical_confidence = 1 - significance["combined_p_value"]
        
        # Combine with empirically derived weights
        overall_confidence = (
            0.3 * semantic_confidence + 
            0.4 * distributional_confidence + 
            0.3 * statistical_confidence
        )
        
        return min(overall_confidence, 1.0)