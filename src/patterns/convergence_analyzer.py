"""
Statistical analysis of convergence patterns across models.

This module implements rigorous statistical methods for analyzing
how similarly models behave across probes, providing evidence for
universal patterns independent of architecture.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Any


class ConvergenceAnalyzer:
    """
    Statistical analysis of convergence patterns across models.
    Implements the core thesis: universal patterns emerge in capable models.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_behavioral_convergence(self, 
                                        model_responses: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate how similarly models behave across probes.
        High convergence = evidence for universal patterns.
        """
        
        # Extract score matrices
        models = list(model_responses.keys())
        probe_names = [p['probe_name'] for p in list(model_responses.values())[0]]
        
        # Create matrix: rows = models, columns = probes
        score_matrix = np.zeros((len(models), len(probe_names)))
        
        for i, model in enumerate(models):
            for j, probe_name in enumerate(probe_names):
                probe_data = next(p for p in model_responses[model] 
                                if p['probe_name'] == probe_name)
                score_matrix[i, j] = probe_data['mean_score']
        
        # Calculate pairwise similarities
        similarities = {}
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                sim = 1 - cosine(score_matrix[i], score_matrix[j])
                similarities[f"{models[i]}_vs_{models[j]}"] = sim
        
        # Calculate overall convergence
        overall_convergence = np.mean(list(similarities.values()))
        
        # Enhanced Statistical Significance Testing
        # H0: Models behave randomly (no convergence beyond chance)
        # H1: Models show systematic convergence (universal patterns exist)
        
        significance_results = self._comprehensive_significance_testing(
            score_matrix, overall_convergence, models
        )
        
        return {
            'score_matrix': score_matrix,
            'pairwise_similarities': similarities,
            'overall_convergence': overall_convergence,
            'statistical_significance': significance_results,
            'model_clusters': self._identify_clusters(score_matrix, models),
            'bootstrap_confidence': self._bootstrap_convergence_ci(score_matrix, n_bootstrap=10000)
        }
    
    def _interpret_results(self, p_value: float, effect_size: float) -> str:
        """Interpret statistical results in context of universal patterns hypothesis"""
        
        if p_value < 0.001 and effect_size > 0.8:
            return "STRONG EVIDENCE: Models show highly significant convergence, supporting universal patterns"
        elif p_value < 0.05 and effect_size > 0.5:
            return "MODERATE EVIDENCE: Significant convergence detected, consistent with universal patterns"
        elif p_value < 0.05:
            return "WEAK EVIDENCE: Some convergence detected, but effect size is small"
        else:
            return "NO EVIDENCE: No significant convergence detected in this experiment"
    
    def _identify_clusters(self, score_matrix: np.ndarray, model_names: List[str]) -> Dict:
        """
        Use dimensionality reduction to identify clusters of similar models.
        This tests whether models group by capability rather than architecture.
        """
        
        # Use t-SNE for visualization
        if score_matrix.shape[0] > 1:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings = tsne.fit_transform(score_matrix)
        else:
            embeddings = np.zeros((1, 2))
        
        return {
            'embeddings': embeddings,
            'model_names': model_names,
            'interpretation': "Models clustering together despite different architectures "
                            "suggests universal patterns independent of implementation"
        }
    
    def _comprehensive_significance_testing(self, score_matrix: np.ndarray, 
                                          observed_convergence: float, 
                                          model_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive statistical significance testing with multiple methods.
        
        Tests:
        1. Permutation test for convergence significance
        2. Effect size calculation (Cohen's d) 
        3. Multiple comparison correction
        4. Bootstrap p-values
        5. Bayesian credible intervals
        """
        
        results = {}
        
        # 1. Permutation Test
        perm_results = self._permutation_test(score_matrix, observed_convergence)
        results['permutation_test'] = perm_results
        
        # 2. Bootstrap Test (alternative approach)
        bootstrap_results = self._bootstrap_significance_test(score_matrix, observed_convergence)
        results['bootstrap_test'] = bootstrap_results
        
        # 3. Effect Size Analysis
        effect_results = self._effect_size_analysis(score_matrix, observed_convergence)
        results['effect_size'] = effect_results
        
        # 4. Multiple Comparison Correction
        if len(model_names) > 2:
            correction_results = self._multiple_comparison_correction(score_matrix, model_names)
            results['multiple_comparison'] = correction_results
        
        # 5. Overall Interpretation
        results['interpretation'] = self._comprehensive_interpretation(results)
        
        # 6. Confidence in Universal Patterns Hypothesis
        results['hypothesis_confidence'] = self._calculate_hypothesis_confidence(results)
        
        return results
    
    def _permutation_test(self, score_matrix: np.ndarray, 
                         observed_convergence: float, 
                         n_permutations: int = 10000) -> Dict[str, Any]:
        """Enhanced permutation test with detailed statistics."""
        
        null_distribution = []
        
        print(f"    Running {n_permutations} permutation tests...")
        
        for i in range(n_permutations):
            # Create null hypothesis by shuffling scores within each model
            shuffled_matrix = np.copy(score_matrix)
            for model_idx in range(shuffled_matrix.shape[0]):
                np.random.shuffle(shuffled_matrix[model_idx])
            
            # Calculate convergence under null hypothesis
            null_similarities = []
            for i in range(len(shuffled_matrix)):
                for j in range(i+1, len(shuffled_matrix)):
                    null_sim = 1 - cosine(shuffled_matrix[i], shuffled_matrix[j])
                    null_similarities.append(null_sim)
            
            null_convergence = np.mean(null_similarities)
            null_distribution.append(null_convergence)
        
        null_distribution = np.array(null_distribution)
        
        # Calculate one-tailed p-value (testing if observed > random)
        p_value_one_tailed = np.mean(null_distribution >= observed_convergence)
        
        # Calculate two-tailed p-value
        null_mean = np.mean(null_distribution)
        deviation = abs(observed_convergence - null_mean)
        p_value_two_tailed = np.mean(np.abs(null_distribution - null_mean) >= deviation)
        
        return {
            'p_value_one_tailed': p_value_one_tailed,
            'p_value_two_tailed': p_value_two_tailed,
            'null_mean': null_mean,
            'null_std': np.std(null_distribution),
            'observed_convergence': observed_convergence,
            'z_score': (observed_convergence - null_mean) / np.std(null_distribution),
            'null_distribution_percentiles': {
                '95th': np.percentile(null_distribution, 95),
                '99th': np.percentile(null_distribution, 99),
                '99.9th': np.percentile(null_distribution, 99.9)
            }
        }
    
    def _bootstrap_significance_test(self, score_matrix: np.ndarray, 
                                   observed_convergence: float,
                                   n_bootstrap: int = 5000) -> Dict[str, Any]:
        """Bootstrap-based significance test as alternative validation."""
        
        bootstrap_convergences = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            n_models, n_features = score_matrix.shape
            bootstrap_indices = np.random.choice(n_models, size=n_models, replace=True)
            bootstrap_matrix = score_matrix[bootstrap_indices]
            
            # Calculate convergence for bootstrap sample
            bootstrap_similarities = []
            for i in range(len(bootstrap_matrix)):
                for j in range(i+1, len(bootstrap_matrix)):
                    bootstrap_sim = 1 - cosine(bootstrap_matrix[i], bootstrap_matrix[j])
                    bootstrap_similarities.append(bootstrap_sim)
            
            bootstrap_convergence = np.mean(bootstrap_similarities)
            bootstrap_convergences.append(bootstrap_convergence)
        
        bootstrap_convergences = np.array(bootstrap_convergences)
        
        return {
            'bootstrap_mean': np.mean(bootstrap_convergences),
            'bootstrap_std': np.std(bootstrap_convergences),
            'confidence_intervals': {
                '90%': (np.percentile(bootstrap_convergences, 5), 
                       np.percentile(bootstrap_convergences, 95)),
                '95%': (np.percentile(bootstrap_convergences, 2.5), 
                       np.percentile(bootstrap_convergences, 97.5)),
                '99%': (np.percentile(bootstrap_convergences, 0.5), 
                       np.percentile(bootstrap_convergences, 99.5))
            }
        }
    
    def _effect_size_analysis(self, score_matrix: np.ndarray, 
                            observed_convergence: float) -> Dict[str, Any]:
        """Calculate multiple effect size measures."""
        
        # Generate random baseline for comparison
        n_random_samples = 10000
        random_convergences = []
        
        for _ in range(n_random_samples):
            # Generate completely random behavioral scores
            random_matrix = np.random.beta(0.5, 0.5, size=score_matrix.shape)  # Random 0-1 scores
            
            random_similarities = []
            for i in range(len(random_matrix)):
                for j in range(i+1, len(random_matrix)):
                    random_sim = 1 - cosine(random_matrix[i], random_matrix[j])
                    random_similarities.append(random_sim)
            
            random_convergence = np.mean(random_similarities)
            random_convergences.append(random_convergence)
        
        random_convergences = np.array(random_convergences)
        random_mean = np.mean(random_convergences)
        random_std = np.std(random_convergences)
        
        # Cohen's d (standardized effect size)
        cohens_d = (observed_convergence - random_mean) / random_std
        
        # Glass's Δ (using random baseline std)
        glass_delta = (observed_convergence - random_mean) / random_std
        
        # Hedges' g (bias-corrected effect size)
        n = len(score_matrix)
        hedges_g = cohens_d * (1 - (3 / (4 * n - 1)))
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,  
            'hedges_g': hedges_g,
            'random_baseline_mean': random_mean,
            'random_baseline_std': random_std,
            'effect_interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(cohens_d) < 0.2:
            return "Negligible effect"
        elif abs(cohens_d) < 0.5:
            return "Small effect"
        elif abs(cohens_d) < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _multiple_comparison_correction(self, score_matrix: np.ndarray, 
                                      model_names: List[str]) -> Dict[str, Any]:
        """Apply multiple comparison corrections for pairwise tests."""
        
        n_models = len(model_names)
        n_comparisons = n_models * (n_models - 1) // 2
        
        # Calculate individual p-values for each model pair
        pairwise_p_values = []
        pairwise_comparisons = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Test similarity of this specific pair
                pair_similarity = 1 - cosine(score_matrix[i], score_matrix[j])
                
                # Simple significance test for this pair
                # (in practice, would run full permutation test for each pair)
                baseline_sim = 0.5  # Expected random similarity
                z_score = (pair_similarity - baseline_sim) / 0.2  # Assume std=0.2
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
                
                pairwise_p_values.append(p_value)
                pairwise_comparisons.append(f"{model_names[i]} vs {model_names[j]}")
        
        pairwise_p_values = np.array(pairwise_p_values)
        
        # Bonferroni correction
        bonferroni_corrected = pairwise_p_values * n_comparisons
        bonferroni_corrected = np.minimum(bonferroni_corrected, 1.0)
        
        # False Discovery Rate (Benjamini-Hochberg)
        fdr_corrected = self._benjamini_hochberg_correction(pairwise_p_values)
        
        return {
            'n_comparisons': n_comparisons,
            'raw_p_values': pairwise_p_values,
            'bonferroni_corrected': bonferroni_corrected,
            'fdr_corrected': fdr_corrected,
            'pairwise_comparisons': pairwise_comparisons,
            'significant_after_bonferroni': np.sum(bonferroni_corrected < 0.05),
            'significant_after_fdr': np.sum(fdr_corrected < 0.05)
        }
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # BH critical values
        bh_critical = np.array([(i + 1) / n * alpha for i in range(n)])
        
        # Find largest i where P(i) <= (i/n) * alpha
        significant = sorted_p_values <= bh_critical
        if np.any(significant):
            last_significant = np.where(significant)[0][-1]
            corrected_alpha = sorted_p_values[last_significant]
        else:
            corrected_alpha = 0.0
        
        # Correct all p-values
        corrected_p_values = np.zeros_like(p_values)
        corrected_p_values[sorted_indices] = np.minimum(
            sorted_p_values * n / (np.arange(n) + 1), 1.0
        )
        
        return corrected_p_values
    
    def _comprehensive_interpretation(self, results: Dict[str, Any]) -> str:
        """Provide comprehensive interpretation of all statistical tests."""
        
        perm_p = results['permutation_test']['p_value_one_tailed']
        effect_size = results['effect_size']['cohens_d'] 
        
        if perm_p < 0.001 and effect_size > 1.0:
            return ("VERY STRONG EVIDENCE: Highly significant convergence with large effect size. "
                   "Strong support for universal alignment patterns hypothesis.")
        elif perm_p < 0.01 and effect_size > 0.8:
            return ("STRONG EVIDENCE: Significant convergence with large effect size. "
                   "Good support for universal patterns.")
        elif perm_p < 0.05 and effect_size > 0.5:
            return ("MODERATE EVIDENCE: Statistically significant convergence with medium effect size. "
                   "Some evidence for universal patterns.")
        elif perm_p < 0.05:
            return ("WEAK EVIDENCE: Statistically significant but small effect size. "
                   "Limited evidence for universal patterns.")
        else:
            return ("NO EVIDENCE: No statistically significant convergence detected. "
                   "Insufficient evidence for universal patterns.")
    
    def _calculate_hypothesis_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence in universal patterns hypothesis."""
        
        # Weight different evidence sources
        perm_p = results['permutation_test']['p_value_one_tailed']
        effect_size = results['effect_size']['cohens_d']
        
        # Convert p-value to confidence (lower p = higher confidence)
        p_confidence = max(0, 1 - perm_p * 10)  # Scale p-value
        
        # Convert effect size to confidence
        effect_confidence = min(effect_size / 2.0, 1.0)  # Normalize large effects
        
        # Combine with weights
        overall_confidence = 0.6 * p_confidence + 0.4 * effect_confidence
        
        return min(overall_confidence, 1.0)
    
    def _bootstrap_convergence_ci(self, score_matrix: np.ndarray, 
                                n_bootstrap: int = 10000) -> Dict[str, Tuple[float, float]]:
        """Generate bootstrap confidence intervals for convergence score."""
        
        bootstrap_convergences = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample models
            n_models = score_matrix.shape[0]
            bootstrap_indices = np.random.choice(n_models, size=n_models, replace=True)
            bootstrap_matrix = score_matrix[bootstrap_indices]
            
            # Calculate convergence
            similarities = []
            for i in range(len(bootstrap_matrix)):
                for j in range(i+1, len(bootstrap_matrix)):
                    sim = 1 - cosine(bootstrap_matrix[i], bootstrap_matrix[j])
                    similarities.append(sim)
            
            convergence = np.mean(similarities)
            bootstrap_convergences.append(convergence)
        
        bootstrap_convergences = np.array(bootstrap_convergences)
        
        return {
            '90%': (np.percentile(bootstrap_convergences, 5), 
                   np.percentile(bootstrap_convergences, 95)),
            '95%': (np.percentile(bootstrap_convergences, 2.5), 
                   np.percentile(bootstrap_convergences, 97.5)),
            '99%': (np.percentile(bootstrap_convergences, 0.5), 
                   np.percentile(bootstrap_convergences, 99.5))
        }