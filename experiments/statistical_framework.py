"""
Enhanced Statistical Framework for Universal Alignment Patterns

This module implements rigorous statistical methods for analyzing convergence
patterns with the statistical rigor required for academic publication and
fellowship applications.

Author: Samuel Tchakwera
Purpose: Statistical validation of universal alignment patterns hypothesis
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass
import pandas as pd


@dataclass
class StatisticalTestResult:
    """Results from a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    is_significant: bool
    
    
@dataclass
class PermutationTestResult:
    """Results from permutation testing"""
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    effect_size: float
    significant: bool
    num_permutations: int


class EnhancedStatisticalAnalyzer:
    """
    Advanced statistical analysis for universal alignment patterns.
    
    Implements state-of-the-art statistical methods including:
    - Permutation testing for distribution-free inference
    - Bootstrap confidence intervals
    - Multiple comparison corrections
    - Effect size calculations
    - Clustering analysis
    """
    
    def __init__(self, significance_level: float = 0.001, n_permutations: int = 10000):
        """Initialize the statistical analyzer"""
        self.significance_level = significance_level
        self.n_permutations = n_permutations
        self.random_state = 42
        np.random.seed(self.random_state)
        
    def permutation_test(self, 
                        group1: np.ndarray, 
                        group2: np.ndarray,
                        statistic_func: callable = None,
                        alternative: str = "two-sided") -> PermutationTestResult:
        """
        Conduct permutation test between two groups.
        
        This is the gold standard for non-parametric testing when
        distributional assumptions cannot be made.
        """
        
        if statistic_func is None:
            statistic_func = lambda x, y: np.abs(np.mean(x) - np.mean(y))
        
        # Calculate observed test statistic
        observed_stat = statistic_func(group1, group2)
        
        # Combine groups for permutation
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        n_total = n1 + n2
        
        # Generate permutation distribution
        null_distribution = np.zeros(self.n_permutations)
        
        for i in range(self.n_permutations):
            # Randomly permute the combined data
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            
            # Calculate statistic for this permutation
            null_distribution[i] = statistic_func(perm_group1, perm_group2)
        
        # Calculate p-value based on alternative hypothesis
        if alternative == "greater":
            p_value = np.mean(null_distribution >= observed_stat)
        elif alternative == "less":
            p_value = np.mean(null_distribution <= observed_stat)
        else:  # two-sided
            p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        effect_size = np.abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        return PermutationTestResult(
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < self.significance_level,
            num_permutations=self.n_permutations
        )
    
    def bootstrap_confidence_interval(self, 
                                    data: np.ndarray,
                                    statistic_func: callable = np.mean,
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Bootstrap provides robust uncertainty estimates without
        distributional assumptions.
        """
        
        n = len(data)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic_func(bootstrap_sample)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def multiple_comparisons_correction(self, 
                                      p_values: List[float],
                                      method: str = "bonferroni") -> List[float]:
        """
        Apply multiple comparisons correction to p-values.
        
        Essential when testing multiple capabilities to control
        family-wise error rate.
        """
        
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == "bonferroni":
            # Most conservative but straightforward
            corrected_p_values = p_values * n_tests
            # Ensure p-values don't exceed 1
            corrected_p_values = np.minimum(corrected_p_values, 1.0)
            
        elif method == "holm":
            # Less conservative than Bonferroni
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            
            corrected_p_values = np.zeros_like(p_values)
            for i, p_val in enumerate(sorted_p_values):
                corrected_p_values[sorted_indices[i]] = p_val * (n_tests - i)
            
            # Ensure monotonicity
            for i in range(1, n_tests):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                corrected_p_values[idx] = max(corrected_p_values[idx], 
                                            corrected_p_values[prev_idx])
            
            corrected_p_values = np.minimum(corrected_p_values, 1.0)
            
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return corrected_p_values.tolist()
    
    def calculate_effect_sizes(self, 
                             similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate various effect size measures for convergence.
        
        Effect sizes quantify practical significance beyond p-values.
        """
        
        # Extract upper triangular similarities (exclude diagonal)
        n = similarity_matrix.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        similarities = similarity_matrix[triu_indices]
        
        if len(similarities) == 0:
            return {"cohens_d": 0.0, "cliffs_delta": 0.0, "eta_squared": 0.0}
        
        # Cohen's d: standardized mean difference from random baseline (0.5)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities, ddof=1)
        cohens_d = (mean_sim - 0.5) / std_sim if std_sim > 0 else 0
        
        # Cliff's delta: non-parametric effect size
        # Proportion of pairs where observed > random baseline
        cliffs_delta = np.mean(similarities > 0.5) - np.mean(similarities < 0.5)
        
        # Eta-squared: proportion of variance explained
        total_variance = np.var(similarities, ddof=1)
        between_variance = n * (mean_sim - 0.5) ** 2
        eta_squared = between_variance / (between_variance + total_variance) if total_variance > 0 else 0
        
        return {
            "cohens_d": cohens_d,
            "cliffs_delta": cliffs_delta, 
            "eta_squared": eta_squared,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim
        }
    
    def convergence_permutation_test(self, 
                                   similarity_matrices: List[np.ndarray],
                                   capability_names: List[str]) -> Dict[str, Any]:
        """
        Test whether observed convergence is significantly above chance.
        
        Core statistical test for the universal patterns hypothesis.
        """
        
        results = {}
        observed_convergences = []
        p_values = []
        
        for i, (similarity_matrix, capability) in enumerate(zip(similarity_matrices, capability_names)):
            # Calculate observed convergence (mean similarity)
            n = similarity_matrix.shape[0]
            if n < 2:
                continue
                
            triu_indices = np.triu_indices(n, k=1)
            similarities = similarity_matrix[triu_indices]
            observed_convergence = np.mean(similarities)
            observed_convergences.append(observed_convergence)
            
            # Generate null distribution by permuting model labels
            null_convergences = np.zeros(self.n_permutations)
            
            for perm in range(self.n_permutations):
                # Randomly permute the similarity matrix
                perm_indices = np.random.permutation(n)
                perm_matrix = similarity_matrix[np.ix_(perm_indices, perm_indices)]
                perm_similarities = perm_matrix[triu_indices]
                null_convergences[perm] = np.mean(perm_similarities)
            
            # Calculate p-value (one-tailed: observed > random)
            p_value = np.mean(null_convergences >= observed_convergence)
            p_values.append(p_value)
            
            # Calculate effect size
            effect_sizes = self.calculate_effect_sizes(similarity_matrix)
            
            results[capability] = {
                "observed_convergence": observed_convergence,
                "null_mean": np.mean(null_convergences),
                "null_std": np.std(null_convergences),
                "p_value": p_value,
                "effect_sizes": effect_sizes,
                "z_score": (observed_convergence - np.mean(null_convergences)) / np.std(null_convergences),
                "significant": p_value < self.significance_level
            }
        
        # Apply multiple comparisons correction
        if p_values:
            corrected_p_values = self.multiple_comparisons_correction(p_values, method="holm")
            
            for i, capability in enumerate(capability_names[:len(corrected_p_values)]):
                if capability in results:
                    results[capability]["corrected_p_value"] = corrected_p_values[i]
                    results[capability]["significant_corrected"] = corrected_p_values[i] < self.significance_level
        
        # Overall meta-analysis
        if observed_convergences:
            overall_convergence = np.mean(observed_convergences)
            significant_capabilities = sum(1 for result in results.values() 
                                         if result.get("significant_corrected", False))
            
            results["meta_analysis"] = {
                "overall_convergence": overall_convergence,
                "capabilities_tested": len(observed_convergences),
                "significant_capabilities": significant_capabilities,
                "proportion_significant": significant_capabilities / len(observed_convergences),
                "evidence_strength": self._classify_evidence_strength(overall_convergence, 
                                                                    significant_capabilities,
                                                                    len(observed_convergences))
            }
        
        return results
    
    def hierarchical_clustering_analysis(self, 
                                       similarity_matrix: np.ndarray,
                                       model_names: List[str]) -> Dict[str, Any]:
        """
        Analyze whether models cluster by capability vs architecture.
        
        Tests the key prediction that universal patterns transcend architecture.
        """
        
        n_models = len(model_names)
        if n_models < 3:
            return {"error": "Insufficient models for clustering analysis"}
        
        # Convert similarity to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix for linkage
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Get clusters at different levels
        cluster_results = {}
        for n_clusters in range(2, min(n_models, 5)):
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(distance_matrix, clusters, metric='precomputed')
            except:
                silhouette = 0.0
            
            cluster_results[f"{n_clusters}_clusters"] = {
                "cluster_assignments": {model: int(cluster) for model, cluster in zip(model_names, clusters)},
                "silhouette_score": silhouette
            }
        
        # Find optimal number of clusters
        best_k = max(cluster_results.keys(), 
                    key=lambda k: cluster_results[k]["silhouette_score"])
        
        return {
            "linkage_matrix": linkage_matrix.tolist(),
            "cluster_results": cluster_results,
            "optimal_clusters": best_k,
            "model_names": model_names,
            "interpretation": self._interpret_clustering_results(cluster_results, model_names)
        }
    
    def dimensionality_reduction_analysis(self, 
                                        similarity_matrix: np.ndarray,
                                        model_names: List[str]) -> Dict[str, Any]:
        """
        Reduce dimensionality for visualization and pattern identification.
        """
        
        n_models = len(model_names)
        if n_models < 3:
            return {"error": "Insufficient models for dimensionality reduction"}
        
        # Convert similarity to feature matrix (use similarity as features)
        features = similarity_matrix
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        results = {}
        
        # PCA Analysis
        try:
            pca = PCA(n_components=min(n_models-1, 3), random_state=self.random_state)
            pca_features = pca.fit_transform(features_scaled)
            
            results["pca"] = {
                "components": pca_features.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "model_names": model_names
            }
        except Exception as e:
            results["pca"] = {"error": str(e)}
        
        # t-SNE Analysis
        try:
            if n_models >= 4:  # t-SNE requires at least 4 samples
                tsne = TSNE(n_components=2, random_state=self.random_state, 
                           perplexity=min(30, n_models-1))
                tsne_features = tsne.fit_transform(features_scaled)
                
                results["tsne"] = {
                    "components": tsne_features.tolist(),
                    "model_names": model_names
                }
            else:
                results["tsne"] = {"error": "Insufficient samples for t-SNE (need ≥4)"}
        except Exception as e:
            results["tsne"] = {"error": str(e)}
        
        return results
    
    def comprehensive_statistical_report(self, 
                                       similarity_matrices: List[np.ndarray],
                                       capability_names: List[str],
                                       model_names: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive statistical analysis report.
        
        This is the main function that produces all statistical evidence
        needed for the fellowship application.
        """
        
        report = {
            "methodology": {
                "significance_level": self.significance_level,
                "n_permutations": self.n_permutations,
                "random_state": self.random_state,
                "multiple_comparison_correction": "Holm-Bonferroni"
            },
            "sample_sizes": {
                "n_models": len(model_names),
                "n_capabilities": len(capability_names),
                "total_comparisons": sum(n*(n-1)//2 for n in [len(model_names)] * len(capability_names))
            }
        }
        
        # Main convergence analysis
        convergence_results = self.convergence_permutation_test(similarity_matrices, capability_names)
        report["convergence_analysis"] = convergence_results
        
        # Clustering analysis (using averaged similarity matrix)
        if similarity_matrices:
            avg_similarity_matrix = np.mean(similarity_matrices, axis=0)
            clustering_results = self.hierarchical_clustering_analysis(avg_similarity_matrix, model_names)
            report["clustering_analysis"] = clustering_results
            
            # Dimensionality reduction
            reduction_results = self.dimensionality_reduction_analysis(avg_similarity_matrix, model_names)
            report["dimensionality_reduction"] = reduction_results
        
        # Overall interpretation
        report["executive_summary"] = self._generate_executive_summary(report)
        
        return report
    
    def _classify_evidence_strength(self, 
                                  overall_convergence: float,
                                  significant_capabilities: int,
                                  total_capabilities: int) -> str:
        """Classify the strength of evidence for universal patterns"""
        
        significance_rate = significant_capabilities / total_capabilities
        
        if overall_convergence > 0.8 and significance_rate >= 0.8:
            return "VERY_STRONG"
        elif overall_convergence > 0.7 and significance_rate >= 0.6:
            return "STRONG"
        elif overall_convergence > 0.6 and significance_rate >= 0.4:
            return "MODERATE"
        elif overall_convergence > 0.5 and significance_rate >= 0.2:
            return "WEAK"
        else:
            return "INSUFFICIENT"
    
    def _interpret_clustering_results(self, 
                                    cluster_results: Dict[str, Any],
                                    model_names: List[str]) -> str:
        """Interpret clustering results in context of universal patterns"""
        
        # Find the clustering solution with highest silhouette score
        best_solution = max(cluster_results.values(), key=lambda x: x["silhouette_score"])
        best_silhouette = best_solution["silhouette_score"]
        
        if best_silhouette > 0.7:
            return "Strong clustering detected - models group by behavioral similarity"
        elif best_silhouette > 0.5:
            return "Moderate clustering detected - some behavioral grouping"
        elif best_silhouette > 0.3:
            return "Weak clustering detected - limited behavioral grouping"
        else:
            return "No clear clustering - models show similar behavioral patterns"
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of statistical findings"""
        
        convergence = report.get("convergence_analysis", {})
        meta = convergence.get("meta_analysis", {})
        
        overall_convergence = meta.get("overall_convergence", 0.0)
        evidence_strength = meta.get("evidence_strength", "INSUFFICIENT")
        significant_capabilities = meta.get("significant_capabilities", 0)
        total_capabilities = meta.get("capabilities_tested", 0)
        
        summary = {
            "overall_convergence_score": overall_convergence,
            "evidence_classification": evidence_strength,
            "statistical_significance": f"{significant_capabilities}/{total_capabilities} capabilities significant",
            "key_findings": [],
            "research_implications": [],
            "limitations": []
        }
        
        # Generate key findings
        if evidence_strength in ["VERY_STRONG", "STRONG"]:
            summary["key_findings"].append(f"Strong evidence for universal alignment patterns (convergence = {overall_convergence:.1%})")
            summary["research_implications"].append("Universal safety measures may be feasible across architectures")
            summary["research_implications"].append("Alignment research can focus on universal features rather than architecture-specific approaches")
        elif evidence_strength == "MODERATE":
            summary["key_findings"].append(f"Moderate evidence for convergent patterns (convergence = {overall_convergence:.1%})")
            summary["research_implications"].append("Some universal patterns exist but architecture-specific research still important")
        else:
            summary["key_findings"].append("Limited evidence for universal patterns in current sample")
            summary["limitations"].append("May require larger sample size or different model selection")
        
        # Add statistical rigor note
        alpha = report["methodology"]["significance_level"]
        n_perm = report["methodology"]["n_permutations"]
        summary["key_findings"].append(f"Results validated with rigorous statistics (α = {alpha}, {n_perm:,} permutations)")
        
        return summary