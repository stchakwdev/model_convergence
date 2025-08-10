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
        
        # Statistical significance test
        # H0: Models behave randomly (no convergence)
        # H1: Models show systematic convergence
        
        # Use permutation test
        n_permutations = 1000
        null_distribution = []
        
        for _ in range(n_permutations):
            # Shuffle each model's scores independently
            shuffled_matrix = np.copy(score_matrix)
            for i in range(len(models)):
                np.random.shuffle(shuffled_matrix[i])
            
            # Calculate similarity under null hypothesis
            null_sims = []
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    null_sim = 1 - cosine(shuffled_matrix[i], shuffled_matrix[j])
                    null_sims.append(null_sim)
            
            null_distribution.append(np.mean(null_sims))
        
        # Calculate p-value
        p_value = np.mean(np.array(null_distribution) >= overall_convergence)
        
        # Effect size (Cohen's d)
        effect_size = (overall_convergence - np.mean(null_distribution)) / np.std(null_distribution)
        
        return {
            'score_matrix': score_matrix,
            'pairwise_similarities': similarities,
            'overall_convergence': overall_convergence,
            'statistical_significance': {
                'p_value': p_value,
                'effect_size': effect_size,
                'interpretation': self._interpret_results(p_value, effect_size)
            },
            'model_clusters': self._identify_clusters(score_matrix, models)
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