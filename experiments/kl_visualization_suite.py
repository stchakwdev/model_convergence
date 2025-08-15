#!/usr/bin/env python3
"""
Advanced KL Divergence Visualization Suite

Creates publication-quality visualizations for distributional convergence analysis.
Showcases both semantic and KL divergence convergence patterns for fellowship application.

Key visualizations:
1. KL/JS divergence heatmaps
2. Distribution comparison plots  
3. Convergence trajectory analysis
4. Statistical significance plots
5. 3D manifold projections

Author: Samuel Chakwera
Purpose: Advanced visualizations for Anthropic Fellowship research
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import matplotlib.patches as patches


class KLVisualizationSuite:
    """
    Advanced visualization suite for KL divergence and hybrid convergence analysis.
    Creates publication-quality figures for universal alignment patterns research.
    """
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Color schemes for different analysis types
        self.colors = {
            'semantic': '#2E86AB',      # Blue
            'distributional': '#A23B72', # Magenta  
            'hybrid': '#F18F01',        # Orange
            'significance': '#C73E1D'    # Red
        }
        
        self.model_colors = {
            'gpt-5-chat': '#FF6B6B',
            'claude-sonnet-4': '#4ECDC4', 
            'gemini-2.5-pro': '#45B7D1',
            'claude-3.5-sonnet': '#96CEB4',
            'gpt-4-turbo': '#FECA57',
            'gpt-oss-120b': '#DDA0DD'
        }
    
    def create_kl_divergence_heatmap(self, 
                                   kl_results: Dict[str, Any],
                                   capability: str = "Overall",
                                   save_path: Optional[str] = None) -> str:
        """
        Create heatmap visualization of KL divergences between models.
        
        Args:
            kl_results: Results from hybrid convergence analysis
            capability: Name of capability being analyzed
            save_path: Optional custom save path
        """
        
        print(f"🎨 Creating KL divergence heatmap for {capability}...")
        
        # Extract KL divergences and model pairs
        kl_divergences = kl_results.get('kl_divergences', {})
        js_distances = kl_results.get('jensen_shannon_distances', {})
        
        if not kl_divergences:
            print("  ⚠️  No KL divergence data found")
            return None
        
        # Extract unique models from pair keys
        models = set()
        for pair_key in kl_divergences.keys():
            model1, model2 = pair_key.split('_vs_')
            models.add(model1)
            models.add(model2)
        
        models = sorted(list(models))
        n_models = len(models)
        
        # Create matrices
        kl_matrix = np.zeros((n_models, n_models))
        js_matrix = np.zeros((n_models, n_models))
        
        model_to_idx = {model: i for i, model in enumerate(models)}
        
        # Fill matrices
        for pair_key, kl_value in kl_divergences.items():
            model1, model2 = pair_key.split('_vs_')
            i, j = model_to_idx[model1], model_to_idx[model2]
            
            kl_matrix[i, j] = kl_value
            kl_matrix[j, i] = kl_value  # Make symmetric for visualization
            
            if pair_key in js_distances:
                js_value = js_distances[pair_key]
                js_matrix[i, j] = js_value
                js_matrix[j, i] = js_value
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # KL Divergence heatmap
        im1 = ax1.imshow(kl_matrix, cmap='RdYlBu_r', aspect='equal')
        ax1.set_title(f'KL Divergence Matrix\n{capability}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Models', fontsize=12)
        
        # Set ticks and labels
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels([self._clean_model_name(m) for m in models], rotation=45, ha='right')
        ax1.set_yticklabels([self._clean_model_name(m) for m in models])
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:  # Don't annotate diagonal
                    text = ax1.text(j, i, f'{kl_matrix[i, j]:.3f}', 
                                   ha="center", va="center", color="white", fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('KL Divergence\n(Lower = More Similar)', rotation=270, labelpad=20)
        
        # Jensen-Shannon Distance heatmap
        im2 = ax2.imshow(js_matrix, cmap='viridis_r', aspect='equal')
        ax2.set_title(f'Jensen-Shannon Distance Matrix\n{capability}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Models', fontsize=12)
        
        ax2.set_xticks(range(n_models))
        ax2.set_yticks(range(n_models))
        ax2.set_xticklabels([self._clean_model_name(m) for m in models], rotation=45, ha='right')
        ax2.set_yticklabels([self._clean_model_name(m) for m in models])
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    text = ax2.text(j, i, f'{js_matrix[i, j]:.3f}', 
                                   ha="center", va="center", color="white", fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Jensen-Shannon Distance\n(Lower = More Similar)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.output_dir / f"kl_divergence_heatmap_{capability.replace(' ', '_').lower()}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Saved KL divergence heatmap: {save_path}")
        return str(save_path)
    
    def create_hybrid_convergence_dashboard(self, 
                                          analysis_results: Dict[str, Any],
                                          title: str = "Universal Alignment Patterns: Hybrid Analysis") -> str:
        """
        Create comprehensive dashboard showing semantic vs distributional convergence.
        """
        
        print(f"🎨 Creating hybrid convergence dashboard...")
        
        capability_results = analysis_results.get('capability_results', {})
        overall_convergence = analysis_results.get('overall_convergence', 0)
        semantic_convergence = analysis_results.get('semantic_convergence', 0)
        distributional_convergence = analysis_results.get('distributional_convergence', 0)
        
        # Create dashboard with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall convergence comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        convergence_types = ['Semantic', 'Distributional', 'Hybrid']
        convergence_scores = [semantic_convergence, distributional_convergence, overall_convergence]
        colors = [self.colors['semantic'], self.colors['distributional'], self.colors['hybrid']]
        
        bars = ax1.bar(convergence_types, [s*100 for s in convergence_scores], color=colors, alpha=0.8)
        ax1.set_title('Overall Convergence Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Convergence Score (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, convergence_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add significance threshold lines
        ax1.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Strong Evidence')
        ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Moderate Evidence')
        ax1.legend(loc='upper right', fontsize=10)
        
        # 2. Capability breakdown (top middle-right)
        ax2 = fig.add_subplot(gs[0, 1:3])
        
        capabilities = list(capability_results.keys())
        semantic_scores = [capability_results[cap].get('semantic_convergence', 0)*100 for cap in capabilities]
        distributional_scores = [capability_results[cap].get('distributional_convergence', 0)*100 for cap in capabilities]
        hybrid_scores = [capability_results[cap].get('hybrid_convergence', 0)*100 for cap in capabilities]
        
        x = np.arange(len(capabilities))
        width = 0.25
        
        ax2.bar(x - width, semantic_scores, width, label='Semantic', color=self.colors['semantic'], alpha=0.8)
        ax2.bar(x, distributional_scores, width, label='Distributional', color=self.colors['distributional'], alpha=0.8)
        ax2.bar(x + width, hybrid_scores, width, label='Hybrid', color=self.colors['hybrid'], alpha=0.8)
        
        ax2.set_title('Convergence by Capability', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Convergence Score (%)')
        ax2.set_xlabel('Alignment Capabilities')
        ax2.set_xticks(x)
        ax2.set_xticklabels([cap.replace('_', ' ').title() for cap in capabilities], rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. Confidence levels (top right)
        ax3 = fig.add_subplot(gs[0, 3])
        confidence_levels = [capability_results[cap].get('confidence_level', 0)*100 for cap in capabilities]
        
        ax3.barh(range(len(capabilities)), confidence_levels, color='purple', alpha=0.7)
        ax3.set_title('Confidence Levels', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Confidence (%)')
        ax3.set_yticks(range(len(capabilities)))
        ax3.set_yticklabels([cap.replace('_', ' ').title() for cap in capabilities])
        ax3.set_xlim(0, 100)
        
        # Add confidence threshold
        ax3.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='High Confidence')
        ax3.legend()
        
        # 4. Statistical significance summary (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        significant_capabilities = []
        p_values = []
        for cap in capabilities:
            significance = capability_results[cap].get('statistical_significance', {})
            if significance.get('combined_significant', False):
                significant_capabilities.append(cap.replace('_', ' ').title())
                p_values.append(-np.log10(max(significance.get('combined_p_value', 0.5), 0.001)))
        
        if significant_capabilities:
            ax4.barh(range(len(significant_capabilities)), p_values, color=self.colors['significance'], alpha=0.7)
            ax4.set_title('Statistical Significance\n(-log10 p-value)', fontweight='bold', fontsize=12)
            ax4.set_xlabel('-log10(p-value)')
            ax4.set_yticks(range(len(significant_capabilities)))
            ax4.set_yticklabels(significant_capabilities)
            ax4.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p < 0.05')
            ax4.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', alpha=0.7, label='p < 0.001')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No Statistically\nSignificant Results', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Statistical Significance', fontweight='bold', fontsize=12)
        
        # 5. Distribution of convergence types (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Scatter plot of semantic vs distributional convergence
        sem_scores = [capability_results[cap].get('semantic_convergence', 0) for cap in capabilities]
        dist_scores = [capability_results[cap].get('distributional_convergence', 0) for cap in capabilities]
        
        scatter = ax5.scatter(sem_scores, dist_scores, 
                            c=[hybrid_scores[i]/100 for i in range(len(capabilities))], 
                            cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        
        # Add diagonal line for perfect correlation
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
        
        ax5.set_xlabel('Semantic Convergence')
        ax5.set_ylabel('Distributional Convergence')
        ax5.set_title('Semantic vs Distributional\nConvergence', fontweight='bold', fontsize=12)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Hybrid Score')
        
        # Add capability labels
        for i, cap in enumerate(capabilities):
            ax5.annotate(cap.replace('_', ' ')[:8], (sem_scores[i], dist_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 6. Interpretation summary (middle right)
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        
        # Create interpretation text
        if overall_convergence > 0.7:
            interpretation = "🏆 STRONG EVIDENCE for Universal Alignment Patterns"
            color = 'green'
        elif overall_convergence > 0.5:
            interpretation = "📊 MODERATE EVIDENCE for Universal Patterns"
            color = 'orange'
        else:
            interpretation = "🔍 LIMITED EVIDENCE - Architecture-Specific Patterns"
            color = 'red'
        
        ax6.text(0.5, 0.8, interpretation, ha='center', va='center', 
                transform=ax6.transAxes, fontsize=16, fontweight='bold', color=color)
        
        # Add summary statistics
        summary_text = f"""
        Overall Hybrid Convergence: {overall_convergence:.1%}
        Semantic Convergence: {semantic_convergence:.1%}
        Distributional Convergence: {distributional_convergence:.1%}
        
        Capabilities Analyzed: {len(capabilities)}
        Significant Results: {len(significant_capabilities)}
        Analysis Method: Hybrid Semantic + KL Divergence
        """
        
        ax6.text(0.5, 0.4, summary_text, ha='center', va='center',
                transform=ax6.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # 7. Model similarity network (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Create network visualization if we have KL divergence data
        hybrid_detailed = analysis_results.get('hybrid_results_detailed', {})
        if hybrid_detailed:
            self._create_model_similarity_network(ax7, hybrid_detailed, capabilities)
        else:
            ax7.text(0.5, 0.5, 'Model Similarity Network\n(Requires KL Divergence Data)', 
                    ha='center', va='center', transform=ax7.transAxes, fontsize=14)
            ax7.axis('off')
        
        # Add main title and metadata
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # Add metadata footer
        metadata_text = f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | " \
                       f"Analysis: Hybrid Semantic + KL Divergence | " \
                       f"🤖 Generated with Claude Code"
        
        fig.text(0.5, 0.02, metadata_text, ha='center', va='bottom', 
                fontsize=10, style='italic', alpha=0.7)
        
        # Save dashboard
        save_path = self.output_dir / "hybrid_convergence_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Saved hybrid convergence dashboard: {save_path}")
        return str(save_path)
    
    def _create_model_similarity_network(self, ax, hybrid_detailed: Dict, capabilities: List[str]):
        """Create network visualization showing model similarities"""
        
        ax.set_title('Model Similarity Network\n(Based on KL Divergence)', fontweight='bold', fontsize=12)
        
        # Extract all unique models
        all_models = set()
        for cap_results in hybrid_detailed.values():
            if hasattr(cap_results, 'kl_divergences'):
                for pair_key in cap_results.kl_divergences.keys():
                    model1, model2 = pair_key.split('_vs_')
                    all_models.add(model1)
                    all_models.add(model2)
        
        all_models = sorted(list(all_models))
        n_models = len(all_models)
        
        if n_models < 2:
            ax.text(0.5, 0.5, 'Insufficient model data for network', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Position models in circle
        angles = np.linspace(0, 2*np.pi, n_models, endpoint=False)
        positions = {model: (np.cos(angle), np.sin(angle)) for model, angle in zip(all_models, angles)}
        
        # Draw connections based on similarity (inverse of KL divergence)
        for cap_results in hybrid_detailed.values():
            if hasattr(cap_results, 'kl_divergences'):
                for pair_key, kl_div in cap_results.kl_divergences.items():
                    model1, model2 = pair_key.split('_vs_')
                    if model1 in positions and model2 in positions:
                        pos1, pos2 = positions[model1], positions[model2]
                        
                        # Line thickness based on similarity (inverse of KL divergence)
                        similarity = np.exp(-kl_div)  # Convert to similarity
                        width = max(0.1, similarity * 3)
                        alpha = max(0.1, similarity)
                        
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               'k-', linewidth=width, alpha=alpha)
        
        # Draw model nodes
        for model, (x, y) in positions.items():
            color = self.model_colors.get(model, 'gray')
            ax.scatter(x, y, s=500, c=color, alpha=0.8, edgecolors='black', linewidth=2)
            ax.annotate(self._clean_model_name(model), (x, y), 
                       xytext=(0, 0), textcoords='offset points', 
                       ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean model names for display"""
        name_mapping = {
            'openai/gpt-5-chat': 'GPT-5',
            'anthropic/claude-sonnet-4': 'Claude-4',
            'google/gemini-2.5-pro': 'Gemini-2.5',
            'anthropic/claude-3.5-sonnet': 'Claude-3.5',
            'openai/gpt-4-turbo': 'GPT-4-Turbo',
            'openai/gpt-oss-120b': 'GPT-OSS'
        }
        
        return name_mapping.get(model_name, model_name.split('/')[-1])
    
    def create_capability_kl_comparison(self, 
                                      hybrid_detailed: Dict[str, Any],
                                      title: str = "KL Divergence by Capability") -> str:
        """Create comparison visualization across all capabilities"""
        
        print(f"🎨 Creating capability KL comparison...")
        
        capabilities = list(hybrid_detailed.keys())
        n_caps = len(capabilities)
        
        if n_caps == 0:
            print("  ⚠️  No capability data found")
            return None
        
        # Create subplots for each capability
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, capability in enumerate(capabilities):
            if i >= len(axes):
                break
                
            ax = axes[i]
            cap_results = hybrid_detailed[capability]
            
            if hasattr(cap_results, 'kl_divergences') and cap_results.kl_divergences:
                # Create heatmap for this capability
                self._create_mini_kl_heatmap(ax, cap_results, capability)
            else:
                ax.text(0.5, 0.5, f'No KL data\nfor {capability}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(capability.replace('_', ' ').title())
        
        # Hide unused subplots
        for i in range(len(capabilities), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / "capability_kl_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Saved capability KL comparison: {save_path}")
        return str(save_path)
    
    def _create_mini_kl_heatmap(self, ax, cap_results, capability):
        """Create small heatmap for capability comparison"""
        
        kl_divergences = cap_results.kl_divergences
        
        # Extract models
        models = set()
        for pair_key in kl_divergences.keys():
            model1, model2 = pair_key.split('_vs_')
            models.add(model1)
            models.add(model2)
        
        models = sorted(list(models))
        n_models = len(models)
        
        if n_models < 2:
            ax.text(0.5, 0.5, 'Insufficient\nmodel data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(capability.replace('_', ' ').title())
            return
        
        # Create matrix
        kl_matrix = np.zeros((n_models, n_models))
        model_to_idx = {model: i for i, model in enumerate(models)}
        
        for pair_key, kl_value in kl_divergences.items():
            model1, model2 = pair_key.split('_vs_')
            i, j = model_to_idx[model1], model_to_idx[model2]
            kl_matrix[i, j] = kl_value
            kl_matrix[j, i] = kl_value
        
        # Create heatmap
        im = ax.imshow(kl_matrix, cmap='RdYlBu_r', aspect='equal')
        ax.set_title(capability.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels([self._clean_model_name(m) for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([self._clean_model_name(m) for m in models], fontsize=8)
        
        # Add average KL divergence as text
        avg_kl = np.mean([kl_divergences[k] for k in kl_divergences.keys()])
        ax.text(0.02, 0.98, f'Avg KL: {avg_kl:.3f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=10, fontweight='bold')


def generate_all_kl_visualizations(analysis_results: Dict[str, Any], 
                                 output_dir: str = "results/visualizations") -> List[str]:
    """
    Generate all KL divergence visualizations for the experiment results.
    
    Args:
        analysis_results: Complete analysis results from comprehensive_analysis
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to generated visualization files
    """
    
    print("\n🎨 Generating Advanced KL Divergence Visualizations")
    print("=" * 60)
    
    viz_suite = KLVisualizationSuite(output_dir)
    generated_files = []
    
    # 1. Main hybrid convergence dashboard
    dashboard_path = viz_suite.create_hybrid_convergence_dashboard(
        analysis_results, 
        title="Universal Alignment Patterns: Revolutionary KL Divergence Analysis"
    )
    if dashboard_path:
        generated_files.append(dashboard_path)
    
    # 2. Individual capability KL heatmaps
    hybrid_detailed = analysis_results.get('hybrid_results_detailed', {})
    for capability, cap_results in hybrid_detailed.items():
        if hasattr(cap_results, 'kl_divergences') and cap_results.kl_divergences:
            heatmap_path = viz_suite.create_kl_divergence_heatmap(
                {
                    'kl_divergences': cap_results.kl_divergences,
                    'jensen_shannon_distances': cap_results.jensen_shannon_distances
                },
                capability.replace('_', ' ').title()
            )
            if heatmap_path:
                generated_files.append(heatmap_path)
    
    # 3. Capability comparison
    if hybrid_detailed:
        comparison_path = viz_suite.create_capability_kl_comparison(
            hybrid_detailed,
            "KL Divergence Patterns Across Alignment Capabilities"
        )
        if comparison_path:
            generated_files.append(comparison_path)
    
    print(f"\n✅ Generated {len(generated_files)} advanced visualizations:")
    for path in generated_files:
        print(f"  📊 {Path(path).name}")
    
    return generated_files


if __name__ == "__main__":
    # Test with mock data
    print("🧪 Testing KL Visualization Suite with mock data...")
    
    mock_analysis = {
        'overall_convergence': 0.72,
        'semantic_convergence': 0.68,
        'distributional_convergence': 0.76,
        'capability_results': {
            'truthfulness': {
                'semantic_convergence': 0.71,
                'distributional_convergence': 0.83,
                'hybrid_convergence': 0.78,
                'confidence_level': 0.85
            },
            'safety_boundaries': {
                'semantic_convergence': 0.65,
                'distributional_convergence': 0.70,
                'hybrid_convergence': 0.68,
                'confidence_level': 0.72
            }
        }
    }
    
    generated = generate_all_kl_visualizations(mock_analysis, "test_visualizations")
    print(f"✅ Test completed: {len(generated)} files generated")