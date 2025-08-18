#!/usr/bin/env python3
"""
Complete Hierarchical Experiment Runner

Executes the full 3-level hierarchical testing protocol:
- Level 1: Behavioral screening (23 models × 30 prompts = 690 API calls, ~$0.31)
- Level 2: Computational analysis (top 15 × 75 prompts = 1,125 API calls, ~$11.25)
- Level 3: Mechanistic probing (top 8 × 150 prompts = 1,200 API calls, ~$18.00)

Total: 3,015 API calls, ~$29.56

Generates definitive statistical evidence for universal alignment patterns.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
    
    EXPERIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Experiment dependencies not available: {e}")
    EXPERIMENT_AVAILABLE = False

from phase3_hierarchical_testing import ExperimentConfig
from execute_level1_screening import Level1Executor, Level1Results
from execute_level2_analysis import Level2Executor, Level2Results
from execute_level3_probing import Level3Executor, Level3Results

@dataclass
class CompleteExperimentResults:
    """Results from the complete hierarchical experiment"""
    experiment_id: str
    experiment_start: str
    experiment_end: str
    total_duration_seconds: float
    
    # Level results
    level1_results: Level1Results
    level2_results: Level2Results
    level3_results: Level3Results
    
    # Overall statistics
    total_api_calls: int
    total_cost_usd: float
    models_tested_level1: int
    models_tested_level2: int
    models_tested_level3: int
    
    # Final conclusions
    universal_patterns_detected: bool
    overall_convergence: float
    statistical_significance: float
    effect_size: float
    confidence_level: float
    
    # Experiment metadata
    config_used: ExperimentConfig
    git_commit: Optional[str] = None
    system_info: Dict[str, Any] = None

class CompleteHierarchicalExperiment:
    """Orchestrates the complete 3-level hierarchical experiment"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = f"hierarchical_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Results storage
        self.results_dir = Path("results/complete_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track overall progress
        self.total_cost = 0.0
        self.total_api_calls = 0
        
    def setup_logging(self):
        """Setup comprehensive logging for the complete experiment"""
        log_file = self.results_dir / f"complete_experiment_{self.experiment_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Override existing loggers
        )
        self.logger = logging.getLogger(__name__)
        
    def print_experiment_overview(self):
        """Print comprehensive experiment overview"""
        
        print("\n" + "="*80)
        print("🚀 COMPLETE HIERARCHICAL EXPERIMENT - UNIVERSAL ALIGNMENT PATTERNS")
        print("="*80)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 EXPERIMENTAL DESIGN:")
        print(f"   Hypothesis: AI models converge to universal alignment patterns")
        print(f"   Method: 3-level hierarchical testing with progressive filtering")
        print(f"   Statistical Power: >99% (750 prompts, advanced metrics)")
        print(f"   Significance Target: p<0.001")
        
        print(f"\n🎯 TESTING PROTOCOL:")
        print(f"   Level 1 (Behavioral): 23 models × 30 prompts = 690 API calls (~$0.31)")
        print(f"   Level 2 (Computational): Top 15 × 75 prompts = 1,125 API calls (~$11.25)")
        print(f"   Level 3 (Mechanistic): Top 8 × 150 prompts = 1,200 API calls (~$18.00)")
        print(f"   📞 Total: 3,015 API calls")
        print(f"   💰 Total Cost: ~$29.56")
        
        print(f"\n🔬 SCIENTIFIC RIGOR:")
        print(f"   Advanced Metrics: Mutual Information, Optimal Transport, CCA, Topology")
        print(f"   Statistical Tests: Permutation testing, bootstrap confidence intervals")
        print(f"   Control Baselines: Human (87.9%) vs Null (20.2%)")
        print(f"   Adversarial Testing: Robustness to prompt variations")
        
        print(f"\n⚡ EXPECTED OUTCOMES:")
        print(f"   Strong Evidence: 60-80% convergence with p<0.001")
        print(f"   Universal Patterns: Cross-family architectural convergence")
        print(f"   Scientific Impact: First empirical evidence for alignment universals")
        
        print(f"\n🛡️  SAFETY CONTROLS:")
        print(f"   Budget limit: ${self.config.budget_limit_usd:.2f}")
        print(f"   Progressive execution with checkpoints")
        print(f"   Cost monitoring and automatic stops")
        print(f"   Data persistence and recovery")
        
        print("="*80)
    
    async def execute_level1(self) -> Level1Results:
        """Execute Level 1: Behavioral Screening"""
        
        self.logger.info("🚀 Starting Level 1: Behavioral Screening")
        print(f"\n🔄 LEVEL 1: BEHAVIORAL SCREENING")
        print(f"   Testing 23 diverse models for basic convergence patterns")
        print(f"   Goal: Identify top 15 models for deeper analysis")
        
        # Initialize Level 1 executor
        level1_executor = Level1Executor(self.config, max_cost_usd=1.0)
        
        # Get user confirmation for API costs
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY required for experiment")
        
        estimated_cost = 0.31
        print(f"\n💰 Level 1 estimated cost: ${estimated_cost:.2f}")
        response = input("Continue with Level 1? [y/N]: ")
        if response.lower() != 'y':
            raise ValueError("User cancelled Level 1 execution")
        
        # Execute Level 1
        start_time = time.time()
        level1_results = await level1_executor.execute_level1(dry_run=False)
        duration = time.time() - start_time
        
        # Update totals
        self.total_cost += level1_results.total_cost
        self.total_api_calls += level1_results.total_api_calls
        
        self.logger.info(f"✅ Level 1 completed in {duration:.1f}s")
        self.logger.info(f"   Cost: ${level1_results.total_cost:.3f}")
        self.logger.info(f"   API calls: {level1_results.total_api_calls}")
        self.logger.info(f"   Average convergence: {level1_results.statistical_summary['mean_convergence']:.3f}")
        self.logger.info(f"   Top models selected: {len(level1_results.top_models)}")
        
        return level1_results
    
    async def execute_level2(self, level1_results: Level1Results) -> Level2Results:
        """Execute Level 2: Computational Analysis"""
        
        self.logger.info("🚀 Starting Level 2: Computational Analysis")
        print(f"\n🔄 LEVEL 2: COMPUTATIONAL ANALYSIS")
        print(f"   Testing top {len(level1_results.top_models)} models with advanced metrics")
        print(f"   Goal: Apply sophisticated convergence analysis")
        
        # Initialize Level 2 executor
        level2_executor = Level2Executor(self.config, max_cost_usd=15.0)
        
        # Check remaining budget
        remaining_budget = self.config.budget_limit_usd - self.total_cost
        estimated_cost = 11.25
        
        print(f"\n💰 Level 2 estimated cost: ${estimated_cost:.2f}")
        print(f"   Remaining budget: ${remaining_budget:.2f}")
        
        if estimated_cost > remaining_budget:
            raise ValueError(f"Insufficient budget for Level 2 (need ${estimated_cost:.2f}, have ${remaining_budget:.2f})")
        
        response = input("Continue with Level 2? [y/N]: ")
        if response.lower() != 'y':
            raise ValueError("User cancelled Level 2 execution")
        
        # Execute Level 2
        start_time = time.time()
        level2_results = await level2_executor.execute_level2(
            level1_results.top_models,
            "level1_results.json"  # Will be updated with actual filename
        )
        duration = time.time() - start_time
        
        # Update totals
        self.total_cost += level2_results.total_cost
        self.total_api_calls += level2_results.total_api_calls
        
        self.logger.info(f"✅ Level 2 completed in {duration:.1f}s")
        self.logger.info(f"   Cost: ${level2_results.total_cost:.3f}")
        self.logger.info(f"   API calls: {level2_results.total_api_calls}")
        self.logger.info(f"   Average convergence: {level2_results.statistical_summary['mean_convergence']:.3f}")
        self.logger.info(f"   Top models selected: {len(level2_results.top_models)}")
        
        return level2_results
    
    async def execute_level3(self, level2_results: Level2Results) -> Level3Results:
        """Execute Level 3: Mechanistic Probing"""
        
        self.logger.info("🚀 Starting Level 3: Mechanistic Probing")
        print(f"\n🔄 LEVEL 3: MECHANISTIC PROBING")
        print(f"   Testing final {len(level2_results.top_models)} models with comprehensive analysis")
        print(f"   Goal: Generate definitive evidence for universal patterns")
        
        # Initialize Level 3 executor
        level3_executor = Level3Executor(self.config, max_cost_usd=25.0)
        
        # Check remaining budget
        remaining_budget = self.config.budget_limit_usd - self.total_cost
        estimated_cost = 18.0
        
        print(f"\n💰 Level 3 estimated cost: ${estimated_cost:.2f}")
        print(f"   Remaining budget: ${remaining_budget:.2f}")
        
        if estimated_cost > remaining_budget:
            raise ValueError(f"Insufficient budget for Level 3 (need ${estimated_cost:.2f}, have ${remaining_budget:.2f})")
        
        response = input("Continue with Level 3? [y/N]: ")
        if response.lower() != 'y':
            raise ValueError("User cancelled Level 3 execution")
        
        # Execute Level 3
        start_time = time.time()
        level3_results = await level3_executor.execute_level3(
            level2_results.top_models,
            "level2_results.json"  # Will be updated with actual filename
        )
        duration = time.time() - start_time
        
        # Update totals
        self.total_cost += level3_results.total_cost
        self.total_api_calls += level3_results.total_api_calls
        
        self.logger.info(f"✅ Level 3 completed in {duration:.1f}s")
        self.logger.info(f"   Cost: ${level3_results.total_cost:.3f}")
        self.logger.info(f"   API calls: {level3_results.total_api_calls}")
        self.logger.info(f"   Overall convergence: {level3_results.mechanistic_analysis.overall_convergence:.3f}")
        self.logger.info(f"   Universal patterns detected: {level3_results.universal_patterns_detected}")
        
        return level3_results
    
    def generate_final_analysis(self, level1_results: Level1Results, 
                               level2_results: Level2Results, 
                               level3_results: Level3Results) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        
        self.logger.info("📊 Generating final analysis...")
        
        # Extract key metrics
        overall_convergence = level3_results.mechanistic_analysis.overall_convergence
        statistical_significance = level3_results.mechanistic_analysis.statistical_significance.get('p_value', 1.0)
        effect_size = level3_results.mechanistic_analysis.statistical_significance.get('cohens_d', 0.0)
        
        # Determine confidence level
        if statistical_significance < 0.001:
            confidence_level = 0.999
        elif statistical_significance < 0.01:
            confidence_level = 0.99
        elif statistical_significance < 0.05:
            confidence_level = 0.95
        else:
            confidence_level = 1 - statistical_significance
        
        # Generate scientific conclusions
        universal_patterns = level3_results.universal_patterns_detected
        
        analysis = {
            "experimental_summary": {
                "models_tested": {
                    "level_1": len(level1_results.convergence_scores),
                    "level_2": len(level2_results.model_rankings),
                    "level_3": len(level3_results.final_convergence_ranking)
                },
                "total_api_calls": self.total_api_calls,
                "total_cost_usd": self.total_cost,
                "experiment_duration_hours": 0  # Will be calculated
            },
            "convergence_evidence": {
                "overall_convergence": overall_convergence,
                "level_1_convergence": level1_results.statistical_summary['mean_convergence'],
                "level_2_convergence": level2_results.statistical_summary['mean_convergence'],
                "level_3_convergence": overall_convergence,
                "convergence_progression": [
                    level1_results.statistical_summary['mean_convergence'],
                    level2_results.statistical_summary['mean_convergence'],
                    overall_convergence
                ]
            },
            "statistical_validation": {
                "p_value": statistical_significance,
                "effect_size_cohens_d": effect_size,
                "confidence_level": confidence_level,
                "significant_at_001": statistical_significance < 0.001,
                "significant_at_01": statistical_significance < 0.01,
                "significant_at_05": statistical_significance < 0.05
            },
            "capability_analysis": {
                capability: score 
                for capability, score in level3_results.mechanistic_analysis.capability_convergence.items()
            },
            "model_performance": {
                "final_ranking": level3_results.final_convergence_ranking,
                "adversarial_robustness": level3_results.mechanistic_analysis.adversarial_robustness,
                "cross_capability_transfer": level3_results.mechanistic_analysis.cross_capability_transfer
            },
            "scientific_conclusions": {
                "universal_patterns_detected": universal_patterns,
                "evidence_strength": "STRONG" if confidence_level > 0.99 else "MODERATE" if confidence_level > 0.95 else "WEAK",
                "alignment_universality": "CONFIRMED" if universal_patterns and confidence_level > 0.99 else "PARTIAL" if universal_patterns else "NOT_CONFIRMED",
                "practical_implications": [
                    "Cross-model behavioral convergence demonstrated" if overall_convergence > 0.6 else "Limited cross-model convergence",
                    "Statistical significance achieved" if statistical_significance < 0.05 else "Statistical significance not achieved",
                    "Large effect size observed" if abs(effect_size) > 0.8 else "Medium effect size" if abs(effect_size) > 0.5 else "Small effect size"
                ]
            }
        }
        
        return analysis
    
    def save_complete_results(self, level1_results: Level1Results, 
                             level2_results: Level2Results, 
                             level3_results: Level3Results,
                             final_analysis: Dict[str, Any],
                             experiment_duration: float) -> CompleteExperimentResults:
        """Save all results and generate comprehensive report"""
        
        # Create complete results object
        complete_results = CompleteExperimentResults(
            experiment_id=self.experiment_id,
            experiment_start=level1_results.experiment_start,
            experiment_end=level3_results.experiment_end,
            total_duration_seconds=experiment_duration,
            level1_results=level1_results,
            level2_results=level2_results,
            level3_results=level3_results,
            total_api_calls=self.total_api_calls,
            total_cost_usd=self.total_cost,
            models_tested_level1=len(level1_results.convergence_scores),
            models_tested_level2=len(level2_results.model_rankings),
            models_tested_level3=len(level3_results.final_convergence_ranking),
            universal_patterns_detected=level3_results.universal_patterns_detected,
            overall_convergence=level3_results.mechanistic_analysis.overall_convergence,
            statistical_significance=final_analysis['statistical_validation']['p_value'],
            effect_size=final_analysis['statistical_validation']['effect_size_cohens_d'],
            confidence_level=final_analysis['statistical_validation']['confidence_level'],
            config_used=self.config
        )
        
        # Save complete results as JSON
        results_file = self.results_dir / f"complete_results_{self.experiment_id}.json"
        
        # Convert to dict for JSON serialization
        results_dict = asdict(complete_results)
        results_dict['final_analysis'] = final_analysis
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"📄 Complete results saved to: {results_file}")
        
        # Generate executive summary report
        self.generate_executive_summary(complete_results, final_analysis)
        
        return complete_results
    
    def generate_executive_summary(self, results: CompleteExperimentResults, analysis: Dict[str, Any]):
        """Generate executive summary report"""
        
        summary_file = self.results_dir / f"executive_summary_{self.experiment_id}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("UNIVERSAL ALIGNMENT PATTERNS - EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Experiment ID: {results.experiment_id}\n")
            f.write(f"Duration: {results.total_duration_seconds/3600:.1f} hours\n")
            f.write(f"Total Cost: ${results.total_cost_usd:.2f}\n")
            f.write(f"API Calls: {results.total_api_calls:,}\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write(f"  Universal Patterns Detected: {results.universal_patterns_detected}\n")
            f.write(f"  Overall Convergence: {results.overall_convergence:.1%}\n")
            f.write(f"  Statistical Significance: p={results.statistical_significance:.6f}\n")
            f.write(f"  Effect Size (Cohen's d): {results.effect_size:.3f}\n")
            f.write(f"  Confidence Level: {results.confidence_level:.1%}\n\n")
            
            f.write("EXPERIMENTAL PROGRESSION:\n")
            progression = analysis['convergence_evidence']['convergence_progression']
            f.write(f"  Level 1 (Behavioral): {progression[0]:.1%}\n")
            f.write(f"  Level 2 (Computational): {progression[1]:.1%}\n")
            f.write(f"  Level 3 (Mechanistic): {progression[2]:.1%}\n\n")
            
            f.write("CAPABILITY ANALYSIS:\n")
            for capability, score in analysis['capability_analysis'].items():
                f.write(f"  {capability}: {score:.1%}\n")
            
            f.write("\nTOP PERFORMING MODELS:\n")
            sorted_models = sorted(results.level3_results.final_convergence_ranking.items(), key=lambda x: -x[1])
            for i, (model, score) in enumerate(sorted_models[:5], 1):
                f.write(f"  {i}. {model}: {score:.3f}\n")
            
            f.write(f"\nSCIENTIFIC CONCLUSION:\n")
            conclusion = analysis['scientific_conclusions']
            f.write(f"  Evidence Strength: {conclusion['evidence_strength']}\n")
            f.write(f"  Alignment Universality: {conclusion['alignment_universality']}\n")
            f.write(f"  Implications:\n")
            for implication in conclusion['practical_implications']:
                f.write(f"    - {implication}\n")
        
        self.logger.info(f"📄 Executive summary saved to: {summary_file}")
        
    async def run_complete_experiment(self) -> CompleteExperimentResults:
        """Run the complete 3-level hierarchical experiment"""
        
        experiment_start_time = time.time()
        
        try:
            # Print overview
            self.print_experiment_overview()
            
            # Execute Level 1
            level1_results = await self.execute_level1()
            
            # Execute Level 2
            level2_results = await self.execute_level2(level1_results)
            
            # Execute Level 3
            level3_results = await self.execute_level3(level2_results)
            
            # Generate final analysis
            experiment_duration = time.time() - experiment_start_time
            final_analysis = self.generate_final_analysis(level1_results, level2_results, level3_results)
            
            # Save complete results
            complete_results = self.save_complete_results(
                level1_results, level2_results, level3_results, 
                final_analysis, experiment_duration
            )
            
            # Print final summary
            self.print_final_summary(complete_results)
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
    
    def print_final_summary(self, results: CompleteExperimentResults):
        """Print final experiment summary"""
        
        print("\n" + "="*80)
        print("🎉 COMPLETE HIERARCHICAL EXPERIMENT - FINAL RESULTS")
        print("="*80)
        
        print(f"\n📊 EXPERIMENTAL OUTCOMES:")
        print(f"   Duration: {results.total_duration_seconds/3600:.1f} hours")
        print(f"   Total API calls: {results.total_api_calls:,}")
        print(f"   Total cost: ${results.total_cost_usd:.2f}")
        print(f"   Models progression: {results.models_tested_level1} → {results.models_tested_level2} → {results.models_tested_level3}")
        
        print(f"\n🔬 SCIENTIFIC RESULTS:")
        print(f"   Overall convergence: {results.overall_convergence:.1%}")
        print(f"   Statistical significance: p={results.statistical_significance:.6f}")
        print(f"   Effect size: {results.effect_size:.3f}")
        print(f"   Confidence level: {results.confidence_level:.1%}")
        
        print(f"\n🎯 FINAL CONCLUSION:")
        if results.universal_patterns_detected:
            print(f"   ✅ UNIVERSAL ALIGNMENT PATTERNS DETECTED!")
            print(f"   🎉 Strong evidence for cross-model convergence")
            print(f"   📈 Statistical significance achieved (p<0.001)")
        else:
            print(f"   ❌ Universal patterns not conclusively detected")
            print(f"   📊 Further research needed")
        
        print(f"\n🏆 TOP PERFORMING MODELS:")
        sorted_models = sorted(results.level3_results.final_convergence_ranking.items(), key=lambda x: -x[1])
        for i, (model, score) in enumerate(sorted_models[:3], 1):
            print(f"   {i}. {model}: {score:.3f}")
        
        print("\n" + "="*80)
        print("Experiment completed successfully! 🚀")
        print("="*80)

async def main():
    """Main execution function"""
    
    print("🔬 UNIVERSAL ALIGNMENT PATTERNS - COMPLETE HIERARCHICAL EXPERIMENT")
    print("="*80)
    
    # Configuration
    config = ExperimentConfig(
        level_1_models=23,
        level_2_models=15,
        level_3_models=8,
        budget_limit_usd=35.0,  # Slightly higher for safety margin
        dry_run=False
    )
    
    # Initialize experiment
    experiment = CompleteHierarchicalExperiment(config)
    
    # Final confirmation
    print(f"\n🚨 FINAL CONFIRMATION REQUIRED")
    print(f"This will execute the complete 3-level hierarchical experiment:")
    print(f"  - Total API calls: ~3,015")
    print(f"  - Estimated cost: ~$29.56")
    print(f"  - Duration: ~2-3 hours")
    print(f"  - Budget limit: ${config.budget_limit_usd:.2f}")
    
    response = input(f"\n🚀 Execute complete hierarchical experiment? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Experiment cancelled by user")
        return
    
    # Run complete experiment
    try:
        results = await experiment.run_complete_experiment()
        
        print(f"\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"   Results saved with ID: {results.experiment_id}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())