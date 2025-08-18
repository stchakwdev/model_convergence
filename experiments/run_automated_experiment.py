#!/usr/bin/env python3
"""
Automated Hierarchical Experiment Runner - No Interactive Prompts

Executes the complete 3-level experiment automatically without user confirmations.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from phase3_hierarchical_testing import ExperimentConfig
from run_complete_hierarchical_experiment import CompleteHierarchicalExperiment

class AutomatedExperiment(CompleteHierarchicalExperiment):
    """Non-interactive version of the complete experiment"""
    
    async def execute_level1(self):
        """Execute Level 1 without user confirmation"""
        
        self.logger.info("🚀 Starting Level 1: Behavioral Screening (AUTOMATED)")
        print(f"\n🔄 LEVEL 1: BEHAVIORAL SCREENING (AUTOMATED)")
        print(f"   Testing 23 diverse models for basic convergence patterns")
        print(f"   Goal: Identify top 15 models for deeper analysis")
        
        # Check API key
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY required for experiment")
        
        # Initialize Level 1 executor
        from execute_level1_screening import Level1Executor
        level1_executor = Level1Executor(self.config, max_cost_usd=1.0)
        
        # Monkey patch the input to return 'y' automatically
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt: 'y'
        
        try:
            # Execute Level 1
            level1_results = await level1_executor.execute_level1(dry_run=False)
            
            # Update totals
            self.total_cost += level1_results.total_cost
            self.total_api_calls += level1_results.total_api_calls
            
            self.logger.info(f"✅ Level 1 completed automatically")
            self.logger.info(f"   Cost: ${level1_results.total_cost:.3f}")
            self.logger.info(f"   API calls: {level1_results.total_api_calls}")
            self.logger.info(f"   Average convergence: {level1_results.statistical_summary['mean_convergence']:.3f}")
            self.logger.info(f"   Top models selected: {len(level1_results.top_models)}")
            
            return level1_results
            
        finally:
            # Restore original input
            builtins.input = original_input
    
    async def execute_level2(self, level1_results):
        """Execute Level 2 without user confirmation"""
        
        self.logger.info("🚀 Starting Level 2: Computational Analysis (AUTOMATED)")
        print(f"\n🔄 LEVEL 2: COMPUTATIONAL ANALYSIS (AUTOMATED)")
        print(f"   Testing top {len(level1_results.top_models)} models with advanced metrics")
        print(f"   Goal: Apply sophisticated convergence analysis")
        
        # Initialize Level 2 executor
        from execute_level2_analysis import Level2Executor
        level2_executor = Level2Executor(self.config, max_cost_usd=15.0)
        
        # Check remaining budget
        remaining_budget = self.config.budget_limit_usd - self.total_cost
        estimated_cost = 11.25
        
        print(f"\n💰 Level 2 estimated cost: ${estimated_cost:.2f}")
        print(f"   Remaining budget: ${remaining_budget:.2f}")
        
        if estimated_cost > remaining_budget:
            raise ValueError(f"Insufficient budget for Level 2 (need ${estimated_cost:.2f}, have ${remaining_budget:.2f})")
        
        print("   Proceeding with Level 2 automatically...")
        
        # Execute Level 2
        level2_results = await level2_executor.execute_level2(
            level1_results.top_models,
            "level1_results.json"
        )
        
        # Update totals
        self.total_cost += level2_results.total_cost
        self.total_api_calls += level2_results.total_api_calls
        
        self.logger.info(f"✅ Level 2 completed automatically")
        self.logger.info(f"   Cost: ${level2_results.total_cost:.3f}")
        self.logger.info(f"   API calls: {level2_results.total_api_calls}")
        self.logger.info(f"   Average convergence: {level2_results.statistical_summary['mean_convergence']:.3f}")
        self.logger.info(f"   Top models selected: {len(level2_results.top_models)}")
        
        return level2_results
    
    async def execute_level3(self, level2_results):
        """Execute Level 3 without user confirmation"""
        
        self.logger.info("🚀 Starting Level 3: Mechanistic Probing (AUTOMATED)")
        print(f"\n🔄 LEVEL 3: MECHANISTIC PROBING (AUTOMATED)")
        print(f"   Testing final {len(level2_results.top_models)} models with comprehensive analysis")
        print(f"   Goal: Generate definitive evidence for universal patterns")
        
        # Initialize Level 3 executor
        from execute_level3_probing import Level3Executor
        level3_executor = Level3Executor(self.config, max_cost_usd=25.0)
        
        # Check remaining budget
        remaining_budget = self.config.budget_limit_usd - self.total_cost
        estimated_cost = 18.0
        
        print(f"\n💰 Level 3 estimated cost: ${estimated_cost:.2f}")
        print(f"   Remaining budget: ${remaining_budget:.2f}")
        
        if estimated_cost > remaining_budget:
            raise ValueError(f"Insufficient budget for Level 3 (need ${estimated_cost:.2f}, have ${remaining_budget:.2f})")
        
        print("   Proceeding with Level 3 automatically...")
        
        # Execute Level 3
        level3_results = await level3_executor.execute_level3(
            level2_results.top_models,
            "level2_results.json"
        )
        
        # Update totals
        self.total_cost += level3_results.total_cost
        self.total_api_calls += level3_results.total_api_calls
        
        self.logger.info(f"✅ Level 3 completed automatically")
        self.logger.info(f"   Cost: ${level3_results.total_cost:.3f}")
        self.logger.info(f"   API calls: {level3_results.total_api_calls}")
        self.logger.info(f"   Overall convergence: {level3_results.mechanistic_analysis.overall_convergence:.3f}")
        self.logger.info(f"   Universal patterns detected: {level3_results.universal_patterns_detected}")
        
        return level3_results

async def main():
    """Main execution function for automated experiment"""
    
    print("🔬 UNIVERSAL ALIGNMENT PATTERNS - AUTOMATED COMPLETE EXPERIMENT")
    print("="*80)
    
    # Configuration
    config = ExperimentConfig(
        level_1_models=23,
        level_2_models=15,
        level_3_models=8,
        budget_limit_usd=35.0,
        dry_run=False
    )
    
    # Initialize automated experiment
    experiment = AutomatedExperiment(config)
    
    print(f"\n🚀 EXECUTING COMPLETE HIERARCHICAL EXPERIMENT AUTOMATICALLY")
    print(f"  - Total API calls: ~3,015")
    print(f"  - Estimated cost: ~$29.56")
    print(f"  - Duration: ~2-3 hours")
    print(f"  - Budget limit: ${config.budget_limit_usd:.2f}")
    print(f"  - No user confirmations required")
    
    # Run complete experiment
    try:
        results = await experiment.run_complete_experiment()
        
        print(f"\n✅ AUTOMATED EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"   Results saved with ID: {results.experiment_id}")
        print(f"   Universal patterns detected: {results.universal_patterns_detected}")
        print(f"   Overall convergence: {results.overall_convergence:.1%}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ AUTOMATED EXPERIMENT FAILED: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())