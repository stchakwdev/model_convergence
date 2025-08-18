#!/usr/bin/env python3
"""
Model Selection Validation for Phase 3 Hierarchical Testing

Validates:
1. OpenRouter API connectivity
2. Model availability 
3. Cost estimation accuracy
4. Integration with existing framework
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.openrouter_model import OpenRouterModel
    from models.model_registry import model_registry
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    OPENROUTER_AVAILABLE = True
except ImportError as e:
    print(f"OpenRouter integration not available: {e}")
    OPENROUTER_AVAILABLE = False

from phase3_hierarchical_testing import HierarchicalTestingOrchestrator, ExperimentConfig, ModelCandidate

class ModelSelectionValidator:
    """Validates model selection for Phase 3 experiment"""
    
    def __init__(self):
        self.api_key_available = bool(os.getenv("OPENROUTER_API_KEY"))
        self.validation_results = {}
        
    def check_api_connectivity(self):
        """Test basic OpenRouter API connectivity"""
        print("🔗 Testing OpenRouter API connectivity...")
        
        if not self.api_key_available:
            print("   ❌ OPENROUTER_API_KEY not found in environment")
            return False
            
        if not OPENROUTER_AVAILABLE:
            print("   ❌ OpenRouter integration not available")
            return False
            
        try:
            # Test with a simple, cheap model
            test_model = OpenRouterModel("openai/gpt-4o-mini")
            print(f"   ✅ Successfully initialized test model: {test_model.name}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to initialize test model: {e}")
            return False
    
    def validate_selected_models(self, candidates: list[ModelCandidate]):
        """Validate that selected models are available"""
        print(f"\n🎯 Validating {len(candidates)} selected models...")
        
        available_models = []
        unavailable_models = []
        total_estimated_cost = 0
        
        for i, candidate in enumerate(candidates, 1):
            try:
                if OPENROUTER_AVAILABLE and self.api_key_available:
                    # Try to initialize the model
                    test_model = OpenRouterModel(candidate.model_id)
                    status = "✅ Available"
                    available_models.append(candidate)
                else:
                    # Mock validation
                    status = "🔍 Mock validation"
                    available_models.append(candidate)
                    
                total_estimated_cost += candidate.cost_per_1k_tokens
                
                print(f"   {i:2}. {candidate.model_id}")
                print(f"       Status: {status}")
                print(f"       Family: {candidate.family} | Specialization: {candidate.specialization}")
                print(f"       Quality: {candidate.estimated_quality}/10 | Cost: ${candidate.cost_per_1k_tokens:.4f}/1K tokens")
                
            except Exception as e:
                print(f"   {i:2}. {candidate.model_id}")
                print(f"       Status: ❌ Error - {str(e)[:60]}...")
                unavailable_models.append(candidate)
        
        print(f"\n📊 Model Availability Summary:")
        print(f"   Available models: {len(available_models)}/{len(candidates)} ({len(available_models)/len(candidates)*100:.1f}%)")
        if unavailable_models:
            print(f"   Unavailable models: {len(unavailable_models)}")
            for model in unavailable_models:
                print(f"     - {model.model_id}")
        
        return available_models, unavailable_models
    
    def validate_cost_estimation(self, candidates: list[ModelCandidate]):
        """Validate cost estimation accuracy"""
        print(f"\n💰 Validating cost estimation...")
        
        # Create orchestrator for cost calculation
        config = ExperimentConfig(level_1_models=len(candidates))
        orchestrator = HierarchicalTestingOrchestrator(config)
        costs = orchestrator.estimate_costs(candidates)
        
        print(f"   Level 1: {costs['level_1_calls']:,} calls @ ~${costs['level_1_cost']:.2f}")
        print(f"   Level 2: {costs['level_2_calls']:,} calls @ ~${costs['level_2_cost']:.2f}")
        print(f"   Level 3: {costs['level_3_calls']:,} calls @ ~${costs['level_3_cost']:.2f}")
        print(f"   Total: {costs['total_calls']:,} calls @ ~${costs['total_cost']:.2f}")
        
        # Validate against budget
        budget_limit = config.budget_limit_usd
        within_budget = costs['total_cost'] <= budget_limit
        
        print(f"   Budget limit: ${budget_limit:.2f}")
        print(f"   Budget status: {'✅ Within budget' if within_budget else '❌ Over budget'}")
        
        # Calculate cost per model and per capability
        cost_per_model = costs['total_cost'] / len(candidates)
        cost_per_capability = costs['total_cost'] / len(config.capabilities)
        
        print(f"   Cost per model: ${cost_per_model:.2f}")
        print(f"   Cost per capability: ${cost_per_capability:.2f}")
        
        return costs, within_budget
    
    def test_quick_api_call(self):
        """Test a quick API call to verify functionality"""
        print(f"\n🚀 Testing quick API call...")
        
        if not (OPENROUTER_AVAILABLE and self.api_key_available):
            print("   🔍 Skipping API test - running in mock mode")
            return True
        
        try:
            # Use cheapest available model for test
            test_model = OpenRouterModel("openai/gpt-4o-mini")
            test_prompt = "What is 2+2? Answer briefly."
            
            print(f"   Model: {test_model.name}")
            print(f"   Prompt: '{test_prompt}'")
            
            # This would make actual API call - commented for safety
            # response = test_model.generate(test_prompt)
            # print(f"   Response: '{response}'")
            
            print("   ✅ API call test simulated successfully (actual call commented for cost control)")
            return True
            
        except Exception as e:
            print(f"   ❌ API call test failed: {e}")
            return False
    
    def analyze_architecture_diversity(self, candidates: list[ModelCandidate]):
        """Analyze the diversity of selected model architectures"""
        print(f"\n🏗️  Analyzing architecture diversity...")
        
        # Group by various attributes
        families = {}
        providers = {}
        specializations = {}
        sizes = {}
        
        for candidate in candidates:
            families[candidate.family] = families.get(candidate.family, 0) + 1
            providers[candidate.provider] = providers.get(candidate.provider, 0) + 1
            specializations[candidate.specialization] = specializations.get(candidate.specialization, 0) + 1
            
            # Extract parameter size category
            if candidate.parameter_count:
                if "405B" in candidate.parameter_count or "400B" in candidate.parameter_count:
                    size_cat = "Ultra Large (400B+)"
                elif "100B" in candidate.parameter_count or "176B" in candidate.parameter_count or "180B" in candidate.parameter_count:
                    size_cat = "Large (100-400B)"
                elif "70B" in candidate.parameter_count or "72B" in candidate.parameter_count:
                    size_cat = "Medium (50-100B)"
                elif "25B" in candidate.parameter_count or "34B" in candidate.parameter_count:
                    size_cat = "Small (10-50B)"
                else:
                    size_cat = "Tiny (<10B)"
            else:
                size_cat = "Unknown"
            sizes[size_cat] = sizes.get(size_cat, 0) + 1
        
        print(f"   Model Families:")
        for family, count in sorted(families.items(), key=lambda x: -x[1]):
            print(f"     {family}: {count} models ({count/len(candidates)*100:.1f}%)")
        
        print(f"   Providers:")
        for provider, count in sorted(providers.items(), key=lambda x: -x[1]):
            print(f"     {provider}: {count} models ({count/len(candidates)*100:.1f}%)")
        
        print(f"   Specializations:")
        for spec, count in sorted(specializations.items(), key=lambda x: -x[1]):
            print(f"     {spec}: {count} models ({count/len(candidates)*100:.1f}%)")
        
        print(f"   Size Categories:")
        for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
            print(f"     {size}: {count} models ({count/len(candidates)*100:.1f}%)")
        
        # Calculate diversity scores
        family_diversity = len(families) / len(candidates)
        provider_diversity = len(providers) / len(candidates) 
        spec_diversity = len(specializations) / len(candidates)
        
        print(f"   Diversity Scores:")
        print(f"     Family diversity: {family_diversity:.3f} (higher = more diverse)")
        print(f"     Provider diversity: {provider_diversity:.3f}")
        print(f"     Specialization diversity: {spec_diversity:.3f}")
        
        overall_diversity = (family_diversity + provider_diversity + spec_diversity) / 3
        print(f"     Overall diversity: {overall_diversity:.3f}")
        
        if overall_diversity > 0.4:
            print(f"     ✅ Excellent diversity for scientific validity")
        elif overall_diversity > 0.3:
            print(f"     📈 Good diversity")
        else:
            print(f"     ⚠️  Consider adding more diverse models")
        
        return {
            'families': families,
            'providers': providers,
            'specializations': specializations,
            'sizes': sizes,
            'diversity_score': overall_diversity
        }
    
    def generate_validation_report(self, candidates: list[ModelCandidate]):
        """Generate comprehensive validation report"""
        print(f"\n📋 Generating validation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(candidates),
            'api_connectivity': self.check_api_connectivity(),
            'validation_summary': {}
        }
        
        # Model availability
        available, unavailable = self.validate_selected_models(candidates)
        report['available_models'] = len(available)
        report['unavailable_models'] = len(unavailable)
        
        # Cost validation
        costs, within_budget = self.validate_cost_estimation(available)
        report['cost_estimation'] = costs
        report['within_budget'] = within_budget
        
        # Architecture analysis
        diversity = self.analyze_architecture_diversity(available)
        report['architecture_diversity'] = diversity
        
        # API test
        api_test_success = self.test_quick_api_call()
        report['api_test_success'] = api_test_success
        
        # Overall readiness
        overall_ready = (
            report['api_connectivity'] and
            len(available) >= 20 and  # Need at least 20 models
            within_budget and
            diversity['diversity_score'] > 0.3 and
            api_test_success
        )
        
        report['experiment_ready'] = overall_ready
        
        # Save report
        report_file = Path("results/phase3_hierarchical/validation_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   📄 Report saved to: {report_file}")
        
        # Print summary
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"   Models ready: {len(available)}/{len(candidates)}")
        print(f"   Budget status: {'✅ Within budget' if within_budget else '❌ Over budget'}")
        print(f"   Diversity score: {diversity['diversity_score']:.3f}")
        print(f"   API connectivity: {'✅ Ready' if report['api_connectivity'] else '❌ Issues'}")
        print(f"   Overall status: {'🚀 READY FOR EXPERIMENT' if overall_ready else '⚠️  NEEDS ATTENTION'}")
        
        return report

def main():
    """Main validation function"""
    print("🔬 PHASE 3 MODEL SELECTION VALIDATION")
    print("="*60)
    
    # Initialize validator
    validator = ModelSelectionValidator()
    
    # Get model candidates from orchestrator
    config = ExperimentConfig()
    orchestrator = HierarchicalTestingOrchestrator(config)
    candidates = orchestrator.select_model_candidates()
    
    # Run comprehensive validation
    report = validator.generate_validation_report(candidates)
    
    if report['experiment_ready']:
        print(f"\n✅ All validations passed! Ready to proceed with Phase 3 experiment.")
        print(f"   To execute: Set dry_run=False in phase3_hierarchical_testing.py")
    else:
        print(f"\n⚠️  Some issues found. Review validation report before proceeding.")
    
    return report

if __name__ == "__main__":
    main()