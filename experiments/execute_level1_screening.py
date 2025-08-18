#!/usr/bin/env python3
"""
Level 1 Execution: Behavioral Screening

Executes the first level of hierarchical testing:
- 23 diverse models × 30 prompts each = 690 API calls
- Cost: ~$0.31 (very low risk)
- Goal: Identify top 15 models for Level 2 analysis

Implements real API calls with comprehensive safety controls.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import random

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.openrouter_model import OpenRouterModel
    from patterns.advanced_metrics import AdvancedConvergenceAnalyzer
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    OPENROUTER_AVAILABLE = True
except ImportError as e:
    print(f"OpenRouter integration not available: {e}")
    OPENROUTER_AVAILABLE = False

from phase3_hierarchical_testing import HierarchicalTestingOrchestrator, ExperimentConfig, ModelCandidate

@dataclass
class Level1Response:
    """Single response from Level 1 testing"""
    model_id: str
    capability: str
    prompt_text: str
    response_text: str
    timestamp: str
    processing_time: float
    api_cost: float

@dataclass
class Level1Results:
    """Complete results from Level 1 testing"""
    experiment_start: str
    experiment_end: str
    total_models: int
    total_prompts: int
    total_api_calls: int
    total_cost: float
    responses: List[Level1Response]
    convergence_scores: Dict[str, float]
    top_models: List[str]
    statistical_summary: Dict

class Level1Executor:
    """Executes Level 1 behavioral screening with safety controls"""
    
    def __init__(self, config: ExperimentConfig, max_cost_usd: float = 1.0):
        self.config = config
        self.max_cost_usd = max_cost_usd
        self.current_cost = 0.0
        self.api_calls_made = 0
        
        # Results storage
        self.results_dir = Path("results/phase3_hierarchical/level1")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load prompts and models
        self.orchestrator = HierarchicalTestingOrchestrator(config)
        self.candidates = self.orchestrator.select_model_candidates()
        self.prompts = self.load_level1_prompts()
        
    def setup_logging(self):
        """Setup comprehensive logging for Level 1"""
        log_file = self.results_dir / f"level1_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_level1_prompts(self) -> Dict[str, List[str]]:
        """Load Level 1 prompts (6 per capability = 30 total)"""
        self.logger.info("Loading Level 1 prompts...")
        
        # Get all prompts from orchestrator
        all_prompts = self.orchestrator.load_prompt_datasets()
        
        # Select 6 prompts per capability for Level 1 screening
        level1_prompts = {}
        for capability, prompts in all_prompts.items():
            # Use systematic sampling to get diverse prompts
            step = len(prompts) // 6
            selected = [prompts[i] for i in range(0, len(prompts), step)][:6]
            level1_prompts[capability] = selected
            
        total_prompts = sum(len(prompts) for prompts in level1_prompts.values())
        self.logger.info(f"Selected {total_prompts} prompts for Level 1 ({len(level1_prompts)} capabilities)")
        
        return level1_prompts
    
    async def test_single_model(self, candidate: ModelCandidate) -> List[Level1Response]:
        """Test a single model with all Level 1 prompts"""
        self.logger.info(f"Testing model: {candidate.model_id}")
        
        if not OPENROUTER_AVAILABLE:
            raise ImportError("OpenRouter integration is required but not available. Fix import errors first.")
        
        try:
            model = OpenRouterModel(candidate.model_id)
            responses = []
            
            for capability, prompts in self.prompts.items():
                for prompt in prompts:
                    # Cost check before each API call
                    if self.current_cost >= self.max_cost_usd:
                        self.logger.warning(f"Cost limit ${self.max_cost_usd} reached. Stopping.")
                        return responses
                    
                    try:
                        start_time = time.time()
                        
                        # Make API call with caching
                        response_text = model.generate(prompt)
                        
                        processing_time = time.time() - start_time
                        api_cost = candidate.cost_per_1k_tokens * 0.05  # Estimate ~50 tokens avg
                        
                        self.current_cost += api_cost
                        self.api_calls_made += 1
                        
                        response = Level1Response(
                            model_id=candidate.model_id,
                            capability=capability,
                            prompt_text=prompt,
                            response_text=response_text,
                            timestamp=datetime.now().isoformat(),
                            processing_time=processing_time,
                            api_cost=api_cost
                        )
                        
                        responses.append(response)
                        
                        self.logger.debug(f"   {capability}: {len(response_text)} chars, ${api_cost:.4f}")
                        
                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.error(f"   Failed prompt in {capability}: {str(e)[:50]}...")
                        continue
                        
            self.logger.info(f"   Completed {len(responses)} responses (${self.current_cost:.3f} total)")
            return responses
            
        except Exception as e:
            self.logger.error(f"   Model {candidate.model_id} failed: {e}")
            return []
    
    def generate_mock_responses(self, candidate: ModelCandidate) -> List[Level1Response]:
        """Generate realistic mock responses for development/testing"""
        responses = []
        
        # Mock response templates by capability
        mock_templates = {
            "truthfulness": [
                "The correct answer is {answer}.",
                "Based on factual information, {answer}.",
                "According to reliable sources, {answer}."
            ],
            "safety_boundaries": [
                "I cannot and will not provide information on {topic}.",
                "I'm not able to assist with {topic} as it could be harmful.",
                "That request involves {topic} which I cannot help with."
            ],
            "instruction_following": [
                "I'll follow your instructions: {response}",
                "As requested: {response}",
                "Here's what you asked for: {response}"
            ],
            "uncertainty_expression": [
                "I'm not certain about {topic}, but {guess}.",
                "That's difficult to predict with certainty. {reasoning}.",
                "I don't have enough information to be sure about {topic}."
            ],
            "context_awareness": [
                "I don't have access to {context_type} information.",
                "I can't recall {context_type} from our conversation.",
                "I don't have information about your {context_type}."
            ]
        }
        
        for capability, prompts in self.prompts.items():
            templates = mock_templates.get(capability, ["I understand your request about {topic}."])
            
            for prompt in prompts:
                # Generate realistic response
                template = random.choice(templates)
                response_text = template.format(
                    answer="[factual response]",
                    topic="that topic", 
                    response="[appropriate response]",
                    guess="here's my best assessment",
                    reasoning="based on general patterns",
                    context_type="personal"
                )
                
                # Add some variability to simulate real models
                if candidate.estimated_quality >= 8:
                    response_text += " [High quality model response with more detail and nuance.]"
                elif candidate.estimated_quality <= 6:
                    response_text += " [Simpler response from smaller model.]"
                
                response = Level1Response(
                    model_id=candidate.model_id,
                    capability=capability,
                    prompt_text=prompt,
                    response_text=response_text,
                    timestamp=datetime.now().isoformat(),
                    processing_time=random.uniform(0.5, 2.0),
                    api_cost=candidate.cost_per_1k_tokens * 0.05
                )
                
                responses.append(response)
        
        return responses
    
    def calculate_convergence_scores(self, all_responses: List[Level1Response]) -> Dict[str, float]:
        """Calculate convergence scores for each model"""
        self.logger.info("Calculating convergence scores...")
        
        # Group responses by model and capability
        model_responses = {}
        for response in all_responses:
            if response.model_id not in model_responses:
                model_responses[response.model_id] = {}
            if response.capability not in model_responses[response.model_id]:
                model_responses[response.model_id][response.capability] = []
            model_responses[response.model_id][response.capability].append(response.response_text)
        
        # Calculate pairwise similarities between models
        convergence_scores = {}
        
        for model_id in model_responses:
            similarities = []
            
            for other_model_id in model_responses:
                if model_id != other_model_id:
                    model_similarity = self.calculate_model_similarity(
                        model_responses[model_id],
                        model_responses[other_model_id]
                    )
                    similarities.append(model_similarity)
            
            # Average similarity with all other models
            convergence_scores[model_id] = sum(similarities) / len(similarities) if similarities else 0.0
        
        self.logger.info(f"Calculated convergence scores for {len(convergence_scores)} models")
        return convergence_scores
    
    def calculate_model_similarity(self, model1_responses: Dict[str, List[str]], 
                                 model2_responses: Dict[str, List[str]]) -> float:
        """Calculate similarity between two models' responses"""
        
        # Simple similarity calculation for Level 1 screening
        # In real implementation, this would use advanced metrics
        
        capability_similarities = []
        
        for capability in model1_responses:
            if capability in model2_responses:
                responses1 = model1_responses[capability]
                responses2 = model2_responses[capability]
                
                # Calculate average response length similarity (proxy for complexity)
                avg_len1 = sum(len(r) for r in responses1) / len(responses1)
                avg_len2 = sum(len(r) for r in responses2) / len(responses2)
                
                length_similarity = 1.0 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2, 1)
                
                # Calculate refusal pattern similarity (for safety boundaries)
                refusal_words = ["cannot", "can't", "unable", "not able", "won't", "will not"]
                refusal1 = sum(1 for r in responses1 if any(word in r.lower() for word in refusal_words)) / len(responses1)
                refusal2 = sum(1 for r in responses2 if any(word in r.lower() for word in refusal_words)) / len(responses2)
                
                refusal_similarity = 1.0 - abs(refusal1 - refusal2)
                
                # Combine similarities
                capability_similarity = (length_similarity + refusal_similarity) / 2
                capability_similarities.append(capability_similarity)
        
        return sum(capability_similarities) / len(capability_similarities) if capability_similarities else 0.0
    
    def select_top_models(self, convergence_scores: Dict[str, float], top_k: int = 15) -> List[str]:
        """Select top models for Level 2 analysis"""
        
        # Sort models by convergence score
        sorted_models = sorted(convergence_scores.items(), key=lambda x: -x[1])
        top_models = [model_id for model_id, score in sorted_models[:top_k]]
        
        self.logger.info(f"Selected top {len(top_models)} models for Level 2:")
        for i, (model_id, score) in enumerate(sorted_models[:top_k], 1):
            self.logger.info(f"   {i:2}. {model_id}: {score:.3f} convergence")
        
        return top_models
    
    async def execute_level1(self, dry_run: bool = False) -> Level1Results:
        """Execute complete Level 1 behavioral screening"""
        
        start_time = datetime.now()
        self.logger.info(f"🚀 Starting Level 1 execution (dry_run={dry_run})")
        self.logger.info(f"   Models: {len(self.candidates)}")
        self.logger.info(f"   Prompts per model: {sum(len(prompts) for prompts in self.prompts.values())}")
        self.logger.info(f"   Total API calls: {len(self.candidates) * sum(len(prompts) for prompts in self.prompts.values())}")
        self.logger.info(f"   Cost limit: ${self.max_cost_usd:.2f}")
        
        if dry_run:
            self.logger.info("   DRY RUN MODE - No real API calls")
        
        all_responses = []
        
        # Test each model
        for i, candidate in enumerate(self.candidates, 1):
            self.logger.info(f"🔄 Testing model {i}/{len(self.candidates)}: {candidate.model_id}")
            
            if dry_run:
                raise ValueError("Dry run mode disabled. Use real API calls only.")
            else:
                # Real API calls
                responses = await self.test_single_model(candidate)
            
            all_responses.extend(responses)
            
            # Progress update
            progress = i / len(self.candidates) * 100
            self.logger.info(f"   Progress: {progress:.1f}% (${self.current_cost:.3f} spent)")
            
            # Safety check
            if not dry_run and self.current_cost >= self.max_cost_usd:
                self.logger.warning("Cost limit reached, stopping execution")
                break
        
        # Calculate convergence scores
        convergence_scores = self.calculate_convergence_scores(all_responses)
        
        # Select top models for Level 2
        top_models = self.select_top_models(convergence_scores, self.config.level_2_models)
        
        # Create results summary
        end_time = datetime.now()
        
        results = Level1Results(
            experiment_start=start_time.isoformat(),
            experiment_end=end_time.isoformat(),
            total_models=len(self.candidates),
            total_prompts=sum(len(prompts) for prompts in self.prompts.values()),
            total_api_calls=len(all_responses),
            total_cost=self.current_cost,
            responses=all_responses,
            convergence_scores=convergence_scores,
            top_models=top_models,
            statistical_summary={
                "mean_convergence": sum(convergence_scores.values()) / len(convergence_scores),
                "max_convergence": max(convergence_scores.values()),
                "min_convergence": min(convergence_scores.values()),
                "std_convergence": self.calculate_std(list(convergence_scores.values()))
            }
        )
        
        # Save results
        self.save_results(results)
        
        self.logger.info(f"✅ Level 1 completed in {(end_time - start_time).total_seconds():.1f}s")
        self.logger.info(f"   Total cost: ${results.total_cost:.3f}")
        self.logger.info(f"   Average convergence: {results.statistical_summary['mean_convergence']:.3f}")
        
        return results
    
    def calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def save_results(self, results: Level1Results):
        """Save Level 1 results to file"""
        
        # Save full results as JSON
        results_file = self.results_dir / f"level1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to dict for JSON serialization
        results_dict = asdict(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"📄 Results saved to: {results_file}")
        
        # Save summary report
        summary_file = self.results_dir / "level1_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("LEVEL 1 BEHAVIORAL SCREENING - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment Period: {results.experiment_start} to {results.experiment_end}\n")
            f.write(f"Models Tested: {results.total_models}\n")
            f.write(f"Prompts per Model: {results.total_prompts}\n")
            f.write(f"Total API Calls: {results.total_api_calls}\n")
            f.write(f"Total Cost: ${results.total_cost:.3f}\n\n")
            
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write(f"  Mean Convergence: {results.statistical_summary['mean_convergence']:.3f}\n")
            f.write(f"  Max Convergence:  {results.statistical_summary['max_convergence']:.3f}\n")
            f.write(f"  Min Convergence:  {results.statistical_summary['min_convergence']:.3f}\n")
            f.write(f"  Std Convergence:  {results.statistical_summary['std_convergence']:.3f}\n\n")
            
            f.write("TOP MODELS SELECTED FOR LEVEL 2:\n")
            for i, model_id in enumerate(results.top_models, 1):
                score = results.convergence_scores[model_id]
                f.write(f"  {i:2}. {model_id}: {score:.3f}\n")
        
        self.logger.info(f"📄 Summary saved to: {summary_file}")

async def main():
    """Main execution function for Level 1"""
    
    print("🔬 LEVEL 1: BEHAVIORAL SCREENING EXECUTION")
    print("=" * 60)
    
    # Configuration
    config = ExperimentConfig(dry_run=False)  # Enable real execution
    executor = Level1Executor(config, max_cost_usd=1.0)  # Conservative cost limit
    
    # Show execution plan
    print(f"📊 EXECUTION PLAN:")
    print(f"   Models to test: {len(executor.candidates)}")
    print(f"   Prompts per model: {sum(len(prompts) for prompts in executor.prompts.values())}")
    print(f"   Total API calls: {len(executor.candidates) * sum(len(prompts) for prompts in executor.prompts.values())}")
    print(f"   Cost limit: ${executor.max_cost_usd:.2f}")
    print(f"   Expected cost: ~${executor.max_cost_usd * 0.31:.2f}")
    
    # Ask for confirmation for real API calls
    if not OPENROUTER_AVAILABLE:
        raise ImportError("OpenRouter integration is required but not available. Fix import errors first.")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable is required for real API calls.")
    
    response = input(f"\n🚨 This will make REAL API calls with REAL costs. Continue? [y/N]: ")
    if response.lower() != 'y':
        print("❌ User cancelled. No API calls made.")
        return None
    else:
        print("🚀 Executing with real API calls...")
        results = await executor.execute_level1(dry_run=False)
    
    # Print final summary
    print(f"\n✅ LEVEL 1 COMPLETE!")
    print(f"   Models tested: {results.total_models}")
    print(f"   API calls made: {results.total_api_calls}")
    print(f"   Total cost: ${results.total_cost:.3f}")
    print(f"   Average convergence: {results.statistical_summary['mean_convergence']:.3f}")
    print(f"   Top models selected: {len(results.top_models)}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())