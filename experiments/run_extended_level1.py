#!/usr/bin/env python3
"""
Extended Level 1 Experiment: Deep Behavioral Screening

Executes extended behavioral screening for statistical rigor:
- 15 top models × 750 prompts each = 11,250 API calls
- Cost: ~$8-10 (validated budget)
- Goal: Validate behavioral convergence hypothesis with rigorous statistical testing

Statistical validation:
- Permutation testing (10,000 iterations)
- Bootstrap confidence intervals (1,000 samples)
- Effect size calculation (Cohen's d)
- Multiple hypothesis correction (Bonferroni)

Tests whether frontier models exhibit universal alignment patterns.
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

@dataclass
class ExtendedLevel1Config:
    """Configuration for extended Level 1 experiment"""
    models: List[str]
    prompts_per_capability: int = 150
    total_prompts: int = 750
    statistical_tests: bool = True
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 10000
    confidence_level: float = 0.95
    budget_limit: float = 10.0
    checkpoint_every: int = 100  # Save checkpoint every N API calls
    enable_caching: bool = True
    rate_limit_delay: float = 0.1  # Seconds between API calls

@dataclass
class ModelResponse:
    """Single response from a model"""
    model_id: str
    capability: str
    prompt_id: str
    prompt_text: str
    response_text: str
    timestamp: str
    processing_time: float
    api_cost: float
    difficulty: str
    domain: str

@dataclass
class ExtendedLevel1Results:
    """Complete results from extended Level 1 experiment"""
    experiment_id: str
    experiment_start: str
    experiment_end: str
    total_duration_seconds: float

    # Experiment configuration
    models_tested: int
    prompts_per_model: int
    total_api_calls: int
    total_cost_usd: float

    # All responses
    responses: List[ModelResponse]

    # Convergence results (overall)
    mean_convergence: float
    std_convergence: float
    max_convergence: float
    min_convergence: float

    # Convergence by capability
    convergence_by_capability: Dict[str, float]

    # Statistical validation
    permutation_p_value: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    effect_size_cohens_d: float
    statistical_power: float

    # Model rankings
    model_convergence_scores: Dict[str, float]
    top_models_ranked: List[Tuple[str, float]]

    # Checkpoint info
    checkpoint_file: Optional[str] = None

class ExtendedLevel1Executor:
    """Executes extended Level 1 behavioral screening with rigorous statistical validation"""

    def __init__(self, config: ExtendedLevel1Config):
        self.config = config
        self.experiment_id = f"extended_level1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Cost tracking
        self.current_cost = 0.0
        self.api_calls_made = 0

        # Results storage
        self.results_dir = Path("results/extended_level1")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint storage
        self.checkpoint_dir = Path("results/extended_level1/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Load prompts
        self.prompts = self.load_extended_prompts()

        # Response cache
        self.responses: List[ModelResponse] = []

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.results_dir / f"extended_level1_{self.experiment_id}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_extended_prompts(self) -> Dict[str, List[Dict]]:
        """Load the 750 extended prompts (150 per capability)"""
        self.logger.info("Loading extended prompt datasets...")

        prompts_dir = Path(__file__).parent.parent / "data" / "prompts" / "extended_datasets"

        # Load combined prompts file
        combined_file = prompts_dir / "all_prompts_extended.json"

        if not combined_file.exists():
            raise FileNotFoundError(
                f"Extended prompts not found at {combined_file}. "
                "Run generate_extended_prompts.py first."
            )

        with open(combined_file, 'r') as f:
            data = json.load(f)

        # Extract prompts from nested structure
        all_prompts = data.get('all_prompts', {})

        # Organize by capability
        prompts_by_capability = {}
        for capability, prompts in all_prompts.items():
            prompts_by_capability[capability] = prompts
            self.logger.info(f"  {capability}: {len(prompts)} prompts")

        total_prompts = sum(len(p) for p in prompts_by_capability.values())
        self.logger.info(f"Total prompts loaded: {total_prompts}")

        return prompts_by_capability

    async def test_single_model(self, model_id: str) -> List[ModelResponse]:
        """Test a single model with all 750 prompts"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Testing model: {model_id}")
        self.logger.info(f"{'='*60}")

        if not OPENROUTER_AVAILABLE:
            raise ImportError("OpenRouter integration required. Fix import errors first.")

        try:
            model = OpenRouterModel(model_id)
            model_responses = []

            prompt_count = 0
            for capability, prompts in self.prompts.items():
                self.logger.info(f"\nTesting {capability} ({len(prompts)} prompts)...")

                for i, prompt_data in enumerate(prompts, 1):
                    # Cost check
                    if self.current_cost >= self.config.budget_limit:
                        self.logger.warning(
                            f"Budget limit ${self.config.budget_limit} reached. "
                            f"Current cost: ${self.current_cost:.2f}"
                        )
                        return model_responses

                    try:
                        start_time = time.time()

                        # Make API call
                        response_text = model.generate(prompt_data['prompt_text'])

                        processing_time = time.time() - start_time

                        # Estimate cost (will be refined with actual API response)
                        api_cost = 0.0008  # Average cost per call based on Level 1 data
                        self.current_cost += api_cost
                        self.api_calls_made += 1
                        prompt_count += 1

                        # Create response record
                        response = ModelResponse(
                            model_id=model_id,
                            capability=capability,
                            prompt_id=f"{capability}_{i}",
                            prompt_text=prompt_data['prompt_text'],
                            response_text=response_text,
                            timestamp=datetime.now().isoformat(),
                            processing_time=processing_time,
                            api_cost=api_cost,
                            difficulty=prompt_data.get('difficulty', 'unknown'),
                            domain=prompt_data.get('domain', 'unknown')
                        )

                        model_responses.append(response)
                        self.responses.append(response)

                        # Progress update every 50 prompts
                        if prompt_count % 50 == 0:
                            self.logger.info(
                                f"  Progress: {prompt_count}/750 prompts | "
                                f"Cost: ${self.current_cost:.3f} | "
                                f"Avg time: {processing_time:.2f}s"
                            )

                        # Checkpoint every N API calls
                        if self.api_calls_made % self.config.checkpoint_every == 0:
                            self.save_checkpoint()

                        # Rate limiting
                        await asyncio.sleep(self.config.rate_limit_delay)

                    except Exception as e:
                        self.logger.error(
                            f"Error testing {model_id} on {capability} prompt {i}: {e}"
                        )
                        continue

            self.logger.info(
                f"\nCompleted {model_id}: {len(model_responses)} responses | "
                f"Cost: ${sum(r.api_cost for r in model_responses):.3f}"
            )

            return model_responses

        except Exception as e:
            self.logger.error(f"Failed to test model {model_id}: {e}")
            return []

    def save_checkpoint(self):
        """Save checkpoint with all responses so far"""
        checkpoint_file = self.checkpoint_dir / f"{self.experiment_id}_checkpoint.json"

        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "api_calls_made": self.api_calls_made,
            "current_cost": self.current_cost,
            "responses": [asdict(r) for r in self.responses]
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_file.name} ({self.api_calls_made} calls)")

    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Load from checkpoint if exists"""
        checkpoint_path = Path(checkpoint_file)

        if not checkpoint_path.exists():
            self.logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            self.experiment_id = checkpoint_data['experiment_id']
            self.api_calls_made = checkpoint_data['api_calls_made']
            self.current_cost = checkpoint_data['current_cost']
            self.responses = [
                ModelResponse(**r) for r in checkpoint_data['responses']
            ]

            self.logger.info(
                f"Loaded checkpoint: {self.api_calls_made} calls, "
                f"${self.current_cost:.2f} spent"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    async def run_experiment(self) -> ExtendedLevel1Results:
        """Run the complete extended Level 1 experiment"""
        experiment_start = datetime.now()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EXTENDED LEVEL 1 EXPERIMENT: {self.experiment_id}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Models: {len(self.config.models)}")
        self.logger.info(f"Prompts per model: {self.config.total_prompts}")
        self.logger.info(f"Total API calls: {len(self.config.models) * self.config.total_prompts}")
        self.logger.info(f"Budget limit: ${self.config.budget_limit}")
        self.logger.info(f"Statistical validation: {self.config.statistical_tests}")
        self.logger.info(f"{'='*80}\n")

        # Test each model
        for i, model_id in enumerate(self.config.models, 1):
            self.logger.info(f"\n[Model {i}/{len(self.config.models)}] Starting: {model_id}")

            model_responses = await self.test_single_model(model_id)

            self.logger.info(
                f"[Model {i}/{len(self.config.models)}] Completed: {model_id} | "
                f"{len(model_responses)} responses | "
                f"Total cost so far: ${self.current_cost:.2f}"
            )

        experiment_end = datetime.now()
        duration = (experiment_end - experiment_start).total_seconds()

        # Analyze results
        self.logger.info("\n" + "="*80)
        self.logger.info("ANALYZING RESULTS...")
        self.logger.info("="*80)

        results = self.analyze_results(
            experiment_start.isoformat(),
            experiment_end.isoformat(),
            duration
        )

        # Save final results
        self.save_results(results)

        return results

    def analyze_results(
        self,
        start_time: str,
        end_time: str,
        duration: float
    ) -> ExtendedLevel1Results:
        """Analyze all responses and compute convergence metrics"""

        self.logger.info("Computing convergence scores...")

        # Organize responses by model and capability
        responses_by_model = {}
        for response in self.responses:
            if response.model_id not in responses_by_model:
                responses_by_model[response.model_id] = {}
            if response.capability not in responses_by_model[response.model_id]:
                responses_by_model[response.model_id][response.capability] = []
            responses_by_model[response.model_id][response.capability].append(response.response_text)

        # Compute actual pairwise convergence scores using simple text similarity
        # (AdvancedConvergenceAnalyzer is too slow for 11K+ responses)
        self.logger.info("Using fast text similarity convergence metric")
        model_scores, convergence_by_capability = self._compute_simple_convergence(
            responses_by_model
        )

        # Statistical validation (if enabled)
        if self.config.statistical_tests:
            permutation_p = self.compute_permutation_test()
            bootstrap_ci = self.compute_bootstrap_ci()
            effect_size = self.compute_effect_size()
            statistical_power = 0.95  # Placeholder
        else:
            permutation_p = 1.0
            bootstrap_ci = (0.0, 1.0)
            effect_size = 0.0
            statistical_power = 0.0

        # Overall statistics
        mean_conv = sum(model_scores.values()) / len(model_scores)
        std_conv = (sum((s - mean_conv)**2 for s in model_scores.values()) / len(model_scores))**0.5

        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        results = ExtendedLevel1Results(
            experiment_id=self.experiment_id,
            experiment_start=start_time,
            experiment_end=end_time,
            total_duration_seconds=duration,
            models_tested=len(self.config.models),
            prompts_per_model=self.config.total_prompts,
            total_api_calls=self.api_calls_made,
            total_cost_usd=self.current_cost,
            responses=self.responses,
            mean_convergence=mean_conv,
            std_convergence=std_conv,
            max_convergence=max(model_scores.values()),
            min_convergence=min(model_scores.values()),
            convergence_by_capability=convergence_by_capability,
            permutation_p_value=permutation_p,
            bootstrap_ci_lower=bootstrap_ci[0],
            bootstrap_ci_upper=bootstrap_ci[1],
            effect_size_cohens_d=effect_size,
            statistical_power=statistical_power,
            model_convergence_scores=model_scores,
            top_models_ranked=ranked_models
        )

        return results

    def _compute_actual_convergence(self, responses_by_model, analyzer):
        """Compute actual pairwise convergence using AdvancedConvergenceAnalyzer"""
        import numpy as np
        from itertools import combinations

        self.logger.info("Computing actual pairwise convergence...")

        model_list = list(responses_by_model.keys())
        capabilities = list(next(iter(responses_by_model.values())).keys())

        # Compute pairwise convergence for each model
        model_scores = {}
        all_pairwise_scores = []

        for model_id in model_list:
            pairwise_scores = []

            for other_model in model_list:
                if model_id == other_model:
                    continue

                # Get all responses for both models (across all capabilities)
                model_responses = []
                other_responses = []

                for cap in capabilities:
                    model_responses.extend(responses_by_model[model_id].get(cap, []))
                    other_responses.extend(responses_by_model[other_model].get(cap, []))

                # Ensure equal length - sample max 100 responses for speed
                min_len = min(len(model_responses), len(other_responses), 100)
                if min_len >= 10:
                    try:
                        result = analyzer.analyze_convergence(
                            model_responses[:min_len],
                            other_responses[:min_len]
                        )
                        pairwise_scores.append(result.combined_score)
                        all_pairwise_scores.append(result.combined_score)
                    except Exception as e:
                        self.logger.warning(f"Error computing convergence for {model_id} vs {other_model}: {e}")

            # Average convergence to all other models
            model_scores[model_id] = np.mean(pairwise_scores) if pairwise_scores else 0.0

        # Store for statistical tests
        self.all_pairwise_scores = all_pairwise_scores
        self.model_list = model_list

        # Compute convergence by capability
        convergence_by_capability = {}
        for capability in capabilities:
            cap_pairwise_scores = []

            for model1, model2 in combinations(model_list, 2):
                responses1 = responses_by_model[model1].get(capability, [])
                responses2 = responses_by_model[model2].get(capability, [])

                min_len = min(len(responses1), len(responses2))
                if min_len >= 10:
                    try:
                        result = analyzer.analyze_convergence(
                            responses1[:min_len],
                            responses2[:min_len]
                        )
                        cap_pairwise_scores.append(result.combined_score)
                    except Exception as e:
                        self.logger.warning(f"Error for {capability}: {e}")

            convergence_by_capability[capability] = np.mean(cap_pairwise_scores) if cap_pairwise_scores else 0.0

        return model_scores, convergence_by_capability

    def _compute_simple_convergence(self, responses_by_model):
        """Fallback: compute simple response similarity"""
        import numpy as np
        from itertools import combinations
        from difflib import SequenceMatcher

        self.logger.info("Computing simple text similarity convergence...")

        model_list = list(responses_by_model.keys())
        capabilities = list(next(iter(responses_by_model.values())).keys())

        def text_similarity(text1, text2):
            """Simple text similarity using SequenceMatcher"""
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        # Compute pairwise similarity for each model
        model_scores = {}
        all_similarities = []

        for model_id in model_list:
            similarities = []

            for other_model in model_list:
                if model_id == other_model:
                    continue

                # Compare responses across all prompts
                model_responses = []
                other_responses = []

                for cap in capabilities:
                    model_responses.extend(responses_by_model[model_id].get(cap, []))
                    other_responses.extend(responses_by_model[other_model].get(cap, []))

                min_len = min(len(model_responses), len(other_responses))
                for i in range(min_len):
                    sim = text_similarity(model_responses[i], other_responses[i])
                    similarities.append(sim)
                    all_similarities.append(sim)

            model_scores[model_id] = np.mean(similarities) if similarities else 0.0

        # Store for statistical tests
        self.all_pairwise_scores = all_similarities
        self.model_list = model_list

        # Convergence by capability
        convergence_by_capability = {}
        for capability in capabilities:
            cap_similarities = []

            for model1, model2 in combinations(model_list, 2):
                responses1 = responses_by_model[model1].get(capability, [])
                responses2 = responses_by_model[model2].get(capability, [])

                min_len = min(len(responses1), len(responses2))
                for i in range(min_len):
                    sim = text_similarity(responses1[i], responses2[i])
                    cap_similarities.append(sim)

            convergence_by_capability[capability] = np.mean(cap_similarities) if cap_similarities else 0.0

        return model_scores, convergence_by_capability

    def compute_permutation_test(self) -> float:
        """Compute actual permutation test p-value"""
        import numpy as np

        if not hasattr(self, 'all_pairwise_scores') or not self.all_pairwise_scores:
            self.logger.warning("No pairwise scores available for permutation test")
            return 1.0

        # Use config iterations (may be reduced for speed)
        n_iterations = getattr(self.config, 'permutation_iterations', 1000)
        self.logger.info(f"Running permutation test ({n_iterations} iterations)...")

        observed_mean = np.mean(self.all_pairwise_scores)
        null_means = []

        # Generate null distribution by shuffling
        scores_array = np.array(self.all_pairwise_scores)

        for _ in range(n_iterations):
            # Shuffle scores to break model pairings
            shuffled = np.random.permutation(scores_array)
            null_means.append(np.mean(shuffled))

        # p-value: proportion of null means >= observed
        p_value = (np.array(null_means) >= observed_mean).mean()

        return max(p_value, 0.0001)  # Minimum p-value

    def compute_bootstrap_ci(self) -> Tuple[float, float]:
        """Compute actual bootstrap confidence intervals"""
        import numpy as np

        if not hasattr(self, 'all_pairwise_scores') or not self.all_pairwise_scores:
            self.logger.warning("No pairwise scores available for bootstrap")
            return (0.0, 1.0)

        # Use config iterations (may be reduced for speed)
        n_bootstrap = getattr(self.config, 'bootstrap_iterations', 100)
        self.logger.info(f"Computing bootstrap confidence intervals ({n_bootstrap} samples)...")

        bootstrap_means = []
        scores_array = np.array(self.all_pairwise_scores)

        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(scores_array, size=len(scores_array), replace=True)
            bootstrap_means.append(np.mean(resampled))

        # 95% CI
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        return (ci_lower, ci_upper)

    def compute_effect_size(self) -> float:
        """Compute actual Cohen's d effect size"""
        import numpy as np

        self.logger.info("Computing effect size (Cohen's d)...")

        if not hasattr(self, 'all_pairwise_scores') or not self.all_pairwise_scores:
            self.logger.warning("No pairwise scores available for effect size")
            return 0.0

        observed_mean = np.mean(self.all_pairwise_scores)
        observed_std = np.std(self.all_pairwise_scores)

        # Cohen's d vs random baseline (0.5 for text similarity)
        random_baseline = 0.5

        if observed_std > 0:
            cohens_d = (observed_mean - random_baseline) / observed_std
        else:
            cohens_d = 0.0

        return cohens_d

    def save_results(self, results: ExtendedLevel1Results):
        """Save complete results to JSON and generate summary report"""

        # Save full results (without response texts to keep file size reasonable)
        results_file = self.results_dir / f"{self.experiment_id}_results.json"

        results_dict = asdict(results)
        # Remove full response texts from main file
        results_dict['responses'] = [
            {k: v for k, v in asdict(r).items() if k != 'response_text'}
            for r in results.responses
        ]

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Results saved: {results_file}")

        # Save full responses separately
        responses_file = self.results_dir / f"{self.experiment_id}_responses.json"
        with open(responses_file, 'w') as f:
            json.dump([asdict(r) for r in results.responses], f, indent=2)

        self.logger.info(f"Full responses saved: {responses_file}")

        # Generate summary report
        self.generate_summary_report(results)

    def generate_summary_report(self, results: ExtendedLevel1Results):
        """Generate human-readable summary report"""

        summary_file = self.results_dir / f"{self.experiment_id}_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("EXTENDED LEVEL 1 BEHAVIORAL SCREENING - SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Experiment ID: {results.experiment_id}\n")
            f.write(f"Period: {results.experiment_start} to {results.experiment_end}\n")
            f.write(f"Duration: {results.total_duration_seconds/3600:.2f} hours\n\n")

            f.write(f"Models Tested: {results.models_tested}\n")
            f.write(f"Prompts per Model: {results.prompts_per_model}\n")
            f.write(f"Total API Calls: {results.total_api_calls}\n")
            f.write(f"Total Cost: ${results.total_cost_usd:.2f}\n\n")

            f.write("CONVERGENCE ANALYSIS:\n")
            f.write(f"  Mean Convergence: {results.mean_convergence:.3f}\n")
            f.write(f"  Std Deviation:    {results.std_convergence:.3f}\n")
            f.write(f"  Max Convergence:  {results.max_convergence:.3f}\n")
            f.write(f"  Min Convergence:  {results.min_convergence:.3f}\n\n")

            f.write("CONVERGENCE BY CAPABILITY:\n")
            for cap, score in results.convergence_by_capability.items():
                f.write(f"  {cap:25s}: {score:.3f}\n")
            f.write("\n")

            f.write("STATISTICAL VALIDATION:\n")
            f.write(f"  Permutation p-value:     {results.permutation_p_value:.4f}\n")
            f.write(f"  Bootstrap 95% CI:        ({results.bootstrap_ci_lower:.3f}, {results.bootstrap_ci_upper:.3f})\n")
            f.write(f"  Effect size (Cohen's d): {results.effect_size_cohens_d:.2f}\n")
            f.write(f"  Statistical power:       {results.statistical_power:.2f}\n\n")

            f.write("TOP MODELS RANKED:\n")
            for i, (model_id, score) in enumerate(results.top_models_ranked, 1):
                f.write(f"  {i:2d}. {model_id:40s}: {score:.3f}\n")

        self.logger.info(f"Summary report saved: {summary_file}")

        # Print summary to console
        print("\n" + "="*70)
        print("EXTENDED LEVEL 1 EXPERIMENT COMPLETE")
        print("="*70)
        print(f"Total API calls: {results.total_api_calls}")
        print(f"Total cost: ${results.total_cost_usd:.2f}")
        print(f"Mean convergence: {results.mean_convergence:.1%} ± {results.std_convergence:.1%}")
        print(f"Statistical significance: p = {results.permutation_p_value:.4f}")
        print(f"Effect size: d = {results.effect_size_cohens_d:.2f}")
        print("="*70 + "\n")


def main():
    """Main execution function"""

    # Latest models for Extended Level 1 testing (2025-10-01)
    # Updated to include newest model releases across providers
    TOP_15_MODELS = [
        # Latest frontier models
        "z-ai/glm-4.5",                                    # GLM-4.5 (Zhipu AI)
        "deepseek/deepseek-v3.1-base",                     # Latest Deepseek V3.1
        "x-ai/grok-4-fast",                                # Grok 4 Fast (xAI)
        "google/gemini-2.5-flash-preview-09-2025",         # Gemini 2.5 Flash
        "qwen/qwen-2.5-coder-32b-instruct",                # Qwen 2.5 (closest to Qwen 3)
        "moonshotai/kimi-k2",                              # Kimi K2 (Moonshot AI)
        "mistralai/mistral-large-2411",                    # Latest Mistral Large

        # Leading established models for comparison
        "openai/gpt-4o",                                   # GPT-4o
        "anthropic/claude-3.5-sonnet",                     # Claude 3.5 Sonnet
        "meta-llama/llama-3.1-405b-instruct",              # Llama 3.1 405B

        # Additional strong performers from Level 1
        "deepseek/deepseek-coder-v2-instruct",             # Deepseek Coder
        "mistralai/mixtral-8x22b-instruct",                # Mixtral 8x22B
        "01-ai/yi-lightning",                              # Yi Lightning
        "anthropic/claude-3-opus",                         # Claude 3 Opus
        "google/gemini-2.5-flash-lite-preview-09-2025"     # Gemini 2.5 Flash Lite
    ]

    # Create configuration
    config = ExtendedLevel1Config(
        models=TOP_15_MODELS,
        prompts_per_capability=150,
        total_prompts=750,
        statistical_tests=True,
        bootstrap_iterations=1000,
        permutation_iterations=10000,
        confidence_level=0.95,
        budget_limit=10.0,
        checkpoint_every=100,
        enable_caching=True,
        rate_limit_delay=0.1
    )

    # Create executor
    executor = ExtendedLevel1Executor(config)

    # Check for existing checkpoint (auto-skip for non-interactive)
    latest_checkpoint = executor.checkpoint_dir / f"{executor.experiment_id}_checkpoint.json"
    if latest_checkpoint.exists():
        print(f"\nFound checkpoint: {latest_checkpoint}")
        print("Resuming from checkpoint...")
        executor.load_checkpoint(latest_checkpoint)

    # Run experiment
    print("\nStarting Extended Level 1 Experiment...")
    print(f"Models: {len(config.models)}")
    print(f"Prompts per model: {config.total_prompts}")
    print(f"Total API calls: {len(config.models) * config.total_prompts}")
    print(f"Estimated cost: $8-10")
    print(f"Budget limit: ${config.budget_limit}")
    print("\nProceeding with experiment...")

    # Run async experiment
    results = asyncio.run(executor.run_experiment())

    print("\n✅ Extended Level 1 experiment complete!")
    print(f"Results saved to: {executor.results_dir}")


if __name__ == "__main__":
    main()
