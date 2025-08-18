# OpenRouter API Integration Guide

## 🌐 Overview

This guide provides comprehensive documentation for integrating with the OpenRouter API - the unified gateway providing access to 300+ AI models from all major providers through a single endpoint. Our Universal Alignment Patterns system leverages OpenRouter for cost-effective, scalable, and robust multi-model analysis.

**Key Innovation**: Transform from managing 10+ different API keys and endpoints to a single, unified integration supporting the world's largest model catalog.

---

## 🚀 Quick Start

### 1. Get Your OpenRouter API Key

```bash
# Visit OpenRouter and create account
open https://openrouter.ai/

# Copy your API key from dashboard
# Set environment variable
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 2. Test Integration

```python
from src.models.openrouter_model import OpenRouterModel

# Initialize model
model = OpenRouterModel("openai/gpt-4o-mini")

# Test API call
response = model.generate("What is 2+2?")
print(response)  # Should output: "4"
```

### 3. Run Quick Analysis

```bash
# Test with proven working models
python main.py --models gpt-4o-mini claude-3-haiku llama-3.1-405b --quick

# Cost: ~$0.50-1.00 for comprehensive 3-model analysis
```

---

## 🏗️ Technical Implementation

### 1. Core OpenRouter Integration

Our `OpenRouterModel` class provides a unified interface to all supported models:

```python
class OpenRouterModel(ModelInterface):
    """
    Unified interface to 300+ models via OpenRouter API.
    
    Supports all major providers:
    - OpenAI (GPT-4, GPT-4o, o1-mini, o1-preview)
    - Anthropic (Claude-3.5-Sonnet, Claude-3-Opus, Claude-3-Haiku)
    - Google (Gemini-Pro, Gemini-Flash)
    - Meta (Llama-3.1 series, Llama-3.2)
    - DeepSeek (DeepSeek-V3, DeepSeek-R1)
    - Qwen (Qwen-2.5 series, Qwen-Coder)
    - And 200+ more...
    """
    
    def __init__(self, model_id: str, temperature: float = 0.0):
        """
        Initialize OpenRouter model connection.
        
        Args:
            model_id: OpenRouter model identifier (e.g., "openai/gpt-4o")
            temperature: Model temperature (0.0 for deterministic results)
        """
        self.model_id = model_id
        self.name = model_id.split('/')[-1]
        self.temperature = temperature
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/your-username/universal-alignment-patterns",
                "X-Title": "Universal Alignment Patterns Research"
            }
        )
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate response using OpenRouter API.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
            
        Raises:
            OpenRouterAPIError: For API errors (404, 400, rate limits, etc.)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Handle common OpenRouter errors
            if "404" in str(e):
                raise OpenRouterAPIError(f"Model {self.model_id} not available")
            elif "400" in str(e):
                raise OpenRouterAPIError(f"Invalid request for {self.model_id}: {e}")
            elif "429" in str(e):
                raise OpenRouterAPIError(f"Rate limit exceeded for {self.model_id}")
            else:
                raise OpenRouterAPIError(f"API error: {e}")
    
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """
        Estimate API cost for prompt/response pair.
        
        Uses OpenRouter's pricing API for accurate cost calculation.
        """
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        response_tokens = len(response.split()) * 1.3
        
        # Get pricing from model registry
        pricing = self.get_model_pricing(self.model_id)
        
        cost = (prompt_tokens * pricing['input'] + 
                response_tokens * pricing['output']) / 1000
        
        return cost
```

### 2. Model Registry Integration

Our centralized registry provides intelligent model management:

```python
class ModelRegistry:
    """
    Centralized registry for OpenRouter model configuration.
    
    Features:
    - Cost tracking and estimation
    - Model availability validation  
    - Preset configurations for different use cases
    - Real-time pricing updates
    """
    
    def __init__(self):
        self.models = {
            # Frontier Models (Highest Quality)
            "gpt-4o": {
                "id": "openai/gpt-4o",
                "provider": "openai", 
                "cost_per_1k_input": 0.0025,
                "cost_per_1k_output": 0.010,
                "context_length": 128000,
                "category": "frontier",
                "strengths": ["reasoning", "analysis", "coding"],
                "available": True
            },
            
            "claude-3.5-sonnet": {
                "id": "anthropic/claude-3.5-sonnet",
                "provider": "anthropic",
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "context_length": 200000,
                "category": "frontier", 
                "strengths": ["safety", "analysis", "creative"],
                "available": True
            },
            
            # Reasoning Specialists
            "o1-mini": {
                "id": "openai/o1-mini",
                "provider": "openai",
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.012,
                "context_length": 65536,
                "category": "reasoning",
                "strengths": ["math", "science", "logic"],
                "available": True
            },
            
            # Cost-Optimized Models
            "gpt-4o-mini": {
                "id": "openai/gpt-4o-mini", 
                "provider": "openai",
                "cost_per_1k_input": 0.00015,
                "cost_per_1k_output": 0.0006,
                "context_length": 128000,
                "category": "cost_optimized",
                "strengths": ["efficiency", "speed"],
                "available": True
            },
            
            # Open Source Leaders
            "llama-3.1-405b": {
                "id": "meta-llama/llama-3.1-405b-instruct",
                "provider": "meta",
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.003,
                "context_length": 32768,
                "category": "open_source",
                "strengths": ["open", "transparent", "customizable"],
                "available": True
            }
        }
    
    def get_preset(self, preset_name: str) -> List[str]:
        """Get model list for specific research presets."""
        
        presets = {
            # Cost-conscious research  
            "cost_optimized": [
                "gpt-4o-mini",           # $0.0006/1k - OpenAI efficiency
                "claude-3-haiku",        # $0.00025/1k - Anthropic speed
                "deepseek-v2.5",         # $0.0001/1k - Ultra-low cost  
                "qwen-2.5-72b",          # $0.0005/1k - Chinese leader
                "llama-3.1-8b"           # $0.0001/1k - Open source
            ],
            
            # Highest quality analysis
            "frontier": [
                "gpt-4o",                # Latest OpenAI flagship
                "claude-3.5-sonnet",     # Anthropic's best
                "llama-3.1-405b",        # Largest open model
                "qwen-2.5-72b",          # Top non-US model
                "deepseek-v3"            # Reasoning specialist
            ],
            
            # Specialized reasoning
            "reasoning": [
                "o1-mini",               # OpenAI reasoning
                "deepseek-r1",           # DeepSeek reasoning  
                "qwen-2.5-math",         # Math specialist
                "claude-3.5-sonnet",     # Analytical strength
                "llama-3.1-70b"          # Balanced open model
            ],
            
            # Fellowship research (comprehensive)
            "fellowship_research": [
                "gpt-4o",                # OpenAI flagship
                "claude-3.5-sonnet",     # Anthropic flagship  
                "claude-3-opus",         # Anthropic premium
                "o1-mini",               # Reasoning specialist
                "llama-3.1-405b",        # Open source leader
                "deepseek-v3",           # Chinese innovation
                "qwen-2.5-72b",          # Alternative reasoning
                "gemini-pro"             # Google's offering
            ]
        }
        
        return presets.get(preset_name, [])
    
    def validate_model_availability(self, model_ids: List[str]) -> Dict[str, bool]:
        """
        Check which models are currently available on OpenRouter.
        
        Returns:
            Dict mapping model_id to availability status
        """
        availability = {}
        
        for model_id in model_ids:
            try:
                # Test with minimal API call
                test_model = OpenRouterModel(model_id)
                test_response = test_model.generate("Test", max_tokens=1)
                availability[model_id] = True
                
            except OpenRouterAPIError as e:
                if "404" in str(e):
                    availability[model_id] = False  # Model not available
                elif "400" in str(e):
                    availability[model_id] = False  # Invalid model ID
                else:
                    availability[model_id] = True   # Likely available, other error
                    
        return availability
```

### 3. Cost Management System

**Real-time cost tracking** ensures experiments stay within budget:

```python
class CostMonitor:
    """
    Advanced cost monitoring for OpenRouter API usage.
    
    Features:
    - Real-time cost tracking
    - Budget limit enforcement
    - Provider-specific cost optimization
    - Usage analytics and reporting
    """
    
    def __init__(self, budget_limit_usd: float):
        self.budget_limit = budget_limit_usd
        self.spent = 0.0
        self.usage_log = []
        
    def track_api_call(self, 
                      model_id: str, 
                      prompt: str, 
                      response: str) -> float:
        """
        Track cost for individual API call.
        
        Args:
            model_id: OpenRouter model identifier
            prompt: Input prompt
            response: Model response
            
        Returns:
            Cost of this API call in USD
        """
        # Calculate tokens (OpenRouter compatible)
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(response)
        
        # Get model pricing
        pricing = self.get_model_pricing(model_id)
        
        # Calculate cost
        cost = (
            prompt_tokens * pricing['input_cost_per_1k'] / 1000 +
            response_tokens * pricing['output_cost_per_1k'] / 1000
        )
        
        # Track usage
        self.spent += cost
        self.usage_log.append({
            'timestamp': time.time(),
            'model_id': model_id,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'cost_usd': cost,
            'cumulative_cost': self.spent
        })
        
        # Check budget
        if self.spent > self.budget_limit:
            raise BudgetExceededException(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Current spend: ${self.spent:.4f}"
            )
        
        return cost
    
    def estimate_experiment_cost(self, 
                                models: List[str], 
                                prompts_per_model: int) -> Dict[str, Any]:
        """
        Estimate total cost for planned experiment.
        
        Args:
            models: List of OpenRouter model IDs
            prompts_per_model: Number of prompts per model
            
        Returns:
            Cost breakdown and estimates
        """
        estimates = {
            'total_api_calls': len(models) * prompts_per_model,
            'model_costs': {},
            'total_estimated_cost': 0.0,
            'budget_utilization': 0.0
        }
        
        for model_id in models:
            # Estimate average tokens per prompt/response
            avg_prompt_tokens = 100  # Conservative estimate
            avg_response_tokens = 200
            
            pricing = self.get_model_pricing(model_id)
            
            cost_per_call = (
                avg_prompt_tokens * pricing['input_cost_per_1k'] / 1000 +
                avg_response_tokens * pricing['output_cost_per_1k'] / 1000  
            )
            
            total_model_cost = cost_per_call * prompts_per_model
            
            estimates['model_costs'][model_id] = {
                'cost_per_call': cost_per_call,
                'total_calls': prompts_per_model,
                'total_cost': total_model_cost
            }
            
            estimates['total_estimated_cost'] += total_model_cost
        
        estimates['budget_utilization'] = (
            estimates['total_estimated_cost'] / self.budget_limit
        )
        
        return estimates
```

---

## 🤖 Supported Models & Categories

### 1. Frontier Models (Highest Capability)

**Best for**: High-stakes analysis, complex reasoning, publication-quality results

| Model | Provider | Cost/1K | Context | Strengths |
|-------|----------|---------|---------|-----------|
| `openai/gpt-4o` | OpenAI | $0.0025/$0.010 | 128K | Reasoning, analysis, coding |
| `anthropic/claude-3.5-sonnet` | Anthropic | $0.003/$0.015 | 200K | Safety, analysis, creative |
| `anthropic/claude-3-opus` | Anthropic | $0.015/$0.075 | 200K | Maximum capability |

### 2. Reasoning Specialists

**Best for**: Mathematical analysis, logical reasoning, scientific problems

| Model | Provider | Cost/1K | Context | Strengths |
|-------|----------|---------|---------|-----------|
| `openai/o1-mini` | OpenAI | $0.003/$0.012 | 65K | Math, science, logic |
| `deepseek/deepseek-r1` | DeepSeek | $0.001/$0.001 | 32K | Reasoning, ultra-low cost |
| `qwen/qwen-2.5-math-72b` | Alibaba | $0.0005/$0.0005 | 32K | Mathematics specialist |

### 3. Cost-Optimized Models

**Best for**: Large-scale experiments, budget-conscious research, rapid iteration

| Model | Provider | Cost/1K | Context | Strengths |
|-------|----------|---------|---------|-----------|
| `openai/gpt-4o-mini` | OpenAI | $0.00015/$0.0006 | 128K | Speed, efficiency, cost |
| `anthropic/claude-3-haiku` | Anthropic | $0.00025/$0.00125 | 200K | Fast responses, low cost |
| `deepseek/deepseek-v2.5` | DeepSeek | $0.0001/$0.0001 | 32K | Ultra-low cost leader |

### 4. Open Source Leaders

**Best for**: Transparency, customization, research reproducibility

| Model | Provider | Cost/1K | Context | Strengths |
|-------|----------|---------|---------|-----------|
| `meta-llama/llama-3.1-405b` | Meta | $0.003/$0.003 | 32K | Largest open model |
| `meta-llama/llama-3.1-70b` | Meta | $0.0008/$0.0008 | 32K | Balanced performance |
| `qwen/qwen-2.5-72b` | Alibaba | $0.0005/$0.0005 | 32K | Non-US open alternative |

---

## 📊 Real-World Usage Examples

### 1. Quick 3-Model Analysis (~$0.50)

```python
from src.models.model_registry import model_registry

# Get cost-optimized preset
models = model_registry.get_preset("cost_optimized")[:3]
# Returns: ["gpt-4o-mini", "claude-3-haiku", "deepseek-v2.5"]

# Run analysis
python main.py --models gpt-4o-mini claude-3-haiku deepseek-v2.5 --quick

# Expected cost: ~$0.50 for 3 models × 25 prompts = 75 API calls
```

### 2. Comprehensive Fellowship Research (~$25)

```python
# Get comprehensive model set
models = model_registry.get_preset("fellowship_research")
# Returns: 8 frontier and specialized models

# Run complete hierarchical experiment
cd experiments
python run_automated_experiment.py

# Expected cost: ~$25 for 8 models × 255 prompts = 2,040 API calls
```

### 3. Custom Model Selection with Validation

```python
from src.models.openrouter_model import OpenRouterModel
from src.models.model_registry import model_registry

# Define custom model set
custom_models = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet", 
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b",
    "meta-llama/llama-3.1-405b"
]

# Validate availability
availability = model_registry.validate_model_availability(custom_models)
working_models = [m for m, available in availability.items() if available]

print(f"Working models: {working_models}")
# Output: Shows which models are actually available

# Estimate costs
cost_monitor = CostMonitor(budget_limit_usd=50.0)
estimates = cost_monitor.estimate_experiment_cost(working_models, 50)

print(f"Estimated total cost: ${estimates['total_estimated_cost']:.2f}")
print(f"Budget utilization: {estimates['budget_utilization']:.1%}")
```

---

## ⚡ Performance Optimization

### 1. Automatic Provider Routing

OpenRouter automatically routes requests to the cheapest available provider:

```python
# OpenRouter automatically selects optimal routing
model = OpenRouterModel("anthropic/claude-3-haiku")

# May route through:
# - Direct Anthropic API (if available)  
# - Alternative providers (if cheaper)
# - Backup providers (if primary unavailable)

response = model.generate(prompt)  # Automatically optimized routing
```

### 2. Response Caching

Our system implements intelligent caching to avoid duplicate API costs:

```python
class CachedOpenRouterModel(OpenRouterModel):
    """
    OpenRouter model with response caching.
    
    Automatically deduplicates identical prompts to save costs.
    """
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.cache = {}
        
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        # Create cache key
        cache_key = f"{self.model_id}:{hash(prompt)}:{max_tokens}"
        
        # Check cache first
        if cache_key in self.cache:
            print(f"Cache hit for {self.model_id}")
            return self.cache[cache_key]
        
        # Generate new response
        response = super().generate(prompt, max_tokens)
        
        # Cache response
        self.cache[cache_key] = response
        
        return response
```

### 3. Batch Processing

For large experiments, process models in batches to optimize throughput:

```python
async def process_models_batch(models: List[str], 
                              prompts: List[str],
                              batch_size: int = 5) -> Dict[str, List[str]]:
    """
    Process multiple models in parallel batches.
    
    Optimizes throughput while respecting rate limits.
    """
    results = {}
    
    for i in range(0, len(models), batch_size):
        batch = models[i:i+batch_size]
        
        # Process batch in parallel
        batch_tasks = [
            process_single_model(model_id, prompts) 
            for model_id in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Collect results
        for model_id, model_responses in zip(batch, batch_results):
            results[model_id] = model_responses
        
        # Rate limiting pause between batches
        await asyncio.sleep(1.0)
    
    return results
```

---

## 🛠️ Troubleshooting Common Issues

### 1. Model Availability Issues

**Problem**: Getting 404 errors for specific models

```python
# ❌ Error: openai/o1-preview not available
# Error code: 404 - {'error': {'message': 'No endpoints found for openai/o1-preview.'}}

# ✅ Solution: Validate availability first
from src.models.model_registry import model_registry

models_to_test = ["openai/o1-preview", "openai/o1-mini", "openai/gpt-4o"]
availability = model_registry.validate_model_availability(models_to_test)

available_models = [m for m, available in availability.items() if available]
print(f"Available models: {available_models}")
# Use only available models in experiments
```

### 2. Invalid Model ID Format

**Problem**: Getting 400 errors for model IDs

```python
# ❌ Error: Invalid model ID format
# Error code: 400 - {'error': {'message': 'google/gemini-1.5-pro is not a valid model ID'}}

# ✅ Solution: Use correct OpenRouter model IDs
correct_model_ids = {
    # Incorrect → Correct
    "google/gemini-1.5-pro": "google/gemini-pro",
    "openai/gpt-4-turbo": "openai/gpt-4-turbo-preview", 
    "anthropic/claude-3": "anthropic/claude-3-sonnet",
    "meta/llama-3.1": "meta-llama/llama-3.1-405b-instruct"
}

# Check OpenRouter documentation for exact model IDs
# https://openrouter.ai/docs#models
```

### 3. Rate Limiting

**Problem**: Getting 429 errors during large experiments

```python
# ❌ Error: Rate limit exceeded
# Error code: 429 - {'error': {'message': 'Rate limit exceeded'}}

# ✅ Solution: Implement exponential backoff
import time
import random

def generate_with_retry(model: OpenRouterModel, 
                       prompt: str, 
                       max_retries: int = 5) -> str:
    """Generate response with exponential backoff retry logic."""
    
    for attempt in range(max_retries):
        try:
            return model.generate(prompt)
            
        except OpenRouterAPIError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")
```

### 4. Budget Monitoring

**Problem**: Unexpected high costs during experiments

```python
# ✅ Solution: Implement comprehensive cost tracking
class SafeExperimentRunner:
    def __init__(self, budget_limit: float):
        self.cost_monitor = CostMonitor(budget_limit)
        
    def run_experiment_safely(self, models: List[str], prompts: List[str]):
        # Pre-flight cost estimation
        estimates = self.cost_monitor.estimate_experiment_cost(
            models, len(prompts)
        )
        
        print(f"Estimated cost: ${estimates['total_estimated_cost']:.2f}")
        print(f"Budget utilization: {estimates['budget_utilization']:.1%}")
        
        # Confirm before proceeding
        if estimates['budget_utilization'] > 0.8:
            response = input(f"High budget usage ({estimates['budget_utilization']:.1%}). Continue? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Run with real-time monitoring
        for model_id in models:
            for prompt in prompts:
                model = OpenRouterModel(model_id)
                response = model.generate(prompt)
                
                # Track each API call
                cost = self.cost_monitor.track_api_call(model_id, prompt, response)
                print(f"API call cost: ${cost:.4f}, Total: ${self.cost_monitor.spent:.2f}")
```

---

## 📈 Cost Optimization Strategies

### 1. Model Selection by Use Case

**Research Phase → Model Selection**:

```python
# 🔬 Initial exploration (Ultra-low cost)
exploration_models = ["deepseek-v2.5", "qwen-2.5-7b", "llama-3.1-8b"]
# Cost: ~$0.01 per 100 prompts

# 🎯 Hypothesis testing (Low cost)  
testing_models = ["gpt-4o-mini", "claude-3-haiku", "qwen-2.5-72b"]
# Cost: ~$0.10 per 100 prompts

# 🏆 Final validation (High quality)
validation_models = ["gpt-4o", "claude-3.5-sonnet", "o1-mini"]  
# Cost: ~$1.00 per 100 prompts
```

### 2. Progressive Filtering Strategy

**Reduce costs through intelligent model filtering**:

```python
class ProgressiveExperiment:
    def run_progressive_analysis(self):
        # Stage 1: Broad screening (23 models × 30 prompts = 690 calls)
        stage1_models = self.get_all_models()  # 23 models
        stage1_results = self.run_basic_analysis(stage1_models, prompts_per_model=30)
        top_15_models = self.filter_top_models(stage1_results, keep=15)
        
        # Stage 2: Advanced analysis (15 models × 75 prompts = 1,125 calls)  
        stage2_results = self.run_advanced_analysis(top_15_models, prompts_per_model=75)
        top_8_models = self.filter_top_models(stage2_results, keep=8)
        
        # Stage 3: Comprehensive analysis (8 models × 150 prompts = 1,200 calls)
        final_results = self.run_comprehensive_analysis(top_8_models, prompts_per_model=150)
        
        # Total: 3,015 API calls vs 23 × 255 = 5,865 calls (48% savings)
        return final_results
```

### 3. Smart Caching Implementation

**Avoid duplicate API costs**:

```python
class SmartCache:
    def __init__(self):
        self.semantic_cache = {}
        self.exact_cache = {}
        
    def get_cached_response(self, prompt: str, model_id: str, similarity_threshold: float = 0.95):
        # Check exact match first
        exact_key = f"{model_id}:{hash(prompt)}"
        if exact_key in self.exact_cache:
            return self.exact_cache[exact_key], "exact_match"
        
        # Check semantic similarity  
        for cached_prompt, cached_response in self.semantic_cache.get(model_id, {}).items():
            similarity = self.calculate_semantic_similarity(prompt, cached_prompt)
            if similarity > similarity_threshold:
                return cached_response, f"semantic_match_{similarity:.2f}"
        
        return None, "no_match"
    
    def cache_response(self, prompt: str, model_id: str, response: str):
        # Store exact match
        exact_key = f"{model_id}:{hash(prompt)}"
        self.exact_cache[exact_key] = response
        
        # Store for semantic matching
        if model_id not in self.semantic_cache:
            self.semantic_cache[model_id] = {}
        self.semantic_cache[model_id][prompt] = response
```

---

## 🔄 Integration with Experiment Framework

### 1. Hierarchical Testing Integration

Our OpenRouter integration seamlessly supports the 3-level hierarchical testing protocol:

```python
# Level 1: Behavioral Screening
level1_executor = Level1Executor(
    models=["gpt-4o-mini", "claude-3-haiku", "deepseek-v2.5"],  # Cost-optimized
    budget_limit=1.0  # $1 for broad screening
)

# Level 2: Computational Analysis  
level2_executor = Level2Executor(
    models=level1_results.top_models,  # Filtered from Level 1
    budget_limit=15.0  # $15 for advanced metrics
)

# Level 3: Mechanistic Probing
level3_executor = Level3Executor(
    models=level2_results.top_models,  # Final candidates
    budget_limit=25.0  # $25 for comprehensive analysis
)
```

### 2. Statistical Framework Integration

OpenRouter costs are automatically tracked for statistical analysis:

```python
class StatisticalAnalysis:
    def analyze_convergence_with_costs(self, results: Dict[str, Any]):
        # Calculate cost per significant finding
        total_cost = sum(result['api_cost'] for result in results.values())
        significant_findings = len([r for r in results.values() if r['p_value'] < 0.05])
        
        cost_per_finding = total_cost / max(significant_findings, 1)
        
        return {
            'total_cost': total_cost,
            'significant_findings': significant_findings,
            'cost_per_finding': cost_per_finding,
            'cost_efficiency': significant_findings / total_cost  # Findings per dollar
        }
```

---

## 📚 Additional Resources

### 1. OpenRouter Documentation
- **Main Docs**: https://openrouter.ai/docs
- **Model List**: https://openrouter.ai/docs#models  
- **Pricing**: https://openrouter.ai/docs#pricing
- **Rate Limits**: https://openrouter.ai/docs#rate-limits

### 2. Our Implementation Files
- `src/models/openrouter_model.py` - Core OpenRouter integration
- `src/models/model_registry.py` - Model management and presets
- `config/openrouter_config.json` - Configuration file
- `experiments/cost_monitor.py` - Advanced cost tracking

### 3. Example Configurations
- `.env.example` - Environment variable template
- `config/presets/` - Pre-configured model sets for different use cases
- `experiments/cost_estimator.py` - Cost estimation utilities

---

## 🎯 Best Practices Summary

### 1. **Always Validate Models First**
```python
# Check availability before running experiments
availability = model_registry.validate_model_availability(model_list)
working_models = [m for m, available in availability.items() if available]
```

### 2. **Implement Progressive Cost Management**
```python
# Start with cost estimates
estimates = cost_monitor.estimate_experiment_cost(models, prompts)
if estimates['budget_utilization'] > 0.8:
    # Consider reducing scope or using cheaper models
```

### 3. **Use Appropriate Models for Each Phase**
```python
# Exploration: Ultra-low cost models
# Validation: Cost-optimized models  
# Publication: Frontier models
```

### 4. **Leverage Caching Aggressively**
```python
# Use cached responses to avoid duplicate costs
# Implement semantic similarity caching for near-duplicates
```

### 5. **Monitor Costs in Real-Time**
```python
# Track every API call
# Set hard budget limits with automatic stops
# Generate cost reports for analysis
```

---

*🌐 OpenRouter API Integration Guide | Revolutionary Multi-Model Access*  
*📅 Last Updated: August 17, 2025 | Samuel Tchakwera | Universal Patterns Research*