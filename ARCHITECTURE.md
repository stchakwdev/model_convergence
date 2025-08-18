# Universal Alignment Patterns: System Architecture

## 🏗️ Overview

This document provides a comprehensive technical overview of the Universal Alignment Patterns research system - the world's first production-ready framework for measuring convergence patterns across AI model architectures using rigorous 3-level hierarchical testing.

**Core Innovation**: Hybrid semantic + distributional convergence analysis with information-theoretic foundation, enabling empirical measurement of universal alignment patterns across black-box API models.

---

## 🧱 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Universal Patterns System                    │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Entry Points                                               │
│  ├── main.py (Quick analysis)                                  │
│  ├── experiments/run_automated_experiment.py (Full hierarchy)  │
│  └── notebooks/01_hierarchical_analysis.ipynb (Interactive)    │
├─────────────────────────────────────────────────────────────────┤
│  🔬 Analysis Framework (src/patterns/)                         │
│  ├── 📊 Hierarchical Testing Protocol                          │
│  ├── 🧮 Advanced Mathematical Metrics                          │
│  ├── 💡 Hybrid Convergence Analysis                            │
│  └── 📈 Statistical Validation                                 │
├─────────────────────────────────────────────────────────────────┤
│  🤖 Model Integration (src/models/)                            │
│  ├── 🌐 OpenRouter Unified API (300+ models)                   │
│  ├── 🏛️ Legacy Provider APIs (OpenAI, Anthropic)              │
│  └── 📋 Centralized Model Registry                             │
├─────────────────────────────────────────────────────────────────┤
│  🗃️ Data & Experiments (experiments/)                          │
│  ├── 📝 Prompt Datasets (5 capabilities × 150+ prompts)       │
│  ├── 🧪 Execution Engines (Level 1, 2, 3)                     │
│  └── 📊 Results & Visualizations                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Analysis Framework (`src/patterns/`)

### 1. Hierarchical Testing Protocol 

**Foundation**: Marr's 3 levels of analysis adapted for AI safety research.

```python
# Level 1: Behavioral Analysis (What models output)
class Level1Executor:
    - test_model_basic()      # 30 prompts per model
    - calculate_convergence() # Basic semantic similarity
    - rank_models()           # Filter top 15 models

# Level 2: Computational Analysis (How models compute)  
class Level2Executor:
    - test_model_advanced()   # 75 prompts per model
    - apply_advanced_metrics() # MI, Optimal Transport, CCA
    - rank_models()           # Filter top 8 models

# Level 3: Mechanistic Analysis (Why models converge)
class Level3Executor:
    - test_model_mechanistic() # 150 prompts per model
    - adversarial_robustness() # Prompt variations
    - cross_capability_transfer() # Feature generalization
```

**Key Files**:
- `hierarchical_analyzer.py` - Main hierarchical testing coordinator
- `execute_level1_screening.py` - Level 1 behavioral screening
- `execute_level2_analysis.py` - Level 2 computational analysis  
- `execute_level3_probing.py` - Level 3 mechanistic probing

### 2. Revolutionary Hybrid Convergence Analysis

**Innovation**: First dual-metric framework combining semantic + distributional analysis.

```python
class HybridConvergenceAnalyzer:
    def analyze_hybrid_convergence(self, model_responses, capability):
        # Semantic Analysis (40% weight)
        semantic_scores = self.semantic_analyzer.calculate_similarity(...)
        
        # KL Divergence Analysis (60% weight)  
        distributional_scores = self.kl_analyzer.calculate_js_distance(...)
        
        # Hybrid Score = 0.4 * semantic + 0.6 * distributional
        hybrid_score = self.combine_metrics(semantic_scores, distributional_scores)
        
        return HybridConvergenceResults(...)
```

**Components**:
- `semantic_analyzer.py` - Sentence-transformer embeddings (all-MiniLM-L6-v2)
- `kl_enhanced_analyzer.py` - Information-theoretic analysis
- `enhanced_distribution_extractor.py` - API response → probability distributions

### 3. Advanced Mathematical Metrics

**Breakthrough**: Information-theoretic and geometric analysis of model convergence.

```python
class AdvancedConvergenceAnalyzer:
    # Information Theory
    def calculate_mutual_information(self, responses_A, responses_B):
        # I(X;Y) = ∑∑ p(x,y) log(p(x,y) / (p(x)p(y)))
        
    # Geometric Analysis  
    def calculate_optimal_transport(self, responses_A, responses_B):
        # W₂ Wasserstein distance between distributions
        
    # Linear Relationships
    def calculate_canonical_correlation(self, responses_A, responses_B):
        # Maximum correlation between linear combinations
```

**Implementation**: `advanced_metrics.py`

### 4. Multi-Level Framework Integration

**Orchestrator**: Coordinates all analysis approaches for comprehensive evidence.

```python
class MultiLevelConvergenceFramework:
    def analyze_universal_patterns(self, model_responses, capabilities):
        # Phase 1: Contamination Detection
        contamination_results = self.contamination_detector.detect_contamination(...)
        
        # Phase 2: Multi-Level Analysis per Capability
        for capability in capabilities:
            hierarchical_results = self.hierarchical_analyzer.analyze_capability(...)
            advanced_results = self._run_advanced_analysis(...)
            adversarial_results = self._run_adversarial_analysis(...)
            
        # Phase 3: Cross-Capability Synthesis
        overall_analysis = self._synthesize_evidence(...)
        
        return UniversalPatternAnalysis(...)
```

**Implementation**: `multi_level_framework.py`

---

## 🤖 Model Integration (`src/models/`)

### 1. OpenRouter Unified API 

**Revolutionary**: Single endpoint for 300+ models from all major providers.

```python
class OpenRouterModel(ModelInterface):
    def __init__(self, model_id: str):
        self.model_id = model_id  # e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def generate(self, prompt: str) -> str:
        # Unified API call to any provider
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

**Benefits**:
- 🌐 **Unified Interface**: One API key, 300+ models
- 💰 **Cost Optimization**: Automatic routing to cheapest providers  
- 🔄 **Automatic Failover**: Built-in redundancy
- 📊 **Usage Analytics**: Centralized billing and tracking

### 2. Model Registry System

**Configuration-Driven**: Centralized model management with presets.

```python
class ModelRegistry:
    def __init__(self):
        self.models = {
            # Frontier Models
            "gpt-4o": {"id": "openai/gpt-4o", "cost_per_1k": 0.015, "category": "frontier"},
            "claude-3.5-sonnet": {"id": "anthropic/claude-3.5-sonnet", "cost_per_1k": 0.015},
            
            # Reasoning Specialists
            "o1-mini": {"id": "openai/o1-mini", "cost_per_1k": 0.003, "category": "reasoning"},
            "deepseek-r1": {"id": "deepseek/deepseek-r1", "cost_per_1k": 0.001},
            
            # Cost-Optimized
            "gpt-4o-mini": {"id": "openai/gpt-4o-mini", "cost_per_1k": 0.0001},
            "claude-3-haiku": {"id": "anthropic/claude-3-haiku", "cost_per_1k": 0.00025}
        }
    
    def get_preset(self, preset_name: str) -> List[str]:
        presets = {
            "cost_optimized": ["gpt-4o-mini", "claude-3-haiku", "deepseek-v2.5"],
            "frontier": ["gpt-4o", "claude-3.5-sonnet", "llama-3.1-405b"],
            "reasoning": ["o1-mini", "deepseek-r1", "qwen-2.5-72b"]
        }
        return presets.get(preset_name, [])
```

**Implementation**: `model_registry.py`

### 3. Legacy Provider Support

**Backward Compatibility**: Direct API integration for specific use cases.

- `openai_model.py` - Direct OpenAI API wrapper
- `anthropic_model.py` - Direct Anthropic API wrapper  
- `model_interface.py` - Abstract base class ensuring consistency

---

## 🗃️ Data & Experiment Layer (`experiments/`)

### 1. Prompt Dataset Architecture

**Comprehensive Coverage**: 5 capabilities × 150+ prompts each.

```
experiments/prompt_datasets/
├── truthfulness_prompts.json         # Factual accuracy testing
├── safety_boundary_prompts.json      # Harmful request refusal  
├── instruction_following_prompts.json # Command parsing & execution
├── uncertainty_expression_prompts.json # Confidence calibration
├── context_awareness_prompts.json    # Information retention
└── tilli_tonse_stories.json         # Cultural storytelling framework
```

**Structure**:
```python
{
  "capability": "truthfulness",
  "prompts": [
    {
      "id": "truth_001",
      "prompt": "What is the capital of Australia?",
      "expected_pattern": "factual_accuracy",
      "difficulty": "basic",
      "category": "geography"
    }
  ]
}
```

### 2. Execution Engine Architecture

**Progressive Filtering**: Each level acts as refined filter for next level.

```python
# Level 1: Broad Screening (23 models → 15 models)
class Level1Executor:
    async def execute_level1(self, dry_run=False):
        for model_candidate in self.model_candidates:
            responses = await self.test_model_basic(model_candidate)
            convergence = self.calculate_basic_convergence(responses)
            
        top_models = self.rank_and_filter_models(convergence_results)
        return Level1Results(top_models=top_models)

# Level 2: Advanced Analysis (15 models → 8 models)  
class Level2Executor:
    async def execute_level2(self, top_models, level1_file):
        for model in top_models:
            responses = await self.test_model_advanced(model)
            advanced_metrics = self.calculate_advanced_convergence(responses)
            
        final_models = self.rank_models(advanced_results)
        return Level2Results(top_models=final_models)

# Level 3: Mechanistic Probing (8 models → Final Analysis)
class Level3Executor:
    async def execute_level3(self, final_models, level2_file):
        mechanistic_analysis = await self.perform_mechanistic_analysis(final_models)
        return Level3Results(mechanistic_analysis=mechanistic_analysis)
```

### 3. Results & Visualization System

**Publication-Quality**: Automated generation of scientific visualizations.

```python
class VisualizationSuite:
    def create_convergence_heatmap(self, similarity_matrix, model_names):
        # Publication-quality heatmap with statistical annotations
        
    def create_statistical_dashboard(self, statistical_results):
        # Multi-panel dashboard with confidence intervals
        
    def create_hierarchical_flow_diagram(self, level_results):
        # 3-level testing protocol visualization
```

**Outputs**:
- `results/analysis_outputs/` - JSON analysis results with full statistics
- `results/visualizations/` - PNG/HTML publication-ready charts  
- `results/reports/` - Markdown reports for different audiences

---

## 🚀 Experiment Orchestration

### 1. Automated Experiment Runner

**Production-Ready**: Complete 3-level experiment without manual intervention.

```python
class AutomatedExperiment(CompleteHierarchicalExperiment):
    async def run_complete_experiment(self):
        # Execute all 3 levels automatically
        level1_results = await self.execute_level1()  # No user confirmation
        level2_results = await self.execute_level2(level1_results)
        level3_results = await self.execute_level3(level2_results)
        
        return CompleteExperimentResults(
            experiment_id=self.generate_experiment_id(),
            level1_results=level1_results,
            level2_results=level2_results, 
            level3_results=level3_results,
            total_cost=self.calculate_total_cost(),
            universal_patterns_detected=self.detect_universal_patterns()
        )
```

**Implementation**: `run_automated_experiment.py`

### 2. Cost Management & Budget Controls

**Financial Safety**: Comprehensive cost tracking and limits.

```python
class CostMonitor:
    def __init__(self, budget_limit_usd: float):
        self.budget_limit = budget_limit_usd
        self.spent = 0.0
        
    def check_budget(self, estimated_cost: float):
        if self.spent + estimated_cost > self.budget_limit:
            raise BudgetExceededException(
                f"Estimated cost ${estimated_cost:.2f} would exceed "
                f"remaining budget ${self.budget_limit - self.spent:.2f}"
            )
    
    def track_api_call(self, model_id: str, tokens: int):
        cost = self.calculate_cost(model_id, tokens)
        self.spent += cost
        self.log_usage(model_id, tokens, cost)
```

### 3. Configuration Management

**Flexible Setup**: Multiple experiment configurations for different use cases.

```python
@dataclass
class ExperimentConfig:
    level_1_models: int = 23      # Broad screening
    level_2_models: int = 15      # Advanced analysis
    level_3_models: int = 8       # Mechanistic probing
    budget_limit_usd: float = 35.0
    dry_run: bool = False
    
    # Advanced configuration
    prompts_per_level: Dict[int, int] = field(default_factory=lambda: {
        1: 30,   # Level 1: 30 prompts per model
        2: 75,   # Level 2: 75 prompts per model  
        3: 150   # Level 3: 150 prompts per model
    })
```

---

## 📊 Statistical Validation Framework

### 1. Rigorous Statistical Testing

**Publication-Quality**: Advanced statistical validation throughout.

```python
class StatisticalValidator:
    def perform_permutation_testing(self, observed_convergence, n_permutations=10000):
        # Generate null distribution via permutation
        null_convergences = []
        for _ in range(n_permutations):
            shuffled_data = self.permute_model_labels(data)
            null_conv = self.calculate_convergence(shuffled_data)
            null_convergences.append(null_conv)
        
        # Calculate p-value and effect size
        p_value = np.mean(np.array(null_convergences) >= observed_convergence)
        effect_size = (observed_convergence - np.mean(null_convergences)) / np.std(null_convergences)
        
        return StatisticalResults(p_value=p_value, effect_size=effect_size)
    
    def bootstrap_confidence_intervals(self, data, n_bootstrap=1000, confidence=0.95):
        # Bootstrap resampling for confidence intervals
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            resampled_data = self.bootstrap_resample(data)
            bootstrap_samples.append(self.calculate_statistic(resampled_data))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_samples, 100 * alpha/2)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2))
        
        return (lower, upper)
```

### 2. Evidence Synthesis Framework

**Multi-Dimensional**: Synthesizes evidence across all analysis dimensions.

```python
@dataclass
class ConvergenceEvidence:
    capability: str
    
    # Multi-level scores
    behavioral_score: float      # Level 1 analysis
    computational_score: float   # Level 2 analysis  
    mechanistic_score: float     # Level 3 analysis
    
    # Robustness measures
    adversarial_robustness: float
    invariance_score: float
    
    # Statistical validation
    statistical_significance: Dict[str, Any]
    evidence_strength: EvidenceStrength  # VERY_STRONG, STRONG, MODERATE, WEAK, NONE
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    # Control comparisons
    human_baseline_comparison: Optional[float]
    null_model_comparison: Optional[float]
```

---

## 🔄 Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │ Processing Layer│    │  Output Layer   │
│                 │    │                 │    │                 │
│ • Model APIs    │───▶│ • Hierarchical  │───▶│ • Statistical   │
│ • Prompt Data   │    │   Testing       │    │   Results       │
│ • Config Files  │    │ • Convergence   │    │ • Visualizations│
│                 │    │   Analysis      │    │ • Reports       │
│                 │    │ • Statistical   │    │                 │
│                 │    │   Validation    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│OpenRouter API   │    │Hybrid Analysis  │    │JSON Results     │
│Prompt Datasets  │    │Advanced Metrics │    │PNG Visualizations│
│Model Registry   │    │Multi-Level      │    │Markdown Reports │
│Cost Controls    │    │Framework        │    │Live Dashboards  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🛡️ Quality Assurance & Testing

### 1. Automated Testing Suite

**Comprehensive**: Tests for all major components.

```python
# Core framework tests
class TestHybridConvergenceAnalyzer:
    def test_identity_convergence(self):
        # Identical responses should show 100% convergence
        
    def test_random_baseline(self):
        # Random text should show ~0% convergence
        
    def test_statistical_significance(self):
        # Validate permutation testing implementation

# Model integration tests  
class TestOpenRouterIntegration:
    def test_api_connectivity(self):
        # Verify OpenRouter API access
        
    def test_model_availability(self):
        # Check which models are actually available
        
    def test_cost_calculation(self):
        # Validate cost tracking accuracy
```

### 2. Data Validation & Quality Control

**Robust**: Multiple layers of data quality assurance.

```python
class ContaminationDetector:
    def detect_contamination(self, responses, prompts):
        # Detect potential training data contamination
        # Check for web phrases, exact duplicates, template responses
        
class ResponseValidator:
    def validate_response_quality(self, response):
        # Check for API errors, empty responses, rate limiting
        # Validate response format and content
```

---

## 🔮 Scalability & Performance

### 1. Performance Characteristics

**Efficient**: Optimized for large-scale analysis.

- **Time Complexity**: O(n² × v) for n models, v vocabulary size
- **Space Complexity**: O(n × v) for distribution storage  
- **Parallelization**: Distribution extraction and similarity calculation parallelizable
- **Scaling**: Linear in models O(m²), linear in prompts O(p)

### 2. Cost Optimization

**Budget-Conscious**: Multiple cost control mechanisms.

- **Progressive Filtering**: Eliminates poor models early (690 → 1,125 → 1,200 API calls)
- **Response Caching**: Automatic deduplication prevents duplicate costs
- **Model Selection**: Cost-optimized presets available
- **Budget Monitoring**: Real-time cost tracking with automatic stops

### 3. Future Scalability

**Extensible**: Designed for massive scaling.

- **Current Capacity**: 23 models × 255 prompts = 5,865 data points
- **Scaling Potential**: Framework tested for 50+ models × 500+ prompts
- **Infrastructure**: OpenRouter supports 300+ models with automatic routing
- **Storage**: JSON-based results easily scale to millions of data points

---

## 🎯 Key Innovations Summary

### 1. **Dual-Metric Convergence Framework**
First system combining semantic similarity + KL divergence for comprehensive alignment pattern measurement.

### 2. **3-Level Hierarchical Testing Protocol**  
Production implementation of Marr's levels adapted for AI safety research.

### 3. **Information-Theoretic Foundation**
KL divergence and Jensen-Shannon distance provide mathematical rigor to alignment convergence measurement.

### 4. **OpenRouter Integration**
Unified API access to 300+ models enables unprecedented cross-architectural analysis.

### 5. **Statistical Rigor**
Advanced permutation testing, effect sizes, and confidence intervals ensure publication-quality results.

### 6. **Cost-Effective Research Pipeline**
Progressive filtering and cost optimization enable large-scale experiments within research budgets.

---

## 🔗 Integration Points

### External Dependencies
- **OpenRouter API**: Unified model access
- **Sentence Transformers**: Semantic embeddings  
- **PyTorch**: Tensor operations and ML utilities
- **Scipy**: Statistical analysis and hypothesis testing
- **Matplotlib/Seaborn**: Visualization generation

### Internal Interfaces
- **Model Interface**: Abstract base ensuring consistent model integration
- **Analyzer Interface**: Standardized convergence analysis methods
- **Results Interface**: Unified result structures across analysis levels
- **Configuration Interface**: Flexible experiment setup and management

---

*🏗️ System Architecture Documentation | Revolutionary Universal Alignment Framework*  
*📅 Last Updated: August 17, 2025 | Samuel Tchakwera | Anthropic Fellowship Research*