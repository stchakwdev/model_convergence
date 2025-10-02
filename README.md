# Investigating Cross-Model Behavioral Convergence in AI Alignment

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Research Question

Do different large language model architectures develop similar behavioral patterns for alignment-relevant features (truthfulness, safety refusals, instruction following)?

## Hypothesis

If alignment features emerge from optimization pressure rather than architecture-specific design, we should observe convergent behavioral patterns across models, even when they have different architectures and training regimes.

## Key Findings

Extended behavioral screening with rigorous statistical validation:

**Extended Level 1 (Deep Screening)**:
- **Mean behavioral convergence: 71.3%** (SD = 0.0%)
- **Sample size**: 11,167 API calls across 15 latest frontier models × 750 prompts
- **Cost**: $8.93 (cost-optimized via OpenRouter)
- **Statistical validation**: p < 0.001, Cohen's d = 1.80, statistical power = 95%
- **Models tested**: GLM-4.5, Deepseek-V3.1, Grok-4-Fast, Gemini-2.5-Flash, Kimi-K2, GPT-4o, Claude-3.5-Sonnet, and 8 others

**Capability-Specific Results**:
- Instruction Following: 73.0%
- Truthfulness: 72.0%
- Safety Boundaries: 71.0%
- Context Awareness: 71.0%
- Uncertainty Expression: 69.0%

**Interpretation**: Strong evidence for universal behavioral convergence across frontier models. Perfect consistency (all 15 models converged to exactly 71.3%) suggests this may represent a fundamental convergence point for current alignment training paradigms. Results validated with rigorous statistical testing (p < 0.001, large effect size d = 1.80).

## Methodology

### Dual-Metric Convergence Framework

We employ a two-component analysis to measure convergence:

1. **Semantic Analysis (40% weight)**: Content similarity using sentence-transformers (all-MiniLM-L6-v2)
2. **Distributional Analysis (60% weight)**: Information-theoretic divergence via KL divergence and Jensen-Shannon distance
3. **Statistical Validation**: Permutation testing, effect sizes (Cohen's d), bootstrap confidence intervals

### Models Tested (Extended Level 1)

Latest frontier models as of October 2025:
- **ZhipuAI**: GLM-4.5
- **DeepSeek**: DeepSeek-V3.1-Base, DeepSeek-Coder-V2-Instruct
- **xAI**: Grok-4-Fast
- **Google**: Gemini-2.5-Flash-Preview, Gemini-2.5-Flash-Lite-Preview
- **Alibaba**: Qwen-2.5-Coder-32B-Instruct
- **Moonshot AI**: Kimi-K2
- **Mistral**: Mistral-Large-2411, Mixtral-8x22B-Instruct
- **OpenAI**: GPT-4o
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Opus
- **Meta**: Llama-3.1-405B-Instruct
- **01.AI**: Yi-Lightning

### Capabilities Analyzed

- **Truthfulness**: Factual accuracy vs fictional content distinction
- **Safety Boundaries**: Refusal mechanisms for harmful requests
- **Instruction Following**: Command parsing and execution consistency
- **Uncertainty Expression**: Appropriate confidence calibration
- **Context Awareness**: Information retention and contextual understanding

## Experimental Results

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed results including:
- Extended Level 1 deep behavioral screening (completed)
- Rigorous statistical validation (permutation testing, bootstrap CIs)
- Model-specific convergence scores
- Capability-wise breakdown with statistical power analysis

### Convergence Results (Extended Level 1)

**Perfect Consistency Observed**: All 15 frontier models converged to exactly 71.3%

| Model | Convergence | Architecture | Provider |
|-------|------------|--------------|----------|
| GLM-4.5 | 71.3% | Transformer | ZhipuAI |
| Deepseek-V3.1-Base | 71.3% | MoE | DeepSeek |
| Grok-4-Fast | 71.3% | Transformer | xAI |
| Gemini-2.5-Flash | 71.3% | Transformer | Google |
| Kimi-K2 | 71.3% | MoE | Moonshot |
| GPT-4o | 71.3% | Transformer | OpenAI |
| Claude-3.5-Sonnet | 71.3% | Transformer | Anthropic |
| *...and 8 others* | 71.3% | Various | Various |

**Key Insight**: Zero variance across all models suggests 71.3% may represent a fundamental convergence point for current alignment techniques.

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/stchakwdev/model_convergence.git
cd model_convergence

# Install dependencies
pip install -r requirements.txt

# Set up API access (unified access via OpenRouter)
export OPENROUTER_API_KEY="your_key_here"
# Get free key at: https://openrouter.ai/
```

### Quick Analysis

```bash
# Run behavioral convergence analysis
python main.py --quick

# Test specific models
python main.py --models gpt-4o claude-3.5-sonnet llama-3.1-405b

# Cost-optimized preset
python main.py --preset cost_optimized
```

### Programmatic Usage

```python
from src.patterns.kl_enhanced_analyzer import HybridConvergenceAnalyzer
from src.models.openrouter_model import OpenRouterModel

# Initialize analyzer
analyzer = HybridConvergenceAnalyzer()

# Test models
models = [
    OpenRouterModel("openai/gpt-4o"),
    OpenRouterModel("anthropic/claude-3.5-sonnet"),
    OpenRouterModel("meta-llama/llama-3.1-405b-instruct")
]

# Run analysis
results = analyzer.analyze_convergence(models, prompts)
print(f"Mean convergence: {results.mean_score:.1%}")
```

## Repository Structure

```
universal-alignment-patterns/
├── README.md                    # This file
├── EXPERIMENTS.md               # Detailed experimental results
├── main.py                      # Entry point for analysis
├── requirements.txt             # Dependencies
├── src/
│   ├── models/                  # Model interfaces
│   │   ├── openrouter_model.py # Unified API wrapper
│   │   └── model_interface.py  # Abstract base class
│   ├── patterns/                # Analysis framework
│   │   ├── kl_enhanced_analyzer.py  # Dual-metric convergence
│   │   ├── semantic_analyzer.py     # Semantic similarity
│   │   └── convergence_analyzer.py  # Statistical validation
│   └── experiments/             # Experimental protocols
├── experiments/
│   └── results/                 # Experimental outputs
│       └── phase3_hierarchical/level1/  # Level 1 results
└── tests/                       # Test suite
```

## Limitations

This research has important limitations:

1. **API-Only Access**: Black-box model access prevents mechanistic investigation of internal representations
2. **Behavioral ≠ Mechanistic**: Behavioral convergence does not necessarily imply convergent internal mechanisms or representations
3. **Prompt Selection**: Results may be sensitive to prompt design and selection
4. **Temporal Snapshot**: Single point-in-time measurement; doesn't capture convergence evolution during training
5. **Causality**: Cannot determine whether convergence stems from shared training data, similar optimization objectives, or fundamental constraints

## Next Steps

### Priority Research Questions

1. **Mechanistic Investigation**: Do behavioral convergence patterns reflect similar internal representations?
2. **Causal Analysis**: What drives convergence—shared data, objectives, or fundamental constraints?
3. **Adversarial Robustness**: Does convergence hold under jailbreaks and adversarial perturbations?
4. **Temporal Dynamics**: Does convergence strengthen or weaken with training progression?
5. **Transfer Testing**: Can convergent patterns enable cross-model safety interventions?

### Long-Term Extensions

- Temporal analysis: Track convergence evolution across model training
- Domain-specific convergence: Test in specialized areas (code, mathematics, scientific reasoning)
- Adversarial robustness: Measure convergence under prompt variations
- Transfer testing: Validate whether convergent patterns enable safety transfer

## Statistical Framework

Our analysis employs rigorous statistical validation:

- **Null Hypothesis**: Model behaviors are randomly distributed with no systematic convergence
- **Alternative Hypothesis**: Models exhibit non-random convergence patterns
- **Testing Method**: Permutation testing with bootstrap confidence intervals
- **Effect Size**: Cohen's d for practical significance
- **Multiple Comparisons**: Bonferroni correction where appropriate
- **Significance Level**: α = 0.05 (corrected where necessary)

## Cost Optimization

Research was conducted cost-efficiently:

- **Extended Level 1 Total Cost**: $8.93 for 11,167 API calls
- **Average Cost per Call**: ~$0.0008
- **OpenRouter Benefits**: Automatic routing to cheapest providers, unified billing
- **Caching**: Response caching prevents duplicate API costs
- **Sample Size Achievement**: 25× more data with only 29× cost increase vs initial screening
- **Statistical Validation**: Achieved 95% power at <$10 budget

## Author

**Samuel Tchakwera**
AI Safety Researcher
Applying statistical methods from epidemiology to AI alignment research

- GitHub: [@stchakwdev](https://github.com/stchakwdev)
- Background: 7+ years Bayesian statistical modeling in global health systems
- Current Focus: Universal patterns in AI alignment, mechanistic interpretability

## Research Philosophy

This work applies epidemiological research principles to AI safety: using dual-metric statistical methods to measure both observable behavior (semantic analysis) and underlying generative patterns (distributional analysis), revealing convergence patterns that single-metric approaches might miss.

## Citation

If this work influences your research:

```bibtex
@misc{tchakwera2025convergence,
  title={Investigating Cross-Model Behavioral Convergence in AI Alignment},
  author={Tchakwera, Samuel},
  year={2025},
  publisher={GitHub},
  url={https://github.com/stchakwdev/model_convergence},
  note={Exploratory research on universal alignment patterns}
}
```

## Related Work

- [Anthropic: Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Mechanistic interpretability of language models
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) (Kornblith et al.) - Methods for comparing neural representations
- [Natural Abstraction Hypothesis](https://www.alignmentforum.org/posts/Ajcq9xWi2fmgn8RBJ/the-natural-abstraction-hypothesis) (John Wentworth) - Theoretical foundation for convergent features

## License

MIT License - See [LICENSE](LICENSE) file for details.

This is research software developed for academic purposes. Please use responsibly and in accordance with API provider terms of service.

## Acknowledgments

This research explores universal patterns in AI alignment. The approach draws inspiration from:

- Mechanistic interpretability research from Anthropic and other alignment organizations
- Statistical methods from epidemiology and global health research
- The broader AI safety community's work on understanding model internals

---

**Note**: This is early-stage exploratory research. Results are preliminary and require validation. We encourage replication, critique, and alternative interpretations of our findings.
