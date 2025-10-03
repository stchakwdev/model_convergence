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
- **Mean behavioral convergence: 11.3%** (SD = 2.6%)
- **Sample size**: 11,167 API calls across 15 latest frontier models × 750 prompts
- **Cost**: $8.93 (cost-optimized via OpenRouter)
- **Statistical validation**: p = 0.596 (not statistically significant), Cohen's d = -2.15
- **Models tested**: GLM-4.5, Deepseek-V3.1, Grok-4-Fast, Gemini-2.5-Flash, Kimi-K2, GPT-4o, Claude-3.5-Sonnet, and 8 others

**Capability-Specific Results**:
- Context Awareness: 22.1%
- Instruction Following: 19.3%
- Truthfulness: 13.7%
- Uncertainty Expression: 8.1%
- Safety Boundaries: 6.2%

**Interpretation**: Low convergence (11.3%, p=0.596) indicates frontier models are largely **divergent** on alignment features. This negative result suggests the hypothesis of strong universal behavioral convergence is not supported by the data. Models show substantial variation in how they handle alignment-relevant tasks, likely reflecting differences in training data, objectives, or architectural constraints.

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

**Substantial Variation Observed**: Models show significant divergence (3.0% - 14.1%)

| Model | Convergence | Architecture | Provider |
|-------|------------|--------------|----------|
| GPT-4o | 14.1% | Transformer | OpenAI |
| Claude-3.5-Sonnet | 13.6% | Transformer | Anthropic |
| Kimi-K2 | 13.5% | MoE | Moonshot |
| Gemini-2.5-Flash | 13.1% | Transformer | Google |
| Llama-3.1-405B | 12.8% | Transformer | Meta |
| Mistral-Large-2411 | 12.5% | MoE | Mistral |
| Mixtral-8x22B | 12.2% | MoE | Mistral |
| Claude-3-Opus | 11.9% | Transformer | Anthropic |
| *...and 7 others* | 3.0%-11.1% | Various | Various |

**Key Insight**: High variance (SD=2.6%, range=11.1%) indicates models are **divergent** in their alignment behaviors. No universal convergence pattern detected. This suggests either: (1) alignment is highly architecture/training-specific, (2) our behavioral metric doesn't capture underlying similarities, or (3) universal patterns don't exist at the behavioral level.

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
2. **Behavioral ≠ Mechanistic**: Behavioral divergence does not necessarily imply divergent internal mechanisms - models may share representations but differ in behavior
3. **Prompt Selection**: Results may be sensitive to prompt design and selection - different prompts might reveal different convergence patterns
4. **Temporal Snapshot**: Single point-in-time measurement; doesn't capture how alignment evolves during training
5. **Causality**: Cannot determine root causes of observed divergence (training data, objectives, architectures, or optimization dynamics)
6. **Metric Limitations**: Text similarity may not capture semantic convergence - models might express similar concepts differently

## Next Steps

Given the low convergence findings, future work should explore:

### Priority Research Questions

1. **Why is convergence low?**
   - Different training data distributions across providers?
   - Different alignment objectives (helpfulness vs. harmlessness tradeoffs)?
   - Architecture-specific alignment mechanisms?
   - Fundamental absence of universal patterns?

2. **Better metrics needed?**
   - Text similarity may miss semantic equivalence
   - Consider embedding-based similarity (sentence transformers, etc.)
   - Behavioral equivalence testing (do models refuse the same prompts?)
   - Clustering analysis to find convergent subgroups

3. **Mechanistic investigation**:
   - Move beyond behavioral analysis to internal representations
   - Activation probing for alignment features
   - Linear representation hypothesis testing
   - Cross-model feature correspondence

4. **Capability-specific analysis**:
   - Context awareness showed highest convergence (22.1%)
   - Safety boundaries showed lowest (6.2%)
   - Focus on specific capabilities instead of averaging

5. **Temporal dynamics**:
   - Track how convergence evolves during training
   - Compare base models to aligned versions
   - Measure convergence at different capability levels

### Alternative Approaches

- **Behavioral equivalence testing**: Do models refuse the same harmful prompts? (binary instead of similarity)
- **Embedding-based convergence**: Use semantic similarity instead of text similarity
- **Adversarial probing**: Test convergence on jailbreak attempts and edge cases
- **Transfer learning experiments**: Can safety features transfer between models?
- **Open model analysis**: Access to weights enables mechanistic interpretability

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
- **Infrastructure Value**: Despite low convergence findings, the data collection pipeline works efficiently and can be repurposed for future experiments

## Author

**Samuel T Chakwera**
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
  author={Tchakwera, Samuel T},
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

**Note**: This is early-stage exploratory research. The negative result (11.3% convergence, not statistically significant) is valuable - it suggests universal behavioral alignment patterns may not exist at the level we hypothesized, or that our text similarity metric doesn't capture underlying convergence. We encourage replication, critique, and alternative interpretations of our findings.
