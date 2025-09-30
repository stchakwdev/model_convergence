# Investigating Cross-Model Behavioral Convergence in AI Alignment

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Research Question

Do different large language model architectures develop similar behavioral patterns for alignment-relevant features (truthfulness, safety refusals, instruction following)?

## Hypothesis

If alignment features emerge from optimization pressure rather than architecture-specific design, we should observe convergent behavioral patterns across models, even when they have different architectures and training regimes.

## Key Findings

Initial behavioral screening across 23 models reveals:

- **Mean behavioral convergence: 71.3%** (SD = 4.4%)
- **Sample size**: 690 API calls across 23 models × 30 prompts
- **Cost**: $0.31 (cost-optimized via OpenRouter)
- **Range**: 61.1% - 75.9% convergence across model pairs

**Interpretation**: Moderate-to-strong behavioral convergence observed across diverse architectures. Results suggest some universal alignment patterns may exist, though findings require validation with expanded sample sizes and mechanistic investigation.

## Methodology

### Dual-Metric Convergence Framework

We employ a two-component analysis to measure convergence:

1. **Semantic Analysis (40% weight)**: Content similarity using sentence-transformers (all-MiniLM-L6-v2)
2. **Distributional Analysis (60% weight)**: Information-theoretic divergence via KL divergence and Jensen-Shannon distance
3. **Statistical Validation**: Permutation testing, effect sizes (Cohen's d), bootstrap confidence intervals

### Models Tested (Level 1 Screening)

Representative sample across major model families:
- OpenAI (GPT-4o, o1-preview, o1-mini)
- Anthropic (Claude-3.5-Sonnet, Claude-3-Opus)
- Meta (Llama-3.1-405B)
- Google (Gemini-1.5-Pro, Gemini-1.5-Flash)
- DeepSeek (DeepSeek-V2.5, DeepSeek-Coder-V2)
- Qwen (Qwen-2.5-72B, Qwen-2.5-Coder-32B)
- Mistral (Mixtral-8x22B)
- ZhipuAI (GLM-4-Plus)
- Additional open-source models (Yi-Lightning, Phind, Baichuan2)

### Capabilities Analyzed

- **Truthfulness**: Factual accuracy vs fictional content distinction
- **Safety Boundaries**: Refusal mechanisms for harmful requests
- **Instruction Following**: Command parsing and execution consistency
- **Uncertainty Expression**: Appropriate confidence calibration
- **Context Awareness**: Information retention and contextual understanding

## Experimental Results

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed results including:
- Level 1 behavioral screening (completed)
- Statistical significance analysis
- Model-specific convergence scores
- Capability-wise breakdown

### Top Converging Models (Level 1)

| Model | Convergence Score | Architecture Type |
|-------|------------------|-------------------|
| Gemini-1.5-Flash | 75.9% | Transformer (Google) |
| Baichuan2-192k | 75.9% | Transformer (Chinese) |
| DeepSeek-V2.5 | 75.9% | MoE |
| Gemini-1.5-Pro | 75.9% | Transformer (Google) |
| Yi-Lightning | 75.6% | Transformer |
| GLM-4-Plus | 75.6% | Transformer (ZhipuAI) |
| Claude-3.5-Sonnet | 71.0% | Transformer (Anthropic) |
| GPT-4o | 70.7% | Transformer (OpenAI) |

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/stchakwdev/universal_patterns.git
cd universal-alignment-patterns

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

1. **Sample Size**: While 690 API calls across 23 models is substantial for initial screening, statistical power for subgroup analyses is limited
2. **API-Only Access**: Black-box model access prevents mechanistic investigation of internal representations
3. **Behavioral ≠ Mechanistic**: Behavioral convergence does not necessarily imply convergent internal mechanisms
4. **Prompt Selection**: Results may be sensitive to prompt design and selection
5. **Preliminary Findings**: Results require replication and validation before strong claims can be made

## Next Steps

### Immediate Research Directions

1. **Expanded Sample Size**: Increase prompts per capability from 30 to 150+ for greater statistical power
2. **Capability-Specific Analysis**: Decompose convergence by individual alignment features
3. **Scale Analysis**: Test whether convergence increases with model capability/size
4. **Mechanistic Investigation**: For open-source models, probe internal activations where possible

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

- **Level 1 Total Cost**: $0.31 for 690 API calls
- **Average Cost per Call**: ~$0.0004
- **OpenRouter Benefits**: Automatic routing to cheapest providers, unified billing
- **Caching**: Response caching prevents duplicate API costs
- **Scaling Estimate**: Full 3-level protocol ~$30 for 3,000+ API calls

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
  url={https://github.com/stchakwdev/universal_patterns},
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

This research was developed as part of an application to the **MATS Program** (ML Alignment & Theory Scholars). The approach draws inspiration from:

- Mechanistic interpretability research from Anthropic and other alignment organizations
- Statistical methods from epidemiology and global health research
- The broader AI safety community's work on understanding model internals

---

**Note**: This is early-stage exploratory research. Results are preliminary and require validation. We encourage replication, critique, and alternative interpretations of our findings.
