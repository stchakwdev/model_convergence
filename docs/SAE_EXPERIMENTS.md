# Gemma Model Convergence Experiment

## Multi-Model SAE-Enhanced Pattern Analysis

This project implements an experimental approach to analyze potential convergence patterns across different Gemma model variants using **Sparse Autoencoders (SAEs)** and **mechanistic interpretability** techniques.

### 🔍 Experimental Approach

We combine three levels of analysis to investigate potential universal alignment patterns:

1. **Behavioral Convergence** - How similarly models respond to prompts
2. **Activation Convergence** - Similarity of internal neural activations  
3. **SAE Feature Convergence** - Interpretable feature overlap using Sparse Autoencoders

This multi-level approach may provide insight into potential alignment mechanisms across model architectures.

## 📊 Experiment Overview

### Models Analyzed
- **Gemma 2 2B** (Base & Instruct variants)
- **Gemma 2 9B** (Base & Instruct variants)

### Analysis Framework
- **TransformerLens** - Clean model access and activation extraction
- **Gemma Scope SAEs** - Sparse Autoencoder interpretability
- **Universal Patterns Framework** - Convergence analysis and statistics

### Capabilities Tested
- Truthfulness and factual accuracy
- Safety boundary recognition
- Instruction following
- Uncertainty expression
- Context awareness

## 📋 Getting Started on Runpod

### 1. Launch Runpod Instance
```bash
# Recommended specs:
# - GPU: RTX 4090 or A100 (16GB+ VRAM)
# - Storage: 50GB+
# - Template: PyTorch 2.0+ with CUDA
```

### 2. Setup Environment
```bash
# SSH into your runpod instance
ssh root@<pod-ip> -p <ssh-port>

# Download the setup script
wget https://raw.githubusercontent.com/your-repo/runpod_setup.sh
chmod +x runpod_setup.sh

# Run automated setup
./runpod_setup.sh
```

### 3. Start Experiment
```bash
# Start Jupyter Lab
./start_experiment.sh

# Or run directly
python gemma_convergence_experiment.py --quick
```

### 4. Open Notebook
Navigate to `gemma_convergence_notebook.ipynb` in Jupyter Lab for interactive analysis.

## 📁 Project Structure

```
gemma_convergence/
├── 🐍 gemma_convergence_experiment.py    # Main experiment framework
├── 📓 gemma_convergence_notebook.ipynb   # Interactive analysis notebook
├── 🔧 runpod_setup.sh                    # Automated environment setup
├── 🚀 start_experiment.sh                # Quick start script
├── 📊 monitor_resources.py               # System monitoring
├── 📋 huggingface_setup.md              # Authentication guide
├── 📈 results/                          # Experiment outputs
│   ├── convergence_results_*.json       # Raw data
│   ├── visualizations/                  # Charts and plots
│   └── logs/                           # Experiment logs
└── 📖 GEMMA_CONVERGENCE_README.md       # This file
```

## 🔐 Prerequisites

### 1. HuggingFace Authentication
```python
# In notebook or Python:
from huggingface_hub import notebook_login
notebook_login()
```

### 2. Gemma Model Access
- Create HuggingFace account: https://huggingface.co/join
- Accept Gemma license: https://huggingface.co/google/gemma-2-2b
- Generate access token: https://huggingface.co/settings/tokens

### 3. Hardware Requirements
- **Minimum**: 16GB GPU VRAM for Quick Mode (2 models)
- **Recommended**: 24GB+ GPU VRAM for Full Experiment (4 models)
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and results

## 🧪 Running Experiments

### Quick Test Mode (Recommended First)
```python
# In notebook or command line:
experiment = GemmaConvergenceExperiment()
results = experiment.run_full_experiment(quick_mode=True)
```

**Quick Mode specs:**
- 2 models (2B Base vs 2B Instruct)
- 3 prompts
- ~5-10 minutes runtime
- ~8GB GPU memory

### Full Experiment
```python
# All 4 models, 5 prompts, comprehensive analysis
results = experiment.run_full_experiment(quick_mode=False)
```

**Full Mode specs:**
- 4 models (2B & 9B, Base & Instruct)
- 5 prompts across alignment capabilities
- ~30-60 minutes runtime
- ~20-24GB GPU memory

### Command Line Usage
```bash
# Quick test
python gemma_convergence_experiment.py --quick

# Full experiment
python gemma_convergence_experiment.py

# Custom output directory
python gemma_convergence_experiment.py --output-dir my_results
```

## 📊 Understanding Results

### 1. Behavioral Convergence
**Metric**: Semantic similarity (0-1)
- **>0.7**: Strong behavioral convergence
- **0.4-0.7**: Moderate convergence
- **<0.4**: Low convergence

### 2. SAE Feature Convergence
**Metric**: Jaccard similarity (0-1)
- Measures overlap of active SAE features
- Higher values = more similar interpretable patterns

### 3. Activation Convergence
**Metric**: Cosine similarity (0-1)
- Raw neural activation similarity
- Provides mechanistic insight into convergence

### Example Output
```
🎯 EXPERIMENT SUMMARY:
📅 Timestamp: 2025-08-20T10:30:45
🤖 Models: 2 (Gemma-2-2B-Base, Gemma-2-2B-Instruct)
📝 Prompts: 3

🗣️ BEHAVIORAL CONVERGENCE:
  Average: 0.756 ± 0.112
  Range: 0.623 - 0.891

🧠 SAE FEATURE CONVERGENCE:
  Average: 0.423 ± 0.089
  Range: 0.312 - 0.534

⚡ ACTIVATION CONVERGENCE:
  Average: 0.678 ± 0.145
  Range: 0.501 - 0.823
```

## 🔍 Research Applications

### Universal Pattern Discovery
- Identify alignment mechanisms that transfer across models
- Understand architecture-invariant safety features
- Build foundation for universal alignment strategies

### Safety Research
- Cross-model safety mechanism analysis
- Robustness testing of alignment approaches
- Safety feature transferability studies

### Interpretability Research
- SAE feature analysis across model variants
- Mechanistic understanding of instruction tuning
- Layer-wise convergence pattern analysis

## 🛠️ Customization & Extensions

### Adding New Models
```python
# Extend the GEMMA_MODELS list in gemma_convergence_experiment.py
new_model = GemmaModel(
    name="Custom-Model",
    model_id="organization/model-name",
    size="2B",
    variant="custom",
    sae_repo="sae-repo-id",
    sae_layers=[10, 15, 20]
)
```

### Custom Prompts
```python
# Define your own prompt set
custom_prompts = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ...
]

experiment.run_full_experiment(prompts=custom_prompts)
```

### Additional Analysis
The framework is modular and extensible:
- Add new convergence metrics
- Implement custom SAE analysis
- Extend visualization capabilities
- Integrate with existing ML pipelines

## 📈 Monitoring & Debugging

### Resource Monitoring
```bash
# Real-time system monitoring
python monitor_resources.py
```

### Common Issues & Solutions

**GPU Memory Issues:**
```bash
# Check GPU usage
nvidia-smi

# Free GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Use Quick Mode for testing
python gemma_convergence_experiment.py --quick
```

**Authentication Errors:**
```bash
# Re-authenticate with HuggingFace
huggingface-cli login

# Check model access
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-2-2b')"
```

**Slow Downloads:**
```bash
# Use HF_HUB_CACHE to specify cache location
export HF_HUB_CACHE="/workspace/hf_cache"
```

## 🤝 Contributing

This framework is designed for collaborative research:

1. **Fork the repository**
2. **Add your models/analysis methods**
3. **Test with Quick Mode**
4. **Submit pull request with results**

### Research Areas We're Looking For:
- Additional model families (LLaMA, Claude, GPT)
- New convergence metrics
- Cross-modal analysis (vision, code)
- Temporal analysis across training
- Adversarial robustness testing

## 📚 Technical Details

### Dependencies
```python
# Core ML
torch>=2.0.0
transformers>=4.30.0
transformer-lens
huggingface-hub

# Scientific Computing
numpy
scipy
scikit-learn
pandas

# Visualization
matplotlib
seaborn
plotly

# Utilities
tqdm
jupyter
ipywidgets
```

### SAE Implementation
Uses JumpReLU SAEs from Gemma Scope:
- Pre-trained on residual stream activations
- Multiple layers per model
- 16k feature dimensions
- L0 sparsity ~70 features per input

### Statistical Framework
- Permutation testing for significance
- Bootstrap confidence intervals
- Multiple comparison correction
- Effect size calculations

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{gemma_convergence_2025,
  author = {Samuel Chakwera},
  title = {Gemma Model Convergence Experiment: SAE-Enhanced Universal Pattern Discovery},
  year = {2025},
  url = {https://github.com/your-repo/gemma-convergence}
}
```

## 📞 Support

**Documentation**: See inline code comments and notebook explanations
**Issues**: Open GitHub issues for bugs or feature requests
**Discussions**: Use GitHub Discussions for research questions

## 🏆 Acknowledgments

- **Google DeepMind** for Gemma models and Gemma Scope SAEs
- **TransformerLens** team for mechanistic interpretability tools
- **Anthropic** for inspiration from mechanistic interpretability research
- **RunPod** for accessible GPU compute infrastructure

---

## 🚀 Ready to Discover Universal Patterns?

```bash
# Start your journey
./runpod_setup.sh
./start_experiment.sh
```

**Happy researching! 🔬✨**