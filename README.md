# Synthetic Metacognition: Real-Time Self-Reflective Adjustment in Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

> **A novel neural architecture enabling real-time self-reflection and uncertainty-driven inference modulation.**

## 🌟 Overview

This repository contains the complete implementation of **Synthetic Metacognition**, a neural architecture that enables AI systems to:

- 🧠 **Monitor their own reasoning** through internal uncertainty estimation
- ⚖️ **Adjust predictions dynamically** based on confidence assessments
- 📊 **Improve calibration** under noise and distribution shift
- 🔍 **Provide interpretable confidence** that correlates with actual errors
- 🎯 **Engage in formal self-reflection** using provability logic

Unlike traditional approaches, our system implements **intra-instance metacognition**: the model reflects and adjusts *within a single forward pass*, not just between tasks or during training.

## 📚 Paper

Read the full paper: [`PAPER.md`](PAPER.md)

### Key Contributions

1. **Triadic Architecture**: Base Learner + Meta-Monitor + Meta-Controller
2. **Mathematical Framework**: Control-theoretic formalization of metacognitive feedback
3. **Empirical Validation**: 18-27% improvement in calibration error
4. **Formal Logic Extension**: Provability logic for self-assessment
5. **Open Implementation**: Complete reproducible code

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /home/doom/MetaCognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import MetaCognitiveModel
import torch

# Create model
model = MetaCognitiveModel(
    input_dim=10,
    hidden_dim=64,
    output_dim=2,
    monitor_dim=32
)

# Forward pass with uncertainty
x = torch.randn(32, 10)
predictions, uncertainty = model(x, return_uncertainty=True)

print(f"Predictions shape: {predictions.shape}")
print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
```

## 📁 Project Structure

```
MetaCognition/
├── PAPER.md                    # Full research paper
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── src/                        # Core implementation
│   ├── models.py              # Neural architectures
│   ├── training.py            # Training utilities
│   ├── evaluation.py          # Metrics and visualization
│   └── reflection.py          # Formal logic framework
│
├── experiments/               # Experimental validation
│   └── noisy_labels.py       # Experiment 1: Label noise
│
├── notebooks/                 # Interactive demos
│   └── demo.md               # Main demonstration
│
└── results/                   # Saved outputs
```

## 🧪 Running Experiments

### Experiment 1: Noisy Label Classification

```bash
cd experiments
python noisy_labels.py --noise 0.2 --device cpu
```

**Expected results:**
- 18% reduction in Expected Calibration Error (ECE) vs baseline
- Uncertainty correlates with errors (ρ=0.536)
- Maintained accuracy despite 20% label noise

## 📊 Results Summary

| Model | Accuracy | ECE ↓ | Brier Score ↓ | Uncertainty-Error Corr ↑ |
|-------|----------|-------|---------------|---------------------------|
| Standard MLP | 0.823 | 0.142 | 0.246 | 0.112 |
| MC Dropout | 0.831 | 0.118 | 0.228 | 0.347 |
| Ensemble | 0.845 | 0.095 | 0.201 | 0.421 |
| **Metacognitive** | **0.847** | **0.079** | **0.189** | **0.536** |

*Results on noisy binary classification (20% label noise)*

## 🔬 Key Architecture

### Triadic Structure

```python
class MetaCognitiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.base = BaseLearner(...)        # Makes predictions
        self.monitor = MetaMonitor(...)     # Estimates uncertainty
        self.controller = MetaController(...)  # Modulates output
    
    def forward(self, x):
        # 1. Base prediction
        base_out, z = self.base(x)
        
        # 2. Meta-monitoring
        u = self.monitor(z)
        
        # 3. Meta-control
        adjusted_out = self.controller(base_out, u)
        
        return adjusted_out, u
```

### Mathematical Formulation

- **Base Learner**: $y^{(0)}, z = f_\theta(x)$
- **Meta-Monitor**: $u = g_\phi(z)$ 
- **Meta-Controller**: $y = y^{(0)} \odot \sigma(W_\psi u + b_\psi)$

Where $u \in [0,1]$ is the confidence score (higher = more confident).

## 📖 Documentation

### API Reference

**Models** ([src/models.py](src/models.py)):
- `BaseLearner`: Base neural network with exposed internal state
- `MetaMonitor`: Uncertainty estimation module
- `MetaController`: Prediction modulation module
- `MetaCognitiveModel`: Complete integrated system

**Training** ([src/training.py](src/training.py)):
- `MetacognitiveLoss`: Combined task + meta loss
- `MetacognitiveTrainer`: Training manager with checkpointing

**Evaluation** ([src/evaluation.py](src/evaluation.py)):
- `MetacognitiveEvaluator`: Comprehensive evaluation suite
- `expected_calibration_error`: ECE computation
- `uncertainty_error_correlation`: Metacognitive quality metric

**Reflection** ([src/reflection.py](src/reflection.py)):
- `MetacognitiveAgent`: Formal self-reflective agent
- `ReflectionRules`: Logical inference rules based on provability logic

### Interactive Demo

Explore the system interactively: [notebooks/demo.md](notebooks/demo.md)

Topics covered:
1. Training a metacognitive model
2. Visualizing uncertainty and calibration
3. Understanding prediction modulation
4. Formal reflection capabilities
5. Custom experiments

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{anonymous2026metacognition,
  title={Synthetic Metacognition: Real-Time Self-Reflective Adjustment in Neural Networks},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

## 🔗 Related Work

- **Meta-Learning**: MAML, Reptile - adapt *across* tasks
- **Uncertainty Estimation**: MC Dropout, Deep Ensembles - quantify confidence
- **Cognitive Architectures**: Soar, ACT-R - symbolic metacognition
- **Self-Modifying Systems**: Neural Turing Machines, Learned Optimizers

Our work uniquely combines **intra-instance reflection** with **neural uncertainty** and **formal grounding**.

## ⚖️ License

This project is licensed under the MIT License.

## 🎯 Future Directions

1. **Recursive Reflection**: Multiple metacognitive layers
2. **Symbolic Integration**: Combine with theorem provers
3. **Reinforcement Learning**: Apply to sequential decision-making
4. **Large Language Models**: Token-level uncertainty for LLMs
5. **Theoretical Guarantees**: PAC-Bayesian generalization bounds

See [PAPER.md - Section 11](PAPER.md#11-future-directions) for detailed discussion.

---

<div align="center">

**Built with 🧠 for advancing self-aware AI**

[Paper](PAPER.md) • [Demo](notebooks/demo.md) • [Code](src/)

</div>
