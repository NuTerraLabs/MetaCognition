# Quick Reference - Synthetic Metacognition

## 🚀 One-Minute Setup

```bash
cd /home/doom/MetaCognition
pip install torch numpy scikit-learn matplotlib seaborn tqdm pandas
python run_tests.py  # Verify installation
```

## 📖 Core Concepts

### The Three Components

```python
# 1. BASE LEARNER - Makes predictions
y_base, z = base_learner(x)  # z = internal representation

# 2. META-MONITOR - Estimates uncertainty
u = meta_monitor(z)  # u ∈ [0,1], higher = more confident

# 3. META-CONTROLLER - Adjusts prediction
y_final = meta_controller(y_base, u)  # Modulates based on confidence
```

### Mathematical Flow

```
Input x
  ↓
Base Learner: y⁽⁰⁾, z = f_θ(x)
  ↓
Meta-Monitor: u = g_φ(z)  
  ↓
Meta-Controller: y = y⁽⁰⁾ ⊙ σ(W_ψ·u)
  ↓
Output: (y, u)
```

## 💻 Usage Examples

### Example 1: Basic Prediction

```python
from src.models import MetaCognitiveModel
import torch

model = MetaCognitiveModel(input_dim=10, hidden_dim=64, output_dim=2)
x = torch.randn(1, 10)
prediction, confidence = model(x, return_uncertainty=True)

print(f"Predicted class: {prediction.argmax().item()}")
print(f"Confidence: {confidence.item():.3f}")
```

### Example 2: Training

```python
from src.training import MetacognitiveTrainer, MetacognitiveLoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = MetacognitiveLoss(lambda_meta=0.1)
trainer = MetacognitiveTrainer(model, optimizer, loss_fn)

history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Example 3: Evaluation

```python
from src.evaluation import MetacognitiveEvaluator

evaluator = MetacognitiveEvaluator(model)
metrics = evaluator.evaluate(test_loader)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ECE: {metrics['ece']:.4f}")
print(f"Uncertainty-Error Corr: {metrics['uncertainty_error_corr']:.4f}")

# Visualizations
evaluator.plot_calibration(test_loader)
evaluator.plot_selective_accuracy(test_loader)
```

### Example 4: Formal Reflection

```python
from src.reflection import MetacognitiveAgent

agent = MetacognitiveAgent("MyAgent")
agent.add_belief("prediction_is_correct", confidence=0.85)
agent.reflect_on_belief("prediction_is_correct")
agent.check_coherence()
```

## 🧪 Running Experiments

```bash
# Experiment 1: Noisy labels
cd experiments
python noisy_labels.py --noise 0.2

# With GPU
python noisy_labels.py --device cuda

# Custom noise level
python noisy_labels.py --noise 0.4
```

## 📊 Key Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Accuracy** | Prediction correctness | Higher ↑ |
| **ECE** | Calibration quality | Lower ↓ |
| **Brier Score** | Probabilistic accuracy | Lower ↓ |
| **Uncertainty-Error Corr** | Does uncertainty predict errors? | Higher ↑ |
| **Selective Accuracy** | Performance when filtering by confidence | Higher ↑ |

## 🔧 Customization

### Custom Architecture

```python
model = MetaCognitiveModel(
    input_dim=784,           # MNIST images
    hidden_dim=256,          # Larger capacity
    output_dim=10,           # 10 classes
    monitor_dim=128,         # Monitor network size
    dropout_rate=0.2,        # Regularization
    control_type='gate'      # Or 'additive'
)
```

### Custom Loss Weights

```python
# More emphasis on calibration
loss_fn = MetacognitiveLoss(lambda_meta=0.5)

# Less emphasis on calibration
loss_fn = MetacognitiveLoss(lambda_meta=0.01)
```

### Custom Training

```python
trainer = MetacognitiveTrainer(
    model, 
    optimizer, 
    loss_fn,
    device='cuda',
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
)
```

## 📂 File Navigator

```
Need to...                        Look in...
├─ Understand theory             → PAPER.md
├─ Quick start                   → README.md  
├─ See examples                  → notebooks/demo.md
├─ Modify architecture           → src/models.py
├─ Change training               → src/training.py
├─ Add new metrics               → src/evaluation.py
├─ Explore logic framework       → src/reflection.py
├─ Run experiments               → experiments/
└─ Check dependencies            → requirements.txt
```

## 🎯 Common Tasks

### Task: Compare Multiple Models

```python
from src.evaluation import compare_models, print_comparison_table

models = {
    'Baseline': baseline_model,
    'Metacognitive': meta_model,
    'MC Dropout': dropout_model
}

results = compare_models(models, test_loader)
print_comparison_table(results)
```

### Task: Get Confidence for Single Input

```python
model.eval()
with torch.no_grad():
    pred, conf, reliable = model.predict_with_confidence(
        x, 
        threshold=0.7
    )
    
if reliable:
    print(f"High confidence prediction: {pred.item()}")
else:
    print(f"Low confidence - consider rejecting")
```

### Task: Analyze What Meta-Monitor Learned

```python
model.eval()
activations = []

def hook(module, input, output):
    activations.append(output.detach())

hook_handle = model.monitor.register_forward_hook(hook)

# Run inference
model(test_x)

# Analyze
uncertainties = torch.cat(activations)
print(f"Mean uncertainty: {uncertainties.mean():.3f}")
print(f"Std uncertainty: {uncertainties.std():.3f}")

hook_handle.remove()
```

## 🐛 Troubleshooting

**Problem**: Training loss not decreasing
- Try lower learning rate (0.0001)
- Increase lambda_meta (0.2-0.5)
- Check data normalization

**Problem**: All uncertainties near 0.5
- Meta-loss weight too high - reduce lambda_meta
- Need more training epochs
- Check if monitor network is too small

**Problem**: Poor calibration
- Increase lambda_meta
- Use larger monitor network
- Train longer

**Problem**: Uncertainty doesn't correlate with errors
- Meta-monitor not learning - check gradients
- May need different architecture
- Try different activation functions in monitor

## 📚 Further Reading

1. **Theory**: Read [PAPER.md](PAPER.md) sections 3-6
2. **Implementation**: Study [src/models.py](src/models.py) docstrings
3. **Experiments**: See [experiments/noisy_labels.py](experiments/noisy_labels.py)
4. **Logic**: Explore [src/reflection.py](src/reflection.py)

## 💡 Tips

1. **Start small**: Test on toy datasets before scaling up
2. **Monitor uncertainty**: Plot confidence distributions frequently
3. **Compare baselines**: Always benchmark against standard models
4. **Tune lambda_meta**: This hyperparameter is crucial (try 0.05-0.2)
5. **Check calibration**: Use ECE as primary quality metric

## 🏆 Expected Performance

**Noisy Binary Classification (20% noise)**:
- Accuracy: ~0.84-0.85
- ECE: ~0.08-0.10 (baseline: 0.14)
- Correlation: ~0.50-0.55 (baseline: 0.11)

**Clean Data**:
- Should match baseline accuracy
- Significantly better ECE
- Strong uncertainty-error correlation

---

**Questions?** Check [README.md](README.md) or run [run_tests.py](run_tests.py)
