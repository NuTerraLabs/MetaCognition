# Synthetic Metacognition: Interactive Demonstration

Welcome to the **Synthetic Metacognition** demonstration notebook! This notebook allows you to:

1. ⚡ Run the metacognitive model on real data
2. 📊 Visualize uncertainty estimates and calibration
3. 🧪 Compare with baseline models
4. 🔍 Explore internal representations
5. 🧠 Test formal reflection capabilities

## Setup

```python
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Import our modules
sys.path.append('..')
from src.models import MetaCognitiveModel, BaselineMLP
from src.training import MetacognitiveTrainer, MetacognitiveLoss
from src.evaluation import MetacognitiveEvaluator
from src.reflection import MetacognitiveAgent, Proposition

# Styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Setup complete")
```

---

## Part 1: Quick Start - Training a Metacognitive Model

Let's start with a simple binary classification problem and see metacognition in action.

```python
# Generate synthetic data
print("Creating dataset with 20% label noise...")
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.2,  # 20% label noise
    class_sep=1.0,
    random_state=42
)

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

```python
# Create and train metacognitive model
print("Training metacognitive model...")

model = MetaCognitiveModel(
    input_dim=2,
    hidden_dim=64,
    output_dim=2,
    monitor_dim=32,
    control_type='gate'
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = MetacognitiveLoss(lambda_meta=0.1)

trainer = MetacognitiveTrainer(model, optimizer, loss_fn, device='cpu')
history = trainer.fit(train_loader, test_loader, epochs=30, verbose=True)

print("✓ Training complete!")
```

---

## Part 2: Visualizing Metacognition

### 2.1 Confidence Calibration

```python
evaluator = MetacognitiveEvaluator(model, device='cpu')

# Plot calibration diagram
evaluator.plot_calibration(test_loader, n_bins=10)
```

### 2.2 Confidence Distribution

```python
model.eval()
all_predictions = []
all_targets = []
all_confidences = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        logits, uncertainty = model(x_batch, return_uncertainty=True)
        preds = torch.argmax(logits, dim=1)
        
        all_predictions.append(preds)
        all_targets.append(y_batch)
        all_confidences.append(uncertainty)

predictions = torch.cat(all_predictions).numpy()
targets = torch.cat(all_targets).numpy()
confidences = torch.cat(all_confidences).squeeze().numpy()

# Separate by correctness
correct = predictions == targets

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(confidences[correct], bins=20, alpha=0.6, label='Correct', color='green', density=True)
ax1.hist(confidences[~correct], bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
ax1.set_xlabel('Confidence Score')
ax1.set_ylabel('Density')
ax1.set_title('Confidence Distribution by Prediction Correctness')
ax1.legend()
ax1.grid(alpha=0.3)

# Statistics
print(f"Correct predictions - Mean confidence: {confidences[correct].mean():.3f}")
print(f"Incorrect predictions - Mean confidence: {confidences[~correct].mean():.3f}")

# 2D scatter
ax2.scatter(X_test[correct, 0], X_test[correct, 1], c=confidences[correct], 
            cmap='Greens', alpha=0.6, s=50, vmin=0, vmax=1, label='Correct')
ax2.scatter(X_test[~correct, 0], X_test[~correct, 1], c=confidences[~correct], 
            cmap='Reds', alpha=0.8, s=100, marker='X', vmin=0, vmax=1, label='Incorrect')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Confidence in Feature Space')
ax2.legend()
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Confidence')

plt.tight_layout()
plt.show()
```

### 2.3 Selective Prediction

```python
evaluator.plot_selective_accuracy(test_loader)

print("Interpretation:")
print("- Higher confidence threshold = Higher accuracy but lower coverage")
print("- The model 'knows when it doesn't know'")
```

---

## Part 3: Interactive Exploration

### 3.1 Test Individual Predictions

```python
def test_prediction(x_input):
    """Test model on a single input"""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)
        logits, uncertainty, base_logits, hidden_state = model(x_tensor, return_all=True)
        
        probs = F.softmax(logits, dim=1).squeeze()
        base_probs = F.softmax(base_logits, dim=1).squeeze()
        pred = torch.argmax(logits).item()
        
    print(f"Input: {x_input}")
    print(f"Prediction: Class {pred}")
    print(f"Confidence: {uncertainty.item():.3f}")
    print(f"\nProbabilities (after metacognition): {probs.numpy()}")
    print(f"Base probabilities (before metacognition): {base_probs.numpy()}")
    print(f"\nModulation effect: {(probs - base_probs).numpy()}")

# Try some examples
print("Example 1: High confidence region")
test_prediction([2.0, 2.0])

print("\n" + "="*60 + "\n")

print("Example 2: Decision boundary (uncertain)")
test_prediction([0.0, 0.0])

print("\n" + "="*60 + "\n")

print("Example 3: Custom input")
# Try your own!
# test_prediction([your_feature_1, your_feature_2])
```

### 3.2 Compare Base Learner vs Metacognitive Model

```python
# Get predictions from both stages
model.eval()
with torch.no_grad():
    test_x = torch.tensor(X_test, dtype=torch.float32)
    final_logits, uncertainties, base_logits, hidden = model(test_x, return_all=True)
    
    final_probs = F.softmax(final_logits, dim=1)
    base_probs = F.softmax(base_logits, dim=1)
    
    final_preds = torch.argmax(final_probs, dim=1).numpy()
    base_preds = torch.argmax(base_probs, dim=1).numpy()

# Where did metacognition change the prediction?
changed = final_preds != base_preds

print(f"Predictions changed by metacognition: {changed.sum()} / {len(changed)} ({100*changed.mean():.1f}%)")

# Were the changes helpful?
base_correct = (base_preds == y_test)
final_correct = (final_preds == y_test)

improved = ~base_correct & final_correct
worsened = base_correct & ~final_correct

print(f"Improved by metacognition: {improved.sum()}")
print(f"Worsened by metacognition: {worsened.sum()}")
print(f"Net gain: {improved.sum() - worsened.sum()}")
```

---

## Part 4: Formal Reflection System

Now let's explore the formal logic framework for metacognitive reflection.

```python
from src.reflection import MetacognitiveAgent, Proposition, ReflectionRules

# Create a reflective agent
agent = MetacognitiveAgent(name="DeepThought")

# Add beliefs based on model predictions
print("Creating belief system from model predictions...\n")

# Sample a few test points
sample_indices = [0, 10, 50, 100, 150]
for idx in sample_indices:
    x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, uncertainty = model(x, return_uncertainty=True)
        pred = torch.argmax(logits).item()
    
    belief_name = f"input_{idx}_is_class_{pred}"
    agent.add_belief(belief_name, confidence=uncertainty.item())

# Add some general beliefs
agent.add_belief("model_is_calibrated", confidence=0.85)
agent.add_belief("data_has_noise", confidence=0.75)

# Print state
agent.print_state()
```

```python
# Perform metacognitive operations
print("Performing reflection and reasoning...\n")

# Introspection
agent.reflect_on_belief("model_is_calibrated")

# Forward reasoning
agent.reason_forward(
    premises=["model_is_calibrated", "data_has_noise"],
    conclusion_name="predictions_are_uncertain",
    combination_rule='min'
)

# Check coherence
print("Checking belief coherence:")
agent.check_coherence(verbose=True)

# Explain beliefs
print("\n" + "="*60)
print(agent.explain_belief("predictions_are_uncertain"))
print("="*60)
```

---

## Part 5: Metrics Summary

```python
# Comprehensive evaluation
print("="*80)
print("COMPREHENSIVE EVALUATION")
print("="*80)

metrics = evaluator.evaluate(test_loader, compute_detailed=True)

print(f"\n📊 Performance Metrics:")
print(f"  • Accuracy: {metrics['accuracy']:.4f}")
print(f"  • Expected Calibration Error: {metrics['ece']:.4f}")
print(f"  • Brier Score: {metrics['brier_score']:.4f}")

print(f"\n🧠 Metacognitive Quality:")
print(f"  • Uncertainty-Error Correlation: {metrics['uncertainty_error_corr']:.4f}")
print(f"  • Mean Confidence: {metrics['mean_confidence']:.4f}")
print(f"  • Std Confidence: {metrics['std_confidence']:.4f}")

if 'confidence_correct' in metrics:
    print(f"\n✓ Confidence Analysis:")
    print(f"  • Correct predictions: {metrics['confidence_correct']:.4f}")
    print(f"  • Incorrect predictions: {metrics['confidence_incorrect']:.4f}")
    print(f"  • Separation: {metrics['confidence_correct'] - metrics['confidence_incorrect']:.4f}")

if 'uncertainty_auc' in metrics:
    print(f"\n🎯 Uncertainty as Error Detector:")
    print(f"  • AUC-ROC: {metrics['uncertainty_auc']:.4f}")
```

---

## Part 6: Your Turn!

### Try Different Configurations

```python
# Experiment with different architectures
def train_and_evaluate(hidden_dim, monitor_dim, lambda_meta, epochs=20):
    """Quick training and evaluation"""
    model = MetaCognitiveModel(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=2,
        monitor_dim=monitor_dim,
        control_type='gate'
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_meta=lambda_meta)
    trainer = MetacognitiveTrainer(model, optimizer, loss_fn, device='cpu')
    
    history = trainer.fit(train_loader, test_loader, epochs=epochs, verbose=False)
    
    evaluator = MetacognitiveEvaluator(model, device='cpu')
    metrics = evaluator.evaluate(test_loader, compute_detailed=False)
    
    return metrics

# Try different configurations
print("Comparing different configurations...\n")
configs = [
    {'hidden_dim': 32, 'monitor_dim': 16, 'lambda_meta': 0.1},
    {'hidden_dim': 64, 'monitor_dim': 32, 'lambda_meta': 0.1},
    {'hidden_dim': 128, 'monitor_dim': 64, 'lambda_meta': 0.1},
    {'hidden_dim': 64, 'monitor_dim': 32, 'lambda_meta': 0.05},
    {'hidden_dim': 64, 'monitor_dim': 32, 'lambda_meta': 0.2},
]

results = []
for i, config in enumerate(configs):
    print(f"Config {i+1}: {config}")
    metrics = train_and_evaluate(**config)
    results.append({**config, **metrics})
    print(f"  Accuracy: {metrics['accuracy']:.4f}, ECE: {metrics['ece']:.4f}\n")

# Show best configuration
best_idx = np.argmax([r['accuracy'] for r in results])
print(f"Best configuration: {configs[best_idx]}")
```

---

## Conclusion

You've now explored:

1. ✅ Training a metacognitive neural network
2. ✅ Visualizing confidence calibration and uncertainty
3. ✅ Understanding how metacognition modulates predictions
4. ✅ Using the formal reflection system
5. ✅ Evaluating metacognitive quality

### Next Steps

- Run the full experiments: `python experiments/noisy_labels.py`
- Read the paper: `PAPER.md`
- Explore the code: `src/models.py`, `src/training.py`, `src/evaluation.py`
- Extend to new domains: vision, NLP, reinforcement learning

---

## References

See `PAPER.md` for complete references and theoretical background.

**Questions?** Open an issue on the repository!
