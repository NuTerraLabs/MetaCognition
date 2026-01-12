#!/usr/bin/env python3
"""
Ground Truth Experiments for Neuro-Symbolic Metacognition

This script tests EVERY claim rigorously:
1. Does uncertainty separate correct/incorrect?
2. Does ECE improve over baseline?
3. Does semantic loss help?
4. Does MoE routing adapt to metacognitive signals?

NO HALLUCINATION - we report exactly what we observe.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.neuro_symbolic import (
    NeuroSymbolicMetaAgent,
    MetacognitiveLoss,
    train_epoch,
    evaluate,
    compute_ece,
    semantic_loss_exactly_one
)


def set_seed(seed=42):
    """Reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_noisy_dataset(n_samples=2000, n_features=20, n_classes=5,
                         noise_rate=0.25, seed=42):
    """Create classification dataset with label noise."""
    np.random.seed(seed)
    
    # Generate structured features with clear class separation
    X = []
    y_true = []
    
    samples_per_class = n_samples // n_classes
    actual_n_samples = samples_per_class * n_classes  # Ensure divisibility
    
    for c in range(n_classes):
        center = np.random.randn(n_features) * 2
        class_samples = center + np.random.randn(samples_per_class, n_features) * 0.8
        X.append(class_samples)
        y_true.extend([c] * samples_per_class)
    
    X = np.vstack(X)
    y_true = np.array(y_true)
    
    # Add noise
    y_noisy = y_true.copy()
    noise_mask = np.random.random(actual_n_samples) < noise_rate
    n_noisy = noise_mask.sum()
    y_noisy[noise_mask] = np.random.randint(0, n_classes, n_noisy)
    
    # Store which samples have clean labels (for analysis)
    clean_mask = (y_true == y_noisy)
    
    return X.astype(np.float32), y_noisy.astype(np.int64), y_true.astype(np.int64), clean_mask


class BaselineMLP(nn.Module):
    """Simple MLP baseline without metacognition."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_baseline(model, train_loader, test_loader, epochs=50, lr=0.001, device='cpu'):
    """Train baseline MLP."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    accuracy = (all_preds == all_targets).mean()
    confidences = all_probs.max(axis=1)
    ece = compute_ece(confidences, all_preds, all_targets)
    conf_std = confidences.std()
    
    # Separation
    correct = (all_preds == all_targets)
    sep = confidences[correct].mean() - confidences[~correct].mean() if (~correct).sum() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'conf_std': conf_std,
        'separation': sep
    }


def train_metacognitive(model, train_loader, test_loader, epochs=50, lr=0.001, device='cpu'):
    """Train neuro-symbolic metacognitive model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = MetacognitiveLoss(
        lambda_evidential=0.5,
        lambda_semantic=0.3,
        lambda_rnd=0.1,
        lambda_gate_entropy=0.01
    )
    
    history = []
    for epoch in range(epochs):
        losses = train_epoch(model, train_loader, optimizer, loss_fn, epoch, epochs, device)
        
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_loader, device)
            history.append({'epoch': epoch + 1, **losses, **metrics})
            print(f"Epoch {epoch+1}: loss={losses['total']:.4f}, acc={metrics['accuracy']:.3f}, "
                  f"ece={metrics['ece']:.4f}, unc_sep={metrics['unc_separation']:.4f}")
    
    # Final evaluation
    return evaluate(model, test_loader, device), history


def experiment_1_basic_functionality():
    """Test: Does the model train without crashing?"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Basic Functionality Test")
    print("="*70)
    
    set_seed(42)
    
    # Small test
    X, y_noisy, y_true, clean_mask = create_noisy_dataset(500, 10, 3, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=32)
    
    model = NeuroSymbolicMetaAgent(input_dim=10, num_classes=3, hidden_dim=64, num_experts=3)
    
    try:
        metrics, _ = train_metacognitive(model, train_loader, test_loader, epochs=20)
        print(f"\n✓ Model trains successfully")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Uncertainty Std: {metrics['unc_std']:.4f}")
        print(f"  Gate Usage: {metrics['gate_usage']}")
        return True
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def experiment_2_vs_baseline():
    """Test: Does metacognition outperform baseline MLP?"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Metacognitive vs Baseline MLP")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, y_true, clean_mask = create_noisy_dataset(2000, 20, 5, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    # Baseline
    print("\nTraining Baseline MLP...")
    baseline = BaselineMLP(20, 5, hidden_dim=128)
    baseline_metrics = train_baseline(baseline, train_loader, test_loader, epochs=50)
    
    # Metacognitive
    print("\nTraining Metacognitive Model...")
    meta_model = NeuroSymbolicMetaAgent(input_dim=20, num_classes=5, hidden_dim=128, num_experts=4)
    meta_metrics, history = train_metacognitive(meta_model, train_loader, test_loader, epochs=50)
    
    # Compare
    print("\n" + "-"*50)
    print("RESULTS COMPARISON:")
    print("-"*50)
    print(f"{'Metric':<25} {'Baseline':>12} {'Metacognitive':>12} {'Change':>12}")
    print("-"*50)
    
    acc_change = (meta_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100
    print(f"{'Accuracy':<25} {baseline_metrics['accuracy']:>12.4f} {meta_metrics['accuracy']:>12.4f} {acc_change:>+11.1f}%")
    
    ece_change = (baseline_metrics['ece'] - meta_metrics['ece']) / baseline_metrics['ece'] * 100
    print(f"{'ECE (↓ better)':<25} {baseline_metrics['ece']:>12.4f} {meta_metrics['ece']:>12.4f} {ece_change:>+11.1f}%")
    
    print(f"{'Confidence Std':<25} {baseline_metrics['conf_std']:>12.4f} {meta_metrics['unc_std']:>12.4f}")
    print(f"{'Separation':<25} {baseline_metrics['separation']:>12.4f} {meta_metrics['unc_separation']:>12.4f}")
    
    # Interpret
    print("\n" + "-"*50)
    print("INTERPRETATION:")
    if meta_metrics['ece'] < baseline_metrics['ece']:
        print(f"✓ ECE IMPROVED by {ece_change:.1f}%")
    else:
        print(f"✗ ECE WORSENED by {-ece_change:.1f}%")
    
    if meta_metrics['unc_std'] > 0.05:
        print(f"✓ Uncertainty has meaningful variance (std={meta_metrics['unc_std']:.4f})")
    else:
        print(f"✗ Uncertainty COLLAPSED (std={meta_metrics['unc_std']:.4f} < 0.05)")
    
    if meta_metrics['unc_separation'] > 0:
        print(f"✓ Higher uncertainty on incorrect predictions (sep={meta_metrics['unc_separation']:.4f})")
    else:
        print(f"✗ Uncertainty does NOT distinguish correct/incorrect")
    
    return baseline_metrics, meta_metrics


def experiment_3_semantic_loss_ablation():
    """Test: Does semantic loss actually help?"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Semantic Loss Ablation")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, _, _ = create_noisy_dataset(1500, 15, 4, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    results = {}
    
    # Without semantic loss
    print("\nTraining WITHOUT semantic loss (λ_sem=0)...")
    model_no_sem = NeuroSymbolicMetaAgent(15, 4, 96, 3)
    optimizer = torch.optim.Adam(model_no_sem.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_semantic=0.0)  # Disable semantic loss
    
    for epoch in range(40):
        train_epoch(model_no_sem, train_loader, optimizer, loss_fn, epoch, 40)
    results['no_semantic'] = evaluate(model_no_sem, test_loader)
    
    # With semantic loss
    print("Training WITH semantic loss (λ_sem=0.3)...")
    model_with_sem = NeuroSymbolicMetaAgent(15, 4, 96, 3)
    optimizer = torch.optim.Adam(model_with_sem.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_semantic=0.3)
    
    for epoch in range(40):
        train_epoch(model_with_sem, train_loader, optimizer, loss_fn, epoch, 40)
    results['with_semantic'] = evaluate(model_with_sem, test_loader)
    
    print("\n" + "-"*50)
    print("SEMANTIC LOSS ABLATION RESULTS:")
    print("-"*50)
    print(f"{'Metric':<20} {'No Semantic':>15} {'With Semantic':>15}")
    print("-"*50)
    print(f"{'Accuracy':<20} {results['no_semantic']['accuracy']:>15.4f} {results['with_semantic']['accuracy']:>15.4f}")
    print(f"{'ECE':<20} {results['no_semantic']['ece']:>15.4f} {results['with_semantic']['ece']:>15.4f}")
    print(f"{'Unc Separation':<20} {results['no_semantic']['unc_separation']:>15.4f} {results['with_semantic']['unc_separation']:>15.4f}")
    
    sem_helps_ece = results['with_semantic']['ece'] < results['no_semantic']['ece']
    print(f"\n→ Semantic loss {'HELPS' if sem_helps_ece else 'HURTS'} ECE")
    
    return results


def experiment_4_uncertainty_quality():
    """Test: Does evidential uncertainty actually work?"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Evidential Uncertainty Analysis")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, y_true, clean_mask = create_noisy_dataset(2000, 20, 5, 0.3)
    
    # Manual split to preserve all arrays
    n_train = int(len(X) * 0.8)
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_noisy[train_idx], y_noisy[test_idx]
    y_test_true = y_true[test_idx]
    clean_test = clean_mask[test_idx]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    model = NeuroSymbolicMetaAgent(20, 5, 128, 4)
    metrics, _ = train_metacognitive(model, train_loader, test_loader, epochs=50)
    
    # Detailed uncertainty analysis
    model.eval()
    all_unc = []
    all_correct = []
    all_clean = []
    
    with torch.no_grad():
        idx = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs['logits'].argmax(dim=1).numpy()
            
            for i in range(len(preds)):
                all_unc.append(outputs['epistemic_uncertainty'][i].item())
                all_correct.append(preds[i] == y_test[idx])
                all_clean.append(clean_test[idx])
                idx += 1
    
    all_unc = np.array(all_unc)
    all_correct = np.array(all_correct)
    all_clean = np.array(all_clean)
    
    print("\n" + "-"*50)
    print("UNCERTAINTY BREAKDOWN:")
    print("-"*50)
    print(f"Overall uncertainty: mean={all_unc.mean():.4f}, std={all_unc.std():.4f}")
    print(f"\nBy Correctness:")
    print(f"  Correct preds:   mean={all_unc[all_correct].mean():.4f}, std={all_unc[all_correct].std():.4f}")
    print(f"  Incorrect preds: mean={all_unc[~all_correct].mean():.4f}, std={all_unc[~all_correct].std():.4f}")
    
    print(f"\nBy Label Quality:")
    print(f"  Clean labels:    mean={all_unc[all_clean].mean():.4f}")
    print(f"  Noisy labels:    mean={all_unc[~all_clean].mean():.4f}")
    
    # Statistical test
    from scipy import stats
    if sum(all_correct) > 0 and sum(~all_correct) > 0:
        t_stat, p_value = stats.ttest_ind(all_unc[~all_correct], all_unc[all_correct])
        print(f"\nt-test (incorrect vs correct): t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05 and t_stat > 0:
            print("✓ Uncertainty is SIGNIFICANTLY higher for incorrect predictions")
        else:
            print("✗ No significant difference in uncertainty")
    
    return all_unc, all_correct, all_clean


def experiment_5_moe_routing():
    """Test: Does MoE actually route based on uncertainty?"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: MoE Routing Analysis")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, _, _ = create_noisy_dataset(1500, 15, 4, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    model = NeuroSymbolicMetaAgent(15, 4, 96, num_experts=4)
    metrics, _ = train_metacognitive(model, train_loader, test_loader, epochs=40)
    
    # Analyze routing
    model.eval()
    all_gates = []
    all_unc = []
    all_correct = []
    
    with torch.no_grad():
        idx = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs['logits'].argmax(dim=1).numpy()
            
            for i in range(len(preds)):
                all_gates.append(outputs['gate_weights'][i].numpy())
                all_unc.append(outputs['epistemic_uncertainty'][i].item())
                all_correct.append(preds[i] == y_test[idx])
                idx += 1
    
    all_gates = np.array(all_gates)
    all_unc = np.array(all_unc)
    all_correct = np.array(all_correct)
    
    # Split by uncertainty level
    unc_median = np.median(all_unc)
    low_unc_mask = all_unc < unc_median
    high_unc_mask = ~low_unc_mask
    
    print("\n" + "-"*50)
    print("MoE ROUTING PATTERNS:")
    print("-"*50)
    print("Expert usage (mean gate weight):")
    print(f"  Overall:      {all_gates.mean(axis=0)}")
    print(f"  Low uncert:   {all_gates[low_unc_mask].mean(axis=0)}")
    print(f"  High uncert:  {all_gates[high_unc_mask].mean(axis=0)}")
    print(f"  Correct:      {all_gates[all_correct].mean(axis=0)}")
    print(f"  Incorrect:    {all_gates[~all_correct].mean(axis=0)}")
    
    # Check if Expert 0 (Exploration) gets more weight when uncertain
    exp0_low = all_gates[low_unc_mask, 0].mean()
    exp0_high = all_gates[high_unc_mask, 0].mean()
    
    print(f"\n→ Expert 0 (Exploration) weight: low_unc={exp0_low:.4f}, high_unc={exp0_high:.4f}")
    if exp0_high > exp0_low:
        print("✓ Exploration expert activated MORE when uncertain (as designed)")
    else:
        print("✗ Routing does NOT adapt to uncertainty as expected")
    
    return all_gates, all_unc, all_correct


def main():
    """Run all experiments."""
    print("="*70)
    print("NEURO-SYMBOLIC METACOGNITION: GROUND TRUTH EXPERIMENTS")
    print("="*70)
    print("\nThis will test EVERY claim with real experiments.")
    print("NO HALLUCINATION - we report exactly what we observe.\n")
    
    results = {}
    
    # Test 1: Basic functionality
    if not experiment_1_basic_functionality():
        print("\n⚠️  ABORTING: Basic functionality test failed")
        return
    
    # Test 2: vs Baseline
    baseline, meta = experiment_2_vs_baseline()
    results['baseline'] = baseline
    results['metacognitive'] = meta
    
    # Test 3: Semantic loss ablation
    results['semantic_ablation'] = experiment_3_semantic_loss_ablation()
    
    # Test 4: Uncertainty quality
    unc_data = experiment_4_uncertainty_quality()
    
    # Test 5: MoE routing
    routing_data = experiment_5_moe_routing()
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL SUMMARY: HONEST ASSESSMENT")
    print("="*70)
    
    claims = []
    
    # ECE claim
    ece_improvement = (baseline['ece'] - meta['ece']) / baseline['ece'] * 100
    if ece_improvement > 5:
        claims.append(f"✓ ECE improved by {ece_improvement:.1f}%")
    elif ece_improvement > 0:
        claims.append(f"△ ECE slightly improved by {ece_improvement:.1f}%")
    else:
        claims.append(f"✗ ECE WORSENED by {-ece_improvement:.1f}%")
    
    # Uncertainty variance
    if meta['unc_std'] > 0.05:
        claims.append(f"✓ Uncertainty has meaningful variance (std={meta['unc_std']:.4f})")
    else:
        claims.append(f"✗ Uncertainty COLLAPSED (std={meta['unc_std']:.4f})")
    
    # Separation
    if meta['unc_separation'] > 0.02:
        claims.append(f"✓ Uncertainty separates correct/incorrect (gap={meta['unc_separation']:.4f})")
    elif meta['unc_separation'] > 0:
        claims.append(f"△ Small uncertainty separation (gap={meta['unc_separation']:.4f})")
    else:
        claims.append(f"✗ NO meaningful uncertainty separation")
    
    print("\nWHAT WE CAN HONESTLY CLAIM:")
    for claim in claims:
        print(f"  {claim}")
    
    print("\nRAW NUMBERS (for paper):")
    print(f"  Baseline ECE: {baseline['ece']:.4f}")
    print(f"  Metacog ECE:  {meta['ece']:.4f}")
    print(f"  Baseline Acc: {baseline['accuracy']:.4f}")
    print(f"  Metacog Acc:  {meta['accuracy']:.4f}")
    print(f"  Unc Std:      {meta['unc_std']:.4f}")
    print(f"  Unc Separation: {meta['unc_separation']:.4f}")


if __name__ == "__main__":
    main()
