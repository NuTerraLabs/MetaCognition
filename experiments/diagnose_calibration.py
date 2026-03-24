#!/usr/bin/env python3
"""
DIAGNOSIS: Why does uncertainty work but ECE worsen?

Key insight from the experiments:
1. Uncertainty separates correct/incorrect ✓
2. But ECE is worse than baseline ✗

This means: the confidence values (softmax probs) are poorly calibrated,
even though the uncertainty signal (Dirichlet) is informative.

The problem: We're combining:
- MoE logits (main prediction)  
- Evidential alpha (small contribution)
- Into final_probs via softmax

This mixing destroys calibration because:
1. MoE experts may have inconsistent scales
2. The +0.1*(alpha-1) term adds noise
3. We're using softmax probs for ECE, not the evidential uncertainty

SOLUTION: Use evidential uncertainty directly for calibration!

Instead of using softmax confidence, use:
- Prediction: argmax(alpha) or argmax(p)
- Confidence: 1 - uncertainty = 1 - K/S

This should naturally be calibrated because:
- Low evidence → high uncertainty → low confidence (correct)
- High evidence → low uncertainty → high confidence (correct)
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
    compute_ece
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_noisy_dataset(n_samples=2000, n_features=20, n_classes=5, noise_rate=0.25, seed=42):
    np.random.seed(seed)
    X = []
    y_true = []
    samples_per_class = n_samples // n_classes
    for c in range(n_classes):
        center = np.random.randn(n_features) * 2
        class_samples = center + np.random.randn(samples_per_class, n_features) * 0.8
        X.append(class_samples)
        y_true.extend([c] * samples_per_class)
    X = np.vstack(X)
    y_true = np.array(y_true)
    y_noisy = y_true.copy()
    n_actual = len(X)
    noise_mask = np.random.random(n_actual) < noise_rate
    y_noisy[noise_mask] = np.random.randint(0, n_classes, noise_mask.sum())
    clean_mask = (y_true == y_noisy)
    return X.astype(np.float32), y_noisy.astype(np.int64), y_true.astype(np.int64), clean_mask


def evaluate_with_multiple_confidences(model, test_loader, device='cpu'):
    """
    Evaluate using different confidence definitions to understand the issue.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_softmax_conf = []
    all_evidential_conf = []
    all_alpha_conf = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            # Predictions from main logits
            preds = outputs['logits'].argmax(dim=1)
            
            # Confidence 1: Softmax of final logits
            softmax_conf = F.softmax(outputs['logits'], dim=1).max(dim=1)[0]
            
            # Confidence 2: 1 - evidential uncertainty
            evidential_conf = 1.0 - outputs['epistemic_uncertainty'].squeeze()
            
            # Confidence 3: From Dirichlet expectation
            alpha = outputs['alpha']
            S = alpha.sum(dim=1, keepdim=True)
            dirichlet_probs = alpha / S
            alpha_conf = dirichlet_probs.max(dim=1)[0]
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())
            all_softmax_conf.append(softmax_conf.cpu().numpy())
            all_evidential_conf.append(evidential_conf.cpu().numpy())
            all_alpha_conf.append(alpha_conf.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_softmax_conf = np.concatenate(all_softmax_conf)
    all_evidential_conf = np.concatenate(all_evidential_conf)
    all_alpha_conf = np.concatenate(all_alpha_conf)
    
    accuracy = (all_preds == all_targets).mean()
    
    # ECE with different confidences
    ece_softmax = compute_ece(all_softmax_conf, all_preds, all_targets)
    ece_evidential = compute_ece(all_evidential_conf, all_preds, all_targets)
    ece_alpha = compute_ece(all_alpha_conf, all_preds, all_targets)
    
    return {
        'accuracy': accuracy,
        'ece_softmax': ece_softmax,
        'ece_evidential': ece_evidential,
        'ece_alpha': ece_alpha,
        'softmax_conf': all_softmax_conf,
        'evidential_conf': all_evidential_conf,
        'alpha_conf': all_alpha_conf,
        'correct': (all_preds == all_targets)
    }


def diagnose():
    """Run diagnostic experiment."""
    print("="*70)
    print("DIAGNOSTIC: Understanding the Calibration Problem")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, _, _ = create_noisy_dataset(2000, 20, 5, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    # Train model
    print("\nTraining model...")
    model = NeuroSymbolicMetaAgent(20, 5, 128, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = MetacognitiveLoss()
    
    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, loss_fn, epoch, 50)
    
    # Evaluate with multiple confidence definitions
    results = evaluate_with_multiple_confidences(model, test_loader)
    
    print("\n" + "-"*70)
    print("CALIBRATION WITH DIFFERENT CONFIDENCE DEFINITIONS:")
    print("-"*70)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"\nECE (Expected Calibration Error) - lower is better:")
    print(f"  Using softmax confidence:     {results['ece_softmax']:.4f}")
    print(f"  Using 1-uncertainty:          {results['ece_evidential']:.4f}")
    print(f"  Using Dirichlet expectation:  {results['ece_alpha']:.4f}")
    
    print("\n" + "-"*70)
    print("CONFIDENCE STATISTICS:")
    print("-"*70)
    
    correct = results['correct']
    
    for name, conf in [('Softmax', results['softmax_conf']),
                       ('1-Uncertainty', results['evidential_conf']),
                       ('Dirichlet', results['alpha_conf'])]:
        print(f"\n{name}:")
        print(f"  Overall:   mean={conf.mean():.4f}, std={conf.std():.4f}")
        print(f"  Correct:   mean={conf[correct].mean():.4f}")
        print(f"  Incorrect: mean={conf[~correct].mean():.4f}")
        print(f"  Separation: {conf[correct].mean() - conf[~correct].mean():.4f}")
    
    # Find best temperature scaling for each
    print("\n" + "-"*70)
    print("TEMPERATURE SCALING ANALYSIS:")
    print("-"*70)
    
    # For each confidence type, find optimal temperature
    for name, conf in [('Softmax', results['softmax_conf']),
                       ('Dirichlet', results['alpha_conf'])]:
        best_t = 1.0
        best_ece = compute_ece(conf, (results['softmax_conf'] == conf.max()).astype(int), 
                               results['correct'].astype(int)) if name == 'Softmax' else 999
        
        # Note: Temperature scaling on raw logits would be more correct
        # but this gives intuition
        print(f"\n{name} already uses implicit temperature of 1.0")


def fix_with_proper_confidence():
    """
    Test: If we use evidential uncertainty properly, do we get better ECE?
    """
    print("\n" + "="*70)
    print("FIX ATTEMPT: Use Evidence-based Confidence")
    print("="*70)
    
    set_seed(42)
    
    X, y_noisy, _, _ = create_noisy_dataset(2000, 20, 5, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    # Train with focus on evidential loss
    print("\nTraining with higher evidential weight...")
    model = NeuroSymbolicMetaAgent(20, 5, 128, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Increase evidential weight, decrease others
    loss_fn = MetacognitiveLoss(
        lambda_evidential=1.0,  # Increased from 0.5
        lambda_semantic=0.1,    # Decreased from 0.3
        lambda_rnd=0.05,        # Decreased from 0.1
        lambda_gate_entropy=0.01
    )
    
    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, loss_fn, epoch, 50)
        if (epoch + 1) % 10 == 0:
            results = evaluate_with_multiple_confidences(model, test_loader)
            print(f"Epoch {epoch+1}: acc={results['accuracy']:.3f}, "
                  f"ece_softmax={results['ece_softmax']:.4f}, "
                  f"ece_dirichlet={results['ece_alpha']:.4f}")
    
    final = evaluate_with_multiple_confidences(model, test_loader)
    
    print("\n" + "-"*70)
    print("FINAL RESULTS with evidence-focused training:")
    print("-"*70)
    print(f"Accuracy: {final['accuracy']:.4f}")
    print(f"ECE (softmax):    {final['ece_softmax']:.4f}")
    print(f"ECE (dirichlet):  {final['ece_alpha']:.4f}")
    print(f"ECE (1-unc):      {final['ece_evidential']:.4f}")
    
    # The key insight: if we USE Dirichlet confidence for decisions,
    # we should also USE Dirichlet confidence for calibration measurement
    
    return final


if __name__ == "__main__":
    diagnose()
    fix_with_proper_confidence()
