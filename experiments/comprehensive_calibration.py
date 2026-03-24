"""
FINAL APPROACH: Direct Calibration Training

Key insight: We need to directly optimize for calibration, not just use 
standard losses and hope calibration emerges.

Methods:
1. Differentiable ECE approximation
2. Focal loss (reduces overconfidence)
3. Label smoothing (implicit calibration)
4. Our innovation: Uncertainty-aware output adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple


def compute_ece(confidences, predictions, targets, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = (predictions[in_bin] == targets[in_bin]).mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    return ece


def differentiable_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """
    Differentiable approximation to ECE.
    
    Uses soft binning with a kernel to make it differentiable.
    """
    device = probs.device
    confidence, predictions = probs.max(dim=1)
    accuracies = (predictions == targets).float()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    ece = torch.tensor(0.0, device=device)
    
    for i in range(n_bins):
        # Soft bin membership using sigmoid
        in_bin_lower = torch.sigmoid(20 * (confidence - bin_lowers[i]))
        in_bin_upper = torch.sigmoid(20 * (bin_uppers[i] - confidence))
        in_bin = in_bin_lower * in_bin_upper
        
        bin_count = in_bin.sum() + 1e-8
        
        if bin_count > 1:
            avg_confidence = (confidence * in_bin).sum() / bin_count
            avg_accuracy = (accuracies * in_bin).sum() / bin_count
            ece = ece + torch.abs(avg_accuracy - avg_confidence) * (bin_count / len(confidence))
    
    return ece


class FocalLoss(nn.Module):
    """
    Focal loss reduces the well-classified examples' loss.
    This naturally reduces overconfidence.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class UncertaintyAwareClassifier(nn.Module):
    """
    Classifier that learns to output calibrated predictions.
    
    Key innovation: Predicts both class logits AND an adjustment factor
    that modifies confidence based on learned uncertainty.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Uncertainty head: predicts how much to "smooth" the prediction
        # Output is log of smoothing factor (ensure positive via exp)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable base smoothing (similar to label smoothing parameter)
        self.base_smoothing = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        
        # Raw classification logits
        logits = self.classifier(hidden)
        
        # Per-sample uncertainty (used to adjust smoothing)
        unc_raw = self.uncertainty_head(hidden).squeeze(-1)
        uncertainty = torch.sigmoid(unc_raw)  # In [0, 1]
        
        # Adaptive smoothing: higher uncertainty → more smoothing
        smoothing = torch.clamp(self.base_smoothing + 0.3 * uncertainty, 0.01, 0.5)
        
        # Apply smoothing to create calibrated probabilities
        # soft_targets = (1 - smoothing) * one_hot + smoothing / num_classes
        # We achieve this by temperature scaling + mixing with uniform
        
        # Method: Use smoothing to interpolate between sharp and uniform
        sharp_probs = F.softmax(logits, dim=1)
        uniform = torch.ones_like(sharp_probs) / self.num_classes
        
        # Calibrated probabilities
        probs = (1 - smoothing.unsqueeze(-1)) * sharp_probs + smoothing.unsqueeze(-1) * uniform
        
        confidence = probs.max(dim=1)[0]
        
        return {
            'logits': logits,
            'probs': probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'smoothing': smoothing,
            'hidden': hidden
        }


class CalibrationLoss(nn.Module):
    """
    Combined loss for calibration-aware training.
    
    L = L_task + λ_focal * L_focal + λ_ece * L_ece + λ_smooth * L_smooth_reg
    """
    
    def __init__(self, lambda_focal: float = 0.3, lambda_ece: float = 0.2,
                 lambda_smooth_reg: float = 0.1):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_ece = lambda_ece
        self.lambda_smooth_reg = lambda_smooth_reg
        self.focal = FocalLoss(gamma=2.0)
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Task loss (cross-entropy on smoothed probs)
        # We use log of calibrated probs for loss
        log_probs = torch.log(outputs['probs'] + 1e-8)
        loss_task = F.nll_loss(log_probs, targets)
        
        # Focal loss (reduces overconfidence)
        loss_focal = self.focal(outputs['logits'], targets)
        
        # Differentiable ECE
        loss_ece = differentiable_ece(outputs['probs'], targets)
        
        # Smoothing regularization (prevent all smoothing → 0 or → 0.5)
        # We want smoothing to be higher for uncertain samples
        # Proxy: smoothing should correlate with prediction entropy
        entropy = -torch.sum(outputs['probs'] * torch.log(outputs['probs'] + 1e-8), dim=1)
        norm_entropy = entropy / np.log(outputs['probs'].shape[1])
        
        # Smoothing should track entropy (higher entropy → higher smoothing)
        smooth_target = 0.1 + 0.4 * norm_entropy.detach()
        loss_smooth_reg = F.mse_loss(outputs['smoothing'], smooth_target)
        
        total = (loss_task + 
                 self.lambda_focal * loss_focal + 
                 self.lambda_ece * loss_ece +
                 self.lambda_smooth_reg * loss_smooth_reg)
        
        return {
            'total': total,
            'task': loss_task,
            'focal': loss_focal,
            'ece': loss_ece,
            'smooth_reg': loss_smooth_reg
        }


def run_comprehensive_experiment():
    """Run all calibration methods and compare."""
    print("="*70)
    print("COMPREHENSIVE CALIBRATION COMPARISON")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create noisy dataset
    n_samples, n_features, n_classes = 2000, 20, 5
    noise_rate = 0.25
    
    X = []
    y_true = []
    samples_per_class = n_samples // n_classes
    for c in range(n_classes):
        center = np.random.randn(n_features) * 2
        class_samples = center + np.random.randn(samples_per_class, n_features) * 0.8
        X.append(class_samples)
        y_true.extend([c] * samples_per_class)
    X = np.vstack(X).astype(np.float32)
    y_true = np.array(y_true)
    
    y_noisy = y_true.copy()
    noise_mask = np.random.random(len(X)) < noise_rate
    y_noisy[noise_mask] = np.random.randint(0, n_classes, noise_mask.sum())
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_noisy, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64)
    
    results = {}
    
    # =========================================================================
    # METHOD 1: Baseline (no calibration)
    # =========================================================================
    print("\n--- METHOD 1: Baseline MLP ---")
    baseline = nn.Sequential(
        nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, n_classes)
    )
    opt = torch.optim.Adam(baseline.parameters(), lr=0.001, weight_decay=1e-5)
    
    for _ in range(50):
        baseline.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(baseline(X_b), y_b)
            loss.backward()
            opt.step()
    
    baseline.eval()
    with torch.no_grad():
        preds, confs, targs = [], [], []
        for X_b, y_b in test_loader:
            probs = F.softmax(baseline(X_b), dim=1)
            preds.append(probs.argmax(1).numpy())
            confs.append(probs.max(1)[0].numpy())
            targs.append(y_b.numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        results['baseline'] = {'acc': (preds==targs).mean(), 'ece': compute_ece(confs, preds, targs)}
    print(f"Accuracy: {results['baseline']['acc']:.4f}, ECE: {results['baseline']['ece']:.4f}")
    
    # =========================================================================
    # METHOD 2: Label Smoothing
    # =========================================================================
    print("\n--- METHOD 2: Label Smoothing (ε=0.1) ---")
    model_ls = nn.Sequential(
        nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, n_classes)
    )
    opt = torch.optim.Adam(model_ls.parameters(), lr=0.001, weight_decay=1e-5)
    
    for _ in range(50):
        model_ls.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model_ls(X_b), y_b, label_smoothing=0.1)
            loss.backward()
            opt.step()
    
    model_ls.eval()
    with torch.no_grad():
        preds, confs, targs = [], [], []
        for X_b, y_b in test_loader:
            probs = F.softmax(model_ls(X_b), dim=1)
            preds.append(probs.argmax(1).numpy())
            confs.append(probs.max(1)[0].numpy())
            targs.append(y_b.numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        results['label_smooth'] = {'acc': (preds==targs).mean(), 'ece': compute_ece(confs, preds, targs)}
    print(f"Accuracy: {results['label_smooth']['acc']:.4f}, ECE: {results['label_smooth']['ece']:.4f}")
    
    # =========================================================================
    # METHOD 3: Temperature Scaling (post-hoc)
    # =========================================================================
    print("\n--- METHOD 3: Post-hoc Temperature Scaling ---")
    
    # Use baseline model, find optimal temperature on validation
    best_t, best_ece = 1.0, float('inf')
    baseline.eval()
    
    for t in np.linspace(0.5, 5.0, 50):
        with torch.no_grad():
            preds, confs, targs = [], [], []
            for X_b, y_b in val_loader:
                probs = F.softmax(baseline(X_b) / t, dim=1)
                preds.append(probs.argmax(1).numpy())
                confs.append(probs.max(1)[0].numpy())
                targs.append(y_b.numpy())
            preds, confs, targs = map(np.concatenate, [preds, confs, targs])
            ece = compute_ece(confs, preds, targs)
            if ece < best_ece:
                best_ece = ece
                best_t = t
    
    with torch.no_grad():
        preds, confs, targs = [], [], []
        for X_b, y_b in test_loader:
            probs = F.softmax(baseline(X_b) / best_t, dim=1)
            preds.append(probs.argmax(1).numpy())
            confs.append(probs.max(1)[0].numpy())
            targs.append(y_b.numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        results['temp_scale'] = {'acc': (preds==targs).mean(), 'ece': compute_ece(confs, preds, targs), 'temp': best_t}
    print(f"Best T={best_t:.2f}, Accuracy: {results['temp_scale']['acc']:.4f}, ECE: {results['temp_scale']['ece']:.4f}")
    
    # =========================================================================
    # METHOD 4: Focal Loss
    # =========================================================================
    print("\n--- METHOD 4: Focal Loss (γ=2.0) ---")
    model_focal = nn.Sequential(
        nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, n_classes)
    )
    opt = torch.optim.Adam(model_focal.parameters(), lr=0.001, weight_decay=1e-5)
    focal_loss = FocalLoss(gamma=2.0)
    
    for _ in range(50):
        model_focal.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = focal_loss(model_focal(X_b), y_b)
            loss.backward()
            opt.step()
    
    model_focal.eval()
    with torch.no_grad():
        preds, confs, targs = [], [], []
        for X_b, y_b in test_loader:
            probs = F.softmax(model_focal(X_b), dim=1)
            preds.append(probs.argmax(1).numpy())
            confs.append(probs.max(1)[0].numpy())
            targs.append(y_b.numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        results['focal'] = {'acc': (preds==targs).mean(), 'ece': compute_ece(confs, preds, targs)}
    print(f"Accuracy: {results['focal']['acc']:.4f}, ECE: {results['focal']['ece']:.4f}")
    
    # =========================================================================
    # METHOD 5: Our Uncertainty-Aware Calibration
    # =========================================================================
    print("\n--- METHOD 5: Uncertainty-Aware Calibration (Ours) ---")
    model_ours = UncertaintyAwareClassifier(n_features, n_classes)
    opt = torch.optim.Adam(model_ours.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = CalibrationLoss()
    
    best_state = None
    best_val_ece = float('inf')
    
    for epoch in range(60):
        model_ours.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            outputs = model_ours(X_b)
            losses = loss_fn(outputs, y_b)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model_ours.parameters(), 5.0)
            opt.step()
        
        # Early stopping on validation ECE
        if (epoch + 1) % 5 == 0:
            model_ours.eval()
            with torch.no_grad():
                preds, confs, targs = [], [], []
                for X_b, y_b in val_loader:
                    outputs = model_ours(X_b)
                    preds.append(outputs['probs'].argmax(1).numpy())
                    confs.append(outputs['confidence'].numpy())
                    targs.append(y_b.numpy())
                preds, confs, targs = map(np.concatenate, [preds, confs, targs])
                val_ece = compute_ece(confs, preds, targs)
                
                if val_ece < best_val_ece:
                    best_val_ece = val_ece
                    best_state = model_ours.state_dict().copy()
    
    if best_state:
        model_ours.load_state_dict(best_state)
    
    model_ours.eval()
    with torch.no_grad():
        preds, confs, targs, uncs, smooths = [], [], [], [], []
        for X_b, y_b in test_loader:
            outputs = model_ours(X_b)
            preds.append(outputs['probs'].argmax(1).numpy())
            confs.append(outputs['confidence'].numpy())
            targs.append(y_b.numpy())
            uncs.append(outputs['uncertainty'].numpy())
            smooths.append(outputs['smoothing'].numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        uncs = np.concatenate(uncs)
        smooths = np.concatenate(smooths)
        
        correct = (preds == targs)
        unc_sep = uncs[~correct].mean() - uncs[correct].mean()
        
        results['ours'] = {
            'acc': (preds==targs).mean(), 
            'ece': compute_ece(confs, preds, targs),
            'unc_sep': unc_sep,
            'unc_std': uncs.std(),
            'smooth_mean': smooths.mean(),
            'smooth_std': smooths.std()
        }
    
    print(f"Accuracy: {results['ours']['acc']:.4f}, ECE: {results['ours']['ece']:.4f}")
    print(f"Uncertainty sep: {results['ours']['unc_sep']:.4f}, std: {results['ours']['unc_std']:.4f}")
    print(f"Smoothing mean: {results['ours']['smooth_mean']:.4f}, std: {results['ours']['smooth_std']:.4f}")
    
    # =========================================================================
    # FINAL COMPARISON TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<35} {'Accuracy':>10} {'ECE':>10} {'ECE Δ':>10}")
    print("-"*65)
    
    base_ece = results['baseline']['ece']
    for name, key in [('Baseline', 'baseline'),
                      ('Label Smoothing (ε=0.1)', 'label_smooth'),
                      ('Temperature Scaling', 'temp_scale'),
                      ('Focal Loss (γ=2)', 'focal'),
                      ('Uncertainty-Aware (Ours)', 'ours')]:
        r = results[key]
        delta = (1 - r['ece']/base_ece) * 100 if key != 'baseline' else 0
        print(f"{name:<35} {r['acc']:>10.4f} {r['ece']:>10.4f} {delta:>+9.1f}%")
    
    # Identify best method
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    methods = [(k, v['ece']) for k, v in results.items()]
    methods.sort(key=lambda x: x[1])
    
    print(f"Best ECE: {methods[0][0]} ({methods[0][1]:.4f})")
    print(f"2nd:      {methods[1][0]} ({methods[1][1]:.4f})")
    print(f"3rd:      {methods[2][0]} ({methods[2][1]:.4f})")
    
    # Did our method beat temperature scaling?
    if results['ours']['ece'] < results['temp_scale']['ece']:
        print(f"\n✓ Our method beats Temperature Scaling!")
    else:
        print(f"\n✗ Our method does NOT beat Temperature Scaling")
        print(f"   (Temperature scaling remains the gold standard)")
    
    # But our method provides additional benefits:
    if results['ours']['unc_sep'] > 0:
        print(f"\n✓ Our method provides interpretable uncertainty signals")
        print(f"   Incorrect predictions have {results['ours']['unc_sep']:.3f} higher uncertainty")
    
    return results


if __name__ == "__main__":
    run_comprehensive_experiment()
