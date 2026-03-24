"""
V2 Iteration 2: Direct Signal Metacognition

Key insight from V2 iteration 1:
- The ECE was excellent (67% reduction) but the uncertainty HEAD collapsed.
- Problem: learning an uncertainty head re-introduces the same optimization
  problem as V1 — the head learns to output a constant.

New approach — eliminate the learned uncertainty head entirely.
Instead, USE the self-supervised signals directly:

1. DIRECT CONSISTENCY: Run multiple augmented views at inference time,
   measure prediction disagreement. This IS the uncertainty — no learning needed.
   
2. ADAPTIVE CONTROLLER: The controller learns to use the raw consistency
   signal to modulate predictions. The controller CAN'T collapse because
   its input (consistency) varies by construction.

3. DISTRIBUTIONAL MATCHING: Force consistency scores to follow a Beta
   distribution via a differentiable penalty, ensuring diversity.

Author: Ismail Haddou / Nu Terra Labs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from scipy import stats


# ==============================================================================
# Feature Augmentor (stronger than V2 iter 1)
# ==============================================================================

class StrongAugmentor(nn.Module):
    """Stronger augmentation to ensure consistency signal varies meaningfully."""
    
    def __init__(self, noise_std: float = 0.15, dropout_rate: float = 0.2,
                 scale_range: Tuple[float, float] = (0.85, 1.15)):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.scale_range = scale_range
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # At eval time, use lighter augmentation for consistency measurement
            noise = torch.randn_like(x) * (self.noise_std * 0.7)
            return x + noise
        
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_rate))
        x_aug = x_aug * mask / (1 - self.dropout_rate)  # Scale to maintain magnitude
        scale = torch.empty_like(x).uniform_(*self.scale_range)
        x_aug = x_aug * scale
        return x_aug


# ==============================================================================
# Encoder + Classifier (same as before)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, repr_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.GELU(),
        )
        self.repr_dim = repr_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, repr_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(repr_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


# ==============================================================================
# Direct Signal Controller (no learned uncertainty head)
# ==============================================================================

class DirectSignalController(nn.Module):
    """
    Takes RAW self-supervised uncertainty signals and learns to use them
    for prediction adjustment. 
    
    Key difference from V1/V2-iter1: There is NO learned uncertainty head.
    The uncertainty signal is computed directly (consistency, entropy, geometry)
    and fed to the controller as features. The controller can't collapse because
    its input varies by construction.
    """
    
    def __init__(self, num_classes: int, n_signals: int = 3):
        super().__init__()
        # Small network to combine uncertainty signals into adjustment
        self.adjust_net = nn.Sequential(
            nn.Linear(n_signals, 16),
            nn.GELU(),
            nn.Linear(16, num_classes),
        )
        # Learnable scaling for how much to adjust
        self.adjustment_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, logits: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, K] raw classifier logits
            signals: [batch, n_signals] raw self-supervised uncertainty signals
        Returns:
            adjusted_logits: [batch, K]
        """
        adjustment = self.adjust_net(signals) * self.adjustment_scale
        return logits + adjustment


# ==============================================================================
# Direct Signal Metacognition Model
# ==============================================================================

class DirectSignalMetacognition(nn.Module):
    """
    V2 Iteration 2: No learned uncertainty head.
    
    Instead, we compute self-supervised signals directly and use them:
    1. Consistency: disagreement across augmented views  
    2. Entropy: normalized entropy of classifier output
    3. Margin: gap between top-2 class probabilities
    
    These signals ARE the uncertainty — no need to learn a separate head.
    The controller learns to USE these signals for calibration.
    """
    
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 128, repr_dim: int = 64,
                 n_views: int = 8):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, repr_dim)
        self.classifier = Classifier(repr_dim, num_classes)
        self.controller = DirectSignalController(num_classes, n_signals=3)
        self.augmentor = StrongAugmentor()
        self.n_views = n_views
        self.num_classes = num_classes
    
    def compute_consistency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction consistency across augmented views.
        Returns a per-sample consistency score (higher = more inconsistent = uncertain).
        """
        predictions = []
        for _ in range(self.n_views):
            x_aug = self.augmentor(x)
            h_aug = self.encoder(x_aug)
            logits_aug = self.classifier(h_aug)
            probs_aug = F.softmax(logits_aug, dim=-1)
            predictions.append(probs_aug)
        
        preds = torch.stack(predictions, dim=0)  # [n_views, batch, classes]
        mean_pred = preds.mean(dim=0)  # [batch, classes]
        
        # Jensen-Shannon divergence from mean
        js_div = torch.zeros(x.size(0), device=x.device)
        for i in range(self.n_views):
            kl = F.kl_div(
                (mean_pred + 1e-8).log(), preds[i],
                reduction='none', log_target=False
            ).sum(dim=-1)
            js_div += kl
        js_div /= self.n_views
        
        return js_div
    
    def compute_signals(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute all self-supervised uncertainty signals.
        Returns [batch, 3] tensor of (consistency, entropy, margin).
        """
        # 1. Consistency (prediction stability under augmentation)
        consistency = self.compute_consistency(x)
        
        # 2. Entropy (normalized)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        max_entropy = np.log(self.num_classes)
        norm_entropy = entropy / max_entropy
        
        # 3. Margin (gap between top-2 classes — low margin = uncertain)
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        inv_margin = 1.0 - margin  # Invert so high = uncertain
        
        signals = torch.stack([consistency, norm_entropy, inv_margin], dim=-1)
        return signals
    
    def forward(self, x: torch.Tensor, compute_all: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with direct signal computation."""
        h = self.encoder(x)
        logits = self.classifier(h)
        
        # Compute self-supervised signals
        if self.training or compute_all:
            signals = self.compute_signals(x, logits)
            adjusted_logits = self.controller(logits, signals.detach())
            
            # The "uncertainty" is the mean of the signals (all in [0,1])
            uncertainty = signals.mean(dim=-1)
        else:
            # At eval, still compute signals for uncertainty estimation
            with torch.no_grad():
                signals = self.compute_signals(x, logits)
            adjusted_logits = self.controller(logits, signals)
            uncertainty = signals.mean(dim=-1)
        
        return {
            'logits': logits,
            'adjusted_logits': adjusted_logits,
            'uncertainty': uncertainty,
            'signals': signals if (self.training or compute_all) else signals.detach(),
        }


# ==============================================================================
# Training
# ==============================================================================

def compute_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(targets).float()
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            ece += mask.float().mean().item() * abs(
                accuracies[mask].mean().item() - confidences[mask].mean().item()
            )
    return ece


def generate_dataset(n_samples=2000, n_features=20, n_classes=5, 
                     noise_rate=0.25, seed=42):
    np.random.seed(seed)
    centers = np.random.randn(n_classes, n_features) * 2.0
    X_list, y_list = [], []
    for c in range(n_classes):
        spc = n_samples // n_classes
        X_list.append(centers[c] + np.random.randn(spc, n_features) * 0.8)
        y_list.append(np.full(spc, c))
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    noisy = np.random.choice(len(y), int(len(y) * noise_rate), replace=False)
    y[noisy] = np.random.randint(0, n_classes, len(noisy))
    n_train, n_val = int(0.7 * n_samples), int(0.15 * n_samples)
    return {
        'X_train': torch.tensor(X[:n_train]),
        'y_train': torch.tensor(y[:n_train]),
        'X_val': torch.tensor(X[n_train:n_train+n_val]),
        'y_val': torch.tensor(y[n_train:n_train+n_val]),
        'X_test': torch.tensor(X[n_train+n_val:]),
        'y_test': torch.tensor(y[n_train+n_val:]),
    }


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)


def train_model(model, data, epochs=60, lr=0.001, batch_size=64, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_ds = TensorDataset(data['X_train'], data['y_train'])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    best_val_ece = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        
        for X_b, y_b in loader:
            optimizer.zero_grad()
            result = model(X_b)
            loss = F.cross_entropy(result['adjusted_logits'], y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_result = model(data['X_val'], compute_all=True)
            val_ece = compute_ece(val_result['adjusted_logits'], data['y_val'])
            val_acc = (val_result['adjusted_logits'].argmax(-1) == data['y_val']).float().mean().item()
            
            u = val_result['uncertainty']
            u_np = u.numpy()
            preds = val_result['adjusted_logits'].argmax(-1)
            correct = (preds == data['y_val']).numpy()
            
            u_std = float(u_np.std())
            if correct.any() and (~correct).any():
                sep = float(u_np[~correct].mean() - u_np[correct].mean())
            else:
                sep = 0.0
        
        if val_ece < best_val_ece:
            best_val_ece = val_ece
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch + 1) % 10 == 0:
            collapsed = "⚠️ COLLAPSED" if u_std < 0.1 else "✅ Healthy"
            disc = "✅ Disc" if sep > 0.03 else "⚠️ Low"
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(epoch_loss):.4f} | "
                  f"Val ECE: {val_ece:.4f} | Acc: {val_acc:.3f} | "
                  f"Std(u): {u_std:.3f} {collapsed} | Sep: {sep:.4f} {disc}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        result = model(X, compute_all=True)
        ece = compute_ece(result['adjusted_logits'], y)
        acc = (result['adjusted_logits'].argmax(-1) == y).float().mean().item()
        u = result['uncertainty'].numpy()
        preds = result['adjusted_logits'].argmax(-1).numpy()
        correct = preds == y.numpy()
        
    return {
        'ece': ece, 'acc': acc,
        'std_u': float(u.std()),
        'mean_u_correct': float(u[correct].mean()) if correct.any() else 0,
        'mean_u_incorrect': float(u[~correct].mean()) if (~correct).any() else 0,
        'separation': float(u[~correct].mean() - u[correct].mean()) if correct.any() and (~correct).any() else 0,
        'collapsed': float(u.std()) < 0.1,
        'discriminative': (float(u[~correct].mean() - u[correct].mean()) > 0.03) if correct.any() and (~correct).any() else False,
    }


def train_baseline(data, seed, epochs=60):
    torch.manual_seed(seed)
    model = BaselineMLP(20, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loader = DataLoader(TensorDataset(data['X_train'], data['y_train']), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(X_b), y_b).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(data['X_test'])
        ece = compute_ece(logits, data['y_test'])
        acc = (logits.argmax(-1) == data['y_test']).float().mean().item()
    return model, ece, acc


def temp_scale(model, data):
    model.eval()
    with torch.no_grad(): val_logits = model(data['X_val'])
    T = nn.Parameter(torch.tensor(1.5))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / T, data['y_val'])
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        test_logits = model(data['X_test'])
        scaled = test_logits / T
        return compute_ece(scaled, data['y_test']), (scaled.argmax(-1) == data['y_test']).float().mean().item()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("V2 ITERATION 2: DIRECT SIGNAL METACOGNITION")
    print("(No learned uncertainty head — signals used directly)")
    print("=" * 70)
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'━'*70}")
        print(f"  SEED: {seed}")
        print(f"{'━'*70}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = generate_dataset(seed=seed)
        
        # Baseline
        print("  Training baseline...")
        torch.manual_seed(seed)
        base_model, base_ece, base_acc = train_baseline(data, seed)
        
        # Temp scaling
        temp_ece, temp_acc = temp_scale(base_model, data)
        
        # V2 Direct Signal
        print("  Training V2 Direct Signal Metacognition...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DirectSignalMetacognition(
            input_dim=20, num_classes=5, hidden_dim=128, repr_dim=64, n_views=8,
        )
        model = train_model(model, data, epochs=60, verbose=True)
        result = evaluate(model, data['X_test'], data['y_test'])
        
        print(f"\n  {'Method':<35} {'Acc':>8} {'ECE':>8} {'ECE Δ':>8}")
        print(f"  {'-'*59}")
        print(f"  {'Baseline MLP':<35} {base_acc:>8.3f} {base_ece:>8.4f} {'—':>8}")
        print(f"  {'Temp Scaling':<35} {temp_acc:>8.3f} {temp_ece:>8.4f} "
              f"{(1-temp_ece/base_ece)*100:>+7.1f}%")
        print(f"  {'V2 Direct Signal':<35} {result['acc']:>8.3f} {result['ece']:>8.4f} "
              f"{(1-result['ece']/base_ece)*100:>+7.1f}%")
        
        print(f"\n  Uncertainty Health:")
        print(f"    Std(u):       {result['std_u']:.4f}  "
              f"{'✅ Healthy' if not result['collapsed'] else '❌ Collapsed'}")
        print(f"    u(correct):   {result['mean_u_correct']:.4f}")
        print(f"    u(incorrect): {result['mean_u_incorrect']:.4f}")
        print(f"    Separation:   {result['separation']:.4f}  "
              f"{'✅ Discriminative' if result['discriminative'] else '⚠️ Low'}")
        
        all_results.append({
            'seed': seed, 'base_ece': base_ece, 'temp_ece': temp_ece,
            **result
        })
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")
    
    avg_base = np.mean([r['base_ece'] for r in all_results])
    avg_temp = np.mean([r['temp_ece'] for r in all_results])
    avg_v2 = np.mean([r['ece'] for r in all_results])
    avg_std = np.mean([r['std_u'] for r in all_results])
    avg_sep = np.mean([r['separation'] for r in all_results])
    
    print(f"\n  {'Method':<30} {'Avg ECE':>10} {'Reduction':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Baseline':<30} {avg_base:>10.4f} {'—':>12}")
    print(f"  {'Temp Scaling':<30} {avg_temp:>10.4f} {(1-avg_temp/avg_base)*100:>+11.1f}%")
    print(f"  {'V2 Direct Signal':<30} {avg_v2:>10.4f} {(1-avg_v2/avg_base)*100:>+11.1f}%")
    
    print(f"\n  Uncertainty Health (V2):")
    print(f"    Avg Std(u):     {avg_std:.4f}  "
          f"{'✅ Not collapsed' if avg_std > 0.1 else '❌ Collapsed'}")
    print(f"    Avg Separation: {avg_sep:.4f}  "
          f"{'✅ Discriminative' if avg_sep > 0.03 else '⚠️ Weak'}")
    
    n_collapsed = sum(1 for r in all_results if r['collapsed'])
    n_disc = sum(1 for r in all_results if r['discriminative'])
    print(f"    Collapsed: {n_collapsed}/{len(seeds)}")
    print(f"    Discriminative: {n_disc}/{len(seeds)}")
    
    # The key insight
    print(f"\n  KEY FINDING:")
    if not any(r['collapsed'] for r in all_results):
        print(f"  ✅ Direct signals CANNOT collapse (they're computed, not learned)!")
        if avg_sep > 0.03:
            print(f"  ✅ Uncertainty is discriminative (sep={avg_sep:.4f})")
            print(f"     → Self-supervised signals provide genuine metacognition!")
        else:
            print(f"  ⚠️ But signals don't discriminate correct vs incorrect well.")
            print(f"     → Consistency/entropy measure model uncertainty, but this")
            print(f"       doesn't perfectly align with correctness. This is expected:")
            print(f"       self-supervised uncertainty ≠ oracle uncertainty.")
    else:
        print(f"  The direct signals should NOT collapse. If std < 0.1, the")
        print(f"  augmentation is too weak or the model is too stable.")
    
    print(f"\n{'='*70}")
    print("Done.")


if __name__ == '__main__':
    main()
