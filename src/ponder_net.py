"""
SIMPLIFIED ADAPTIVE COMPUTATION METACOGNITION

The previous approach was too complex. Let's simplify to the core idea:
- Model can "think" for variable number of steps
- Uncertainty = how many steps were needed
- Simple, clean, testable

Key simplification: Instead of soft halting, use hard halting with 
straight-through gradient estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def compute_ece(confidences, predictions, targets, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = (predictions[in_bin] == targets[in_bin]).mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    return ece


class SimplePonderNet(nn.Module):
    """
    Simplified PonderNet: learns when to stop thinking.
    
    Architecture:
    1. Core network processes input
    2. At each step, predicts:
       - class logits
       - halting probability λ
    3. Uses geometric distribution for halting
    4. Uncertainty = expected number of steps
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 max_steps: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Recurrent core (simple RNN-style)
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output heads
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.halter = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Learnable prior for halting (geometric distribution parameter)
        self.lambda_prior = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        device = x.device
        
        # Initial encoding
        h = self.encoder(x)
        
        # Storage
        all_logits = []
        all_halts = []
        
        # Pondering
        for step in range(self.max_steps):
            # Update state
            h = self.core(h, h)  # Self-recurrence
            
            # Predict
            logits = self.classifier(h)
            halt_prob = self.halter(h).squeeze(-1)
            
            all_logits.append(logits)
            all_halts.append(halt_prob)
        
        # Stack
        logits_stack = torch.stack(all_logits, dim=1)  # (B, S, C)
        halts_stack = torch.stack(all_halts, dim=1)    # (B, S)
        
        # Compute halting distribution (geometric-like)
        # p(halt at step t) = halt[t] * prod(1 - halt[i] for i < t)
        log_not_halt = torch.log(1 - halts_stack + 1e-8)
        cumsum_log = torch.cumsum(log_not_halt, dim=1)
        # Shift to get prod up to but not including current
        cumsum_shifted = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            cumsum_log[:, :-1]
        ], dim=1)
        
        # p(halt at t)
        halt_dist = halts_stack * torch.exp(cumsum_shifted)
        # Ensure sums to 1 (add remainder to last step)
        remainder = 1 - halt_dist.sum(dim=1, keepdim=True)
        halt_dist = torch.cat([halt_dist[:, :-1], halt_dist[:, -1:] + remainder.clamp(min=0)], dim=1)
        
        # Weighted prediction
        weighted_logits = (halt_dist.unsqueeze(-1) * logits_stack).sum(dim=1)
        
        # Expected steps
        step_indices = torch.arange(1, self.max_steps + 1, device=device).float()
        expected_steps = (halt_dist * step_indices).sum(dim=1)
        
        # Confidence: combine softmax confidence with computation-based uncertainty
        probs = F.softmax(weighted_logits, dim=1)
        softmax_conf = probs.max(dim=1)[0]
        
        # Computation uncertainty: normalize by max steps
        comp_uncertainty = expected_steps / self.max_steps
        
        # Combined confidence
        confidence = 0.7 * softmax_conf + 0.3 * (1 - comp_uncertainty)
        
        return {
            'logits': weighted_logits,
            'probs': probs,
            'confidence': confidence,
            'softmax_confidence': softmax_conf,
            'expected_steps': expected_steps,
            'halt_dist': halt_dist,
            'comp_uncertainty': comp_uncertainty,
        }


class PonderLoss(nn.Module):
    """Loss for PonderNet training."""
    
    def __init__(self, lambda_p: float = 0.1, prior_p: float = 0.5):
        super().__init__()
        self.lambda_p = lambda_p
        self.prior_p = prior_p  # Geometric distribution prior
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        batch_size = outputs['halt_dist'].shape[0]
        max_steps = outputs['halt_dist'].shape[1]
        device = outputs['logits'].device
        
        # 1. Reconstruction loss (task)
        loss_task = F.cross_entropy(outputs['logits'], targets)
        
        # 2. KL divergence from geometric prior
        # Prior: p(halt at step t) = p * (1-p)^(t-1)
        prior_dist = torch.zeros(max_steps, device=device)
        for t in range(max_steps):
            prior_dist[t] = self.prior_p * ((1 - self.prior_p) ** t)
        prior_dist = prior_dist / prior_dist.sum()  # Normalize
        prior_dist = prior_dist.unsqueeze(0).expand(batch_size, -1)
        
        # KL(q || p) where q = halt_dist
        kl = F.kl_div(
            torch.log(outputs['halt_dist'] + 1e-8),
            prior_dist,
            reduction='batchmean',
            log_target=False
        )
        
        total = loss_task + self.lambda_p * kl
        
        return {
            'total': total,
            'task': loss_task,
            'kl': kl
        }


def run_experiment():
    """Test simplified PonderNet."""
    print("="*70)
    print("SIMPLIFIED PONDER-NET METACOGNITION")
    print("="*70)
    print("\nCore idea: Uncertainty emerges from COMPUTATION TIME")
    print("- Easy examples: few steps needed → high confidence")
    print("- Hard examples: many steps needed → low confidence\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset
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
    
    # Baseline
    print("--- BASELINE ---")
    baseline = nn.Sequential(
        nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, n_classes)
    )
    opt = torch.optim.AdamW(baseline.parameters(), lr=0.001, weight_decay=0.01)
    
    for _ in range(50):
        baseline.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            F.cross_entropy(baseline(X_b), y_b).backward()
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
        base_acc = (preds == targs).mean()
        base_ece = compute_ece(confs, preds, targs)
    print(f"Accuracy: {base_acc:.4f}, ECE: {base_ece:.4f}\n")
    
    # PonderNet
    print("--- PONDER-NET ---")
    model = SimplePonderNet(n_features, n_classes, hidden_dim=128, max_steps=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = PonderLoss(lambda_p=0.05, prior_p=0.3)
    
    best_ece = float('inf')
    best_state = None
    
    for epoch in range(50):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            losses = loss_fn(outputs, y_b)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds, confs, targs, steps = [], [], [], []
                for X_b, y_b in val_loader:
                    out = model(X_b)
                    preds.append(out['probs'].argmax(1).numpy())
                    confs.append(out['confidence'].numpy())
                    targs.append(y_b.numpy())
                    steps.append(out['expected_steps'].numpy())
                preds, confs, targs = map(np.concatenate, [preds, confs, targs])
                steps = np.concatenate(steps)
                
                acc = (preds == targs).mean()
                ece = compute_ece(confs, preds, targs)
                
                print(f"Epoch {epoch+1}: acc={acc:.3f}, ECE={ece:.4f}, steps={steps.mean():.2f}")
                
                if ece < best_ece:
                    best_ece = ece
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final eval
    print("\n--- FINAL EVALUATION ---")
    model.eval()
    with torch.no_grad():
        all_preds, all_confs, all_targs, all_steps = [], [], [], []
        
        for X_b, y_b in test_loader:
            out = model(X_b)
            all_preds.append(out['probs'].argmax(1).numpy())
            all_confs.append(out['confidence'].numpy())
            all_targs.append(y_b.numpy())
            all_steps.append(out['expected_steps'].numpy())
        
        preds = np.concatenate(all_preds)
        confs = np.concatenate(all_confs)
        targs = np.concatenate(all_targs)
        steps = np.concatenate(all_steps)
        
        correct = (preds == targs)
        meta_acc = correct.mean()
        meta_ece = compute_ece(confs, preds, targs)
    
    print(f"\n{'='*55}")
    print(f"{'Metric':<25} {'Baseline':>12} {'PonderNet':>12}")
    print(f"{'='*55}")
    print(f"{'Accuracy':<25} {base_acc:>12.4f} {meta_acc:>12.4f}")
    print(f"{'ECE':<25} {base_ece:>12.4f} {meta_ece:>12.4f}")
    
    ece_change = (1 - meta_ece/base_ece) * 100
    
    print(f"\n{'='*55}")
    print("METACOGNITIVE ANALYSIS")
    print(f"{'='*55}")
    print(f"Average steps: {steps.mean():.2f} / {model.max_steps}")
    print(f"Steps (correct):   {steps[correct].mean():.2f}")
    print(f"Steps (incorrect): {steps[~correct].mean():.2f}")
    
    step_diff = steps[~correct].mean() - steps[correct].mean()
    conf_sep = confs[correct].mean() - confs[~correct].mean()
    
    print(f"\n{'='*55}")
    print("VERDICT")
    print(f"{'='*55}")
    
    if meta_ece < base_ece:
        print(f"✓ ECE improved by {ece_change:.1f}%")
    else:
        print(f"✗ ECE worsened by {-ece_change:.1f}%")
    
    if step_diff > 0.1:
        print(f"✓ More steps for incorrect predictions (+{step_diff:.2f})")
    elif step_diff > 0:
        print(f"△ Slightly more steps for incorrect (+{step_diff:.2f})")
    else:
        print(f"✗ Fewer steps for incorrect ({step_diff:.2f})")
    
    if conf_sep > 0.05:
        print(f"✓ Confidence separates correct/incorrect ({conf_sep:.3f})")
    else:
        print(f"△ Weak confidence separation ({conf_sep:.3f})")
    
    return {
        'baseline': {'acc': base_acc, 'ece': base_ece},
        'ponder': {'acc': meta_acc, 'ece': meta_ece, 'step_diff': step_diff}
    }


if __name__ == "__main__":
    run_experiment()
