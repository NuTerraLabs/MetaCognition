"""
REPRODUCIBILITY TEST: Verify PonderNet results across multiple seeds
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
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 max_steps: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.halter = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        device = x.device
        
        h = self.encoder(x)
        all_logits, all_halts = [], []
        
        for step in range(self.max_steps):
            h = self.core(h, h)
            all_logits.append(self.classifier(h))
            all_halts.append(self.halter(h).squeeze(-1))
        
        logits_stack = torch.stack(all_logits, dim=1)
        halts_stack = torch.stack(all_halts, dim=1)
        
        log_not_halt = torch.log(1 - halts_stack + 1e-8)
        cumsum_log = torch.cumsum(log_not_halt, dim=1)
        cumsum_shifted = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            cumsum_log[:, :-1]
        ], dim=1)
        
        halt_dist = halts_stack * torch.exp(cumsum_shifted)
        remainder = 1 - halt_dist.sum(dim=1, keepdim=True)
        halt_dist = torch.cat([halt_dist[:, :-1], halt_dist[:, -1:] + remainder.clamp(min=0)], dim=1)
        
        weighted_logits = (halt_dist.unsqueeze(-1) * logits_stack).sum(dim=1)
        
        step_indices = torch.arange(1, self.max_steps + 1, device=device).float()
        expected_steps = (halt_dist * step_indices).sum(dim=1)
        
        probs = F.softmax(weighted_logits, dim=1)
        softmax_conf = probs.max(dim=1)[0]
        comp_uncertainty = expected_steps / self.max_steps
        confidence = 0.7 * softmax_conf + 0.3 * (1 - comp_uncertainty)
        
        return {
            'logits': weighted_logits,
            'probs': probs,
            'confidence': confidence,
            'expected_steps': expected_steps,
            'halt_dist': halt_dist,
        }


class PonderLoss(nn.Module):
    def __init__(self, lambda_p: float = 0.1, prior_p: float = 0.5):
        super().__init__()
        self.lambda_p = lambda_p
        self.prior_p = prior_p
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        batch_size = outputs['halt_dist'].shape[0]
        max_steps = outputs['halt_dist'].shape[1]
        device = outputs['logits'].device
        
        loss_task = F.cross_entropy(outputs['logits'], targets)
        
        prior_dist = torch.zeros(max_steps, device=device)
        for t in range(max_steps):
            prior_dist[t] = self.prior_p * ((1 - self.prior_p) ** t)
        prior_dist = prior_dist / prior_dist.sum()
        prior_dist = prior_dist.unsqueeze(0).expand(batch_size, -1)
        
        kl = F.kl_div(
            torch.log(outputs['halt_dist'] + 1e-8),
            prior_dist,
            reduction='batchmean',
            log_target=False
        )
        
        return {'total': loss_task + self.lambda_p * kl, 'task': loss_task, 'kl': kl}


def run_single_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_noisy, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64)
    
    # Baseline
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
    
    # PonderNet
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
                preds, confs, targs = [], [], []
                for X_b, y_b in val_loader:
                    out = model(X_b)
                    preds.append(out['probs'].argmax(1).numpy())
                    confs.append(out['confidence'].numpy())
                    targs.append(y_b.numpy())
                preds, confs, targs = map(np.concatenate, [preds, confs, targs])
                ece = compute_ece(confs, preds, targs)
                if ece < best_ece:
                    best_ece = ece
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        preds, confs, targs, steps = [], [], [], []
        for X_b, y_b in test_loader:
            out = model(X_b)
            preds.append(out['probs'].argmax(1).numpy())
            confs.append(out['confidence'].numpy())
            targs.append(y_b.numpy())
            steps.append(out['expected_steps'].numpy())
        preds, confs, targs = map(np.concatenate, [preds, confs, targs])
        steps = np.concatenate(steps)
        
        correct = (preds == targs)
        meta_acc = correct.mean()
        meta_ece = compute_ece(confs, preds, targs)
        step_diff = steps[~correct].mean() - steps[correct].mean()
    
    return {
        'base_acc': base_acc,
        'base_ece': base_ece,
        'meta_acc': meta_acc,
        'meta_ece': meta_ece,
        'step_diff': step_diff,
        'ece_improvement': (1 - meta_ece/base_ece) * 100
    }


def main():
    print("="*70)
    print("REPRODUCIBILITY TEST: Multiple Seeds")
    print("="*70)
    
    seeds = [42, 123, 456, 789, 1000]
    results = []
    
    for seed in seeds:
        print(f"\nSeed {seed}...")
        r = run_single_seed(seed)
        results.append(r)
        print(f"  Base ECE: {r['base_ece']:.4f}, Meta ECE: {r['meta_ece']:.4f}, "
              f"Improvement: {r['ece_improvement']:.1f}%")
    
    print("\n" + "="*70)
    print("SUMMARY ACROSS SEEDS")
    print("="*70)
    
    base_ece = np.array([r['base_ece'] for r in results])
    meta_ece = np.array([r['meta_ece'] for r in results])
    improvements = np.array([r['ece_improvement'] for r in results])
    step_diffs = np.array([r['step_diff'] for r in results])
    
    print(f"\nBaseline ECE:  {base_ece.mean():.4f} ± {base_ece.std():.4f}")
    print(f"PonderNet ECE: {meta_ece.mean():.4f} ± {meta_ece.std():.4f}")
    print(f"\nECE Improvement: {improvements.mean():.1f}% ± {improvements.std():.1f}%")
    print(f"Step difference: {step_diffs.mean():.3f} ± {step_diffs.std():.3f}")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(base_ece, meta_ece)
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
    else:
        print("✗ Not statistically significant")
    
    # How many seeds showed improvement?
    improved = sum(1 for r in results if r['meta_ece'] < r['base_ece'])
    print(f"\nImproved in {improved}/{len(seeds)} seeds")


if __name__ == "__main__":
    main()
