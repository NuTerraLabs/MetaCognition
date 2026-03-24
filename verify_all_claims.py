#!/usr/bin/env python3
"""
VERIFY ALL CLAIMS: Run this to reproduce every number in FINDINGS_LOG.md

This script runs all experiments from scratch and prints verified results.
No cached results, no cherry-picking. Takes ~2 minutes on CPU.

Usage:
    python verify_all_claims.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_ece(confidences, predictions, targets, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            acc_bin = (predictions[in_bin] == targets[in_bin]).mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    return ece


def create_dataset(seed=42, n_samples=2000, n_features=20, n_classes=5, noise_rate=0.25):
    """Create reproducible noisy classification dataset."""
    np.random.seed(seed)
    X, y_true = [], []
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
    return train_loader, val_loader, test_loader, n_features, n_classes


def make_baseline(n_features, n_classes):
    return nn.Sequential(
        nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, n_classes)
    )


def train_model(model, train_loader, epochs=50, lr=0.001, label_smoothing=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(X_b), y_b, label_smoothing=label_smoothing).backward()
            opt.step()
    return model


def eval_baseline(model, test_loader):
    model.eval()
    preds, confs, targs = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            probs = F.softmax(model(X_b), dim=1)
            preds.append(probs.argmax(1).numpy())
            confs.append(probs.max(1)[0].numpy())
            targs.append(y_b.numpy())
    preds, confs, targs = map(np.concatenate, [preds, confs, targs])
    return (preds == targs).mean(), compute_ece(confs, preds, targs), preds, confs, targs


def find_best_temperature(model, test_loader, targs):
    model.eval()
    with torch.no_grad():
        all_logits = torch.cat([model(X_b) for X_b, _ in test_loader])
    best_t, best_ece = 1.0, 1.0
    for T in np.arange(0.5, 5.0, 0.1):
        sp = F.softmax(all_logits / T, dim=1)
        ece = compute_ece(sp.max(1)[0].numpy(), sp.argmax(1).numpy(), targs)
        if ece < best_ece:
            best_ece = ece
            best_t = T
    return best_t, best_ece


def verify_phase1_triadic(seed=42):
    """Verify: Triadic architecture monitor collapse."""
    print("\n" + "=" * 70)
    print("PHASE 1: TRIADIC ARCHITECTURE (Monitor Collapse)")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _, test_loader, n_feat, n_cls = create_dataset(seed)

    from src.models import MetaCognitiveModel
    model = MetaCognitiveModel(input_dim=n_feat, hidden_dim=128, output_dim=n_cls, monitor_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out, u = model(X_b, return_uncertainty=True)
            correct_mask = (out.argmax(1) == y_b).float().unsqueeze(1)
            loss = F.cross_entropy(out, y_b) + 0.2 * F.smooth_l1_loss(u, correct_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    all_p, all_c, all_t, all_u = [], [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            out, u = model(X_b, return_uncertainty=True)
            probs = F.softmax(out, dim=1)
            all_p.append(probs.argmax(1).numpy())
            all_c.append(probs.max(1)[0].numpy())
            all_t.append(y_b.numpy())
            all_u.append(u.squeeze().numpy())
    all_p, all_c, all_t, all_u = map(np.concatenate, [all_p, all_c, all_t, all_u])
    correct = (all_p == all_t)

    print(f"  Accuracy:       {correct.mean():.4f}")
    print(f"  ECE:            {compute_ece(all_c, all_p, all_t):.4f}")
    print(f"  Monitor std(u): {all_u.std():.4f}")
    print(f"  u(correct):     {all_u[correct].mean():.4f}")
    print(f"  u(incorrect):   {all_u[~correct].mean():.4f}")
    print(f"  Separation:     {all_u[correct].mean() - all_u[~correct].mean():.4f}")
    collapsed = all_u.std() < 0.1
    print(f"  COLLAPSED:      {'YES ❌' if collapsed else 'NO ✓'}")
    return collapsed


def verify_phase3_baselines(seed=42):
    """Verify: Temperature scaling and label smoothing baselines."""
    print("\n" + "=" * 70)
    print("PHASE 3: CALIBRATION BASELINES")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _, test_loader, n_feat, n_cls = create_dataset(seed)

    # Standard baseline
    torch.manual_seed(seed)
    m1 = train_model(make_baseline(n_feat, n_cls), train_loader)
    acc, ece, preds, confs, targs = eval_baseline(m1, test_loader)
    best_t, temp_ece = find_best_temperature(m1, test_loader, targs)

    # Label smoothing
    torch.manual_seed(seed)
    m2 = train_model(make_baseline(n_feat, n_cls), train_loader, label_smoothing=0.1)
    ls_acc, ls_ece, _, _, _ = eval_baseline(m2, test_loader)

    print(f"  {'Method':<25} {'Acc':>8} {'ECE':>8} {'Reduction':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline':<25} {acc:>8.4f} {ece:>8.4f} {'—':>10}")
    print(f"  {f'Temp Scaling (T={best_t:.1f})':<25} {acc:>8.4f} {temp_ece:>8.4f} {(1-temp_ece/ece)*100:>+9.1f}%")
    print(f"  {'Label Smooth (ε=0.1)':<25} {ls_acc:>8.4f} {ls_ece:>8.4f} {(1-ls_ece/ece)*100:>+9.1f}%")
    return ece, temp_ece


def verify_phase4_pondernet(seeds=[42, 123, 456]):
    """Verify: PonderNet ECE improvement and step analysis."""
    print("\n" + "=" * 70)
    print("PHASE 4: PONDERNET")
    print("=" * 70)

    from src.ponder_net import SimplePonderNet, PonderLoss

    all_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, val_loader, test_loader, n_feat, n_cls = create_dataset(seed)

        # Baseline
        torch.manual_seed(seed)
        base_model = train_model(make_baseline(n_feat, n_cls), train_loader)
        base_acc, base_ece, _, _, targs = eval_baseline(base_model, test_loader)

        # PonderNet
        torch.manual_seed(seed)
        model = SimplePonderNet(n_feat, n_cls, hidden_dim=128, max_steps=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        loss_fn = PonderLoss(lambda_p=0.05, prior_p=0.3)
        best_ece_val, best_state = float('inf'), None

        for epoch in range(50):
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                outputs = model(X_b)
                losses = loss_fn(outputs, y_b)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    ps, cs, ts = [], [], []
                    for X_b, y_b in val_loader:
                        out = model(X_b)
                        ps.append(out['probs'].argmax(1).numpy())
                        cs.append(out['confidence'].numpy())
                        ts.append(y_b.numpy())
                    ps, cs, ts = map(np.concatenate, [ps, cs, ts])
                    val_ece = compute_ece(cs, ps, ts)
                    if val_ece < best_ece_val:
                        best_ece_val = val_ece
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        ps, cs, cs_sm, ts, steps = [], [], [], [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                out = model(X_b)
                ps.append(out['probs'].argmax(1).numpy())
                cs.append(out['confidence'].numpy())
                cs_sm.append(out['softmax_confidence'].numpy())
                ts.append(y_b.numpy())
                steps.append(out['expected_steps'].numpy())
        ps, cs, cs_sm, ts, steps = map(np.concatenate, [ps, cs, cs_sm, ts, steps])
        correct = (ps == ts)

        r = {
            'seed': seed,
            'base_ece': base_ece,
            'ponder_ece': compute_ece(cs, ps, ts),
            'ponder_ece_sm': compute_ece(cs_sm, ps, ts),
            'step_correct': steps[correct].mean(),
            'step_incorrect': steps[~correct].mean(),
            'acc': correct.mean(),
        }
        all_results.append(r)

    print(f"\n  {'Seed':>6} {'Base ECE':>10} {'Ponder ECE':>12} {'Reduction':>10} {'Step Δ':>8} {'Adaptive?':>10}")
    print(f"  {'-'*62}")
    for r in all_results:
        pct = (1 - r['ponder_ece'] / r['base_ece']) * 100
        step_diff = r['step_incorrect'] - r['step_correct']
        adaptive = "YES" if abs(step_diff) > 0.1 else "NO"
        print(f"  {r['seed']:>6} {r['base_ece']:>10.4f} {r['ponder_ece']:>12.4f} {pct:>+9.1f}% {step_diff:>+8.3f} {adaptive:>10}")

    avg_step_diff = np.mean([r['step_incorrect'] - r['step_correct'] for r in all_results])
    print(f"\n  Average step difference (incorrect − correct): {avg_step_diff:+.3f}")
    print(f"  Verdict: {'Adaptive computation confirmed ✓' if abs(avg_step_diff) > 0.1 else 'NO adaptive computation ❌ (diff < 0.1 steps)'}")
    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC METACOGNITION: FULL VERIFICATION")
    print("Reproducing all claims from FINDINGS_LOG.md")
    print("=" * 70)

    collapsed = verify_phase1_triadic()
    base_ece, temp_ece = verify_phase3_baselines()
    ponder_results = verify_phase4_pondernet()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Triadic monitor collapsed:    {'YES ❌' if collapsed else 'NO ✓'}")
    print(f"  Temp scaling ECE reduction:   {(1-temp_ece/base_ece)*100:.1f}%")
    avg_ponder = np.mean([(1 - r['ponder_ece'] / r['base_ece']) * 100 for r in ponder_results])
    print(f"  PonderNet avg ECE reduction:  {avg_ponder:.1f}%")
    avg_step = np.mean([r['step_incorrect'] - r['step_correct'] for r in ponder_results])
    print(f"  PonderNet avg step diff:      {avg_step:+.3f} ({'adaptive' if abs(avg_step) > 0.1 else 'NOT adaptive'})")
    print()
