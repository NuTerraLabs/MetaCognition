"""
Metacognitive Agent v2: Properly Calibrated

Key fixes from diagnosis:
1. Use Dirichlet-based confidence for predictions (not softmax)
2. Separate task accuracy from calibration training
3. Better loss balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


class EvidentialClassifier(nn.Module):
    """
    Simplified evidential classifier that directly outputs calibrated predictions.
    
    Key insight: Rather than complex MoE routing, focus on getting the 
    evidential head to work well, then add metacognitive components.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Evidential head: outputs log-evidence to ensure stability
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            - evidence: Non-negative evidence per class
            - alpha: Dirichlet parameters
            - probs: Expected class probabilities (use for predictions)
            - uncertainty: Epistemic uncertainty (use for confidence)
            - confidence: 1 - uncertainty (use for ECE)
        """
        hidden = self.encoder(x)
        
        # Evidence via softplus ensures non-negativity
        evidence = F.softplus(self.evidence_head(hidden))
        
        # Dirichlet parameters
        alpha = evidence + 1
        
        # Strength (total evidence)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected probabilities from Dirichlet
        probs = alpha / S
        
        # Epistemic uncertainty: K/S (high when low evidence)
        uncertainty = self.num_classes / S
        
        # Confidence for calibration
        confidence = 1 - uncertainty
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty.squeeze(-1),
            'confidence': confidence.squeeze(-1),
            'hidden': hidden
        }
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction and calibrated confidence."""
        out = self.forward(x)
        pred = out['probs'].argmax(dim=1)
        # Confidence is the max probability, scaled by certainty
        # This gives calibrated confidence
        max_prob = out['probs'].max(dim=1)[0]
        # Combine max_prob with uncertainty-based confidence
        conf = 0.5 * max_prob + 0.5 * out['confidence']
        return pred, conf


def evidential_loss_mse(evidence: torch.Tensor, targets: torch.Tensor,
                        epoch: int = 0, total_epochs: int = 50,
                        annealing_step: int = 10) -> torch.Tensor:
    """
    Evidential MSE loss with KL regularization (annealed).
    
    This is the proper evidential loss from Sensoy et al. 2018.
    """
    num_classes = evidence.shape[1]
    device = evidence.device
    
    # One-hot encode targets
    y = F.one_hot(targets, num_classes=num_classes).float().to(device)
    
    # Dirichlet parameters
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Expected probability
    p = alpha / S
    
    # MSE loss component (Bayes risk)
    err = (y - p) ** 2
    var = p * (1 - p) / (S + 1)
    loss_mse = torch.sum(err + var, dim=1)
    
    # KL divergence regularization (annealed)
    annealing_coef = min(1.0, epoch / annealing_step)
    
    # Remove evidence from correct class
    alpha_tilde = evidence * (1 - y) + y  # evidence for wrong classes + 1
    alpha_tilde = alpha_tilde + 1  # Add prior
    
    # KL(Dir(alpha_tilde) || Dir(1,1,...,1))
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    lnB = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
    lnB_uniform = num_classes * torch.lgamma(torch.tensor(1.0).to(device)) - torch.lgamma(torch.tensor(float(num_classes)).to(device))
    
    dg0 = torch.digamma(S_tilde)
    dg1 = torch.digamma(alpha_tilde)
    
    kl = torch.sum((alpha_tilde - 1) * (dg1 - dg0), dim=1, keepdim=True) + lnB - lnB_uniform
    
    loss = torch.mean(loss_mse) + annealing_coef * torch.mean(kl)
    
    return loss


class MetaCognitiveMonitor(nn.Module):
    """
    Learns to predict whether the model will be correct.
    
    Key difference from before: Uses evidential uncertainty as input,
    not just hidden representations.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Input: hidden features + evidence + uncertainty
        self.monitor = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),  # +2 for max_evidence and uncertainty
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden: torch.Tensor, max_evidence: torch.Tensor, 
                uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of being correct.
        
        Returns:
            p_correct: Probability model predicts this sample correctly
        """
        features = torch.cat([
            hidden,
            max_evidence.unsqueeze(-1),
            uncertainty.unsqueeze(-1)
        ], dim=1)
        return self.monitor(features).squeeze(-1)


class CalibratedMetaAgent(nn.Module):
    """
    Complete metacognitive agent with proper calibration.
    
    Architecture:
    1. EvidentialClassifier: Main prediction with built-in uncertainty
    2. MetaCognitiveMonitor: Predicts correctness probability
    3. Confidence = combination of evidential and monitor signals
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = EvidentialClassifier(input_dim, num_classes, hidden_dim)
        self.monitor = MetaCognitiveMonitor(hidden_dim)
        
        # Learnable combination weights
        self.alpha_evid = nn.Parameter(torch.tensor(0.7))
        self.alpha_monitor = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Base classifier
        class_out = self.classifier(x)
        
        # Metacognitive monitoring
        max_evidence = class_out['evidence'].max(dim=1)[0]
        p_correct = self.monitor(
            class_out['hidden'].detach(),  # Detach to prevent gradient flow
            max_evidence.detach(),
            class_out['uncertainty'].detach()
        )
        
        # Combined confidence
        evid_conf = class_out['confidence']
        # Weighted combination (softmax for normalization)
        w = F.softmax(torch.stack([self.alpha_evid, self.alpha_monitor]), dim=0)
        combined_conf = w[0] * evid_conf + w[1] * p_correct
        
        return {
            **class_out,
            'p_correct': p_correct,
            'combined_confidence': combined_conf
        }


class CalibratedLoss(nn.Module):
    """Loss function for calibrated metacognitive agent."""
    
    def __init__(self, lambda_monitor: float = 0.5):
        super().__init__()
        self.lambda_monitor = lambda_monitor
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                epoch: int = 0, total_epochs: int = 50) -> Dict[str, torch.Tensor]:
        
        # 1. Evidential classification loss
        loss_evid = evidential_loss_mse(
            outputs['evidence'], targets, epoch, total_epochs
        )
        
        # 2. Monitor loss (binary classification: correct vs incorrect)
        with torch.no_grad():
            preds = outputs['probs'].argmax(dim=1)
            is_correct = (preds == targets).float()
        
        loss_monitor = F.binary_cross_entropy(
            outputs['p_correct'], is_correct
        )
        
        total = loss_evid + self.lambda_monitor * loss_monitor
        
        return {
            'total': total,
            'evidential': loss_evid,
            'monitor': loss_monitor
        }


def compute_ece(confidences, predictions, targets, n_bins=15):
    """Compute Expected Calibration Error."""
    import numpy as np
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = (predictions[in_bin] == targets[in_bin]).mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    
    return ece


def train_and_evaluate():
    """Test the calibrated agent."""
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    print("="*70)
    print("CALIBRATED METACOGNITIVE AGENT: Ground Truth Test")
    print("="*70)
    
    # Reproducibility
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    # Baseline MLP
    print("\n--- BASELINE MLP ---")
    baseline = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, n_classes)
    )
    opt_base = torch.optim.Adam(baseline.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(50):
        baseline.train()
        for X_batch, y_batch in train_loader:
            opt_base.zero_grad()
            logits = baseline(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            opt_base.step()
    
    baseline.eval()
    with torch.no_grad():
        all_preds, all_conf, all_targets = [], [], []
        for X_batch, y_batch in test_loader:
            logits = baseline(X_batch)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.argmax(dim=1).numpy())
            all_conf.append(probs.max(dim=1)[0].numpy())
            all_targets.append(y_batch.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_conf = np.concatenate(all_conf)
        all_targets = np.concatenate(all_targets)
        
        baseline_acc = (all_preds == all_targets).mean()
        baseline_ece = compute_ece(all_conf, all_preds, all_targets)
        baseline_sep = all_conf[all_preds == all_targets].mean() - all_conf[all_preds != all_targets].mean()
        
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"Baseline ECE:      {baseline_ece:.4f}")
        print(f"Baseline Conf Sep: {baseline_sep:.4f}")
    
    # Calibrated MetaAgent
    print("\n--- CALIBRATED META-AGENT ---")
    model = CalibratedMetaAgent(n_features, n_classes, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = CalibratedLoss(lambda_monitor=0.3)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = loss_fn(outputs, y_batch, epoch, epochs)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += losses['total'].item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                all_preds, all_evid_conf, all_comb_conf, all_targets = [], [], [], []
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    preds = outputs['probs'].argmax(dim=1)
                    all_preds.append(preds.numpy())
                    all_evid_conf.append(outputs['confidence'].numpy())
                    all_comb_conf.append(outputs['combined_confidence'].numpy())
                    all_targets.append(y_batch.numpy())
                
                preds_np = np.concatenate(all_preds)
                evid_conf_np = np.concatenate(all_evid_conf)
                comb_conf_np = np.concatenate(all_comb_conf)
                targets_np = np.concatenate(all_targets)
                
                acc = (preds_np == targets_np).mean()
                ece_evid = compute_ece(evid_conf_np, preds_np, targets_np)
                ece_comb = compute_ece(comb_conf_np, preds_np, targets_np)
                
                print(f"Epoch {epoch+1}: acc={acc:.3f}, ECE_evid={ece_evid:.4f}, ECE_comb={ece_comb:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds, all_evid_conf, all_comb_conf, all_unc, all_targets = [], [], [], [], []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs['probs'].argmax(dim=1)
            all_preds.append(preds.numpy())
            all_evid_conf.append(outputs['confidence'].numpy())
            all_comb_conf.append(outputs['combined_confidence'].numpy())
            all_unc.append(outputs['uncertainty'].numpy())
            all_targets.append(y_batch.numpy())
        
        preds_np = np.concatenate(all_preds)
        evid_conf_np = np.concatenate(all_evid_conf)
        comb_conf_np = np.concatenate(all_comb_conf)
        unc_np = np.concatenate(all_unc)
        targets_np = np.concatenate(all_targets)
        
        meta_acc = (preds_np == targets_np).mean()
        meta_ece_evid = compute_ece(evid_conf_np, preds_np, targets_np)
        meta_ece_comb = compute_ece(comb_conf_np, preds_np, targets_np)
        
        correct = (preds_np == targets_np)
        meta_unc_sep = unc_np[~correct].mean() - unc_np[correct].mean()
        meta_conf_sep = comb_conf_np[correct].mean() - comb_conf_np[~correct].mean()
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':>12} {'MetaAgent':>12} {'Δ':>10}")
    print("-"*60)
    print(f"{'Accuracy':<25} {baseline_acc:>12.4f} {meta_acc:>12.4f} {(meta_acc-baseline_acc)*100:>+9.1f}%")
    print(f"{'ECE (↓ better)':<25} {baseline_ece:>12.4f} {meta_ece_comb:>12.4f} {(1-meta_ece_comb/baseline_ece)*100:>+9.1f}%")
    print(f"{'Confidence Separation':<25} {baseline_sep:>12.4f} {meta_conf_sep:>12.4f}")
    print(f"{'Uncertainty Std':<25} {'-':>12} {unc_np.std():>12.4f}")
    
    print("\n" + "="*70)
    print("VERDICT:")
    print("="*70)
    
    if meta_ece_comb < baseline_ece:
        ece_improv = (1 - meta_ece_comb/baseline_ece) * 100
        print(f"✓ ECE IMPROVED by {ece_improv:.1f}%")
    else:
        print(f"✗ ECE worsened")
    
    if unc_np.std() > 0.05:
        print(f"✓ Uncertainty has meaningful variance (std={unc_np.std():.4f})")
    else:
        print(f"✗ Uncertainty collapsed (std={unc_np.std():.4f})")
    
    if meta_unc_sep > 0:
        print(f"✓ Higher uncertainty on incorrect predictions (gap={meta_unc_sep:.4f})")
    else:
        print(f"✗ Uncertainty does not separate correct/incorrect")
    
    return {
        'baseline': {'acc': baseline_acc, 'ece': baseline_ece},
        'meta': {'acc': meta_acc, 'ece': meta_ece_comb, 'unc_std': unc_np.std(), 'unc_sep': meta_unc_sep}
    }


if __name__ == "__main__":
    train_and_evaluate()
