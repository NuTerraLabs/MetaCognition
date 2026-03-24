"""
BACK TO BASICS: What Actually Works for Calibration?

From our experiments:
1. Temperature scaling works (ECE: 0.183 → 0.055)
2. Evidential uncertainty has signal (separates correct/incorrect)
3. But combining them naively hurts calibration

NEW APPROACH: "Uncertainty-Aware Temperature Scaling"
Key insight: Instead of learning a single temperature T, learn to predict
the optimal temperature per-sample based on uncertainty signals.

This is simpler and builds on what we KNOW works.
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


class BaseClassifier(nn.Module):
    """Standard MLP classifier."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
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
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and hidden features."""
        hidden = self.encoder(x)
        logits = self.classifier(hidden)
        return logits, hidden


class AdaptiveTemperatureScaler(nn.Module):
    """
    Learns to predict optimal temperature per-sample.
    
    Key innovation: Temperature is a function of:
    1. Model uncertainty (entropy of logits)
    2. Prediction confidence
    3. Hidden features (sample difficulty)
    """
    
    def __init__(self, hidden_dim: int, base_temperature: float = 1.5):
        super().__init__()
        self.base_temperature = base_temperature
        
        # Temperature predictor: input is hidden + uncertainty signals
        self.temp_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, 32),  # +3 for entropy, max_logit, confidence
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Temperature must be > 0
        )
        
        # Offset to center around base temperature
        self.temp_offset = nn.Parameter(torch.tensor(base_temperature))
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Predict per-sample temperature.
        
        Args:
            hidden: Encoder features
            logits: Raw classifier logits
            
        Returns:
            temperature: Per-sample temperature values
        """
        # Compute uncertainty signals
        probs = F.softmax(logits, dim=1)
        
        # Entropy (higher = more uncertain)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        
        # Max logit (higher = more confident)
        max_logit = logits.max(dim=1, keepdim=True)[0]
        
        # Max probability
        max_prob = probs.max(dim=1, keepdim=True)[0]
        
        # Concatenate features
        features = torch.cat([hidden, entropy, max_logit, max_prob], dim=1)
        
        # Predict temperature deviation from base
        temp_delta = self.temp_net(features).squeeze(-1)
        
        # Final temperature (ensure > 0.1 for stability)
        temperature = self.temp_offset + temp_delta
        temperature = torch.clamp(temperature, min=0.1, max=10.0)
        
        return temperature


class MetaCognitiveCalibrator(nn.Module):
    """
    Complete model: Classifier + Adaptive Temperature Scaling
    
    Two-stage training:
    1. Train classifier normally
    2. Train temperature scaler to minimize calibration error
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = BaseClassifier(input_dim, num_classes, hidden_dim)
        self.temp_scaler = AdaptiveTemperatureScaler(hidden_dim)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor, use_adaptive_temp: bool = True) -> Dict[str, torch.Tensor]:
        logits, hidden = self.classifier(x)
        
        if use_adaptive_temp:
            temperature = self.temp_scaler(hidden.detach(), logits.detach())
            scaled_logits = logits / temperature.unsqueeze(-1)
        else:
            temperature = torch.ones(x.shape[0], device=x.device)
            scaled_logits = logits
        
        probs = F.softmax(scaled_logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Uncertainty from entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        uncertainty = entropy / np.log(self.num_classes)  # Normalized to [0, 1]
        
        return {
            'logits': logits,
            'scaled_logits': scaled_logits,
            'probs': probs,
            'confidence': confidence,
            'temperature': temperature,
            'uncertainty': uncertainty,
            'hidden': hidden
        }


def train_classifier_stage(model, train_loader, epochs=30, lr=0.001):
    """Stage 1: Train the classifier."""
    # Only train classifier parameters
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, use_adaptive_temp=False)
            loss = F.cross_entropy(outputs['logits'], y_batch)
            loss.backward()
            optimizer.step()


def train_temperature_stage(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Stage 2: Train temperature scaler for calibration."""
    # Freeze classifier, only train temperature scaler
    for param in model.classifier.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.temp_scaler.parameters(), lr=lr)
    
    best_ece = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, use_adaptive_temp=True)
            
            # Calibration loss: We want confidence to match accuracy
            # Use NLL on temperature-scaled logits
            loss = F.cross_entropy(outputs['scaled_logits'], y_batch)
            
            # Add regularization to keep temperature reasonable
            temp_reg = 0.01 * torch.mean((outputs['temperature'] - 1.5) ** 2)
            
            total_loss = loss + temp_reg
            total_loss.backward()
            optimizer.step()
        
        # Evaluate ECE on validation set
        if (epoch + 1) % 5 == 0:
            ece = evaluate_ece(model, val_loader)
            if ece < best_ece:
                best_ece = ece
                best_state = model.temp_scaler.state_dict().copy()
    
    # Restore best
    if best_state is not None:
        model.temp_scaler.load_state_dict(best_state)
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True


def evaluate_ece(model, loader):
    """Evaluate Expected Calibration Error."""
    model.eval()
    all_preds, all_conf, all_targets = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch, use_adaptive_temp=True)
            preds = outputs['probs'].argmax(dim=1)
            all_preds.append(preds.numpy())
            all_conf.append(outputs['confidence'].numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_conf = np.concatenate(all_conf)
    all_targets = np.concatenate(all_targets)
    
    return compute_ece(all_conf, all_preds, all_targets)


def run_experiment():
    """Full experiment with ground truth verification."""
    print("="*70)
    print("METACOGNITIVE CALIBRATION via Adaptive Temperature Scaling")
    print("="*70)
    
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
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
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_noisy, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), 
                            batch_size=64)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), 
                             batch_size=64)
    
    # =========================================================================
    # BASELINE: Standard MLP without calibration
    # =========================================================================
    print("\n--- BASELINE (No Calibration) ---")
    baseline = BaseClassifier(n_features, n_classes)
    opt = torch.optim.Adam(baseline.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(50):
        baseline.train()
        for X_batch, y_batch in train_loader:
            opt.zero_grad()
            logits, _ = baseline(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            opt.step()
    
    baseline.eval()
    with torch.no_grad():
        all_preds, all_conf, all_targets = [], [], []
        for X_batch, y_batch in test_loader:
            logits, _ = baseline(X_batch)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.argmax(dim=1).numpy())
            all_conf.append(probs.max(dim=1)[0].numpy())
            all_targets.append(y_batch.numpy())
        
        base_preds = np.concatenate(all_preds)
        base_conf = np.concatenate(all_conf)
        base_targets = np.concatenate(all_targets)
        
        base_acc = (base_preds == base_targets).mean()
        base_ece = compute_ece(base_conf, base_preds, base_targets)
    
    print(f"Accuracy: {base_acc:.4f}")
    print(f"ECE:      {base_ece:.4f}")
    
    # =========================================================================
    # GLOBAL Temperature Scaling (standard approach)
    # =========================================================================
    print("\n--- GLOBAL Temperature Scaling ---")
    
    # Find best global temperature on validation set
    best_t, best_ece_global = 1.0, float('inf')
    baseline.eval()
    
    for t in np.linspace(0.5, 5.0, 50):
        with torch.no_grad():
            all_preds, all_conf, all_targets = [], [], []
            for X_batch, y_batch in val_loader:
                logits, _ = baseline(X_batch)
                scaled_logits = logits / t
                probs = F.softmax(scaled_logits, dim=1)
                all_preds.append(probs.argmax(dim=1).numpy())
                all_conf.append(probs.max(dim=1)[0].numpy())
                all_targets.append(y_batch.numpy())
            
            preds = np.concatenate(all_preds)
            conf = np.concatenate(all_conf)
            targets = np.concatenate(all_targets)
            
            ece = compute_ece(conf, preds, targets)
            if ece < best_ece_global:
                best_ece_global = ece
                best_t = t
    
    # Evaluate on test set with best temperature
    with torch.no_grad():
        all_preds, all_conf, all_targets = [], [], []
        for X_batch, y_batch in test_loader:
            logits, _ = baseline(X_batch)
            scaled_logits = logits / best_t
            probs = F.softmax(scaled_logits, dim=1)
            all_preds.append(probs.argmax(dim=1).numpy())
            all_conf.append(probs.max(dim=1)[0].numpy())
            all_targets.append(y_batch.numpy())
        
        global_preds = np.concatenate(all_preds)
        global_conf = np.concatenate(all_conf)
        global_targets = np.concatenate(all_targets)
        
        global_acc = (global_preds == global_targets).mean()
        global_ece = compute_ece(global_conf, global_preds, global_targets)
    
    print(f"Best Temperature: {best_t:.2f}")
    print(f"Accuracy: {global_acc:.4f}")
    print(f"ECE:      {global_ece:.4f}")
    
    # =========================================================================
    # ADAPTIVE Temperature Scaling (our approach)
    # =========================================================================
    print("\n--- ADAPTIVE Temperature Scaling (MetaCognitive) ---")
    
    model = MetaCognitiveCalibrator(n_features, n_classes)
    
    # Stage 1: Train classifier
    print("Stage 1: Training classifier...")
    train_classifier_stage(model, train_loader, epochs=50)
    
    # Evaluate before temperature scaling
    with torch.no_grad():
        model.eval()
        all_preds, all_conf, all_targets = [], [], []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch, use_adaptive_temp=False)
            all_preds.append(outputs['probs'].argmax(dim=1).numpy())
            all_conf.append(outputs['confidence'].numpy())
            all_targets.append(y_batch.numpy())
        
        pre_preds = np.concatenate(all_preds)
        pre_conf = np.concatenate(all_conf)
        pre_targets = np.concatenate(all_targets)
        
        pre_acc = (pre_preds == pre_targets).mean()
        pre_ece = compute_ece(pre_conf, pre_preds, pre_targets)
    
    print(f"Before temp scaling - Accuracy: {pre_acc:.4f}, ECE: {pre_ece:.4f}")
    
    # Stage 2: Train adaptive temperature
    print("Stage 2: Training adaptive temperature scaler...")
    train_temperature_stage(model, train_loader, val_loader, epochs=50)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds, all_conf, all_unc, all_temps, all_targets = [], [], [], [], []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch, use_adaptive_temp=True)
            all_preds.append(outputs['probs'].argmax(dim=1).numpy())
            all_conf.append(outputs['confidence'].numpy())
            all_unc.append(outputs['uncertainty'].numpy())
            all_temps.append(outputs['temperature'].numpy())
            all_targets.append(y_batch.numpy())
        
        adapt_preds = np.concatenate(all_preds)
        adapt_conf = np.concatenate(all_conf)
        adapt_unc = np.concatenate(all_unc)
        adapt_temps = np.concatenate(all_temps)
        adapt_targets = np.concatenate(all_targets)
        
        adapt_acc = (adapt_preds == adapt_targets).mean()
        adapt_ece = compute_ece(adapt_conf, adapt_preds, adapt_targets)
        
        correct = (adapt_preds == adapt_targets)
        adapt_sep = adapt_unc[~correct].mean() - adapt_unc[correct].mean()
    
    print(f"After temp scaling - Accuracy: {adapt_acc:.4f}, ECE: {adapt_ece:.4f}")
    print(f"Temperature stats: mean={adapt_temps.mean():.2f}, std={adapt_temps.std():.2f}")
    print(f"Uncertainty separation (incorrect-correct): {adapt_sep:.4f}")
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<30} {'Accuracy':>10} {'ECE':>10} {'ECE Δ':>10}")
    print("-"*60)
    print(f"{'Baseline (no calibration)':<30} {base_acc:>10.4f} {base_ece:>10.4f} {'-':>10}")
    print(f"{'Global Temp Scaling (T={:.2f})':<30} {global_acc:>10.4f} {global_ece:>10.4f} {(1-global_ece/base_ece)*100:>+9.1f}%".format(best_t))
    print(f"{'Adaptive Temp Scaling':<30} {adapt_acc:>10.4f} {adapt_ece:>10.4f} {(1-adapt_ece/base_ece)*100:>+9.1f}%")
    
    print("\n" + "="*70)
    print("VERDICT:")
    print("="*70)
    
    # Compare to global temperature scaling (the real competitor)
    if adapt_ece < global_ece:
        improv = (1 - adapt_ece/global_ece) * 100
        print(f"✓ Adaptive beats Global Temp Scaling by {improv:.1f}%")
    else:
        deficit = (adapt_ece/global_ece - 1) * 100
        print(f"✗ Adaptive WORSE than Global Temp Scaling by {deficit:.1f}%")
    
    if adapt_ece < base_ece:
        improv = (1 - adapt_ece/base_ece) * 100
        print(f"✓ Adaptive beats Baseline by {improv:.1f}%")
    else:
        print(f"✗ Adaptive WORSE than Baseline")
    
    if adapt_sep > 0.01:
        print(f"✓ Uncertainty separates correct/incorrect (gap={adapt_sep:.4f})")
    else:
        print(f"△ Weak uncertainty separation (gap={adapt_sep:.4f})")
    
    return {
        'baseline': {'acc': base_acc, 'ece': base_ece},
        'global': {'acc': global_acc, 'ece': global_ece, 'temp': best_t},
        'adaptive': {'acc': adapt_acc, 'ece': adapt_ece, 'unc_sep': adapt_sep}
    }


if __name__ == "__main__":
    run_experiment()
