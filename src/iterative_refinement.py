"""
ITERATIVE SELF-REFINEMENT METACOGNITION

A fundamentally new approach to synthetic metacognition.

Key Insight: Real thinking is ITERATIVE and ADAPTIVE.
- Humans don't just output answers - they reflect and revise
- The NUMBER of refinement steps is itself a metacognitive signal
- Knowing WHEN to stop thinking is core to metacognition

Architecture: Adaptive Computation Time (ACT) meets Self-Reflection

1. Initial prediction (System 1 - fast/intuitive)
2. Reflection module evaluates: "Should I think more?"
3. If yes: Refinement step updates the prediction
4. Repeat until halting probability > threshold OR max steps
5. Final prediction aggregates all steps (pondering)

This is fundamentally different because:
- Uncertainty emerges from BEHAVIOR (how many steps needed)
- No explicit uncertainty head that can collapse
- The model learns to allocate computation adaptively
- More "thinking" on hard examples, less on easy ones

Inspired by:
- Adaptive Computation Time (Graves, 2016)
- PonderNet (Banino et al., 2021)
- Universal Transformers
- System 1 / System 2 (Kahneman)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


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


class ReflectionModule(nn.Module):
    """
    Decides whether to continue thinking or halt.
    
    Input: current state + prediction confidence signals
    Output: halting probability (λ ∈ [0, 1])
    
    Key: This is learned end-to-end, so the model learns WHEN
    more computation helps vs when it's wasteful.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),  # +2 for entropy and max_prob
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Current hidden state
            logits: Current prediction logits
            
        Returns:
            halt_prob: Probability of halting (not thinking more)
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        
        features = torch.cat([state, entropy, max_prob], dim=-1)
        return self.net(features).squeeze(-1)


class RefinementCell(nn.Module):
    """
    Single refinement step - updates the hidden state and prediction.
    
    Like an RNN/GRU cell but for iterative refinement.
    Uses residual connections to allow gradual updates.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # Input size: hidden + num_classes (prev logits) + input_dim
        gate_input_size = hidden_dim + num_classes + input_dim
        
        # State update (GRU-like)
        self.update_gate = nn.Linear(gate_input_size, hidden_dim)
        self.reset_gate = nn.Linear(gate_input_size, hidden_dim)
        self.candidate = nn.Linear(hidden_dim + input_dim, hidden_dim)
        
        # Prediction head
        self.predictor = nn.Linear(hidden_dim, num_classes)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, state: torch.Tensor, 
                prev_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of refinement.
        
        Args:
            x: Original input features
            state: Current hidden state
            prev_logits: Previous prediction logits
            
        Returns:
            new_state: Updated hidden state
            new_logits: Updated prediction logits
        """
        # Concatenate all information
        combined = torch.cat([state, prev_logits, x], dim=-1)
        
        # GRU-style gating
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        
        # Candidate state
        candidate_input = torch.cat([r * state, x], dim=-1)
        h_tilde = torch.tanh(self.candidate(candidate_input))
        
        # New state
        new_state = (1 - z) * state + z * h_tilde
        new_state = self.layer_norm(new_state)
        
        # New prediction (residual)
        delta_logits = self.predictor(new_state)
        new_logits = prev_logits + 0.5 * delta_logits  # Soft residual
        
        return new_state, new_logits


class IterativeSelfRefinement(nn.Module):
    """
    Main architecture: Iterative Self-Refinement with Learned Halting.
    
    The model learns to:
    1. Make initial predictions (fast, intuitive)
    2. Reflect on whether more thinking is needed
    3. Refine predictions iteratively
    4. Halt when confident enough
    
    Metacognition emerges from:
    - Number of steps taken (behavioral uncertainty)
    - Halting probability trajectory
    - Prediction stability across steps
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 max_steps: int = 5, halt_threshold: float = 0.8):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        
        # Initial encoder (System 1 - fast)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Initial predictor
        self.initial_predictor = nn.Linear(hidden_dim, num_classes)
        
        # Reflection module (decides when to halt)
        self.reflector = ReflectionModule(hidden_dim)
        
        # Refinement cell (updates predictions)
        self.refiner = RefinementCell(input_dim, hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor, 
                return_all_steps: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive computation.
        
        Args:
            x: Input features (batch, input_dim)
            return_all_steps: If True, return predictions from all steps
            
        Returns:
            Dict with:
                - logits: Final aggregated logits
                - probs: Final probabilities
                - confidence: Calibrated confidence
                - n_steps: Number of steps taken (per sample)
                - halt_probs: Halting probabilities at each step
                - all_logits: (optional) Logits from all steps
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initial encoding and prediction (System 1)
        state = self.encoder(x)
        logits = self.initial_predictor(state)
        
        # Storage for pondering
        all_logits = [logits]
        all_halt_probs = []
        
        # Cumulative halting probability (for probabilistic halting)
        cumulative_halt = torch.zeros(batch_size, device=device)
        remainder = torch.ones(batch_size, device=device)
        
        # Weighted logits for final prediction
        weighted_logits = torch.zeros_like(logits)
        
        # Iterative refinement
        for step in range(self.max_steps):
            # Reflect: Should I think more?
            halt_prob = self.reflector(state, logits)
            all_halt_probs.append(halt_prob)
            
            # Probabilistic halting (soft attention over steps)
            # This is differentiable and allows gradient flow
            step_weight = remainder * halt_prob
            remainder = remainder * (1 - halt_prob)
            
            # Accumulate weighted logits
            weighted_logits = weighted_logits + step_weight.unsqueeze(-1) * logits
            
            # Update cumulative halt
            cumulative_halt = cumulative_halt + step_weight
            
            # Refine (even if "halted" - we use soft halting)
            if step < self.max_steps - 1:
                state, logits = self.refiner(x, state, logits)
                all_logits.append(logits)
        
        # Add final remainder to last step
        weighted_logits = weighted_logits + remainder.unsqueeze(-1) * logits
        
        # Final prediction
        final_probs = F.softmax(weighted_logits, dim=-1)
        final_confidence = final_probs.max(dim=-1)[0]
        
        # Compute behavioral metacognitive signals
        halt_probs_tensor = torch.stack(all_halt_probs, dim=1)  # (batch, max_steps)
        
        # Expected number of steps (soft)
        step_indices = torch.arange(1, self.max_steps + 1, device=device).float()
        # Compute expected steps from halting distribution
        halt_distribution = self._compute_halt_distribution(halt_probs_tensor)
        expected_steps = (halt_distribution * step_indices).sum(dim=-1)
        
        # Uncertainty from computation: more steps = more uncertain
        computation_uncertainty = expected_steps / self.max_steps
        
        # Prediction stability: how much did logits change?
        if len(all_logits) > 1:
            logits_tensor = torch.stack(all_logits, dim=1)  # (batch, steps, classes)
            stability = 1.0 - torch.std(logits_tensor, dim=1).mean(dim=-1)
        else:
            stability = torch.ones(batch_size, device=device)
        
        # Combined metacognitive confidence
        # High confidence = high stability, low computation, high softmax confidence
        metacog_confidence = (
            0.4 * final_confidence +
            0.3 * stability +
            0.3 * (1 - computation_uncertainty)
        )
        
        result = {
            'logits': weighted_logits,
            'probs': final_probs,
            'confidence': metacog_confidence,
            'softmax_confidence': final_confidence,
            'expected_steps': expected_steps,
            'computation_uncertainty': computation_uncertainty,
            'stability': stability,
            'halt_probs': halt_probs_tensor,
        }
        
        if return_all_steps:
            result['all_logits'] = torch.stack(all_logits, dim=1)
        
        return result
    
    def _compute_halt_distribution(self, halt_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the distribution over halting at each step.
        
        P(halt at step t) = halt_prob[t] * Π_{i<t}(1 - halt_prob[i])
        """
        batch_size, n_steps = halt_probs.shape
        device = halt_probs.device
        
        # Compute cumulative product of (1 - halt_prob)
        not_halted = torch.cumprod(1 - halt_probs, dim=1)
        
        # Shift to get product up to (but not including) current step
        not_halted_before = torch.cat([
            torch.ones(batch_size, 1, device=device),
            not_halted[:, :-1]
        ], dim=1)
        
        # P(halt at step t)
        halt_distribution = halt_probs * not_halted_before
        
        # Add remainder probability to last step
        remainder = not_halted[:, -1:]
        halt_distribution = torch.cat([
            halt_distribution[:, :-1],
            halt_distribution[:, -1:] + remainder
        ], dim=1)
        
        return halt_distribution


class PonderingLoss(nn.Module):
    """
    Loss function for iterative self-refinement.
    
    Components:
    1. Task loss (classification)
    2. Ponder cost (regularize computation)
    3. Calibration alignment (confidence should match accuracy)
    """
    
    def __init__(self, lambda_ponder: float = 0.01, lambda_calibration: float = 0.1):
        super().__init__()
        self.lambda_ponder = lambda_ponder
        self.lambda_calibration = lambda_calibration
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 1. Task loss
        loss_task = F.cross_entropy(outputs['logits'], targets)
        
        # 2. Ponder cost: penalize excessive computation
        # But allow more computation when needed (via task loss gradient)
        loss_ponder = outputs['expected_steps'].mean()
        
        # 3. Calibration loss: confidence should match correctness
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=-1)
            correct = (preds == targets).float()
        
        # Binary cross-entropy between confidence and correctness
        loss_calibration = F.binary_cross_entropy(
            outputs['confidence'].clamp(0.01, 0.99),
            correct
        )
        
        total = (loss_task + 
                 self.lambda_ponder * loss_ponder +
                 self.lambda_calibration * loss_calibration)
        
        return {
            'total': total,
            'task': loss_task,
            'ponder': loss_ponder,
            'calibration': loss_calibration
        }


def run_experiment():
    """Test the iterative self-refinement architecture."""
    print("="*70)
    print("ITERATIVE SELF-REFINEMENT METACOGNITION")
    print("="*70)
    print("\nKey Innovation: Metacognition emerges from BEHAVIOR")
    print("- Number of refinement steps = uncertainty signal")
    print("- Prediction stability across steps = confidence signal")
    print("- Halting probability = self-assessed difficulty\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset with varying difficulty
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
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    print("--- BASELINE MLP ---")
    baseline = nn.Sequential(
        nn.Linear(n_features, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
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
        base_acc = (preds == targs).mean()
        base_ece = compute_ece(confs, preds, targs)
    
    print(f"Accuracy: {base_acc:.4f}, ECE: {base_ece:.4f}\n")
    
    # =========================================================================
    # ITERATIVE SELF-REFINEMENT
    # =========================================================================
    print("--- ITERATIVE SELF-REFINEMENT ---")
    model = IterativeSelfRefinement(
        input_dim=n_features,
        num_classes=n_classes,
        hidden_dim=128,
        max_steps=5,
        halt_threshold=0.8
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = PonderingLoss(lambda_ponder=0.01, lambda_calibration=0.1)
    
    best_val_ece = float('inf')
    best_state = None
    
    for epoch in range(60):
        model.train()
        epoch_losses = {'total': 0, 'task': 0, 'ponder': 0, 'calibration': 0}
        
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            losses = loss_fn(outputs, y_b)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds, confs, targs, steps = [], [], [], []
                for X_b, y_b in val_loader:
                    outputs = model(X_b)
                    preds.append(outputs['probs'].argmax(1).numpy())
                    confs.append(outputs['confidence'].numpy())
                    targs.append(y_b.numpy())
                    steps.append(outputs['expected_steps'].numpy())
                
                preds = np.concatenate(preds)
                confs = np.concatenate(confs)
                targs = np.concatenate(targs)
                steps = np.concatenate(steps)
                
                val_acc = (preds == targs).mean()
                val_ece = compute_ece(confs, preds, targs)
                
                print(f"Epoch {epoch+1}: acc={val_acc:.3f}, ECE={val_ece:.4f}, "
                      f"avg_steps={steps.mean():.2f}")
                
                if val_ece < best_val_ece:
                    best_val_ece = val_ece
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    print("\n--- FINAL EVALUATION ---")
    model.eval()
    with torch.no_grad():
        all_preds, all_confs, all_targs = [], [], []
        all_steps, all_stability, all_softmax_conf = [], [], []
        
        for X_b, y_b in test_loader:
            outputs = model(X_b)
            
            all_preds.append(outputs['probs'].argmax(1).numpy())
            all_confs.append(outputs['confidence'].numpy())
            all_softmax_conf.append(outputs['softmax_confidence'].numpy())
            all_targs.append(y_b.numpy())
            all_steps.append(outputs['expected_steps'].numpy())
            all_stability.append(outputs['stability'].numpy())
        
        preds = np.concatenate(all_preds)
        confs = np.concatenate(all_confs)
        softmax_confs = np.concatenate(all_softmax_conf)
        targs = np.concatenate(all_targs)
        steps = np.concatenate(all_steps)
        stability = np.concatenate(all_stability)
        
        correct = (preds == targs)
        
        meta_acc = correct.mean()
        meta_ece = compute_ece(confs, preds, targs)
        meta_ece_softmax = compute_ece(softmax_confs, preds, targs)
    
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Baseline':>12} {'Iterative':>12}")
    print(f"{'-'*55}")
    print(f"{'Accuracy':<30} {base_acc:>12.4f} {meta_acc:>12.4f}")
    print(f"{'ECE (metacog confidence)':<30} {base_ece:>12.4f} {meta_ece:>12.4f}")
    print(f"{'ECE (softmax confidence)':<30} {base_ece:>12.4f} {meta_ece_softmax:>12.4f}")
    
    print(f"\n{'='*60}")
    print("METACOGNITIVE SIGNALS (Novel Metrics)")
    print(f"{'='*60}")
    print(f"Average computation steps: {steps.mean():.2f} (max={model.max_steps})")
    print(f"Steps for correct preds:   {steps[correct].mean():.2f}")
    print(f"Steps for incorrect preds: {steps[~correct].mean():.2f}")
    print(f"Step difference:           {steps[~correct].mean() - steps[correct].mean():.3f}")
    
    print(f"\nStability (1 = stable, 0 = changing):")
    print(f"  Overall:   {stability.mean():.4f}")
    print(f"  Correct:   {stability[correct].mean():.4f}")
    print(f"  Incorrect: {stability[~correct].mean():.4f}")
    
    print(f"\nConfidence Analysis:")
    print(f"  Confidence for correct:   {confs[correct].mean():.4f}")
    print(f"  Confidence for incorrect: {confs[~correct].mean():.4f}")
    print(f"  Separation:               {confs[correct].mean() - confs[~correct].mean():.4f}")
    
    # ECE improvement
    ece_change = (1 - meta_ece / base_ece) * 100
    
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    
    if meta_ece < base_ece:
        print(f"✓ ECE IMPROVED by {ece_change:.1f}%")
    else:
        print(f"✗ ECE worsened by {-ece_change:.1f}%")
    
    step_diff = steps[~correct].mean() - steps[correct].mean()
    if step_diff > 0.1:
        print(f"✓ Model uses MORE steps on incorrect predictions (diff={step_diff:.3f})")
        print("  → Computation serves as uncertainty signal!")
    else:
        print(f"△ Step difference is small ({step_diff:.3f})")
    
    stab_diff = stability[correct].mean() - stability[~correct].mean()
    if stab_diff > 0.01:
        print(f"✓ Correct predictions are MORE stable (diff={stab_diff:.4f})")
    else:
        print(f"△ Stability difference is small ({stab_diff:.4f})")
    
    conf_sep = confs[correct].mean() - confs[~correct].mean()
    if conf_sep > 0.05:
        print(f"✓ Confidence separates correct/incorrect (sep={conf_sep:.4f})")
    else:
        print(f"△ Confidence separation is weak ({conf_sep:.4f})")
    
    return {
        'baseline': {'acc': base_acc, 'ece': base_ece},
        'iterative': {
            'acc': meta_acc, 
            'ece': meta_ece,
            'avg_steps': steps.mean(),
            'step_diff': step_diff,
            'conf_sep': conf_sep
        }
    }


if __name__ == "__main__":
    results = run_experiment()
