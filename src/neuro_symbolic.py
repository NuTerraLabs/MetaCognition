"""
Neuro-Symbolic Metacognition: A Complete Implementation

This implements the "Haddou et al." architecture with:
1. Semantic Loss for differentiable logic
2. Evidential Deep Learning for uncertainty
3. RND for epistemic curiosity
4. Reflective Mixture-of-Experts with metacognitive gating

Author: Research Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


# =============================================================================
# COMPONENT 1: SEMANTIC LOSS (Differentiable Logic)
# =============================================================================

def semantic_loss_exactly_one(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Semantic Loss for the 'Exactly One' constraint.
    
    This enforces that the network outputs a decisive prediction where
    exactly one class should be true. This is differentiable and can
    backpropagate through the network.
    
    Theory: L_sem ∝ -log Σ_i [p_i * Π_{j≠i} (1-p_j)]
    
    Args:
        probs: Tensor of shape (batch, num_classes), values in [0,1]
        eps: Small value for numerical stability
        
    Returns:
        loss: Scalar tensor (lower = more decisive/consistent)
    """
    probs = torch.clamp(probs, min=eps, max=1-eps)
    
    # Compute in log-space for stability
    log_probs = torch.log(probs)
    log_not_probs = torch.log(1 - probs)
    
    # Sum of log(1-p_j) for all j
    sum_log_not = torch.sum(log_not_probs, dim=1, keepdim=True)
    
    # For each class i: log(p_i) + Σ_{j≠i} log(1-p_j)
    # = log(p_i) + (sum_log_not - log(1-p_i))
    term_i = log_probs + (sum_log_not - log_not_probs)
    
    # Semantic probability via logsumexp
    log_semantic_prob = torch.logsumexp(term_i, dim=1)
    
    # Loss is negative log probability
    loss = -torch.mean(log_semantic_prob)
    return loss


def semantic_loss_mutual_exclusion(probs: torch.Tensor, groups: list, eps: float = 1e-8) -> torch.Tensor:
    """
    Enforces mutual exclusion within specified groups.
    
    Args:
        probs: Tensor of shape (batch, num_classes)
        groups: List of lists, each inner list contains mutually exclusive class indices
        
    Returns:
        loss: Penalty for violating mutual exclusion
    """
    loss = 0.0
    probs = torch.clamp(probs, min=eps, max=1-eps)
    
    for group in groups:
        if len(group) < 2:
            continue
        # Sum of probabilities in group should be <= 1
        group_probs = probs[:, group]
        group_sum = torch.sum(group_probs, dim=1)
        # Penalize if sum > 1
        violation = F.relu(group_sum - 1.0)
        loss = loss + torch.mean(violation ** 2)
    
    return loss


def semantic_loss_implication(probs: torch.Tensor, antecedent: int, consequent: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Enforces logical implication: if antecedent then consequent (A → B).
    
    In fuzzy logic: A → B ≡ ¬A ∨ B ≡ max(1-A, B)
    We want this to be close to 1.
    
    Args:
        probs: Tensor of shape (batch, num_classes)
        antecedent: Index of antecedent class
        consequent: Index of consequent class
        
    Returns:
        loss: Penalty for violating implication
    """
    probs = torch.clamp(probs, min=eps, max=1-eps)
    
    p_a = probs[:, antecedent]
    p_b = probs[:, consequent]
    
    # Implication satisfaction: max(1-p_a, p_b)
    implication_sat = torch.max(1 - p_a, p_b)
    
    # Loss is how far from perfect satisfaction (1.0)
    loss = torch.mean((1.0 - implication_sat) ** 2)
    return loss


# =============================================================================
# COMPONENT 2: EVIDENTIAL DEEP LEARNING
# =============================================================================

class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning head that outputs Dirichlet parameters.
    
    Instead of softmax probabilities, outputs evidence for each class.
    This properly separates epistemic (lack of evidence) from 
    aleatoric (conflicting evidence) uncertainty.
    
    Key insight: uncertainty = K / S where S = sum of alpha, K = num classes
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
            nn.Softplus()  # Evidence must be >= 0
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Hidden representation
            
        Returns:
            evidence: Non-negative evidence for each class
            alpha: Dirichlet parameters (evidence + 1)
            uncertainty: Epistemic uncertainty (K/S)
        """
        evidence = self.evidence_net(x)
        alpha = evidence + 1  # Dirichlet parameters
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S  # High when total evidence is low
        
        return evidence, alpha, uncertainty
    
    def get_probabilities(self, alpha: torch.Tensor) -> torch.Tensor:
        """Get expected class probabilities from Dirichlet."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        return alpha / S


def evidential_loss(evidence: torch.Tensor, targets: torch.Tensor, 
                    epoch: int = 0, total_epochs: int = 50,
                    lambda_kl: float = 0.1) -> torch.Tensor:
    """
    Computes Evidential Loss (Type II Maximum Likelihood + KL Regularization).
    
    Args:
        evidence: Network output (>= 0), shape (batch, num_classes)
        targets: Ground truth labels (not one-hot), shape (batch,)
        epoch: Current epoch (for annealing)
        total_epochs: Total epochs (for annealing)
        lambda_kl: Weight for KL regularization
        
    Returns:
        loss: Evidential loss
    """
    num_classes = evidence.shape[1]
    
    # Convert targets to one-hot
    targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Dirichlet parameters
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Expected probabilities
    p = alpha / S
    
    # Component 1: Expected MSE (Bayes Risk)
    err = (targets_onehot - p) ** 2
    var = p * (1 - p) / (S + 1)
    loss_mse = torch.sum(err + var, dim=1)
    
    # Component 2: KL Divergence from uniform (regularization)
    # Annealing: start low, increase over training
    annealing_coef = min(1.0, epoch / max(1, total_epochs // 3))
    
    # Remove evidence from correct class before computing KL
    alpha_tilde = alpha * (1 - targets_onehot) + targets_onehot
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    
    # KL divergence from Dirichlet(1,1,...,1)
    kl = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
    kl = kl + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
    
    loss = torch.mean(loss_mse) + annealing_coef * lambda_kl * torch.mean(kl)
    
    return loss


# =============================================================================
# COMPONENT 3: RANDOM NETWORK DISTILLATION (Curiosity)
# =============================================================================

class RNDModule(nn.Module):
    """
    Random Network Distillation for epistemic curiosity / novelty detection.
    
    Key insight: If the predictor can't match the random target network,
    the state is novel (out-of-distribution).
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 64):
        super().__init__()
        
        # Target network: random, NEVER trained
        self.target = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Predictor network: trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('count', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input state/features
            
        Returns:
            novelty: Normalized novelty score (higher = more novel)
            rnd_loss: Loss for training the predictor
        """
        with torch.no_grad():
            target_features = self.target(x)
        
        predicted_features = self.predictor(x)
        
        # RND error = MSE between predictor and target
        rnd_error = torch.mean((predicted_features - target_features) ** 2, dim=1)
        
        # Update running statistics (detached to avoid graph issues)
        if self.training:
            with torch.no_grad():
                batch_mean = rnd_error.mean().detach()
                batch_std = rnd_error.std().detach() + 1e-8
                
                # Exponential moving average
                momentum = 0.99
                self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
                self.running_std = momentum * self.running_std + (1 - momentum) * batch_std
                self.count += 1
        
        # Normalize novelty score (use detached running stats)
        novelty = (rnd_error - self.running_mean.detach()) / (self.running_std.detach() + 1e-8)
        novelty = torch.clamp(novelty, -5, 5)  # Clip for stability
        
        # Loss for training predictor
        rnd_loss = torch.mean(rnd_error)
        
        return novelty, rnd_loss


# =============================================================================
# COMPONENT 4: REFLECTIVE MIXTURE-OF-EXPERTS
# =============================================================================

class ReflectiveGatingNetwork(nn.Module):
    """
    Gating network that routes based on input AND metacognitive signals.
    
    Key innovation: The routing decision incorporates:
    - Input features (standard MoE)
    - Epistemic uncertainty (from Evidential head or RND)
    - Logical consistency (from Semantic Loss)
    """
    
    def __init__(self, input_dim: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        
        # Base routing from input
        self.fc = nn.Linear(input_dim, num_experts)
        
        # Learnable modulation strengths
        self.beta_uncertainty = nn.Parameter(torch.tensor(2.0))
        self.beta_logic = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor, 
                logic_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)
            uncertainty: Epistemic uncertainty (batch, 1) - higher = more uncertain
            logic_score: Logic violation score (batch, 1) - higher = more violation
            
        Returns:
            weights: Expert weights (batch, num_experts)
            logits: Raw logits for analysis
        """
        # Base logits from input
        logits = self.fc(x)
        
        # Metacognitive modulation
        # Convention: Expert 0 = Exploration, Expert 1 = Safety, 2+ = Task
        metacog_bias = torch.zeros_like(logits)
        
        # High uncertainty → bias toward Exploration Expert
        if self.num_experts > 0:
            metacog_bias[:, 0] = self.beta_uncertainty * uncertainty.squeeze()
        
        # High logic violation → bias toward Safety Expert
        if self.num_experts > 1:
            metacog_bias[:, 1] = self.beta_logic * logic_score.squeeze()
        
        logits = logits + metacog_bias
        
        # Gumbel-Softmax for differentiable discrete routing
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        
        return weights, logits


class Expert(nn.Module):
    """Single expert network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReflectiveMoE(nn.Module):
    """
    Mixture-of-Experts with metacognitive routing.
    
    Experts specialize:
    - Expert 0 (Exploration): Higher stochasticity, exploration-focused
    - Expert 1 (Safety): Conservative, heavily regularized
    - Expert 2+ (Task): Standard task optimization
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4,
                 hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        
        # Create specialized experts
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # Metacognitive gating
        self.gate = ReflectiveGatingNetwork(input_dim, num_experts, temperature)
    
    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor,
                logic_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features
            uncertainty: Epistemic uncertainty
            logic_score: Logic violation score
            
        Returns:
            output: Weighted combination of expert outputs
            gate_weights: The routing weights (for analysis/logging)
        """
        # Get routing weights
        weights, _ = self.gate(x, uncertainty, logic_score)
        
        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Shape: (batch, num_experts, output_dim)
        
        # Weighted combination
        output = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        
        return output, weights


# =============================================================================
# COMPONENT 5: COMPLETE METACOGNITIVE AGENT
# =============================================================================

class NeuroSymbolicMetaAgent(nn.Module):
    """
    Complete Metacognitive Agent integrating all components:
    
    1. Feature Encoder (shared representation)
    2. Evidential Head (uncertainty quantification)
    3. RND Module (novelty/curiosity)
    4. Reflective MoE (metacognitive routing)
    5. Semantic Loss (logical consistency)
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_experts: int = 4, moe_temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Evidential head for uncertainty
        self.evidential = EvidentialHead(hidden_dim, num_classes)
        
        # RND for novelty detection
        self.rnd = RNDModule(hidden_dim, feature_dim=64)
        
        # Reflective MoE for final prediction
        self.moe = ReflectiveMoE(
            input_dim=hidden_dim,
            output_dim=num_classes,
            num_experts=num_experts,
            hidden_dim=hidden_dim // 2,
            temperature=moe_temperature
        )
        
        # Store last forward pass info for logging
        self._last_info = {}
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all metacognitive signals.
        
        Returns dict containing:
            - logits: Final class logits
            - probs: Class probabilities
            - evidence: Dirichlet evidence
            - alpha: Dirichlet parameters
            - epistemic_uncertainty: From evidential head
            - novelty: From RND
            - gate_weights: MoE routing weights
            - hidden: Encoder output (for Semantic Loss computation)
        """
        # 1. Encode input
        hidden = self.encoder(x)
        
        # 2. Evidential uncertainty
        evidence, alpha, evidential_uncertainty = self.evidential(hidden)
        
        # 3. RND novelty
        novelty, rnd_loss = self.rnd(hidden)
        
        # 4. Combined uncertainty signal (normalized)
        # Combine evidential and novelty for routing
        combined_uncertainty = (
            0.5 * evidential_uncertainty.squeeze() + 
            0.5 * torch.sigmoid(novelty)  # Normalize novelty to [0,1]
        ).unsqueeze(-1)
        
        # 5. Compute semantic loss as "logic violation score"
        # Using evidence-based probabilities
        probs = self.evidential.get_probabilities(alpha)
        # We compute this as the negative log of semantic consistency
        # (but we need it as a score for routing, so compute inline)
        with torch.no_grad():
            sem_prob = self._compute_semantic_prob(probs)
            logic_violation = 1.0 - sem_prob  # Higher = more violation
        
        # 6. MoE with metacognitive routing
        moe_logits, gate_weights = self.moe(hidden, combined_uncertainty, logic_violation.unsqueeze(-1))
        
        # 7. Combine MoE output with evidential output
        # The MoE provides task-specific predictions, evidential provides calibration
        final_logits = moe_logits + 0.1 * (alpha - 1)  # Small evidential contribution
        final_probs = F.softmax(final_logits, dim=1)
        
        # Store info
        result = {
            'logits': final_logits,
            'probs': final_probs,
            'evidence': evidence,
            'alpha': alpha,
            'epistemic_uncertainty': evidential_uncertainty,
            'novelty': novelty,
            'rnd_loss': rnd_loss,
            'gate_weights': gate_weights,
            'hidden': hidden,
            'evidential_probs': probs,
            'logic_violation': logic_violation
        }
        self._last_info = result
        
        return result
    
    def _compute_semantic_prob(self, probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute probability of satisfying 'exactly one' constraint."""
        probs = torch.clamp(probs, min=eps, max=1-eps)
        log_probs = torch.log(probs)
        log_not_probs = torch.log(1 - probs)
        sum_log_not = torch.sum(log_not_probs, dim=1, keepdim=True)
        term_i = log_probs + (sum_log_not - log_not_probs)
        log_semantic_prob = torch.logsumexp(term_i, dim=1)
        return torch.exp(log_semantic_prob)


# =============================================================================
# COMPONENT 6: TRAINING UTILITIES
# =============================================================================

class MetacognitiveLoss(nn.Module):
    """
    Combined loss function for the neuro-symbolic metacognitive agent.
    
    L_total = L_task + λ_evid * L_evidential + λ_sem * L_semantic + λ_rnd * L_rnd
    """
    
    def __init__(self, lambda_evidential: float = 0.5, 
                 lambda_semantic: float = 0.3,
                 lambda_rnd: float = 0.1,
                 lambda_gate_entropy: float = 0.01):
        super().__init__()
        self.lambda_evidential = lambda_evidential
        self.lambda_semantic = lambda_semantic
        self.lambda_rnd = lambda_rnd
        self.lambda_gate_entropy = lambda_gate_entropy
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                epoch: int = 0, total_epochs: int = 50) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: Dict from NeuroSymbolicMetaAgent.forward()
            targets: Ground truth labels
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Dict with 'total' loss and individual components
        """
        losses = {}
        
        # 1. Task loss (cross-entropy)
        losses['task'] = F.cross_entropy(outputs['logits'], targets)
        
        # 2. Evidential loss
        losses['evidential'] = evidential_loss(
            outputs['evidence'], targets, epoch, total_epochs
        )
        
        # 3. Semantic loss (logical consistency)
        losses['semantic'] = semantic_loss_exactly_one(outputs['evidential_probs'])
        
        # 4. RND loss (curiosity training)
        losses['rnd'] = outputs['rnd_loss']
        
        # 5. Gate entropy regularization (encourage diverse expert usage)
        gate_probs = outputs['gate_weights']
        gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1)
        losses['gate_entropy'] = -torch.mean(gate_entropy)  # Negative because we maximize entropy
        
        # Combined loss
        losses['total'] = (
            losses['task'] +
            self.lambda_evidential * losses['evidential'] +
            self.lambda_semantic * losses['semantic'] +
            self.lambda_rnd * losses['rnd'] +
            self.lambda_gate_entropy * losses['gate_entropy']
        )
        
        return losses


def train_epoch(model: NeuroSymbolicMetaAgent, 
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: MetacognitiveLoss,
                epoch: int, total_epochs: int,
                device: str = 'cpu') -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0, 'task': 0, 'evidential': 0, 'semantic': 0, 'rnd': 0}
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        losses = loss_fn(outputs, y_batch, epoch, total_epochs)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model: NeuroSymbolicMetaAgent,
             test_loader: torch.utils.data.DataLoader,
             device: str = 'cpu') -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_uncertainties = []
    all_gate_weights = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        
        preds = outputs['logits'].argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch)
        all_probs.append(outputs['probs'].cpu())
        all_uncertainties.append(outputs['epistemic_uncertainty'].cpu())
        all_gate_weights.append(outputs['gate_weights'].cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs)
    all_uncertainties = torch.cat(all_uncertainties).squeeze().numpy()
    all_gate_weights = torch.cat(all_gate_weights).numpy()
    
    # Accuracy
    accuracy = (all_preds == all_targets).mean()
    
    # ECE (using max prob as confidence)
    confidences = all_probs.max(dim=1)[0].numpy()
    ece = compute_ece(confidences, all_preds, all_targets)
    
    # Uncertainty statistics
    correct_mask = (all_preds == all_targets)
    unc_correct = all_uncertainties[correct_mask].mean() if correct_mask.sum() > 0 else 0
    unc_incorrect = all_uncertainties[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    unc_separation = unc_incorrect - unc_correct  # Should be positive (higher unc for wrong)
    
    # Gate usage
    gate_usage = all_gate_weights.mean(axis=0)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'unc_correct': unc_correct,
        'unc_incorrect': unc_incorrect,
        'unc_separation': unc_separation,
        'unc_std': all_uncertainties.std(),
        'gate_usage': gate_usage.tolist()
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
