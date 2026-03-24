"""
V2: Self-Supervised Metacognitive Architecture

Key idea: Derive uncertainty from self-supervised signals (prediction consistency,
representation geometry, entropy matching) instead of binary correct/incorrect labels.
This directly addresses the monitor collapse problem from V1.

Author: Ismail Haddou / Nu Terra Labs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ==============================================================================
# Data Augmentation for Consistency-Based Uncertainty
# ==============================================================================

class FeatureAugmentor(nn.Module):
    """
    Augments input features to create multiple 'views' for consistency estimation.
    For tabular data, this means noise injection + feature dropout + scaling.
    
    The key insight: if the model gives different predictions for slightly different
    versions of the same input, it's uncertain about that input.
    """
    
    def __init__(self, noise_std: float = 0.1, dropout_rate: float = 0.15, 
                 scale_range: Tuple[float, float] = (0.9, 1.1)):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.scale_range = scale_range
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create an augmented view of the input."""
        if not self.training:
            return x
        
        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise
        
        # Feature dropout (zero out random features)
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_rate))
        x_aug = x_aug * mask
        
        # Random scaling per feature
        scale = torch.empty_like(x).uniform_(*self.scale_range)
        x_aug = x_aug * scale
        
        return x_aug


# ==============================================================================
# Shared Encoder
# ==============================================================================

class SharedEncoder(nn.Module):
    """
    Shared feature extractor. Produces representations used by both the classifier
    and the uncertainty head. Uses LayerNorm + GELU for stable training.
    """
    
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


# ==============================================================================
# Classifier Head
# ==============================================================================

class ClassifierHead(nn.Module):
    """Standard classifier. We keep its weight matrix accessible for 
    geometric boundary distance computation."""
    
    def __init__(self, repr_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(repr_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
    
    @property
    def weight(self) -> torch.Tensor:
        return self.fc.weight
    
    @property
    def bias(self) -> torch.Tensor:
        return self.fc.bias


# ==============================================================================
# Uncertainty Head (Self-Supervised)
# ==============================================================================

class UncertaintyHead(nn.Module):
    """
    Predicts uncertainty from the representation. Unlike V1's meta-monitor,
    this is trained with SELF-SUPERVISED signals (consistency, geometry, entropy)
    — not binary correct/incorrect labels.
    
    Outputs u ∈ (0, 1) where higher = more uncertain.
    """
    
    def __init__(self, repr_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns uncertainty score u ∈ (0, 1)."""
        return self.net(h).squeeze(-1)


# ==============================================================================
# Metacognitive Controller
# ==============================================================================

class MetacognitiveController(nn.Module):
    """
    Uses the uncertainty signal to modulate predictions.
    When uncertain → push predictions toward uniform distribution.
    When confident → keep original predictions.
    
    Unlike V1's controller (which had a collapsed uncertainty signal),
    V2's controller receives a self-supervised uncertainty signal that
    varies meaningfully per sample.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # Learnable blending temperature
        self.blend_temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, logits: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Blend between model prediction and uniform distribution based on uncertainty.
        
        Args:
            logits: [batch, num_classes] raw classifier output
            uncertainty: [batch] uncertainty scores in (0, 1)
        
        Returns:
            adjusted_logits: [batch, num_classes]
        """
        # Smooth blending factor
        blend = torch.sigmoid(self.blend_temp * (uncertainty - 0.5)).unsqueeze(-1)
        
        # Uniform logits (equal probability for all classes)
        uniform = torch.zeros_like(logits)
        
        # Blend: high uncertainty → toward uniform, low uncertainty → keep original
        adjusted = (1 - blend) * logits + blend * uniform
        
        return adjusted


# ==============================================================================
# Self-Supervised Signal Computers
# ==============================================================================

class ConsistencyEstimator:
    """
    Computes prediction consistency across augmented views.
    
    Core idea: Create multiple augmented versions of each input, run them through
    the model, and measure how much the predictions vary. High variance → uncertain.
    
    This is a SELF-SUPERVISED signal — it requires no labels.
    """
    
    def __init__(self, n_views: int = 4):
        self.n_views = n_views
    
    def compute(self, encoder: nn.Module, classifier: nn.Module,
                augmentor: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency score for each sample.
        
        Returns:
            consistency_uncertainty: [batch] — higher means more inconsistent (uncertain)
        """
        predictions = []
        
        encoder_training = encoder.training
        classifier_training = classifier.training
        
        # Generate multiple views and collect predictions
        for _ in range(self.n_views):
            x_aug = augmentor(x)
            h_aug = encoder(x_aug)
            logits_aug = classifier(h_aug)
            probs_aug = F.softmax(logits_aug, dim=-1)
            predictions.append(probs_aug)
        
        # Stack: [n_views, batch, classes]
        preds = torch.stack(predictions, dim=0)
        
        # Mean prediction across views
        mean_pred = preds.mean(dim=0)  # [batch, classes]
        
        # Consistency = average KL divergence from mean (Jensen-Shannon-like)
        kl_sum = torch.zeros(x.size(0), device=x.device)
        for i in range(self.n_views):
            kl = F.kl_div(
                (mean_pred + 1e-8).log(), 
                preds[i], 
                reduction='none',
                log_target=False
            ).sum(dim=-1)
            kl_sum += kl
        
        consistency_uncertainty = kl_sum / self.n_views
        
        # Normalize to (0, 1) range using sigmoid
        consistency_uncertainty = torch.sigmoid(consistency_uncertainty * 10)
        
        return consistency_uncertainty.detach()


class GeometricUncertainty:
    """
    Computes uncertainty from distance to decision boundaries in representation space.
    
    Points near decision boundaries (where the classifier is torn between classes)
    are inherently uncertain. We compute this geometrically from the classifier's
    weight vectors without any labels.
    """
    
    @staticmethod
    def compute(h: torch.Tensor, classifier: ClassifierHead) -> torch.Tensor:
        """
        Compute normalized distance to nearest decision boundary.
        
        For a linear classifier with weights W and biases b, the decision boundary
        between classes i and j is the hyperplane:
            (w_i - w_j)^T h + (b_i - b_j) = 0
        
        Distance from h to this boundary:
            d_ij = |((w_i - w_j)^T h + (b_i - b_j))| / ||w_i - w_j||
        
        Returns:
            boundary_uncertainty: [batch] — closer to boundary = higher uncertainty
        """
        W = classifier.weight  # [num_classes, repr_dim]
        b = classifier.bias    # [num_classes]
        K = W.size(0)
        
        min_distances = torch.full((h.size(0),), float('inf'), device=h.device)
        
        for i in range(K):
            for j in range(i + 1, K):
                w_diff = W[i] - W[j]  # [repr_dim]
                b_diff = b[i] - b[j]  # scalar
                
                # Distance to boundary
                numerator = torch.abs(h @ w_diff + b_diff)  # [batch]
                denominator = w_diff.norm() + 1e-8
                distance = numerator / denominator
                
                min_distances = torch.min(min_distances, distance)
        
        # Normalize: small distance → high uncertainty
        # Use sigmoid to map to (0, 1), invert so close=high
        boundary_uncertainty = torch.sigmoid(-min_distances + min_distances.median())
        
        return boundary_uncertainty.detach()


class EntropyUncertainty:
    """
    Maps classifier entropy to uncertainty.
    
    High entropy in the classifier's output → the model is "confused" among classes.
    This is a richer signal than binary correct/incorrect.
    """
    
    @staticmethod
    def compute(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized entropy of classifier predictions.
        
        Returns:
            entropy_uncertainty: [batch] — in (0, 1), where 1 = max entropy
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch]
        
        # Normalize by max entropy (log K)
        max_entropy = np.log(logits.size(-1))
        normalized = entropy / max_entropy
        
        return normalized.detach()


# ==============================================================================
# EMA Teacher for Self-Distillation
# ==============================================================================

class EMATeacher:
    """
    Exponential Moving Average teacher for progressive self-distillation.
    
    Every update, the teacher slowly tracks the student. The teacher's predictions
    serve as soft targets for the uncertainty head — providing a stable, slowly-evolving
    signal about what the model "should" predict.
    
    When the student's current predictions diverge from the teacher's, that indicates
    the model is still learning about those samples → high uncertainty.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    @torch.no_grad()
    def apply(self, model: nn.Module):
        """Temporarily apply EMA weights to model."""
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        return backup
    
    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        """Restore original weights after EMA inference."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ==============================================================================
# Full V2 Model
# ==============================================================================

class SelfSupervisedMetacognition(nn.Module):
    """
    V2 Self-Supervised Metacognitive Architecture.
    
    Unlike V1 (which used binary correct/incorrect labels → collapse),
    V2 derives uncertainty from:
        1. Prediction consistency under augmentation
        2. Distance to decision boundaries
        3. Classifier entropy
        4. Divergence from EMA teacher (self-distillation)
    
    ALL self-supervised. No labels needed for the uncertainty signal.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dim: int = 128, repr_dim: int = 64,
                 n_consistency_views: int = 4):
        super().__init__()
        
        # Core components
        self.encoder = SharedEncoder(input_dim, hidden_dim, repr_dim)
        self.classifier = ClassifierHead(repr_dim, num_classes)
        self.uncertainty_head = UncertaintyHead(repr_dim)
        self.controller = MetacognitiveController(num_classes)
        self.augmentor = FeatureAugmentor()
        
        # Self-supervised signal computers
        self.consistency_estimator = ConsistencyEstimator(n_views=n_consistency_views)
        
        # Config
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.repr_dim = repr_dim
    
    def forward(self, x: torch.Tensor, 
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, input_dim] input features
            return_details: if True, return all intermediate values
        
        Returns:
            dict with keys:
                - 'logits': raw classifier logits
                - 'adjusted_logits': metacognitively adjusted logits
                - 'uncertainty': predicted uncertainty ∈ (0, 1)
                - 'representation': hidden representation
                + (if return_details) self-supervised targets
        """
        # Encode
        h = self.encoder(x)
        
        # Classify
        logits = self.classifier(h)
        
        # Predict uncertainty (self-supervised head)
        uncertainty = self.uncertainty_head(h)
        
        # Controller adjusts logits based on uncertainty
        adjusted_logits = self.controller(logits, uncertainty)
        
        result = {
            'logits': logits,
            'adjusted_logits': adjusted_logits,
            'uncertainty': uncertainty,
            'representation': h,
        }
        
        if return_details and self.training:
            # Compute self-supervised targets
            consistency_target = self.consistency_estimator.compute(
                self.encoder, self.classifier, self.augmentor, x
            )
            geometry_target = GeometricUncertainty.compute(h, self.classifier)
            entropy_target = EntropyUncertainty.compute(logits)
            
            result['consistency_target'] = consistency_target
            result['geometry_target'] = geometry_target
            result['entropy_target'] = entropy_target
        
        return result
    
    def get_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and calibrated confidence for inference.
        
        Returns:
            predictions: [batch] predicted class indices
            confidence: [batch] confidence scores ∈ (0, 1)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x)
            probs = F.softmax(result['adjusted_logits'], dim=-1)
            confidence = 1.0 - result['uncertainty']
            predictions = probs.argmax(dim=-1)
        return predictions, confidence


# ==============================================================================
# Loss Functions
# ==============================================================================

class V2MetacognitiveLoss(nn.Module):
    """
    Combined loss for V2 training.
    
    L = L_task + α·L_consistency + β·L_geometry + γ·L_entropy + δ·L_distill
    
    All uncertainty losses are SELF-SUPERVISED.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, 
                 gamma: float = 0.5, delta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Consistency weight
        self.beta = beta    # Geometry weight
        self.gamma = gamma  # Entropy weight
        self.delta = delta  # Self-distillation weight
    
    def forward(self, result: Dict[str, torch.Tensor], 
                targets: torch.Tensor,
                teacher_probs: Optional[torch.Tensor] = None,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            result: output from model.forward(x, return_details=True)
            targets: [batch] ground truth class indices (for task loss only)
            teacher_probs: [batch, classes] EMA teacher predictions (for distillation)
            epoch: current epoch (for progressive scheduling)
        
        Returns:
            dict with 'total' and individual loss components
        """
        losses = {}
        
        # Task loss (standard cross-entropy — we still need the model to learn)
        losses['task'] = F.cross_entropy(result['adjusted_logits'], targets)
        
        # === Self-supervised uncertainty losses ===
        uncertainty = result['uncertainty']
        
        # 1. Consistency loss: uncertainty should match prediction instability
        if 'consistency_target' in result:
            losses['consistency'] = F.mse_loss(uncertainty, result['consistency_target'])
        else:
            losses['consistency'] = torch.tensor(0.0, device=uncertainty.device)
        
        # 2. Geometry loss: uncertainty should match boundary proximity
        if 'geometry_target' in result:
            losses['geometry'] = F.mse_loss(uncertainty, result['geometry_target'])
        else:
            losses['geometry'] = torch.tensor(0.0, device=uncertainty.device)
        
        # 3. Entropy loss: uncertainty should match prediction entropy
        if 'entropy_target' in result:
            losses['entropy'] = F.mse_loss(uncertainty, result['entropy_target'])
        else:
            losses['entropy'] = torch.tensor(0.0, device=uncertainty.device)
        
        # 4. Self-distillation loss (progressive — ramps up over training)
        if teacher_probs is not None and epoch >= 5:
            student_probs = F.softmax(result['adjusted_logits'], dim=-1)
            distill_uncertainty = F.kl_div(
                (student_probs + 1e-8).log(),
                teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1)
            # Normalize and use as additional uncertainty target
            distill_target = torch.sigmoid(distill_uncertainty * 5)
            losses['distill'] = F.mse_loss(uncertainty, distill_target.detach())
            # Ramp up distillation weight over epochs
            distill_scale = min(1.0, (epoch - 5) / 15.0)
            losses['distill'] = losses['distill'] * distill_scale
        else:
            losses['distill'] = torch.tensor(0.0, device=uncertainty.device)
        
        # Total loss
        losses['total'] = (
            losses['task'] 
            + self.alpha * losses['consistency']
            + self.beta * losses['geometry']
            + self.gamma * losses['entropy']
            + self.delta * losses['distill']
        )
        
        return losses


# ==============================================================================
# Diagnostic Tools
# ==============================================================================

def diagnose_uncertainty(model: SelfSupervisedMetacognition, 
                         X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Run health diagnostics on the uncertainty head.
    
    Same criteria as V1, but we expect V2 to pass them:
    - Std(u) > 0.1 (not collapsed)
    - Separation > 0.03 (discriminative)
    - u(incorrect) > u(correct) (higher uncertainty on wrong predictions)
    """
    model.eval()
    with torch.no_grad():
        result = model.forward(X)
        uncertainty = result['uncertainty'].numpy()
        preds = result['adjusted_logits'].argmax(dim=-1).numpy()
        y_np = y.numpy()
    
    correct_mask = preds == y_np
    incorrect_mask = ~correct_mask
    
    diagnostics = {
        'mean_uncertainty': float(uncertainty.mean()),
        'std_uncertainty': float(uncertainty.std()),
        'mean_u_correct': float(uncertainty[correct_mask].mean()) if correct_mask.any() else 0.0,
        'mean_u_incorrect': float(uncertainty[incorrect_mask].mean()) if incorrect_mask.any() else 0.0,
        'separation': 0.0,
        'accuracy': float(correct_mask.mean()),
        'collapsed': uncertainty.std() < 0.1,
        'discriminative': False,
    }
    
    if incorrect_mask.any() and correct_mask.any():
        diagnostics['separation'] = (
            diagnostics['mean_u_incorrect'] - diagnostics['mean_u_correct']
        )
        diagnostics['discriminative'] = diagnostics['separation'] > 0.03
    
    return diagnostics


def compute_ece(logits: torch.Tensor, targets: torch.Tensor, 
                n_bins: int = 15) -> float:
    """Expected Calibration Error (same as V1 for fair comparison)."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(targets).float()
    
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].mean().item()
            ece += mask.float().mean().item() * abs(avg_acc - avg_conf)
    
    return ece
