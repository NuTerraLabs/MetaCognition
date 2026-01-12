"""
Core Neural Architecture for Synthetic Metacognition

This module implements the triadic structure:
1. Base Learner: Makes predictions and exposes internal state
2. Meta-Monitor: Estimates epistemic uncertainty from hidden representations
3. Meta-Controller: Modulates predictions based on uncertainty

Author: Anonymous
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BaseLearner(nn.Module):
    """
    Base learning network that produces predictions and exposes intermediate representations.
    
    Architecture:
    - Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
    - Returns both final output and hidden state z for metacognitive monitoring
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output (number of classes)
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        super(BaseLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using He initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through base learner
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            out: Logits of shape (batch_size, output_dim)
            z: Hidden representation of shape (batch_size, hidden_dim)
        """
        # First hidden layer
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout1(h1)
        
        # Second hidden layer (this is our intermediate representation z)
        z = F.relu(self.bn2(self.fc2(h1)))
        z = self.dropout2(z)
        
        # Output layer
        out = self.fc3(z)
        
        return out, z


class MetaMonitor(nn.Module):
    """
    Meta-monitoring network that estimates epistemic uncertainty from internal representations.
    
    Maps hidden state z to uncertainty score u ∈ [0,1]
    - High u: Model is confident
    - Low u: Model is uncertain
    
    Architecture:
    - Hidden state -> Dense layer -> Sigmoid -> Uncertainty score
    
    Args:
        hidden_dim (int): Dimension of hidden representation from base learner
        monitor_dim (int): Dimension of monitoring network (default: hidden_dim // 2)
    """
    
    def __init__(self, hidden_dim: int, monitor_dim: Optional[int] = None):
        super(MetaMonitor, self).__init__()
        
        if monitor_dim is None:
            monitor_dim = hidden_dim // 2
        
        self.fc1 = nn.Linear(hidden_dim, monitor_dim)
        self.fc2 = nn.Linear(monitor_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)  # Start with neutral uncertainty
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty from hidden representation
        
        Args:
            z: Hidden state of shape (batch_size, hidden_dim)
            
        Returns:
            u: Uncertainty score of shape (batch_size, 1), values in [0,1]
               Higher values indicate higher confidence
        """
        h = F.relu(self.fc1(z))
        uncertainty = torch.sigmoid(self.fc2(h))
        return uncertainty


class MetaController(nn.Module):
    """
    Meta-controller that modulates predictions based on uncertainty assessment.
    
    Implements gating mechanism:
        y_adjusted = y_base ⊙ σ(W·u + b)
    
    When uncertainty is low (model unsure):
    - Gating signal dampens extreme predictions
    - Pushes output toward uniform distribution
    
    When uncertainty is high (model confident):
    - Gating signal ≈ 1, minimal modulation
    
    Args:
        output_dim (int): Dimension of base learner output
        control_type (str): Type of control mechanism ('gate' or 'additive')
    """
    
    def __init__(self, output_dim: int, control_type: str = 'gate'):
        super(MetaController, self).__init__()
        
        self.control_type = control_type
        self.gate = nn.Linear(1, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to start with minimal modulation"""
        nn.init.xavier_normal_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)  # Bias toward confidence initially
    
    def forward(
        self, 
        out: torch.Tensor, 
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate predictions based on uncertainty
        
        Args:
            out: Base learner logits of shape (batch_size, output_dim)
            uncertainty: Uncertainty score of shape (batch_size, 1)
            
        Returns:
            modulated_out: Adjusted logits of shape (batch_size, output_dim)
        """
        if self.control_type == 'gate':
            # Gating mechanism: multiply by confidence-based weights
            control_signal = torch.sigmoid(self.gate(uncertainty))
            modulated_out = out * control_signal
        elif self.control_type == 'additive':
            # Additive mechanism: add confidence-based correction
            control_signal = torch.tanh(self.gate(uncertainty))
            modulated_out = out + control_signal
        else:
            raise ValueError(f"Unknown control_type: {self.control_type}")
        
        return modulated_out


class MetaCognitiveModel(nn.Module):
    """
    Complete metacognitive system integrating Base Learner, Meta-Monitor, and Meta-Controller.
    
    Forward Pass:
    1. Base learner produces initial prediction and hidden state: y⁽⁰⁾, z = f_θ(x)
    2. Meta-monitor estimates uncertainty: u = g_φ(z)
    3. Meta-controller adjusts prediction: y = m_ψ(y⁽⁰⁾, u)
    
    This creates a single-loop feedback mechanism for real-time metacognition.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (number of classes)
        monitor_dim (int): Meta-monitor hidden dimension
        dropout_rate (float): Dropout rate for base learner
        control_type (str): Type of control mechanism ('gate' or 'additive')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        monitor_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        control_type: str = 'gate'
    ):
        super(MetaCognitiveModel, self).__init__()
        
        self.base = BaseLearner(input_dim, hidden_dim, output_dim, dropout_rate)
        self.monitor = MetaMonitor(hidden_dim, monitor_dim)
        self.controller = MetaController(output_dim, control_type)
        
        # Store configuration
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'monitor_dim': monitor_dim,
            'dropout_rate': dropout_rate,
            'control_type': control_type
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Complete metacognitive forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_uncertainty: If True, return (output, uncertainty)
            return_all: If True, return (output, uncertainty, base_output, hidden_state)
            
        Returns:
            If return_uncertainty=False: adjusted_out of shape (batch_size, output_dim)
            If return_uncertainty=True: (adjusted_out, uncertainty)
            If return_all=True: (adjusted_out, uncertainty, base_out, hidden_state)
        """
        # Base prediction
        base_out, z = self.base(x)
        
        # Meta-monitoring
        u = self.monitor(z)
        
        # Meta-control
        adjusted_out = self.controller(base_out, u)
        
        # Return based on flags
        if return_all:
            return adjusted_out, u, base_out, z
        elif return_uncertainty:
            return adjusted_out, u
        else:
            return adjusted_out
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only uncertainty estimate for input
        
        Args:
            x: Input tensor
            
        Returns:
            u: Uncertainty scores
        """
        with torch.no_grad():
            _, z = self.base(x)
            u = self.monitor(z)
        return u
    
    def predict_with_confidence(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions and indicate which ones are reliable
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for reliable predictions
            
        Returns:
            predictions: Class predictions
            confidence: Confidence scores
            reliable: Boolean mask indicating reliable predictions (u > threshold)
        """
        with torch.no_grad():
            out, u = self(x, return_uncertainty=True)
            probs = F.softmax(out, dim=1)
            predictions = torch.argmax(probs, dim=1)
            confidence = u.squeeze()
            reliable = confidence > threshold
        
        return predictions, confidence, reliable


class BaselineMLP(nn.Module):
    """
    Baseline MLP without metacognition for comparison
    
    Same architecture as BaseLearner but without exposing internal state
    or metacognitive components.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout1(h1)
        
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout2(h2)
        
        out = self.fc3(h2)
        return out


def test_models():
    """Test function to verify model architectures"""
    print("Testing Synthetic Metacognition Models...\n")
    
    # Configuration
    batch_size = 32
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    
    # Generate random data
    x = torch.randn(batch_size, input_dim)
    
    # Test BaseLearner
    print("1. Testing BaseLearner...")
    base = BaseLearner(input_dim, hidden_dim, output_dim)
    out, z = base(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Hidden state shape: {z.shape}")
    assert out.shape == (batch_size, output_dim)
    assert z.shape == (batch_size, hidden_dim)
    print("   ✓ BaseLearner working correctly\n")
    
    # Test MetaMonitor
    print("2. Testing MetaMonitor...")
    monitor = MetaMonitor(hidden_dim)
    u = monitor(z)
    print(f"   Hidden state shape: {z.shape}")
    print(f"   Uncertainty shape: {u.shape}")
    print(f"   Uncertainty range: [{u.min():.3f}, {u.max():.3f}]")
    assert u.shape == (batch_size, 1)
    assert (u >= 0).all() and (u <= 1).all()
    print("   ✓ MetaMonitor working correctly\n")
    
    # Test MetaController
    print("3. Testing MetaController...")
    controller = MetaController(output_dim)
    adjusted_out = controller(out, u)
    print(f"   Base output shape: {out.shape}")
    print(f"   Adjusted output shape: {adjusted_out.shape}")
    assert adjusted_out.shape == out.shape
    print("   ✓ MetaController working correctly\n")
    
    # Test MetaCognitiveModel
    print("4. Testing MetaCognitiveModel...")
    model = MetaCognitiveModel(input_dim, hidden_dim, output_dim)
    
    # Test basic forward
    final_out = model(x)
    print(f"   Basic forward output shape: {final_out.shape}")
    assert final_out.shape == (batch_size, output_dim)
    
    # Test with uncertainty
    final_out, u = model(x, return_uncertainty=True)
    print(f"   With uncertainty - output: {final_out.shape}, u: {u.shape}")
    assert final_out.shape == (batch_size, output_dim)
    assert u.shape == (batch_size, 1)
    
    # Test with all outputs
    final_out, u, base_out, z = model(x, return_all=True)
    print(f"   With all outputs - final: {final_out.shape}, u: {u.shape}, base: {base_out.shape}, z: {z.shape}")
    
    # Test prediction with confidence
    predictions, confidence, reliable = model.predict_with_confidence(x, threshold=0.5)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Confidence shape: {confidence.shape}")
    print(f"   Reliable predictions: {reliable.sum().item()}/{batch_size}")
    print("   ✓ MetaCognitiveModel working correctly\n")
    
    # Test BaselineMLP
    print("5. Testing BaselineMLP...")
    baseline = BaselineMLP(input_dim, hidden_dim, output_dim)
    baseline_out = baseline(x)
    print(f"   Baseline output shape: {baseline_out.shape}")
    assert baseline_out.shape == (batch_size, output_dim)
    print("   ✓ BaselineMLP working correctly\n")
    
    print("All model tests passed! ✓")


if __name__ == "__main__":
    test_models()
