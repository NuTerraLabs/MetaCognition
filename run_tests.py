"""
Comprehensive test suite for Synthetic Metacognition

Tests all components to ensure correct functionality.

Run: python run_tests.py
"""

import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

print("="*80)
print("SYNTHETIC METACOGNITION - TEST SUITE")
print("="*80)

# Test 1: Import all modules
print("\n[1/7] Testing imports...")
try:
    from src.models import MetaCognitiveModel, BaselineMLP, BaseLearner, MetaMonitor, MetaController
    from src.training import MetacognitiveTrainer, MetacognitiveLoss
    from src.evaluation import MetacognitiveEvaluator, expected_calibration_error
    from src.reflection import MetacognitiveAgent, Proposition, ReflectionRules
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model instantiation
print("\n[2/7] Testing model instantiation...")
try:
    model = MetaCognitiveModel(
        input_dim=10,
        hidden_dim=32,
        output_dim=2,
        monitor_dim=16
    )
    print(f"✓ MetaCognitiveModel created: {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n[3/7] Testing forward pass...")
try:
    x = torch.randn(16, 10)
    
    # Basic forward
    out = model(x)
    assert out.shape == (16, 2), f"Wrong output shape: {out.shape}"
    
    # Forward with uncertainty
    out, u = model(x, return_uncertainty=True)
    assert u.shape == (16, 1), f"Wrong uncertainty shape: {u.shape}"
    assert (u >= 0).all() and (u <= 1).all(), "Uncertainty out of range"
    
    # Forward with all outputs
    out, u, base_out, z = model(x, return_all=True)
    assert z.shape == (16, 32), f"Wrong hidden shape: {z.shape}"
    
    print("✓ Forward pass working correctly")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Training
print("\n[4/7] Testing training loop...")
try:
    # Create dummy dataset
    X_train = torch.randn(200, 10)
    y_train = torch.randint(0, 2, (200,))
    X_val = torch.randn(50, 10)
    y_val = torch.randint(0, 2, (50,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_meta=0.1)
    trainer = MetacognitiveTrainer(model, optimizer, loss_fn, device='cpu')
    
    history = trainer.fit(train_loader, val_loader, epochs=3, verbose=False)
    
    assert 'train_loss' in history, "Missing training history"
    assert len(history['train_loss']) == 3, "Wrong number of epochs recorded"
    
    print(f"✓ Training completed: Final val_acc = {history['val_accuracy'][-1]:.4f}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    sys.exit(1)

# Test 5: Evaluation
print("\n[5/7] Testing evaluation metrics...")
try:
    evaluator = MetacognitiveEvaluator(model, device='cpu')
    metrics = evaluator.evaluate(val_loader, compute_detailed=True)
    
    required_metrics = ['accuracy', 'ece', 'brier_score', 'uncertainty_error_corr']
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], float), f"Metric {metric} is not float"
    
    print(f"✓ Evaluation metrics computed:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - ECE: {metrics['ece']:.4f}")
    print(f"  - Uncertainty-Error Corr: {metrics['uncertainty_error_corr']:.4f}")
except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    sys.exit(1)

# Test 6: Reflection system
print("\n[6/7] Testing formal reflection system...")
try:
    agent = MetacognitiveAgent(name="TestAgent")
    
    # Add beliefs
    agent.add_belief("test_belief_1", confidence=0.9)
    agent.add_belief("test_belief_2", confidence=0.7)
    agent.add_belief("¬test_belief_2", confidence=0.4)
    
    # Reflect
    meta_prop = agent.reflect_on_belief("test_belief_1")
    assert meta_prop is not None, "Reflection failed"
    
    # Check coherence
    violations = agent.check_coherence(verbose=False)
    
    # Forward reasoning
    conclusion = agent.reason_forward(
        premises=["test_belief_1", "test_belief_2"],
        conclusion_name="combined",
        combination_rule='min'
    )
    assert conclusion.confidence == min(0.9, 0.7), "Uncertainty propagation failed"
    
    # Assess uncertainty
    stats = agent.assess_uncertainty()
    assert 'mean' in stats and 'std' in stats, "Missing uncertainty statistics"
    
    print(f"✓ Reflection system working:")
    print(f"  - Beliefs: {len(agent.beliefs)}")
    print(f"  - Meta-beliefs: {len(agent.meta_beliefs)}")
    print(f"  - Coherence violations: {len(violations)}")
except Exception as e:
    print(f"✗ Reflection system failed: {e}")
    sys.exit(1)

# Test 7: Baseline comparison
print("\n[7/7] Testing baseline model...")
try:
    baseline = BaselineMLP(input_dim=10, hidden_dim=32, output_dim=2)
    x = torch.randn(16, 10)
    out = baseline(x)
    assert out.shape == (16, 2), f"Wrong baseline output shape: {out.shape}"
    print("✓ Baseline model working")
except Exception as e:
    print(f"✗ Baseline model failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
print("\nNext steps:")
print("1. Run full experiments: cd experiments && python noisy_labels.py")
print("2. Explore demo notebook: notebooks/demo.md")
print("3. Read the paper: PAPER.md")
print("\nThe system is ready for research and experimentation!")
print("="*80)
