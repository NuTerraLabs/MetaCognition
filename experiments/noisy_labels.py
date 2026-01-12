"""
Experiment 1: Noisy Label Classification

Tests metacognitive model's ability to handle label noise and maintain calibration.

Task: Binary classification with synthetic data containing 20% label noise
Evaluation: Accuracy, ECE, uncertainty-error correlation

Author: Anonymous
Date: January 2026
"""

import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import MetaCognitiveModel, BaselineMLP
from src.training import MetacognitiveTrainer, MetacognitiveLoss, train_baseline_model
from src.evaluation import MetacognitiveEvaluator, compare_models, print_comparison_table


def create_noisy_dataset(
    n_samples: int = 2000,
    n_features: int = 2,
    label_noise: float = 0.2,
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Create synthetic dataset with label noise
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        label_noise: Fraction of labels to flip
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=label_noise,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, val_loader, test_loader


def run_experiment(
    label_noise: float = 0.2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_results: bool = True
):
    """
    Run noisy label experiment
    
    Args:
        label_noise: Fraction of labels to flip
        device: Device to run on
        save_results: Whether to save results
    """
    print("="*80)
    print(f"EXPERIMENT 1: Noisy Label Classification (noise={label_noise})")
    print("="*80)
    
    # Create dataset
    print("\n1. Creating dataset...")
    train_loader, val_loader, test_loader = create_noisy_dataset(
        n_samples=2000,
        n_features=2,
        label_noise=label_noise
    )
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Train baseline model
    print("\n2. Training baseline MLP...")
    baseline_model = BaselineMLP(
        input_dim=2,
        hidden_dim=128,
        output_dim=2,
        dropout_rate=0.1
    )
    baseline_history = train_baseline_model(
        baseline_model,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        device=device,
        verbose=False
    )
    print(f"   Final validation accuracy: {baseline_history['val_accuracy'][-1]:.4f}")
    
    # Train metacognitive model
    print("\n3. Training metacognitive model...")
    meta_model = MetaCognitiveModel(
        input_dim=2,
        hidden_dim=128,
        output_dim=2,
        monitor_dim=64,
        dropout_rate=0.1,
        control_type='gate'
    )
    
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_meta=0.1)
    trainer = MetacognitiveTrainer(meta_model, optimizer, loss_fn, device=device)
    
    meta_history = trainer.fit(
        train_loader,
        val_loader,
        epochs=50,
        verbose=False,
        save_best=True,
        checkpoint_path='../results/noisy_labels_checkpoint.pt'
    )
    print(f"   Final validation accuracy: {meta_history['val_accuracy'][-1]:.4f}")
    
    # Evaluate both models
    print("\n4. Evaluating on test set...")
    
    models_dict = {
        'Baseline MLP': baseline_model,
        'Metacognitive': meta_model
    }
    
    results = compare_models(models_dict, test_loader, device=device)
    print_comparison_table(results)
    
    # Detailed analysis
    print("\n5. Detailed Analysis")
    print("-"*80)
    
    evaluator = MetacognitiveEvaluator(meta_model, device=device)
    
    # Plot calibration
    print("   Generating calibration plot...")
    evaluator.plot_calibration(
        test_loader,
        save_path='../results/noisy_labels_calibration.png' if save_results else None
    )
    
    # Plot selective accuracy
    print("   Generating selective accuracy plot...")
    evaluator.plot_selective_accuracy(
        test_loader,
        save_path='../results/noisy_labels_selective.png' if save_results else None
    )
    
    # Plot training curves
    print("   Generating training curves...")
    plot_training_curves(
        baseline_history,
        meta_history,
        save_path='../results/noisy_labels_training.png' if save_results else None
    )
    
    # Confidence distribution analysis
    print("   Analyzing confidence distributions...")
    analyze_confidence_distribution(
        meta_model,
        test_loader,
        device=device,
        save_path='../results/noisy_labels_confidence.png' if save_results else None
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return results


def plot_training_curves(
    baseline_history,
    meta_history,
    save_path=None
):
    """Plot training curves for comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    ax.plot(baseline_history['train_loss'], label='Baseline Train', alpha=0.7)
    ax.plot(baseline_history['val_loss'], label='Baseline Val', alpha=0.7)
    ax.plot(meta_history['train_loss'], label='Metacognitive Train', alpha=0.7, linestyle='--')
    ax.plot(meta_history['val_loss'], label='Metacognitive Val', alpha=0.7, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Accuracy curves
    ax = axes[1]
    ax.plot(baseline_history['train_accuracy'], label='Baseline Train', alpha=0.7)
    ax.plot(baseline_history['val_accuracy'], label='Baseline Val', alpha=0.7)
    ax.plot(meta_history['val_accuracy'], label='Metacognitive Val', alpha=0.7, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_confidence_distribution(
    model,
    data_loader,
    device='cpu',
    save_path=None
):
    """Analyze confidence distribution for correct vs incorrect predictions"""
    model.eval()
    
    confidences_correct = []
    confidences_incorrect = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            logits, uncertainty = model(x_batch, return_uncertainty=True)
            preds = torch.argmax(logits, dim=1)
            
            correct = preds == y_batch.to(device)
            
            confidences_correct.extend(uncertainty[correct].cpu().numpy())
            confidences_incorrect.extend(uncertainty[~correct].cpu().numpy())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(confidences_correct, bins=20, alpha=0.6, label='Correct', density=True, color='green')
    ax.hist(confidences_incorrect, bins=20, alpha=0.6, label='Incorrect', density=True, color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution by Correctness')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Box plot
    ax = axes[1]
    ax.boxplot([confidences_correct, confidences_incorrect], labels=['Correct', 'Incorrect'])
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Statistics')
    ax.grid(alpha=0.3)
    
    # Print statistics
    print(f"      Correct predictions - Mean: {np.mean(confidences_correct):.3f}, Std: {np.std(confidences_correct):.3f}")
    print(f"      Incorrect predictions - Mean: {np.mean(confidences_incorrect):.3f}, Std: {np.std(confidences_incorrect):.3f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run noisy label experiment')
    parser.add_argument('--noise', type=float, default=0.2, help='Label noise level')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    results = run_experiment(
        label_noise=args.noise,
        device=args.device,
        save_results=not args.no_save
    )
