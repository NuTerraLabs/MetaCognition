"""
Evaluation metrics and analysis tools for Synthetic Metacognition

Implements:
- Calibration metrics (ECE, Brier score)
- Uncertainty-error correlation
- Selective accuracy
- Visualization utilities

Author: Anonymous
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def expected_calibration_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    confidences: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE)
    
    ECE measures the difference between model confidence and actual accuracy:
    ECE = Σ (|Bm|/N) · |acc(Bm) - conf(Bm)|
    
    Args:
        predictions: Predicted class labels
        targets: True labels
        confidences: Model confidence scores
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better)
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def brier_score(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute Brier Score for probabilistic predictions
    
    BS = (1/N) Σ ||ŷ - y||²
    
    Args:
        predictions: Probability predictions (batch_size, num_classes)
        targets: True labels (batch_size,)
        
    Returns:
        Brier score (lower is better)
    """
    # Convert targets to one-hot
    num_classes = predictions.shape[1]
    targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Compute squared error
    bs = torch.mean((predictions - targets_onehot) ** 2).item()
    return bs


def uncertainty_error_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor
) -> float:
    """
    Compute correlation between uncertainty and prediction errors
    
    A good uncertainty estimator should assign high uncertainty to incorrect predictions.
    
    Args:
        predictions: Predicted class labels
        targets: True labels
        uncertainties: Uncertainty scores (higher = more confident)
        
    Returns:
        Pearson correlation coefficient (positive means uncertainty correlates with errors)
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.cpu().numpy().flatten()
    
    # Compute errors (1 if incorrect, 0 if correct)
    errors = (predictions != targets).astype(float)
    
    # Invert uncertainties so higher values mean less confident
    inverted_uncertainties = 1.0 - uncertainties
    
    # Compute correlation
    correlation = np.corrcoef(inverted_uncertainties, errors)[0, 1]
    
    return float(correlation)


def selective_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    confidences: torch.Tensor,
    thresholds: Optional[List[float]] = None
) -> Dict[float, Tuple[float, float]]:
    """
    Compute accuracy when only retaining predictions above confidence threshold
    
    Args:
        predictions: Predicted class labels
        targets: True labels
        confidences: Confidence scores
        thresholds: List of confidence thresholds to evaluate
        
    Returns:
        Dictionary mapping threshold -> (accuracy, coverage)
        where coverage is the fraction of samples retained
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy().flatten()
    
    results = {}
    for threshold in thresholds:
        # Select samples above threshold
        mask = confidences >= threshold
        
        if mask.sum() == 0:
            results[threshold] = (0.0, 0.0)
        else:
            selected_preds = predictions[mask]
            selected_targets = targets[mask]
            accuracy = (selected_preds == selected_targets).mean()
            coverage = mask.mean()
            results[threshold] = (float(accuracy), float(coverage))
    
    return results


class MetacognitiveEvaluator:
    """
    Comprehensive evaluation suite for metacognitive models
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        compute_detailed: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on a dataset
        
        Args:
            data_loader: DataLoader for evaluation data
            compute_detailed: Whether to compute detailed metrics (slower)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_probabilities = []
        all_logits = []
        
        # Collect predictions
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            
            logits, uncertainty = self.model(x_batch, return_uncertainty=True)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_logits.append(logits.cpu())
            all_predictions.append(preds.cpu())
            all_targets.append(y_batch)
            all_confidences.append(uncertainty.cpu())
            all_probabilities.append(probs.cpu())
        
        # Concatenate
        all_logits = torch.cat(all_logits)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_confidences = torch.cat(all_confidences).squeeze()
        all_probabilities = torch.cat(all_probabilities)
        
        # Basic metrics
        accuracy = (all_predictions == all_targets).float().mean().item()
        
        # Calibration metrics
        ece = expected_calibration_error(
            all_predictions, all_targets, all_confidences
        )
        bs = brier_score(all_probabilities, all_targets)
        
        # Uncertainty-error correlation
        corr = uncertainty_error_correlation(
            all_predictions, all_targets, all_confidences
        )
        
        metrics = {
            'accuracy': accuracy,
            'ece': ece,
            'brier_score': bs,
            'uncertainty_error_corr': corr,
            'mean_confidence': all_confidences.mean().item(),
            'std_confidence': all_confidences.std().item()
        }
        
        # Detailed metrics
        if compute_detailed:
            # Selective accuracy
            selective_acc = selective_accuracy(
                all_predictions, all_targets, all_confidences
            )
            metrics['selective_accuracy'] = selective_acc
            
            # Separate confidence for correct/incorrect
            correct_mask = (all_predictions == all_targets)
            if correct_mask.sum() > 0:
                metrics['confidence_correct'] = all_confidences[correct_mask].mean().item()
            if (~correct_mask).sum() > 0:
                metrics['confidence_incorrect'] = all_confidences[~correct_mask].mean().item()
            
            # AUC for uncertainty as error detector
            errors = (all_predictions != all_targets).cpu().numpy()
            inverted_conf = (1.0 - all_confidences).cpu().numpy()
            if len(np.unique(errors)) > 1:
                metrics['uncertainty_auc'] = roc_auc_score(errors, inverted_conf)
        
        return metrics
    
    def plot_calibration(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_bins: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plot calibration diagram
        
        Args:
            data_loader: DataLoader for evaluation
            n_bins: Number of bins
            save_path: Path to save figure (optional)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                logits, uncertainty = self.model(x_batch, return_uncertainty=True)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_predictions.append(preds.cpu())
                all_targets.append(y_batch)
                all_confidences.append(uncertainty.cpu())
        
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        confidences = torch.cat(all_confidences).squeeze().numpy()
        
        # Compute bin statistics
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append((predictions[in_bin] == targets[in_bin]).mean())
                bin_counts.append(in_bin.sum())
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.bar(bin_confidences, bin_accuracies, width=1.0/n_bins, 
                alpha=0.6, edgecolor='black', label='Model')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Confidence histogram
        correct = predictions == targets
        ax2.hist(confidences[correct], bins=20, alpha=0.5, label='Correct', density=True)
        ax2.hist(confidences[~correct], bins=20, alpha=0.5, label='Incorrect', density=True)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_selective_accuracy(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_path: Optional[str] = None
    ):
        """
        Plot selective accuracy vs coverage curve
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                logits, uncertainty = self.model(x_batch, return_uncertainty=True)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.append(preds.cpu())
                all_targets.append(y_batch)
                all_confidences.append(uncertainty.cpu())
        
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        confidences = torch.cat(all_confidences).squeeze()
        
        # Compute selective accuracy
        thresholds = np.linspace(0, 1, 50)
        results = selective_accuracy(predictions, targets, confidences, thresholds.tolist())
        
        accuracies = [results[t][0] for t in thresholds]
        coverages = [results[t][1] for t in thresholds]
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(coverages, accuracies, 'b-', linewidth=2)
        ax.axhline(y=(predictions == targets).float().mean().item(), 
                   color='r', linestyle='--', label='Full dataset accuracy')
        ax.set_xlabel('Coverage (fraction of predictions retained)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Selective Accuracy vs Coverage', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def compare_models(
    models_dict: Dict[str, torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same dataset
    
    Args:
        models_dict: Dictionary of {model_name: model}
        data_loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Dictionary of {model_name: metrics}
    """
    results = {}
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        evaluator = MetacognitiveEvaluator(model, device)
        metrics = evaluator.evaluate(data_loader, compute_detailed=True)
        results[name] = metrics
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print formatted comparison table of model results
    """
    import pandas as pd
    
    # Extract main metrics
    metrics_to_show = ['accuracy', 'ece', 'brier_score', 'uncertainty_error_corr']
    
    data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for metric in metrics_to_show:
            if metric in metrics:
                row[metric] = f"{metrics[metric]:.4f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    print("Testing evaluation utilities...")
    
    # Create dummy predictions
    predictions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 0, 0, 1, 1, 1])
    confidences = torch.tensor([0.9, 0.85, 0.95, 0.6, 0.8, 0.9, 0.55, 0.75])
    probs = torch.tensor([
        [0.9, 0.1], [0.15, 0.85], [0.95, 0.05], [0.4, 0.6],
        [0.8, 0.2], [0.1, 0.9], [0.45, 0.55], [0.25, 0.75]
    ])
    
    # Test ECE
    ece = expected_calibration_error(predictions, targets, confidences)
    print(f"ECE: {ece:.4f}")
    
    # Test Brier score
    bs = brier_score(probs, targets)
    print(f"Brier Score: {bs:.4f}")
    
    # Test correlation
    corr = uncertainty_error_correlation(predictions, targets, confidences)
    print(f"Uncertainty-Error Correlation: {corr:.4f}")
    
    # Test selective accuracy
    sel_acc = selective_accuracy(predictions, targets, confidences)
    print("\nSelective Accuracy:")
    for threshold, (acc, cov) in sel_acc.items():
        print(f"  Threshold {threshold:.1f}: Acc={acc:.3f}, Coverage={cov:.3f}")
    
    print("\n✓ Evaluation utilities working correctly")
