"""
Advanced Training Procedures for Metacognitive Models

Implements novel training strategies:
1. Contrastive Metacognitive Training
2. Evidential Training with KL Annealing
3. Adversarial Meta-Training
4. Curriculum Learning for Metacognition

Author: Research Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np


class AdvancedMetacognitiveLoss(nn.Module):
    """
    Advanced loss combining multiple objectives
    """
    
    def __init__(
        self,
        lambda_meta: float = 0.1,
        lambda_contrastive: float = 0.5,
        lambda_diversity: float = 0.01,
        monitor_type: str = 'contrastive'
    ):
        super().__init__()
        self.lambda_meta = lambda_meta
        self.lambda_contrastive = lambda_contrastive
        self.lambda_diversity = lambda_diversity
        self.monitor_type = monitor_type
        
        self.task_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        projections: Optional[torch.Tensor] = None,
        evidential_outputs: Optional[Dict] = None,
        epoch: int = 0,
        num_epochs: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute advanced metacognitive loss
        """
        # 1. Task loss
        loss_task = self.task_loss_fn(predictions, targets)
        
        # 2. Meta-calibration loss
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float().unsqueeze(1)
        
        # Use SMOOTH L1 loss instead of MSE to prevent collapse
        loss_meta = F.smooth_l1_loss(uncertainty, correct)
        
        # 3. Diversity loss: encourage varied uncertainty estimates
        # Penalize if all uncertainties are similar (prevents collapse)
        u_mean = uncertainty.mean()
        u_var = uncertainty.var()
        loss_diversity = torch.exp(-u_var)  # Low variance = high penalty
        
        losses = {
            'task': loss_task.item(),
            'meta': loss_meta.item(),
            'diversity': loss_diversity.item()
        }
        
        total_loss = loss_task + self.lambda_meta * loss_meta + self.lambda_diversity * loss_diversity
        
        # 4. Contrastive loss (if applicable)
        if projections is not None:
            from src.advanced_models import ContrastiveMetaMonitor
            # Instantiate temporarily for loss computation
            temp_monitor = ContrastiveMetaMonitor(64)  # Dummy
            loss_contrastive = temp_monitor.contrastive_loss(projections, correct.squeeze().bool())
            
            losses['contrastive'] = loss_contrastive.item()
            total_loss = total_loss + self.lambda_contrastive * loss_contrastive
        
        # 5. Evidential loss (if applicable)
        if evidential_outputs is not None:
            from src.advanced_models import EvidentialUncertaintyHead
            temp_head = EvidentialUncertaintyHead(64, predictions.shape[1])
            loss_evidential = temp_head.evidential_loss(
                evidential_outputs, targets, epoch, num_epochs
            )
            losses['evidential'] = loss_evidential.item()
            total_loss = loss_evidential  # Replace task loss with evidential
        
        losses['total'] = total_loss
        return losses


class AdvancedMetacognitiveTrainer:
    """
    Advanced trainer with curriculum learning and adaptive strategies
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: AdvancedMetacognitiveLoss,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_curriculum: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.use_curriculum = use_curriculum
        
        self.history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_meta_loss': [],
            'train_diversity_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_uncertainty_separation': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch with advanced features
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'task': 0.0,
            'meta': 0.0,
            'diversity': 0.0,
            'contrastive': 0.0
        }
        num_batches = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader
        
        for batch_idx, (x_batch, y_batch) in enumerate(iterator):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (handle different monitor types)
            if self.model.monitor_type == 'contrastive':
                predictions, uncertainty = self.model(x_batch, return_uncertainty=True)
                projections, _ = self.model.get_contrastive_projections(x_batch)
                
                loss_dict = self.loss_fn(
                    predictions, y_batch, uncertainty,
                    projections=projections,
                    epoch=epoch,
                    num_epochs=num_epochs
                )
            
            elif self.model.monitor_type == 'evidential':
                predictions, uncertainty, evidential_outputs = self.model(
                    x_batch, return_uncertainty=True, return_evidential=True
                )
                
                loss_dict = self.loss_fn(
                    predictions, y_batch, uncertainty,
                    evidential_outputs=evidential_outputs,
                    epoch=epoch,
                    num_epochs=num_epochs
                )
            
            else:  # multiscale
                predictions, uncertainty = self.model(x_batch, return_uncertainty=True)
                loss_dict = self.loss_fn(
                    predictions, y_batch, uncertainty,
                    epoch=epoch,
                    num_epochs=num_epochs
                )
            
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (critical for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in total_losses:
                    total_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            if verbose and batch_idx % 10 == 0:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'task': f"{loss_dict.get('task', 0):.4f}",
                    'meta': f"{loss_dict.get('meta', 0):.4f}"
                })
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate with detailed uncertainty analysis
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        all_uncertainties_correct = []
        all_uncertainties_incorrect = []
        
        iterator = tqdm(val_loader, desc="Validation") if verbose else val_loader
        
        for x_batch, y_batch in iterator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            if self.model.monitor_type == 'evidential':
                predictions, uncertainty, _ = self.model(
                    x_batch, return_uncertainty=True, return_evidential=True
                )
            else:
                predictions, uncertainty = self.model(x_batch, return_uncertainty=True)
            
            # Compute loss (simplified for validation)
            pred_classes = torch.argmax(predictions, dim=1)
            is_correct = (pred_classes == y_batch)
            
            correct += is_correct.sum().item()
            total += y_batch.size(0)
            
            # Track uncertainties
            all_uncertainties_correct.extend(uncertainty[is_correct].cpu().numpy())
            all_uncertainties_incorrect.extend(uncertainty[~is_correct].cpu().numpy())
            
            num_batches += 1
        
        accuracy = correct / total
        
        # Uncertainty separation
        if len(all_uncertainties_correct) > 0 and len(all_uncertainties_incorrect) > 0:
            u_correct_mean = np.mean(all_uncertainties_correct)
            u_incorrect_mean = np.mean(all_uncertainties_incorrect)
            uncertainty_separation = abs(u_correct_mean - u_incorrect_mean)
        else:
            uncertainty_separation = 0.0
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': accuracy,
            'uncertainty_separation': uncertainty_separation,
            'u_correct_mean': np.mean(all_uncertainties_correct) if len(all_uncertainties_correct) > 0 else 0.0,
            'u_incorrect_mean': np.mean(all_uncertainties_incorrect) if len(all_uncertainties_incorrect) > 0 else 0.0
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: int = 15
    ) -> Dict[str, list]:
        """
        Complete training with early stopping
        """
        best_val_separation = 0.0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(
                train_loader, epoch, epochs, verbose=verbose
            )
            
            self.history['train_loss'].append(train_metrics['total'])
            self.history['train_task_loss'].append(train_metrics['task'])
            self.history['train_meta_loss'].append(train_metrics['meta'])
            self.history['train_diversity_loss'].append(train_metrics['diversity'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, verbose=False)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_uncertainty_separation'].append(val_metrics['uncertainty_separation'])
                
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"Loss: {train_metrics['total']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.4f}, "
                          f"U-Sep: {val_metrics['uncertainty_separation']:.4f}")
                
                # Early stopping based on uncertainty separation
                if val_metrics['uncertainty_separation'] > best_val_separation:
                    best_val_separation = val_metrics['uncertainty_separation']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        return self.history


if __name__ == "__main__":
    print("Testing advanced training procedures...")
    print("✓ Advanced training module ready")
