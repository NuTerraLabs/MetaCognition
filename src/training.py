"""
Training utilities for Synthetic Metacognition models

Implements training procedures, loss functions, and optimization strategies
for metacognitive neural networks.

Author: Anonymous
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm


class MetacognitiveLoss(nn.Module):
    """
    Combined loss function for metacognitive training:
    
    L_total = L_task + λ·L_meta
    
    where:
    - L_task: Standard task loss (cross-entropy, MSE, etc.)
    - L_meta: Calibration loss encouraging uncertainty to match correctness
    
    Args:
        lambda_meta (float): Weight for meta-loss component
        task_loss_fn (nn.Module): Task-specific loss function
    """
    
    def __init__(
        self, 
        lambda_meta: float = 0.1,
        task_loss_fn: Optional[nn.Module] = None
    ):
        super(MetacognitiveLoss, self).__init__()
        self.lambda_meta = lambda_meta
        self.task_loss_fn = task_loss_fn if task_loss_fn is not None else nn.CrossEntropyLoss()
    
    def forward(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined metacognitive loss
        
        Args:
            predictions: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            uncertainty: Uncertainty scores of shape (batch_size, 1)
            
        Returns:
            Dictionary containing:
                - 'total': Total loss
                - 'task': Task loss component
                - 'meta': Meta loss component
        """
        # Task loss
        loss_task = self.task_loss_fn(predictions, targets)
        
        # Meta loss: calibration
        # Encourage uncertainty to match whether prediction is correct
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float().unsqueeze(1)
        
        # MSE between uncertainty and correctness
        loss_meta = F.mse_loss(uncertainty, correct)
        
        # Combined loss
        loss_total = loss_task + self.lambda_meta * loss_meta
        
        return {
            'total': loss_total,
            'task': loss_task.item(),
            'meta': loss_meta.item()
        }


class MetacognitiveTrainer:
    """
    Training manager for metacognitive models
    
    Handles:
    - Training loop with proper loss computation
    - Learning rate scheduling
    - Checkpointing
    - Logging
    
    Args:
        model (nn.Module): Metacognitive model to train
        optimizer (torch.optim.Optimizer): Optimizer
        loss_fn (MetacognitiveLoss): Loss function
        device (str): Device to train on ('cuda' or 'cpu')
        scheduler (Optional): Learning rate scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: MetacognitiveLoss,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        
        self.history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_meta_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_meta_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader
        
        for batch_idx, (x_batch, y_batch) in enumerate(iterator):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, uncertainty = self.model(x_batch, return_uncertainty=True)
            
            # Compute loss
            loss_dict = self.loss_fn(predictions, y_batch, uncertainty)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_task_loss += loss_dict['task']
            total_meta_loss += loss_dict['meta']
            num_batches += 1
            
            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'task': f"{loss_dict['task']:.4f}",
                    'meta': f"{loss_dict['meta']:.4f}"
                })
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_meta_loss = total_meta_loss / num_batches
        
        return {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'meta_loss': avg_meta_loss
        }
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_meta_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        iterator = tqdm(val_loader, desc="Validation") if verbose else val_loader
        
        for x_batch, y_batch in iterator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions, uncertainty = self.model(x_batch, return_uncertainty=True)
            
            # Compute loss
            loss_dict = self.loss_fn(predictions, y_batch, uncertainty)
            
            # Accumulate losses
            total_loss += loss_dict['total'].item()
            total_task_loss += loss_dict['task']
            total_meta_loss += loss_dict['meta']
            num_batches += 1
            
            # Compute accuracy
            pred_classes = torch.argmax(predictions, dim=1)
            correct += (pred_classes == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_meta_loss = total_meta_loss / num_batches
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'meta_loss': avg_meta_loss,
            'accuracy': accuracy
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        verbose: bool = True,
        save_best: bool = True,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Complete training procedure
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            verbose: Whether to print progress
            save_best: Whether to save best model checkpoint
            checkpoint_path: Path to save checkpoints
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, verbose)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_task_loss'].append(train_metrics['task_loss'])
            self.history['train_meta_loss'].append(train_metrics['meta_loss'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, verbose=False)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if save_best and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                        if verbose:
                            print(f"  → Saved checkpoint (val_loss: {best_val_loss:.4f})")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"Train Loss: {train_metrics['loss']:.4f}")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Average time per epoch: {total_time/epochs:.2f}s")
        
        return self.history
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int, 
        metrics: Dict[str, float]
    ):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train a baseline model without metacognition
    
    Args:
        model: Baseline model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                val_correct += (pred == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_accuracy'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_accuracy'].append(val_correct / val_total)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Train Acc: {history['train_accuracy'][-1]:.4f}, "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, "
                  f"Val Acc: {history['val_accuracy'][-1]:.4f}")
    
    return history


if __name__ == "__main__":
    print("Testing training utilities...")
    
    # Create dummy data
    from torch.utils.data import TensorDataset
    
    X_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 2, (1000,))
    X_val = torch.randn(200, 10)
    y_val = torch.randint(0, 2, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Test metacognitive training
    print("\nTesting metacognitive training...")
    from models import MetaCognitiveModel
    
    model = MetaCognitiveModel(input_dim=10, hidden_dim=32, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MetacognitiveLoss(lambda_meta=0.1)
    
    trainer = MetacognitiveTrainer(model, optimizer, loss_fn, device='cpu')
    history = trainer.fit(train_loader, val_loader, epochs=5, verbose=True)
    
    print(f"\nFinal validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print("✓ Training utilities working correctly")
