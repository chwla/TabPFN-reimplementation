"""
Training script for Improved TabPFN - FINAL WORKING VERSION
Place this in: TabPFN/scripts/train_improved_tabpfn.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from dataset_loader import create_dataloaders
from improved_tabpfn_model import ImprovedTabPFNModel


class ImprovedTabPFNTrainingConfig:
    """Configuration for improved TabPFN training"""
    
    def __init__(
        self,
        data_dir: str = "./tabpfn_replicated_datasets",
        train_split: float = 0.8,
        max_samples_per_dataset: int = 1000,
        train_test_split: float = 0.7,
        difficulty_filter: Optional[list] = None,
        task_filter: Optional[str] = None,
        
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_features: int = 100,
        max_classes: int = 10,
        
        num_epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        
        save_dir: str = "./checkpoints_improved",
        save_every: int = 5,
        eval_every: int = 1,
        
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers: int = 4,
        mixed_precision: bool = True,
        
        log_every: int = 10,
    ):
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / f"training_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
        
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ImprovedTabPFNTrainer:
    """Trainer for improved TabPFN"""
    
    def __init__(self, config: ImprovedTabPFNTrainingConfig):
        self.config = config
        
        self.log(f"🚀 Initializing Improved TabPFN Trainer")
        self.log(f"Device: {config.device}")
        self.log(f"Mixed Precision: {config.mixed_precision}")
        
        self.log("\n📦 Loading datasets...")
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=config.data_dir,
            train_split=config.train_split,
            batch_size=config.batch_size,
            difficulty_filter=config.difficulty_filter,
            task_filter=config.task_filter,
            max_samples_per_dataset=config.max_samples_per_dataset,
            num_workers=config.num_workers
        )
        
        self.log("\n🏗️  Building improved model...")
        self.model = ImprovedTabPFNModel(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_features=config.max_features,
            max_classes=config.max_classes
        ).to(config.device)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"   Model parameters: {n_params:,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and config.device == 'cuda' else None
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        config.save(self.config.save_dir / "config.json")
        self.log(f"\n✓ Configuration saved")
    
    def log(self, message: str):
        print(message)
        with open(self.config.log_file, 'a') as f:
            f.write(message + '\n')
    
    def split_dataset(self, X, y, split_ratio=0.7):
        """Split dataset into train and test portions"""
        n_samples = len(X)
        n_train = int(n_samples * split_ratio)
        
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        X = batch['X']
        y = batch['y']
        task_type = batch['task_type']
        
        # Convert to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y) if task_type == 'classification' else torch.FloatTensor(y)
        
        X = X.to(self.config.device)
        y = y.to(self.config.device)
        
        # Split into train/test
        X_train, y_train, X_test, y_test = self.split_dataset(X, y, self.config.train_test_split)
        
        # Add batch dimension
        X_train = X_train.unsqueeze(0)
        y_train = y_train.unsqueeze(0)
        X_test = X_test.unsqueeze(0)
        y_test = y_test.unsqueeze(0)
        
        # Forward pass
        with torch.amp.autocast(device_type=self.config.device, enabled=self.config.mixed_precision):
            if task_type == 'classification':
                n_classes = int(torch.max(y).item() + 1)
                logits = self.model(X_train, y_train, X_test, 
                                   task_type='classification', n_classes=n_classes)
                y_test_flat = y_test.reshape(-1)
                loss = nn.CrossEntropyLoss()(logits, y_test_flat)
                
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_test_flat).float().mean()
            else:
                preds = self.model(X_train, y_train, X_test, task_type='regression')
                y_test_flat = y_test.reshape(-1)
                loss = nn.MSELoss()(preds, y_test_flat)
                acc = None
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler and self.config.device == 'cuda':
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        self.global_step += 1
        
        metrics = {'loss': loss.item()}
        if acc is not None:
            metrics['accuracy'] = acc.item()
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_acc = 0
        n_batches = 0
        n_classification = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            try:
                X = batch['X']
                y = batch['y']
                task_type = batch['task_type']
                
                if not isinstance(X, torch.Tensor):
                    X = torch.FloatTensor(X)
                if not isinstance(y, torch.Tensor):
                    y = torch.LongTensor(y) if task_type == 'classification' else torch.FloatTensor(y)
                
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                
                X_train, y_train, X_test, y_test = self.split_dataset(X, y, self.config.train_test_split)
                
                X_train = X_train.unsqueeze(0)
                y_train = y_train.unsqueeze(0)
                X_test = X_test.unsqueeze(0)
                y_test = y_test.unsqueeze(0)
                
                if task_type == 'classification':
                    n_classes = int(torch.max(y).item() + 1)
                    logits = self.model(X_train, y_train, X_test,
                                       task_type='classification', n_classes=n_classes)
                    y_test_flat = y_test.reshape(-1)
                    loss = nn.CrossEntropyLoss()(logits, y_test_flat)
                    
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == y_test_flat).float().mean()
                    total_acc += acc.item()
                    n_classification += 1
                else:
                    preds = self.model(X_train, y_train, X_test, task_type='regression')
                    y_test_flat = y_test.reshape(-1)
                    loss = nn.MSELoss()(preds, y_test_flat)
                
                total_loss += loss.item()
                n_batches += 1
                
            except Exception as e:
                self.log(f"⚠️  Eval error: {e}")
                continue
        
        metrics = {'val_loss': total_loss / max(n_batches, 1)}
        if n_classification > 0:
            metrics['val_accuracy'] = total_acc / n_classification
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        path = self.config.save_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, path)
        self.log(f"💾 Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.config.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log(f"⭐ New best model saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        self.log("\n" + "="*60)
        self.log("🎯 Starting Training")
        self.log("="*60)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.log(f"\n📍 Epoch {epoch+1}/{self.config.num_epochs}")
            
            epoch_losses = []
            epoch_accs = []
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    metrics = self.train_step(batch)
                    epoch_losses.append(metrics['loss'])
                    if 'accuracy' in metrics:
                        epoch_accs.append(metrics['accuracy'])
                    
                    postfix = {'loss': f"{metrics['loss']:.4f}"}
                    if 'accuracy' in metrics:
                        postfix['acc'] = f"{metrics['accuracy']:.3f}"
                    pbar.set_postfix(postfix)
                    
                    if batch_idx % self.config.log_every == 0:
                        log_msg = f"  Step {self.global_step}: loss={metrics['loss']:.4f}"
                        if 'accuracy' in metrics:
                            log_msg += f", acc={metrics['accuracy']:.3f}"
                        self.log(log_msg)
                
                except Exception as e:
                    self.log(f"⚠️  Training error on batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            self.log(f"\n  📊 Epoch {epoch+1} Summary:")
            self.log(f"     Average Train Loss: {avg_train_loss:.4f}")
            if epoch_accs:
                avg_train_acc = np.mean(epoch_accs)
                self.log(f"     Average Train Accuracy: {avg_train_acc:.3f}")
            
            if (epoch + 1) % self.config.eval_every == 0:
                self.log(f"\n  🔍 Running validation...")
                val_metrics = self.evaluate()
                val_loss = val_metrics['val_loss']
                self.log(f"     Validation Loss: {val_loss:.4f}")
                if 'val_accuracy' in val_metrics:
                    self.log(f"     Validation Accuracy: {val_metrics['val_accuracy']:.3f}")
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.save_checkpoint({'train_loss': avg_train_loss, **val_metrics}, is_best)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log(f"     Learning Rate: {current_lr:.6f}")
        
        self.log("\n" + "="*60)
        self.log("✅ Training Complete!")
        self.log(f"Best Validation Loss: {self.best_val_loss:.4f}")
        self.log("="*60)


def main():
    """Main training function"""
    
    config = ImprovedTabPFNTrainingConfig(
        data_dir="./tabpfn_replicated_datasets",
        train_split=0.5,
        max_samples_per_dataset=200,
        train_test_split=0.7,
        
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_features=50,
        
        num_epochs=3,
        batch_size=1,
        learning_rate=1e-4,
        
        save_dir="./checkpoints_improved",
        save_every=1,
        eval_every=1,
        
        device="cpu",
        num_workers=0,
        mixed_precision=False,
    )
    
    if not os.path.exists(config.data_dir):
        print(f"❌ Data directory not found: {config.data_dir}")
        print(f"   Please run dataset.py first to generate training data")
        return
    
    trainer = ImprovedTabPFNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()