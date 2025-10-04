"""
Improved TabPFN-like Architecture - FINAL WORKING VERSION
Based on "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"

Place this in: TabPFN/scripts/improved_tabpfn_model.py
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class TabPFNEncoder(nn.Module):
    """Encodes tabular data into transformer-ready format"""
    
    def __init__(
        self,
        d_model: int = 512,
        max_features: int = 100,
        max_classes: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_features = max_features
        self.max_classes = max_classes
        
        # Feature value embedding
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Feature index embedding
        self.feature_idx_embedding = nn.Embedding(max_features, d_model)
        
        # Label embedding (for classification only)
        self.label_embedding = nn.Embedding(max_classes + 1, d_model)
        
        # Train/test position embedding
        self.is_train_embedding = nn.Embedding(2, d_model)
        
        # Sample position embedding
        self.pos_embedding = nn.Embedding(10000, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,  # (batch, seq_len, n_features)
        labels: Optional[torch.Tensor] = None,  # (batch, seq_len)
        is_train_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
    ) -> torch.Tensor:
        """Encode tabular data with all necessary embeddings"""
        
        if len(features.shape) != 3:
            raise ValueError(f"Expected features shape (batch, seq_len, n_features), got {features.shape}")
            
        batch_size, seq_len, n_features = features.shape
        device = features.device
        
        # Flatten to process each feature separately
        features_flat = features.reshape(batch_size, seq_len * n_features, 1)
        
        # Embed feature values
        feature_emb = self.feature_embedding(features_flat)
        
        # Add feature index embeddings
        feature_indices = torch.arange(n_features, device=device).repeat(seq_len)
        feature_idx_emb = self.feature_idx_embedding(feature_indices)
        feature_emb = feature_emb + feature_idx_emb.unsqueeze(0)
        
        # Add sample position embeddings
        sample_positions = torch.arange(seq_len, device=device).repeat_interleave(n_features)
        pos_emb = self.pos_embedding(sample_positions)
        feature_emb = feature_emb + pos_emb.unsqueeze(0)
        
        # Add train/test embeddings
        if is_train_mask is not None:
            is_train_flat = is_train_mask.unsqueeze(-1).repeat(1, 1, n_features).reshape(batch_size, -1)
            is_train_emb = self.is_train_embedding(is_train_flat.long())
            feature_emb = feature_emb + is_train_emb
        
        # Add label embeddings (only for classification with integer labels)
        if labels is not None and labels.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            labels_flat = labels.unsqueeze(-1).repeat(1, 1, n_features).reshape(batch_size, -1)
            if is_train_mask is not None:
                labels_flat = labels_flat * is_train_flat + (self.max_classes) * (1 - is_train_flat)
            label_emb = self.label_embedding(labels_flat.long())
            feature_emb = feature_emb + label_emb
        
        return self.dropout(feature_emb)


class ImprovedTabPFNModel(nn.Module):
    """TabPFN-like model with in-context learning"""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_features: int = 100,
        max_classes: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        
        print("Initializing Improved TabPFN Model")
        print(f"   d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        
        self.d_model = d_model
        self.max_features = max_features
        
        self.encoder = TabPFNEncoder(
            d_model=d_model,
            max_features=max_features,
            max_classes=max_classes,
            dropout=dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        X_train: torch.Tensor,  # (batch, n_train, n_features)
        y_train: torch.Tensor,  # (batch, n_train)
        X_test: torch.Tensor,   # (batch, n_test, n_features)
        task_type: str = 'classification',
        n_classes: int = 2
    ) -> torch.Tensor:
        """Forward pass with in-context learning"""
        
        # Ensure all tensors have correct shapes
        if len(X_train.shape) != 3:
            raise ValueError(f"X_train must have shape (batch, n_train, n_features), got {X_train.shape}")
        if len(X_test.shape) != 3:
            raise ValueError(f"X_test must have shape (batch, n_test, n_features), got {X_test.shape}")
        
        batch_size = X_train.shape[0]
        n_train = X_train.shape[1]
        n_test = X_test.shape[1]
        n_features = X_train.shape[2]
        device = X_train.device
        
        # Ensure y_train has shape (batch, n_train)
        while len(y_train.shape) > 2:
            y_train = y_train.squeeze(-1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(0)
        
        # Concatenate train and test data
        X_all = torch.cat([X_train, X_test], dim=1)
        
        # Create train/test mask
        is_train_mask = torch.cat([
            torch.ones(batch_size, n_train, device=device),
            torch.zeros(batch_size, n_test, device=device)
        ], dim=1)
        
        # Create y_all with proper shape
        test_labels = torch.zeros(batch_size, n_test, device=device, dtype=y_train.dtype)
        y_all = torch.cat([y_train, test_labels], dim=1)
        
        # Encode inputs
        embeddings = self.encoder(X_all, y_all, is_train_mask)
        
        # Transformer
        hidden = self.transformer(embeddings)
        hidden = self.output_norm(hidden)
        
        # Extract test sample representations
        hidden_reshaped = hidden.reshape(batch_size, n_train + n_test, n_features, self.d_model)
        sample_representations = hidden_reshaped.mean(dim=2)
        
        # Get only test representations
        test_representations = sample_representations[:, n_train:, :]
        
        # Output head
        if task_type == 'classification':
            if not hasattr(self, f'clf_head_{n_classes}'):
                head = nn.Linear(self.d_model, n_classes).to(device)
                # Better initialization for classification
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
                setattr(self, f'clf_head_{n_classes}', head)
            clf_head = getattr(self, f'clf_head_{n_classes}')
            logits = clf_head(test_representations)
            return logits.reshape(-1, n_classes)
        else:
            if not hasattr(self, 'reg_head'):
                head = nn.Linear(self.d_model, 1).to(device)
                # Better initialization for regression
                nn.init.xavier_uniform_(head.weight, gain=0.01)
                nn.init.zeros_(head.bias)
                self.reg_head = head
            preds = self.reg_head(test_representations)
            return preds.reshape(-1)


def test_model():
    """Test the improved model"""
    print("\n" + "="*60)
    print("Testing Improved TabPFN Model")
    print("="*60)
    
    model = ImprovedTabPFNModel(
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_features=20
    )
    
    batch_size = 2
    n_train = 100
    n_test = 50
    n_features = 10
    
    X_train = torch.randn(batch_size, n_train, n_features)
    y_train = torch.randint(0, 3, (batch_size, n_train))
    X_test = torch.randn(batch_size, n_test, n_features)
    y_test = torch.randint(0, 3, (batch_size, n_test))
    
    print(f"\nTest Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train samples: {n_train}")
    print(f"   Test samples: {n_test}")
    print(f"   Features: {n_features}")
    
    print("\nRunning forward pass (classification)...")
    logits = model(X_train, y_train, X_test, task_type='classification', n_classes=3)
    print(f"   Output shape: {logits.shape} (expected: {(batch_size * n_test, 3)})")
    
    y_test_flat = y_test.reshape(-1)
    loss = nn.CrossEntropyLoss()(logits, y_test_flat)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\nRunning forward pass (regression)...")
    y_train_reg = torch.randn(batch_size, n_train)
    y_test_reg = torch.randn(batch_size, n_test)
    preds = model(X_train, y_train_reg, X_test, task_type='regression')
    print(f"   Output shape: {preds.shape} (expected: {batch_size * n_test})")
    
    y_test_flat_reg = y_test_reg.reshape(-1)
    loss_reg = nn.MSELoss()(preds, y_test_flat_reg)
    print(f"   Loss: {loss_reg.item():.4f}")
    
    print("\nModel test passed!")
    print("="*60)


if __name__ == "__main__":
    test_model()