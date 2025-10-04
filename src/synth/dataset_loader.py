"""
Dataset Loader for TabPFN Training
Place this in: TabPFN/scripts/dataset_loader.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List


class TabularDataset(Dataset):
    """Dataset for loading TabPFN-generated datasets"""
    
    def __init__(
        self,
        data_dir: str,
        dataset_files: List[str],
        max_samples: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_files = dataset_files
        self.max_samples = max_samples
        self.datasets = []
        
        for file in dataset_files:
            csv_path = self.data_dir / file
            json_path = self.data_dir / file.replace('.csv', '.json')
            
            if not csv_path.exists() or not json_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Limit samples
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42)
            
            # Separate features and target
            if 'target' in df.columns:
                X = df.drop('target', axis=1)
                y = df['target']
            else:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            # Handle categorical columns
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
            
            # Convert to numpy
            X_np = X.values.astype(np.float32)
            
            # Normalize features
            X_mean = np.mean(X_np, axis=0, keepdims=True)
            X_std = np.std(X_np, axis=0, keepdims=True) + 1e-8
            X_np = (X_np - X_mean) / X_std
            
            # Determine task type
            is_classification = metadata['config'].get('is_classification', False)
            
            if is_classification:
                y_np = y.values.astype(np.int64)
            else:
                # Normalize regression targets
                y_np = y.values.astype(np.float32)
                y_mean = np.mean(y_np)
                y_std = np.std(y_np) + 1e-8
                y_np = (y_np - y_mean) / y_std
            
            self.datasets.append({
                'X': X_np,
                'y': y_np,
                'task_type': 'classification' if is_classification else 'regression',
                'n_samples': len(X_np),
                'n_features': X_np.shape[1],
                'metadata': metadata
            })
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        dataset = self.datasets[idx]
        return {
            'X': dataset['X'],
            'y': dataset['y'],
            'task_type': dataset['task_type'],
            'n_samples': dataset['n_samples'],
            'n_features': dataset['n_features']
        }


def collate_fn(batch):
    """Custom collate function - returns single dataset at a time"""
    # Each batch contains one complete dataset
    return batch[0]


def create_dataloaders(
    data_dir: str,
    train_split: float = 0.8,
    batch_size: int = 1,
    difficulty_filter: Optional[List[str]] = None,
    task_filter: Optional[str] = None,
    max_samples_per_dataset: int = 1000,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Find all dataset files
    csv_files = sorted([f.name for f in data_path.glob("*.csv")])
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Filter datasets
    filtered_files = []
    dataset_stats = {'total': len(csv_files), 'by_difficulty': {}, 'by_task': {}}
    
    for csv_file in csv_files:
        json_file = csv_file.replace('.csv', '.json')
        json_path = data_path / json_file
        
        if not json_path.exists():
            continue
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        config = metadata.get('config', {})
        difficulty = config.get('difficulty', 'unknown')
        is_classification = config.get('is_classification', False)
        task_type = 'classification' if is_classification else 'regression'
        
        # Apply filters
        if difficulty_filter and difficulty not in difficulty_filter:
            continue
        if task_filter and task_type != task_filter:
            continue
        
        filtered_files.append(csv_file)
        dataset_stats['by_difficulty'][difficulty] = dataset_stats['by_difficulty'].get(difficulty, 0) + 1
        dataset_stats['by_task'][task_type] = dataset_stats['by_task'].get(task_type, 0) + 1
    
    if not filtered_files:
        raise ValueError("No datasets match the specified filters")
    
    print(f"Found {len(filtered_files)} datasets from {data_dir}")
    
    # Split into train and validation
    n_train = max(1, int(len(filtered_files) * train_split))
    train_files = filtered_files[:n_train]
    val_files = filtered_files[n_train:]
    
    # Create datasets
    train_dataset = TabularDataset(data_dir, train_files, max_samples_per_dataset)
    val_dataset = TabularDataset(data_dir, val_files, max_samples_per_dataset) if val_files else None
    
    # Calculate statistics
    if len(train_dataset) > 0:
        avg_samples = np.mean([d['n_samples'] for d in train_dataset.datasets])
        avg_features = np.mean([d['n_features'] for d in train_dataset.datasets])
    else:
        avg_samples = 0
        avg_features = 0
    
    print(f"\nDataset Statistics:")
    print(f"  Total datasets: {len(filtered_files)}")
    print(f"  By difficulty: {dataset_stats['by_difficulty']}")
    print(f"  By task: {dataset_stats['by_task']}")
    print(f"  Avg samples per dataset: {avg_samples:.1f}")
    print(f"  Avg features per dataset: {avg_features:.1f}")
    print(f"\nSplit: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} validation datasets")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader