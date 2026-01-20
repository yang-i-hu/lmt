"""
DNN Factor Reweight Model Training Pipeline

Complete pipeline for training a Deep Neural Network on factor data:
1. Load aligned factor and label data
2. Preprocess (handle NaN, scale features)
3. Time-series train/val/test split
4. Train DNN with early stopping
5. Evaluate and document results
6. Save model and artifacts

Requirements:
    pip install torch pandas numpy scikit-learn pyyaml scipy

Usage:
    python train_dnn.py --config config_dnn_full.yaml
    
    # With GPU
    python train_dnn.py --config config_dnn_full.yaml --device cuda
    
    # Override learning rate
    python train_dnn.py --config config_dnn_full.yaml --lr 0.0005
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import our dataloader
from modeling.dataloader import FactorDataLoader

# Import lmt_data_api for evaluation
try:
    from lmt_data_api.api import DataApi
    LMT_API_AVAILABLE = True
except ImportError:
    LMT_API_AVAILABLE = False


# =============================================================================
# Setup Logging
# =============================================================================

def setup_logging(log_file: Path = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("DNNTrainer")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DNN Model Definition
# =============================================================================

class FactorDNN(nn.Module):
    """
    Deep Neural Network for Factor Reweight Prediction.
    
    Architecture:
        Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float = 0.3,
        activation: str = "leaky_relu",
        batch_norm: bool = True,
        output_activation: Optional[str] = None
    ):
        super(FactorDNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Activation functions
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(activations[activation])
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        # Output activation (optional)
        if output_activation:
            if output_activation in activations:
                layers.append(activations[output_activation])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Custom Loss Functions
# =============================================================================

class ICLoss(nn.Module):
    """
    Information Coefficient (IC) based loss.
    
    Minimizes negative Spearman correlation between predictions and targets.
    """
    
    def __init__(self):
        super(ICLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use Pearson correlation as differentiable proxy
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        cov = (pred_centered * target_centered).mean()
        pred_std = pred_centered.std()
        target_std = target_centered.std()
        
        correlation = cov / (pred_std * target_std + 1e-8)
        
        # Minimize negative correlation
        return -correlation


class CombinedLoss(nn.Module):
    """Combined MSE + IC loss."""
    
    def __init__(self, mse_weight: float = 0.5, ic_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ic_loss = ICLoss()
        self.mse_weight = mse_weight
        self.ic_weight = ic_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(pred, target)
        ic = self.ic_loss(pred, target)
        return self.mse_weight * mse + self.ic_weight * ic


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_data(config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load factor and label data using the FactorDataLoader."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)
    
    config_path = Path(".")
    
    data_dir = config.get('data_dir', 'data/')
    if not Path(data_dir).is_absolute():
        data_dir = config_path / data_dir
    
    universe_file = config.get('universe_file')
    if universe_file and not Path(universe_file).is_absolute():
        universe_file = config_path / universe_file
    
    loader = FactorDataLoader(
        data_dir=data_dir,
        factor_key=config.get('factor_key', '0'),
        start_date=config.get('start_date'),
        end_date=config.get('end_date'),
        universe_file=universe_file,
        universe_list=config.get('universe_list'),
        aligned_only=config.get('aligned_only', True),
        drop_na_labels=config.get('drop_na_labels', True),
    )
    
    logger.info(f"DataLoader configuration:")
    logger.info(f"  - Data directory: {data_dir}")
    logger.info(f"  - Factor key: {config.get('factor_key', '0')}")
    logger.info(f"  - Date range: {config.get('start_date')} to {config.get('end_date')}")
    logger.info(f"  - Universe: {'All instruments' if not universe_file else universe_file}")
    
    X, y = loader.load()
    
    logger.info(f"Data loaded:")
    logger.info(f"  - X shape: {X.shape}")
    logger.info(f"  - y shape: {y.shape}")
    logger.info(f"  - Memory usage: {X.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    dates = X.index.get_level_values('date').unique().sort_values()
    instruments = X.index.get_level_values('instrument').unique()
    
    logger.info(f"  - Unique dates: {len(dates)}")
    logger.info(f"  - Date range: {dates.min()} to {dates.max()}")
    logger.info(f"  - Unique instruments: {len(instruments)}")
    
    return X, y


def preprocess_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int, Dict, pd.DataFrame, pd.Series]:
    """
    Preprocess data and create PyTorch DataLoaders.
    
    Returns
    -------
    train_loader, val_loader, test_loader, scaler, input_size, split_info, X_test_orig, y_test_orig
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 60)
    
    train_params = config.get('training', {})
    train_ratio = train_params.get('train_ratio', 0.7)
    val_ratio = train_params.get('val_ratio', 0.15)
    batch_size = train_params.get('batch_size', 4096)
    random_seed = train_params.get('random_seed', 42)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Handle NaN in features
    nan_count_before = X.isna().sum().sum()
    if nan_count_before > 0:
        logger.info(f"NaN values in features before fillna: {nan_count_before}")
        X = X.fillna(X.median())
        nan_count_after = X.isna().sum().sum()
        if nan_count_after > 0:
            X = X.fillna(0)
        logger.info(f"NaN values after fillna: {X.isna().sum().sum()}")
    else:
        logger.info("No NaN values in features")
    
    # Time-series split
    dates = X.index.get_level_values('date').unique().sort_values()
    n_dates = len(dates)
    
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]
    
    train_mask = X.index.get_level_values('date').isin(train_dates)
    val_mask = X.index.get_level_values('date').isin(val_dates)
    test_mask = X.index.get_level_values('date').isin(test_dates)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    split_info = {
        'train': {
            'dates': (int(train_dates.min()), int(train_dates.max())),
            'n_dates': len(train_dates),
            'n_samples': len(X_train)
        },
        'val': {
            'dates': (int(val_dates.min()), int(val_dates.max())),
            'n_dates': len(val_dates),
            'n_samples': len(X_val)
        },
        'test': {
            'dates': (int(test_dates.min()), int(test_dates.max())),
            'n_dates': len(test_dates),
            'n_samples': len(X_test)
        }
    }
    
    logger.info(f"Time-series split:")
    logger.info(f"  - Train: {train_dates.min()} to {train_dates.max()} ({len(train_dates)} days, {len(X_train)} samples)")
    logger.info(f"  - Val:   {val_dates.min()} to {val_dates.max()} ({len(val_dates)} days, {len(X_val)} samples)")
    logger.info(f"  - Test:  {test_dates.min()} to {test_dates.max()} ({len(test_dates)} days, {len(X_test)} samples)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    logger.info(f"Features scaled with StandardScaler")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test.values)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"DataLoaders created with batch_size={batch_size}")
    
    input_size = X_train.shape[1]
    
    # Return X_test and y_test with original indices for API evaluation
    return train_loader, val_loader, test_loader, scaler, input_size, split_info, X_test, y_test


# =============================================================================
# Model Training
# =============================================================================

def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Get loss function from config."""
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'mse')
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        delta = loss_config.get('huber_delta', 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_type == 'ic_loss':
        return ICLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler from config."""
    scheduler_config = config.get('training', {}).get('scheduler', {})
    scheduler_type = scheduler_config.get('type')
    
    if not scheduler_type:
        return None
    
    if scheduler_type == 'cosine':
        epochs = config.get('training', {}).get('epochs', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        patience = scheduler_config.get('patience', 10)
        factor = scheduler_config.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    else:
        return None


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Calculate IC
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ic, _ = stats.spearmanr(all_targets, all_preds)
    
    return total_loss / n_batches, ic


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, List]:
    """
    Train the DNN model.
    
    Returns
    -------
    history : dict
        Training history with losses and metrics
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training DNN Model")
    logger.info("=" * 60)
    
    train_params = config.get('training', {})
    epochs = train_params.get('epochs', 100)
    lr = train_params.get('learning_rate', 0.001)
    weight_decay = train_params.get('weight_decay', 0.0001)
    grad_clip = train_params.get('grad_clip')
    log_interval = config.get('output', {}).get('log_interval', 100)
    
    # Loss function
    criterion = get_loss_function(config)
    logger.info(f"Loss function: {criterion.__class__.__name__}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"Scheduler: {scheduler.__class__.__name__}")
    
    # Early stopping
    es_config = train_params.get('early_stopping', {})
    early_stopping = None
    if es_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=es_config.get('patience', 15),
            min_delta=es_config.get('min_delta', 0.0001)
        )
        logger.info(f"Early stopping: patience={es_config.get('patience', 15)}")
    
    logger.info(f"Training for max {epochs} epochs...")
    logger.info("-" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ic': [],
        'lr': []
    }
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        epoch_start = datetime.now()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        # Validate
        val_loss, val_ic = validate(model, val_loader, criterion, device)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ic'].append(val_ic)
        history['lr'].append(current_lr)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val IC: {val_ic:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                early_stopping.load_best_model(model)
                break
    
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info("-" * 60)
    logger.info(f"Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Load best model if early stopping was used
    if early_stopping and early_stopping.best_model_state:
        logger.info(f"Loaded best model with val_loss={early_stopping.best_score:.6f}")
    
    return history


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    X_test_orig: pd.DataFrame = None,
    y_test_orig: pd.Series = None,
    scaler: StandardScaler = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Evaluate model on all splits, including lmt_data_api metrics."""
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating Model")
    logger.info("=" * 60)
    
    model.eval()
    metrics = {}
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    for split_name, loader in loaders.items():
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        y = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        ic, ic_pvalue = stats.spearmanr(y, y_pred)
        pearson_r, pearson_pvalue = stats.pearsonr(y, y_pred)
        
        metrics[split_name] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'ic': float(ic),
            'ic_pvalue': float(ic_pvalue),
            'pearson_r': float(pearson_r),
            'pearson_pvalue': float(pearson_pvalue),
            'n_samples': int(len(y))
        }
        
        logger.info(f"\n{split_name.upper()} Metrics:")
        logger.info(f"  - RMSE:     {rmse:.6f}")
        logger.info(f"  - MAE:      {mae:.6f}")
        logger.info(f"  - R²:       {r2:.6f}")
        logger.info(f"  - IC:       {ic:.6f} (p={ic_pvalue:.2e})")
        logger.info(f"  - Pearson:  {pearson_r:.6f} (p={pearson_pvalue:.2e})")
        logger.info(f"  - Samples:  {len(y)}")
    
    # -------------------------------------------------------------------------
    # LMT Data API Evaluation (Test Set Only)
    # -------------------------------------------------------------------------
    if LMT_API_AVAILABLE and X_test_orig is not None and y_test_orig is not None and scaler is not None:
        logger.info("\n" + "-" * 40)
        logger.info("LMT Data API Evaluation (Test Set)")
        logger.info("-" * 40)
        
        eval_config = config.get('evaluation', {}) if config else {}
        label_period = eval_config.get('label_period', 10)
        alpha = eval_config.get('alpha', 1)
        
        # Check if we have enough dates for API evaluation
        test_dates = X_test_orig.index.get_level_values('date').unique()
        n_dates = len(test_dates)
        
        if n_dates < label_period:
            logger.warning(
                f"Skipping API evaluation: test set has only {n_dates} dates, "
                f"but label_period={label_period} requires at least {label_period} dates."
            )
            metrics["lmt_api"] = {"error": f"Insufficient test dates: {n_dates} < {label_period}"}
        else:
            try:
                # Generate predictions on test set
                X_test_scaled = scaler.transform(X_test_orig.values)
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                
                model.eval()
                with torch.no_grad():
                    test_preds = model(X_test_tensor).cpu().numpy()
                
                # Build pred_esem Series with proper format
                # Index needs: date (int), code (str)
                test_index = X_test_orig.index
                dates_raw = test_index.get_level_values('date' if 'date' in test_index.names else 'instrument')
                codes_raw = test_index.get_level_values('instrument' if 'instrument' in test_index.names else 'code')
                
                # Handle index level names
                if 'date' in test_index.names and 'instrument' in test_index.names:
                    dates_raw = test_index.get_level_values('date')
                    codes_raw = test_index.get_level_values('instrument')
                
                dates_int = pd.to_numeric(dates_raw, errors='coerce').astype(int)
                codes_str = codes_raw.astype(str)
                
                new_index = pd.MultiIndex.from_arrays(
                    [dates_int, codes_str],
                    names=["date", "code"]
                )
                
                pred_esem = pd.Series(
                    test_preds.astype(np.float64),
                    index=new_index,
                    name="factor"
                )
                
                # Remove duplicates if any
                if pred_esem.index.duplicated().any():
                    logger.debug(f"Removing {pred_esem.index.duplicated().sum()} duplicate index entries")
                    pred_esem = pred_esem[~pred_esem.index.duplicated(keep='first')]
                
                logger.debug(f"pred_esem shape: {pred_esem.shape}")
                logger.debug(f"pred_esem index names: {pred_esem.index.names}")
                logger.debug(f"pred_esem dtype: {pred_esem.dtype}")
                logger.debug(f"pred_esem name: {pred_esem.name}")
                
                api = DataApi()
                
                # Initialize API results
                group_re, group_ir, group_hs, ic_df = None, None, None, None
                
                # Group Return Evaluation
                try:
                    logger.info(f"Calling api.da_eva_group_return (alpha={alpha}, label_period={label_period})...")
                    group_re, group_ir, group_hs = api.da_eva_group_return(
                        pred_esem, "factor", alpha=alpha, label_period=label_period
                    )
                    logger.debug(f"group_re type: {type(group_re)}, shape: {getattr(group_re, 'shape', None)}")
                    logger.debug(f"group_ir type: {type(group_ir)}, shape: {getattr(group_ir, 'shape', None)}")
                    logger.debug(f"group_hs type: {type(group_hs)}, shape: {getattr(group_hs, 'shape', None)}")
                except Exception as e:
                    logger.warning(f"api.da_eva_group_return failed: {e}")
                
                # IC Evaluation
                try:
                    logger.info(f"Calling api.da_eva_ic (label_period={label_period})...")
                    ic_df = api.da_eva_ic(pred_esem, "factor", label_period)
                    logger.debug(f"ic_df type: {type(ic_df)}, shape: {getattr(ic_df, 'shape', None)}")
                except Exception as e:
                    logger.warning(f"api.da_eva_ic failed: {e}")
                
                # Store raw API metrics
                metrics["lmt_api"] = {
                    "ic": str(ic_df),
                    "group_return": str(group_re),
                    "group_ir": str(group_ir),
                    "group_hs": str(group_hs),
                }
                
                # Build unified stats table (as per API guide)
                can_build = True
                if ic_df is None or (hasattr(ic_df, 'empty') and ic_df.empty):
                    logger.warning("ic_df is None or empty")
                    can_build = False
                if group_re is None or (hasattr(group_re, 'empty') and group_re.empty):
                    logger.warning("group_re is None or empty")
                    can_build = False
                if group_ir is None or (hasattr(group_ir, 'empty') and group_ir.empty):
                    logger.warning("group_ir is None or empty")
                    can_build = False
                if group_hs is None or (hasattr(group_hs, 'empty') and group_hs.empty):
                    logger.warning("group_hs is None or empty")
                    can_build = False
                
                if can_build:
                    try:
                        stats_all = pd.concat(
                            objs=[
                                ic_df,
                                group_re[["group0", "group9", "ls"]],
                                group_ir[["group0", "group9", "ls"]],
                                group_hs[["group0", "group9"]]
                            ],
                            axis=1
                        )
                        
                        stats_all.columns = [
                            "IC", "ICIR",
                            "Short", "Long", "LS",
                            "ShortIr", "LongIr", "LSIR",
                            "ShortHS", "LongHS"
                        ]
                        
                        logger.info("\nLMT API Evaluation Results:")
                        logger.info(f"\n{stats_all.to_string()}")
                        
                        metrics["lmt_api"]["stats_all"] = stats_all.to_dict()
                        
                    except Exception as concat_err:
                        logger.warning(f"Failed to build stats_all: {concat_err}")
                else:
                    logger.warning("Cannot build stats_all - some API returns are None or empty")
                    
            except Exception as e:
                logger.warning(f"API evaluation failed: {e}")
                metrics["lmt_api"] = {"error": str(e)}
    elif not LMT_API_AVAILABLE:
        logger.info("\nSkipping API evaluation: lmt_data_api not available")
        metrics["lmt_api"] = {"error": "lmt_data_api not available"}
    
    return metrics


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    model: nn.Module,
    scaler: StandardScaler,
    metrics: Dict[str, Any],
    history: Dict[str, List],
    config: Dict[str, Any],
    feature_names: list,
    split_info: Dict,
    logger: logging.Logger,
    output_dir: Path = None
) -> Dict[str, str]:
    """Save model, scaler, and results."""
    logger.info("=" * 60)
    logger.info("STEP 5: Saving Results")
    logger.info("=" * 60)
    
    output_config = config.get('output', {})
    if output_dir is None:
        output_dir = Path(output_config.get('output_dir', 'outputs_dnn/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = output_dir / output_config.get('model_file', 'dnn_model.pt')
    scaler_file = output_dir / output_config.get('scaler_file', 'scaler.pkl')
    results_file = output_dir / output_config.get('results_file', 'training_results.json')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_sizes': model.hidden_sizes,
            'dropout': config.get('model', {}).get('dropout', 0.3),
            'activation': config.get('model', {}).get('activation', 'leaky_relu'),
            'batch_norm': config.get('model', {}).get('batch_norm', True)
        }
    }, model_file)
    logger.info(f"Model saved to: {model_file}")
    
    # Save scaler
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to: {scaler_file}")
    
    # Save feature names
    feature_file = output_dir / 'feature_names.json'
    with open(feature_file, 'w') as f:
        json.dump(feature_names, f)
    logger.info(f"Feature names saved to: {feature_file}")
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f)
    logger.info(f"Training history saved to: {history_file}")
    
    # Compile results
    model_params = config.get('model', {})
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_dir': str(config.get('data_dir')),
            'factor_key': config.get('factor_key'),
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date'),
            'model': model_params,
            'training': config.get('training', {}),
            'loss': config.get('loss', {})
        },
        'model_info': {
            'n_features': len(feature_names),
            'hidden_sizes': model_params.get('hidden_sizes'),
            'n_parameters': model.count_parameters()
        },
        'split_info': split_info,
        'metrics': metrics,
        'training_summary': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss']),
            'best_val_ic': max(history['val_ic']),
            'epochs_trained': len(history['train_loss'])
        },
        'files': {
            'model': str(model_file),
            'scaler': str(scaler_file),
            'features': str(feature_file),
            'history': str(history_file)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    return {
        'model': str(model_file),
        'scaler': str(scaler_file),
        'features': str(feature_file),
        'history': str(history_file),
        'results': str(results_file)
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    config_path: str, 
    device: str = None,
    override_lr: float = None,
    override_epochs: int = None
):
    """
    Run the full DNN training pipeline.
    """
    # Load config
    config = load_config(config_path)
    
    # Apply overrides
    if override_lr is not None:
        config.setdefault('training', {})['learning_rate'] = override_lr
    if override_epochs is not None:
        config.setdefault('training', {})['epochs'] = override_epochs
    
    # Setup output directory and logging (use config filename for output folder)
    config_dir = Path(config_path).resolve().parent
    config_filename = Path(config_path).stem  # e.g., "config_dnn_full"
    output_dir_name = f"output_{config_filename}"
    output_dir = config_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'training.log'
    logger = setup_logging(log_file)
    
    # Device setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    logger.info("=" * 60)
    logger.info("DNN Factor Reweight Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    try:
        # Step 1: Load data
        X, y = load_data(config, logger)
        feature_names = list(X.columns)
        
        # Step 2: Preprocess
        train_loader, val_loader, test_loader, scaler, input_size, split_info, X_test_orig, y_test_orig = preprocess_data(
            X, y, config, logger
        )
        
        # Create model
        model_config = config.get('model', {})
        model = FactorDNN(
            input_size=input_size,
            hidden_sizes=model_config.get('hidden_sizes', [512, 256, 128, 64]),
            dropout=model_config.get('dropout', 0.3),
            activation=model_config.get('activation', 'leaky_relu'),
            batch_norm=model_config.get('batch_norm', True)
        ).to(device)
        
        logger.info(f"\nModel Architecture:")
        logger.info(f"  - Input size: {input_size}")
        logger.info(f"  - Hidden sizes: {model_config.get('hidden_sizes')}")
        logger.info(f"  - Parameters: {model.count_parameters():,}")
        logger.info(f"  - Dropout: {model_config.get('dropout')}")
        logger.info(f"  - Activation: {model_config.get('activation')}")
        logger.info(f"  - Batch norm: {model_config.get('batch_norm')}")
        
        # Step 3: Train
        history = train_model(model, train_loader, val_loader, config, device, logger)
        
        # Step 4: Evaluate
        metrics = evaluate_model(
            model, train_loader, val_loader, test_loader, device, logger,
            X_test_orig=X_test_orig, y_test_orig=y_test_orig, scaler=scaler, config=config
        )
        
        # Step 5: Save
        saved_files = save_results(
            model, scaler, metrics, history, config, feature_names, split_info, logger,
            output_dir=output_dir
        )
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: DNN {model_config.get('hidden_sizes')}")
        print(f"Parameters: {model.count_parameters():,}")
        print(f"Epochs trained: {len(history['train_loss'])}")
        print(f"\nTrain Metrics:")
        print(f"  RMSE: {metrics['train']['rmse']:.6f}")
        print(f"  R²:   {metrics['train']['r2']:.6f}")
        print(f"  IC:   {metrics['train']['ic']:.6f}")
        print(f"\nVal Metrics:")
        print(f"  RMSE: {metrics['val']['rmse']:.6f}")
        print(f"  R²:   {metrics['val']['r2']:.6f}")
        print(f"  IC:   {metrics['val']['ic']:.6f}")
        print(f"\nTest Metrics:")
        print(f"  RMSE: {metrics['test']['rmse']:.6f}")
        print(f"  R²:   {metrics['test']['r2']:.6f}")
        print(f"  IC:   {metrics['test']['ic']:.6f}")
        print("=" * 60)
        
        return model, scaler, metrics, history
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DNN Factor Reweight Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python train_dnn.py --config config_dnn_full.yaml
    
    # Use GPU
    python train_dnn.py --config config_dnn_full.yaml --device cuda
    
    # Override learning rate
    python train_dnn.py --config config_dnn_full.yaml --lr 0.0005
    
    # Quick test with fewer epochs
    python train_dnn.py --config config_dnn_full.yaml --epochs 10
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_dnn_full.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (auto-detect if not specified)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        device=args.device,
        override_lr=args.lr,
        override_epochs=args.epochs
    )
