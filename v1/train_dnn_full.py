"""
DNN Factor Reweight Model Training - Full IS/OOS Pipeline

This script trains a DNN model on FULL In-Sample (IS) data and evaluates on 
FULL Out-of-Sample (OOS) data, following the multi-snapshot workflow.

Key Differences from train_dnn.py:
1. Trains on ALL IS data (no train/val/test split within IS)
2. Uses separate OOS parquet files for evaluation
3. Evaluates ALL 3 keys (0, 1, 2) and computes ensemble
4. Runs LMT API evaluation on full OOS period

Workflow:
1. Load IS data (train_X.parquet from data/combined/)
2. Train model on 100% IS with cross-validation for early stopping
3. Load OOS data (test_oos_X.parquet from data/combined/)
4. Generate predictions and ensemble
5. Evaluate with LMT API on full OOS period

Usage:
    python train_dnn_full.py --config config_dnn_full.yaml
    python train_dnn_full.py --config config_dnn_full.yaml --snapshot 20201231
    python train_dnn_full.py --config config_dnn_full.yaml --device cuda
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
from sklearn.model_selection import TimeSeriesSplit
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# LMT API
try:
    from lmt_data_api.api import DataApi
    LMT_API_AVAILABLE = True
except ImportError:
    LMT_API_AVAILABLE = False
    print("Warning: lmt_data_api not available. API evaluation will be skipped.")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "data_dir": "data/combined",
    "snapshot": "20201231",  # Which snapshot to use
    "model": {
        "hidden_sizes": [512, 256, 128, 64],
        "dropout": 0.3,
        "activation": "leaky_relu",
        "batch_norm": True
    },
    "training": {
        "epochs": 100,
        "batch_size": 4096,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "early_stopping_patience": 15,
        "val_ratio": 0.15,  # Use last 15% of IS for validation
        "random_seed": 42
    },
    "evaluation": {
        "label_period": 10,
        "alpha": 1
    },
    "output": {
        "output_dir": "outputs_dnn_full"
    }
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger("DNNFullTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
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
# DNN Model
# =============================================================================

class FactorDNN(nn.Module):
    """Deep Neural Network for Factor Reweight Prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float = 0.3,
        activation: str = "leaky_relu",
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        act_fn = activations.get(activation, nn.LeakyReLU(0.1))
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def load_best_model(self, model: nn.Module):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# Data Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and merge with defaults."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}
    
    # Merge with defaults
    config = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


def load_is_data(data_dir: Path, key: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load In-Sample training data."""
    file_path = data_dir / f"train_{key}.parquet"
    
    if not file_path.exists():
        # Try alternative path
        file_path = data_dir / f"factors_{key}_is.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"IS data not found: {file_path}")
    
    logger.info(f"Loading IS data from: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Separate features and labels
    label_cols = ['labelValue', 'endDate', 'source_snapshot']
    feature_cols = [c for c in df.columns if c not in label_cols]
    
    X = df[feature_cols]
    y = df['labelValue']
    
    # Drop NaN labels
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"  Shape: {X.shape}, Labels: {len(y)}")
    
    dates = X.index.get_level_values('date')
    logger.info(f"  Date range: {dates.min()} to {dates.max()} ({dates.nunique()} days)")
    
    return X, y


def load_oos_data(data_dir: Path, key: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Out-of-Sample test data."""
    file_path = data_dir / f"test_oos_{key}.parquet"
    
    if not file_path.exists():
        # Try alternative path
        file_path = data_dir / f"factors_{key}_oos.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"OOS data not found: {file_path}")
    
    logger.info(f"Loading OOS data from: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Separate features and labels
    label_cols = ['labelValue', 'endDate', 'source_snapshot']
    feature_cols = [c for c in df.columns if c not in label_cols]
    
    X = df[feature_cols]
    y = df['labelValue']
    
    # Drop NaN labels
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"  Shape: {X.shape}, Labels: {len(y)}")
    
    dates = X.index.get_level_values('date')
    logger.info(f"  Date range: {dates.min()} to {dates.max()} ({dates.nunique()} days)")
    
    return X, y


# =============================================================================
# Training
# =============================================================================

def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Prepare data for training with time-series val split."""
    train_params = config.get('training', {})
    val_ratio = train_params.get('val_ratio', 0.15)
    batch_size = train_params.get('batch_size', 4096)
    random_seed = train_params.get('random_seed', 42)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Handle NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with median")
        X = X.fillna(X.median()).fillna(0)
    
    # Time-series split: last val_ratio for validation
    dates = X.index.get_level_values('date').unique().sort_values()
    n_dates = len(dates)
    val_start_idx = int(n_dates * (1 - val_ratio))
    
    train_dates = dates[:val_start_idx]
    val_dates = dates[val_start_idx:]
    
    train_mask = X.index.get_level_values('date').isin(train_dates)
    val_mask = X.index.get_level_values('date').isin(val_dates)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    logger.info(f"Train: {len(X_train)} samples ({len(train_dates)} days)")
    logger.info(f"Val:   {len(X_val)} samples ({len(val_dates)} days)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, List]:
    """Train the DNN model."""
    train_params = config.get('training', {})
    epochs = train_params.get('epochs', 100)
    lr = train_params.get('learning_rate', 0.001)
    weight_decay = train_params.get('weight_decay', 0.0001)
    patience = train_params.get('early_stopping_patience', 15)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': [], 'val_ic': []}
    
    logger.info(f"Training for max {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_ic, _ = stats.spearmanr(all_targets, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ic'].append(val_ic)
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val IC: {val_ic:.4f}"
            )
        
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            early_stopping.load_best_model(model)
            break
    
    return history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_on_oos(
    model: nn.Module,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    device: torch.device,
    logger: logging.Logger,
    config: Dict[str, Any]
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Generate predictions on OOS data and compute metrics."""
    
    # Handle NaN
    X_oos_clean = X_oos.fillna(X_oos.median()).fillna(0)
    
    # Scale and predict
    X_scaled = scaler.transform(X_oos_clean.values)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Create prediction Series
    pred_series = pd.Series(predictions, index=X_oos.index, name='prediction')
    
    # Basic metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_oos, predictions))),
        'mae': float(mean_absolute_error(y_oos, predictions)),
        'r2': float(r2_score(y_oos, predictions)),
        'ic': float(stats.spearmanr(y_oos, predictions)[0]),
        'n_samples': len(predictions),
        'n_dates': X_oos.index.get_level_values('date').nunique()
    }
    
    logger.info(f"OOS Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  MAE:  {metrics['mae']:.6f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")
    logger.info(f"  IC:   {metrics['ic']:.6f}")
    
    return pred_series, metrics


def run_lmt_api_evaluation(
    pred_ensemble: pd.Series,
    logger: logging.Logger,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run LMT API evaluation on ensemble predictions."""
    
    if not LMT_API_AVAILABLE:
        logger.warning("LMT API not available, skipping API evaluation")
        return {"error": "lmt_data_api not available"}
    
    eval_config = config.get('evaluation', {})
    label_period = eval_config.get('label_period', 10)
    alpha = eval_config.get('alpha', 1)
    
    # Prepare pred_esem format
    pred_esem = pred_ensemble.copy()
    pred_esem.name = 'factor'
    
    # Rename index: (date, instrument) -> (date, code)
    if pred_esem.index.names == ['date', 'instrument']:
        pred_esem.index = pred_esem.index.rename(['date', 'code'])
    
    # Remove duplicates
    pred_esem = pred_esem[~pred_esem.index.duplicated(keep='last')]
    
    # Check date count
    n_dates = pred_esem.index.get_level_values('date').nunique()
    logger.info(f"Running LMT API on {n_dates} dates, {len(pred_esem)} samples")
    
    if n_dates < label_period:
        return {"error": f"Insufficient dates: {n_dates} < {label_period}"}
    
    try:
        api = DataApi()
        
        # Group return evaluation
        logger.info(f"Calling api.da_eva_group_return (alpha={alpha}, label_period={label_period})...")
        group_re, group_ir, group_hs = api.da_eva_group_return(
            pred_esem, "factor", alpha=alpha, label_period=label_period
        )
        
        # IC evaluation
        logger.info(f"Calling api.da_eva_ic (label_period={label_period})...")
        ic_df = api.da_eva_ic(pred_esem, "factor", label_period)
        
        # Build stats table
        results = {
            "ic_df": ic_df.to_dict() if ic_df is not None else None,
            "group_re": group_re.to_dict() if group_re is not None else None,
            "group_ir": group_ir.to_dict() if group_ir is not None else None,
            "group_hs": group_hs.to_dict() if group_hs is not None else None,
        }
        
        # Try to build unified stats
        try:
            if all(x is not None for x in [ic_df, group_re, group_ir, group_hs]):
                stats_all = pd.concat([
                    ic_df,
                    group_re[["group0", "group9", "ls"]],
                    group_ir[["group0", "group9", "ls"]],
                    group_hs[["group0", "group9"]]
                ], axis=1)
                
                stats_all.columns = [
                    "IC", "ICIR",
                    "Short", "Long", "LS",
                    "ShortIr", "LongIr", "LSIR",
                    "ShortHS", "LongHS"
                ]
                
                logger.info("\n" + "=" * 60)
                logger.info("LMT API EVALUATION RESULTS")
                logger.info("=" * 60)
                logger.info(f"\n{stats_all.to_string()}")
                
                results["stats_all"] = stats_all.to_dict()
        except Exception as e:
            logger.warning(f"Failed to build stats table: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"LMT API evaluation failed: {e}")
        return {"error": str(e)}


# =============================================================================
# Main Pipeline
# =============================================================================

def main(config_path: str, device: str = None, snapshot: str = None):
    """
    Run full IS training and OOS evaluation for all 3 keys.
    """
    # Load config
    config = load_config(config_path)
    
    if snapshot:
        config['snapshot'] = snapshot
    
    # Setup output directory
    config_dir = Path(config_path).resolve().parent if Path(config_path).exists() else Path(".")
    output_dir = config_dir / config.get('output', {}).get('output_dir', 'outputs_dnn_full')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'training.log'
    logger = setup_logging(log_file)
    
    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    logger.info("=" * 60)
    logger.info("DNN FULL IS/OOS TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Snapshot: {config.get('snapshot', 'latest')}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    data_dir = Path(config.get('data_dir', 'data/combined'))
    
    # Train and evaluate each key
    all_predictions = {}
    all_metrics = {}
    models = {}
    scalers = {}
    
    for key in ['0', '1', '2']:
        logger.info("\n" + "=" * 60)
        logger.info(f"PROCESSING KEY: {key}")
        logger.info("=" * 60)
        
        try:
            # Load IS data
            logger.info("\nStep 1: Loading IS data...")
            X_is, y_is = load_is_data(data_dir, key, logger)
            feature_names = list(X_is.columns)
            
            # Prepare data
            logger.info("\nStep 2: Preparing data...")
            train_loader, val_loader, scaler = prepare_data(X_is, y_is, config, logger)
            
            # Create model
            model_config = config.get('model', {})
            model = FactorDNN(
                input_size=len(feature_names),
                hidden_sizes=model_config.get('hidden_sizes', [512, 256, 128, 64]),
                dropout=model_config.get('dropout', 0.3),
                activation=model_config.get('activation', 'leaky_relu'),
                batch_norm=model_config.get('batch_norm', True)
            ).to(device)
            
            logger.info(f"Model parameters: {model.count_parameters():,}")
            
            # Train
            logger.info("\nStep 3: Training model...")
            history = train_model(model, train_loader, val_loader, config, device, logger)
            
            # Load OOS data
            logger.info("\nStep 4: Loading OOS data...")
            X_oos, y_oos = load_oos_data(data_dir, key, logger)
            
            # Evaluate on OOS
            logger.info("\nStep 5: Evaluating on OOS...")
            predictions, metrics = evaluate_on_oos(
                model, X_oos, y_oos, scaler, device, logger, config
            )
            
            all_predictions[key] = predictions
            all_metrics[key] = metrics
            models[key] = model
            scalers[key] = scaler
            
            # Save individual model
            model_path = output_dir / f'model_key{key}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'feature_names': feature_names
            }, model_path)
            
            scaler_path = output_dir / f'scaler_key{key}.pkl'
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Saved model to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to process key {key}: {e}")
            raise
    
    # Compute ensemble predictions
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING ENSEMBLE")
    logger.info("=" * 60)
    
    # Align predictions and compute mean
    pred_df = pd.DataFrame(all_predictions)
    pred_ensemble = pred_df.mean(axis=1)
    pred_ensemble.name = 'prediction'
    
    logger.info(f"Ensemble shape: {pred_ensemble.shape}")
    logger.info(f"Ensemble date range: {pred_ensemble.index.get_level_values('date').min()} to "
                f"{pred_ensemble.index.get_level_values('date').max()}")
    
    # Run LMT API evaluation on ensemble
    logger.info("\n" + "=" * 60)
    logger.info("LMT API EVALUATION (ENSEMBLE)")
    logger.info("=" * 60)
    
    lmt_results = run_lmt_api_evaluation(pred_ensemble, logger, config)
    
    # Save ensemble predictions
    ensemble_path = output_dir / 'pred_ensemble.parquet'
    pred_ensemble.to_frame().to_parquet(ensemble_path)
    logger.info(f"Saved ensemble predictions to {ensemble_path}")
    
    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics_by_key': all_metrics,
        'lmt_api_results': lmt_results,
        'ensemble_stats': {
            'n_samples': len(pred_ensemble),
            'n_dates': pred_ensemble.index.get_level_values('date').nunique()
        }
    }
    
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for key in ['0', '1', '2']:
        if key in all_metrics:
            m = all_metrics[key]
            print(f"\nKey {key} OOS Metrics:")
            print(f"  IC: {m['ic']:.6f}")
            print(f"  R²: {m['r2']:.6f}")
            print(f"  RMSE: {m['rmse']:.6f}")
    print("=" * 60)
    
    return models, scalers, pred_ensemble, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DNN on full IS data and evaluate on full OOS")
    parser.add_argument("--config", type=str, default="config_dnn_full.yaml")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--snapshot", type=str, default=None, help="Snapshot to use (e.g., 20201231)")
    
    args = parser.parse_args()
    main(args.config, args.device, args.snapshot)
