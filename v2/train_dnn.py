"""
DNN Factor Reweight — Rolling Per-Snapshot Training & Evaluation (v2)

This script implements a rolling per-snapshot pipeline:

  For each snapshot folder independently:
    1. Load IS data → train 3 DNN models (keys 0, 1, 2)
    2. Load OOS data → predict → compute ensemble
    3. Run LMT API evaluation on that snapshot's OOS
    4. Save per-snapshot artifacts

  After all snapshots:
    5. Concatenate all OOS predictions chronologically
    6. Run aggregate LMT API evaluation
    7. Generate rolling backtest report

⚠️ Data from different snapshot folders is NEVER merged for training.
   Each snapshot is a self-contained training + evaluation unit.

Usage:
    python train_dnn.py --config config_dnn.yaml
    python train_dnn.py --config config_dnn.yaml --device cuda
    python train_dnn.py --config config_dnn.yaml --snapshots 20181228 20191231
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

# LMT API
try:
    from lmt_data_api.api import DataApi
    LMT_API_AVAILABLE = True
except ImportError:
    LMT_API_AVAILABLE = False
    print("Warning: lmt_data_api not available. API evaluation will be skipped.")


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "data_dir": "data/",
    "snapshots": ["20181228", "20191231", "20201231"],
    "snapshot_oos_end": {
        "20181228": 20191231,
        "20191231": 20201231,
        "20201231": 20211231,
    },
    "factor_keys": ["0", "1", "2"],
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
        "val_ratio": 0.15,
        "random_seed": 42
    },
    "evaluation": {
        "label_period": 10,
        "alpha": 1
    },
    "output": {
        "output_dir": "outputs_dnn"
    }
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path = None, name: str = "DNNRolling") -> logging.Logger:
    logger = logging.getLogger(name)
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

        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        act_fn = activations.get(activation, nn.LeakyReLU(0.1))

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

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
    """Load config from YAML file, merging with defaults."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}

    config = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def load_snapshot_data(
    data_dir: Path, snapshot: str, key: str, split: str, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load IS or OOS data for a specific snapshot and key.

    Args:
        data_dir: Base data directory
        snapshot: Snapshot name (e.g., '20201231')
        key: Factor key ('0', '1', '2')
        split: 'is' or 'oos'
        logger: Logger instance
    """
    file_path = data_dir / snapshot / f"factors_{key}_{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading {split.upper()} data: {file_path}")
    df = pd.read_parquet(file_path)

    label_cols = ['labelValue', 'endDate']
    feature_cols = [c for c in df.columns if c not in label_cols]
    X = df[feature_cols]
    y = df['labelValue']

    # Drop NaN labels
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    dates = X.index.get_level_values('date')
    logger.info(f"  Shape: {X.shape}, date range: {dates.min()} to {dates.max()} ({dates.nunique()} days)")

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
    """Prepare IS data into train/val DataLoaders with temporal split."""
    train_params = config.get('training', {})
    val_ratio = train_params.get('val_ratio', 0.15)
    batch_size = train_params.get('batch_size', 4096)
    random_seed = train_params.get('random_seed', 42)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Fill NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with median")
        X = X.fillna(X.median()).fillna(0)

    # Temporal split for validation
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

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)

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
    """Train the DNN model with early stopping."""
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
# Prediction & Evaluation
# =============================================================================

def predict_oos(
    model: nn.Module,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Generate OOS predictions and compute basic metrics."""
    X_oos_clean = X_oos.fillna(X_oos.median()).fillna(0)
    X_scaled = scaler.transform(X_oos_clean.values)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    pred_series = pd.Series(predictions, index=X_oos.index, name='prediction')

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_oos, predictions))),
        'mae': float(mean_absolute_error(y_oos, predictions)),
        'r2': float(r2_score(y_oos, predictions)),
        'ic': float(stats.spearmanr(y_oos, predictions)[0]),
        'n_samples': len(predictions),
        'n_dates': int(X_oos.index.get_level_values('date').nunique())
    }

    logger.info(f"  OOS — IC: {metrics['ic']:.4f} | R²: {metrics['r2']:.6f} | RMSE: {metrics['rmse']:.6f}")
    return pred_series, metrics


def run_lmt_api_evaluation(
    pred_ensemble: pd.Series,
    logger: logging.Logger,
    config: Dict[str, Any],
    label: str = ""
) -> Dict[str, Any]:
    """Run LMT API evaluation on an ensemble prediction series."""
    if not LMT_API_AVAILABLE:
        logger.warning("LMT API not available, skipping evaluation")
        return {"error": "lmt_data_api not available"}

    eval_config = config.get('evaluation', {})
    label_period = eval_config.get('label_period', 10)
    alpha_param = eval_config.get('alpha', 1)

    pred_esem = pred_ensemble.copy()
    pred_esem.name = 'factor'
    if pred_esem.index.names == ['date', 'instrument']:
        pred_esem.index = pred_esem.index.rename(['date', 'code'])
    pred_esem = pred_esem[~pred_esem.index.duplicated(keep='last')]

    n_dates = pred_esem.index.get_level_values('date').nunique()
    logger.info(f"LMT API {label}: {n_dates} dates, {len(pred_esem)} samples")

    if n_dates < label_period:
        logger.warning(f"Insufficient dates for LMT API: {n_dates} < {label_period}")
        return {"error": f"Insufficient dates: {n_dates} < {label_period}"}

    try:
        api = DataApi()

        group_re, group_ir, group_hs = api.da_eva_group_return(
            pred_esem, "factor", alpha=alpha_param, label_period=label_period
        )
        ic_df = api.da_eva_ic(pred_esem, "factor", label_period)

        results = {
            "ic_df": ic_df.to_dict() if ic_df is not None else None,
            "group_re": group_re.to_dict() if group_re is not None else None,
            "group_ir": group_ir.to_dict() if group_ir is not None else None,
            "group_hs": group_hs.to_dict() if group_hs is not None else None,
        }

        # Build summary table
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
                logger.info(f"\n{'='*60}\nLMT API RESULTS {label}\n{'='*60}")
                logger.info(f"\n{stats_all.to_string()}")
                results["stats_all"] = stats_all.to_dict()
        except Exception as e:
            logger.warning(f"Failed to build stats table: {e}")

        return results

    except Exception as e:
        logger.error(f"LMT API evaluation failed: {e}")
        return {"error": str(e)}


# =============================================================================
# Per-Snapshot Processing
# =============================================================================

def process_snapshot(
    snapshot: str,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a single snapshot: train models, predict OOS, evaluate.

    Returns a dict with predictions, metrics, and LMT results for this snapshot.
    """
    data_dir = Path(config.get('data_dir', 'data/'))
    output_base = Path(config.get('output', {}).get('output_dir', 'outputs_dnn'))
    snapshot_output = output_base / snapshot
    snapshot_output.mkdir(parents=True, exist_ok=True)

    factor_keys = config.get('factor_keys', ['0', '1', '2'])
    cutoff = int(snapshot)
    oos_end = config.get('snapshot_oos_end', {}).get(snapshot, cutoff + 10000)

    logger.info("\n" + "=" * 70)
    logger.info(f"SNAPSHOT: {snapshot}")
    logger.info(f"  IS:  dates <= {cutoff}")
    logger.info(f"  OOS: dates > {cutoff} and <= {oos_end}")
    logger.info("=" * 70)

    all_predictions = {}
    all_metrics = {}

    for key in factor_keys:
        logger.info(f"\n--- Key {key} ---")

        # Load IS data
        X_is, y_is = load_snapshot_data(data_dir, snapshot, key, 'is', logger)
        feature_names = list(X_is.columns)

        # Prepare data (temporal train/val split within IS)
        train_loader, val_loader, scaler = prepare_data(X_is, y_is, config, logger)

        # Build model
        model_config = config.get('model', {})
        model = FactorDNN(
            input_size=len(feature_names),
            hidden_sizes=model_config.get('hidden_sizes', [512, 256, 128, 64]),
            dropout=model_config.get('dropout', 0.3),
            activation=model_config.get('activation', 'leaky_relu'),
            batch_norm=model_config.get('batch_norm', True)
        ).to(device)
        logger.info(f"  Model parameters: {model.count_parameters():,}")

        # Train
        history = train_model(model, train_loader, val_loader, config, device, logger)

        # Load OOS data
        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, 'oos', logger)

        # Predict OOS
        predictions, metrics = predict_oos(model, X_oos, y_oos, scaler, device, logger)

        all_predictions[key] = predictions
        all_metrics[key] = metrics

        # Save model & scaler
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'feature_names': feature_names,
            'snapshot': snapshot,
            'key': key
        }, snapshot_output / f'model_key{key}.pt')
        joblib.dump(scaler, snapshot_output / f'scaler_key{key}.pkl')

        # Free memory
        del X_is, y_is, train_loader, val_loader, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Compute ensemble
    logger.info(f"\n--- Ensemble (snapshot {snapshot}) ---")
    pred_df = pd.DataFrame(all_predictions)
    pred_ensemble = pred_df.mean(axis=1)
    pred_ensemble.name = 'prediction'

    dates = pred_ensemble.index.get_level_values('date')
    logger.info(f"Ensemble: {len(pred_ensemble)} samples, "
                f"{dates.nunique()} dates ({dates.min()} to {dates.max()})")

    # Save ensemble predictions
    pred_ensemble.to_frame().to_parquet(snapshot_output / 'pred_ensemble.parquet')

    # LMT API evaluation for this snapshot
    lmt_results = run_lmt_api_evaluation(
        pred_ensemble, logger, config, label=f"[Snapshot {snapshot}]"
    )

    # Build snapshot report
    snapshot_report = {
        'snapshot': snapshot,
        'cutoff_date': cutoff,
        'oos_end_date': oos_end,
        'oos_date_range': [int(dates.min()), int(dates.max())],
        'metrics_by_key': all_metrics,
        'ensemble_stats': {
            'n_samples': len(pred_ensemble),
            'n_dates': int(dates.nunique())
        },
        'lmt_api_results': lmt_results,
        'timestamp': datetime.now().isoformat()
    }

    # Save per-snapshot report
    with open(snapshot_output / 'snapshot_report.json', 'w') as f:
        json.dump(snapshot_report, f, indent=2, default=str)

    logger.info(f"✅ Snapshot {snapshot} complete → {snapshot_output}/")

    return {
        'report': snapshot_report,
        'pred_ensemble': pred_ensemble
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main(config_path: str, device: str = None, snapshots_override: List[str] = None):
    config = load_config(config_path)

    config_dir = Path(config_path).resolve().parent if Path(config_path).exists() else Path(".")
    output_dir = config_dir / config.get('output', {}).get('output_dir', 'outputs_dnn')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve relative data_dir
    data_dir = Path(config.get('data_dir', 'data/'))
    if not data_dir.is_absolute():
        config['data_dir'] = str(config_dir / data_dir)

    # Resolve relative output_dir in config
    config['output']['output_dir'] = str(output_dir)

    log_file = output_dir / 'training.log'
    logger = setup_logging(log_file)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    snapshots = snapshots_override or config.get('snapshots', ["20181228", "20191231", "20201231"])

    logger.info("=" * 70)
    logger.info("DNN ROLLING PER-SNAPSHOT TRAINING PIPELINE (v2)")
    logger.info("=" * 70)
    logger.info(f"Config:    {config_path}")
    logger.info(f"Device:    {device}")
    logger.info(f"Snapshots: {snapshots}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # =========================================================================
    # Phase 1: Rolling per-snapshot training
    # =========================================================================

    all_snapshot_results = []
    all_oos_predictions = []

    for snapshot in snapshots:
        result = process_snapshot(snapshot, config, device, logger)
        all_snapshot_results.append(result['report'])
        all_oos_predictions.append(result['pred_ensemble'])

    # =========================================================================
    # Phase 2: Aggregate all OOS predictions
    # =========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("ROLLING AGGREGATION — ALL SNAPSHOTS")
    logger.info("=" * 70)

    combined_oos = pd.concat(all_oos_predictions).sort_index()
    # Remove any duplicates (shouldn't happen with proper OOS boundaries)
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep='last')]
    combined_oos.name = 'prediction'

    dates = combined_oos.index.get_level_values('date')
    logger.info(f"Combined OOS: {len(combined_oos)} samples, "
                f"{dates.nunique()} dates ({dates.min()} to {dates.max()})")

    # Save combined predictions
    combined_oos.to_frame().to_parquet(output_dir / 'pred_ensemble_all.parquet')

    # LMT API aggregate evaluation
    aggregate_lmt = run_lmt_api_evaluation(
        combined_oos, logger, config, label="[AGGREGATE]"
    )

    # =========================================================================
    # Phase 3: Generate rolling report
    # =========================================================================

    rolling_report = {
        'pipeline': 'DNN Rolling Per-Snapshot (v2)',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'snapshots_processed': snapshots,
        'per_snapshot_results': all_snapshot_results,
        'aggregate': {
            'n_samples': len(combined_oos),
            'n_dates': int(dates.nunique()),
            'date_range': [int(dates.min()), int(dates.max())],
            'lmt_api_results': aggregate_lmt
        }
    }

    with open(output_dir / 'rolling_report.json', 'w') as f:
        json.dump(rolling_report, f, indent=2, default=str)

    # =========================================================================
    # Summary
    # =========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")

    print("\n" + "=" * 70)
    print("ROLLING TRAINING SUMMARY")
    print("=" * 70)

    for report in all_snapshot_results:
        snapshot = report['snapshot']
        oos_range = report.get('oos_date_range', ['?', '?'])
        print(f"\n📊 Snapshot {snapshot} (OOS: {oos_range[0]} → {oos_range[1]})")
        for key in config.get('factor_keys', ['0', '1', '2']):
            if key in report.get('metrics_by_key', {}):
                m = report['metrics_by_key'][key]
                print(f"   Key {key}: IC={m['ic']:.4f}  R²={m['r2']:.6f}  RMSE={m['rmse']:.6f}")

    print(f"\n📈 Aggregate OOS: {len(combined_oos)} samples, {dates.nunique()} dates")
    print(f"   Date range: {dates.min()} → {dates.max()}")
    print("=" * 70)

    return rolling_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNN rolling per-snapshot training & evaluation (v2)"
    )
    parser.add_argument("--config", type=str, default="config_dnn.yaml",
                        help="Config YAML file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu")
    parser.add_argument("--snapshots", nargs="+", default=None,
                        help="Override snapshot list")
    args = parser.parse_args()
    main(args.config, args.device, args.snapshots)
