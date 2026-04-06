"""
DNN Factor Reweight — Rolling Per-Snapshot Training & Evaluation (v2)

Pipeline:
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

Usage:
    python train_dnn.py --config configs/dnn.yaml
    python train_dnn.py --config configs/dnn.yaml --device cuda:0
    python train_dnn.py --config configs/dnn.yaml --snapshots 20181228 20191231
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
        "batch_norm": True,
    },
    "training": {
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "warmup_epochs": 5,
        "early_stopping_patience": 15,
        "val_ratio": 0.15,
        "random_seed": 42,
    },
    "evaluation": {"label_period": 10, "alpha": 1},
    "output": {"output_dir": "outputs_dnn"},
}


# =============================================================================
# Logging helpers
# =============================================================================

SEPARATOR = "─" * 72
DOUBLE_SEP = "═" * 72


def setup_logging(log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger("DNN")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s │ %(levelname)-5s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    return logger


def log_banner(logger, title: str, char: str = "═"):
    line = char * 72
    logger.info("")
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def log_kv(logger, key: str, value, indent: int = 4):
    logger.info(f"{' ' * indent}{key + ':':<22s} {value}")


def log_section(logger, title: str):
    logger.info("")
    logger.info(f"── {title} " + "─" * max(0, 68 - len(title)))


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
        batch_norm: bool = True,
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        act_fn = activations.get(activation, nn.LeakyReLU(0.1))

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
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
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
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
    if Path(config_path).exists():
        with open(config_path, "r") as f:
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


def load_universe(path: str, logger: logging.Logger) -> Optional[set]:
    """Load instrument universe from a text file (one code per line)."""
    p = Path(path)
    if not path or not p.exists():
        return None
    codes = set()
    for line in p.read_text().strip().splitlines():
        code = line.strip()
        if code:
            codes.add(code)
    logger.info(f"    Universe loaded: {len(codes)} instruments from {p}")
    return codes


def load_snapshot_data(
    data_dir: Path, snapshot: str, key: str, split: str,
    logger: logging.Logger, universe: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    file_path = data_dir / snapshot / f"factors_{key}_{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    label_cols = ["labelValue", "endDate"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    X = df[feature_cols]
    y = df["labelValue"]

    valid = y.notna()
    X, y = X[valid], y[valid]

    # Filter by universe
    if universe is not None:
        instruments = X.index.get_level_values("instrument")
        mask = instruments.isin(universe)
        n_before = len(X)
        X, y = X[mask], y[mask]
        logger.info(f"    Universe filter: {n_before:,} → {len(X):,} rows "
                    f"({len(X)/max(1,n_before):.0%} kept)")

    dates = X.index.get_level_values("date")
    logger.info(f"    Loaded {split.upper()} key={key}: "
                f"{X.shape[0]:>8,} rows × {X.shape[1]} cols  │  "
                f"dates {dates.min()}→{dates.max()} ({dates.nunique()} days)")
    return X, y


# =============================================================================
# Training
# =============================================================================

def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    tp = config.get("training", {})
    val_ratio = tp.get("val_ratio", 0.15)
    batch_size = tp.get("batch_size", 4096)
    seed = tp.get("random_seed", 42)

    np.random.seed(seed)
    torch.manual_seed(seed)

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"    Filling {nan_count:,} NaN values with column median")
        X = X.fillna(X.median()).fillna(0)

    # Temporal split for validation
    dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)
    val_start = int(n_dates * (1 - val_ratio))
    # Gap buffer: skip label_period days between train and val to
    # prevent forward-return overlap (data leakage)
    label_period = config.get("evaluation", {}).get("label_period", 10)
    train_end = max(0, val_start - label_period)
    train_dates = dates[:train_end]
    val_dates = dates[val_start:]

    train_mask = X.index.get_level_values("date").isin(train_dates)
    val_mask = X.index.get_level_values("date").isin(val_dates)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    logger.info(f"    Train split: {len(X_train):>8,} samples ({len(train_dates)} days)")
    logger.info(f"    Val   split: {len(X_val):>8,} samples ({len(val_dates)} days)")
    logger.info(f"    Gap buffer:  {label_period} days dropped between train/val")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train.values))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val.values))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, List]:
    tp = config.get("training", {})
    epochs = tp.get("epochs", 100)
    lr = tp.get("learning_rate", 0.0001)
    wd = tp.get("weight_decay", 0.001)
    patience = tp.get("early_stopping_patience", 15)
    warmup_epochs = tp.get("warmup_epochs", 5)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Warmup + cosine annealing scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_ic": []}

    # Epoch log header
    logger.info(f"    {'Epoch':>6s}  │  {'Train Loss':>10s}  │  "
                f"{'Val Loss':>10s}  │  {'Val IC':>8s}  │  {'LR':>10s}")
    logger.info(f"    {'─'*6}──┼──{'─'*10}──┼──{'─'*10}──┼──{'─'*8}──┼──{'─'*10}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
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
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())
        val_loss /= len(val_loader)

        val_ic = float(stats.spearmanr(all_targets, all_preds)[0])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"    {epoch+1:4d}/{epochs:d}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  {cur_lr:10.2e}"
            )

        if early_stopping(val_loss, model):
            logger.info(
                f"    {epoch+1:4d}/{epochs:d}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  early stop"
            )
            early_stopping.load_best_model(model)
            logger.info(f"    ⤷ Restored best model (val_loss={early_stopping.best_score:.6f})")
            break

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_ic = history["val_ic"][best_epoch - 1]
    logger.info(f"    Best epoch: {best_epoch}  │  "
                f"val_loss {early_stopping.best_score:.6f}  │  val_ic {best_ic:+.4f}")

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
    logger: logging.Logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    X_clean = X_oos.fillna(X_oos.median()).fillna(0)
    X_scaled = scaler.transform(X_clean.values)
    X_t = torch.FloatTensor(X_scaled).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    pred_s = pd.Series(preds, index=X_oos.index, name="prediction")

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_oos, preds))),
        "mae": float(mean_absolute_error(y_oos, preds)),
        "r2": float(r2_score(y_oos, preds)),
        "ic": float(stats.spearmanr(y_oos, preds)[0]),
        "n_samples": len(preds),
        "n_dates": int(X_oos.index.get_level_values("date").nunique()),
    }

    logger.info(f"    OOS metrics  │  IC {metrics['ic']:+.4f}  │  "
                f"R² {metrics['r2']:.6f}  │  RMSE {metrics['rmse']:.6f}  │  "
                f"MAE {metrics['mae']:.6f}")
    return pred_s, metrics


def run_lmt_api_evaluation(
    pred_ensemble: pd.Series,
    logger: logging.Logger,
    config: Dict[str, Any],
    label: str = "",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if not LMT_API_AVAILABLE:
        logger.warning("  lmt_data_api not available — skipping API evaluation")
        return {"status": "skipped", "reason": "lmt_data_api not installed"}

    eval_cfg = config.get("evaluation", {})
    label_period = eval_cfg.get("label_period", 10)
    alpha_param = eval_cfg.get("alpha", 1)

    pred_esem = pred_ensemble.copy()
    pred_esem.name = "factor"
    if pred_esem.index.names == ["date", "instrument"]:
        pred_esem.index = pred_esem.index.rename(["date", "code"])
    pred_esem = pred_esem[~pred_esem.index.duplicated(keep="last")]

    n_dates = pred_esem.index.get_level_values("date").nunique()
    if n_dates < label_period:
        logger.warning(f"  Insufficient dates for LMT API: {n_dates} < {label_period}")
        return {"status": "skipped", "reason": f"dates {n_dates} < {label_period}"}

    try:
        api = DataApi()
        group_re, group_ir, group_hs = api.da_eva_group_return(
            pred_esem, "factor", alpha=alpha_param, label_period=label_period
        )
        ic_df = api.da_eva_ic(pred_esem, "factor", label_period)

        results = {
            "status": "success",
            "ic_df": ic_df.to_dict() if ic_df is not None else None,
            "group_re": group_re.to_dict() if group_re is not None else None,
            "group_ir": group_ir.to_dict() if group_ir is not None else None,
            "group_hs": group_hs.to_dict() if group_hs is not None else None,
        }

        # ── Save full API results to CSV ──
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for name, obj in [("lmt_ic", ic_df), ("lmt_group_return", group_re),
                              ("lmt_group_ir", group_ir), ("lmt_group_hs", group_hs)]:
                if obj is None:
                    continue
                csv_p = output_dir / f"{name}.csv"
                if isinstance(obj, pd.Series):
                    obj.to_frame().to_csv(csv_p)
                else:
                    obj.to_csv(csv_p)
                saved.append(csv_p.name)
            if saved:
                logger.info(f"    API data saved → {', '.join(saved)}")

        if all(x is not None for x in [ic_df, group_re, group_ir, group_hs]):
            try:
                logger.debug(f"  API return shapes — ic_df: {getattr(ic_df, 'shape', '?')}, "
                             f"group_re: {getattr(group_re, 'shape', '?')}, "
                             f"group_ir: {getattr(group_ir, 'shape', '?')}, "
                             f"group_hs: {getattr(group_hs, 'shape', '?')}")

                # Build summary pieces dynamically
                parts = []
                col_names = []

                # IC piece — could be Series (1 col) or DataFrame (2 cols)
                if isinstance(ic_df, pd.Series):
                    parts.append(ic_df.rename("IC"))
                    col_names.append("IC")
                elif isinstance(ic_df, pd.DataFrame):
                    parts.append(ic_df)
                    ic_cols = list(ic_df.columns)
                    if len(ic_cols) == 1:
                        col_names.extend(["IC"])
                    elif len(ic_cols) == 2:
                        col_names.extend(["IC", "ICIR"])
                    else:
                        col_names.extend([f"IC_{c}" for c in ic_cols])

                # Group return / IR / HS — pick available columns
                grp_targets = ["group0", "group9", "ls"]
                for df, prefix, cols_wanted in [
                    (group_re, "", grp_targets),
                    (group_ir, "IR_", grp_targets),
                    (group_hs, "HS_", ["group0", "group9"]),
                ]:
                    avail = [c for c in cols_wanted if c in df.columns]
                    if avail:
                        parts.append(df[avail])
                        name_map = {"group0": "Short", "group9": "Long", "ls": "LS"}
                        col_names.extend([prefix + name_map.get(c, c) for c in avail])

                if parts:
                    summary = pd.concat(parts, axis=1)
                    summary.columns = col_names
                    log_section(logger, f"LMT API Results {label}")
                    logger.info(f"\n{summary.to_string()}")
                    results["summary"] = summary.to_dict()

                    # Save summary CSV
                    if output_dir is not None:
                        sum_path = output_dir / "lmt_summary.csv"
                        summary.to_csv(sum_path)
                        logger.info(f"    Summary saved → {sum_path.name}")
                else:
                    logger.warning("  No valid data pieces for summary table")
            except Exception as e:
                logger.warning(f"  Failed to build summary table: {e}")

        return results

    except Exception as e:
        logger.error(f"  LMT API evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# Per-Snapshot Processing
# =============================================================================

def process_snapshot(
    snapshot: str,
    config: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    universe: Optional[set] = None,
) -> Dict[str, Any]:
    """Process a single snapshot: train → predict → evaluate → save."""
    data_dir = Path(config.get("data_dir", "data/"))
    factor_keys = config.get("factor_keys", ["0", "1", "2"])
    cutoff = int(snapshot)
    oos_end = config.get("snapshot_oos_end", {}).get(snapshot, cutoff + 10000)

    # Per-snapshot output directory
    snap_dir = run_dir / f"snapshot_{snapshot}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    log_banner(logger, f"SNAPSHOT {snapshot}  │  IS ≤ {cutoff}  │  OOS ({cutoff+1}, {oos_end}]")

    all_predictions = {}
    all_metrics = {}
    model_config = config.get("model", {})

    for key in factor_keys:
        log_section(logger, f"Key {key}  —  Snapshot {snapshot}")

        # Load IS data
        X_is, y_is = load_snapshot_data(data_dir, snapshot, key, "is", logger, universe)
        feature_names = list(X_is.columns)

        # Prepare data
        train_loader, val_loader, scaler = prepare_data(X_is, y_is, config, logger)

        # Build model
        model = FactorDNN(
            input_size=len(feature_names),
            hidden_sizes=model_config.get("hidden_sizes", [512, 256, 128, 64]),
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "leaky_relu"),
            batch_norm=model_config.get("batch_norm", True),
        ).to(device)
        logger.info(f"    Model: {model.count_parameters():,} parameters  │  device={device}")

        # Train
        history = train_model(model, train_loader, val_loader, config, device, logger)

        # Load OOS & predict
        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, "oos", logger, universe)
        preds, metrics = predict_oos(model, X_oos, y_oos, scaler, device, logger)

        all_predictions[key] = preds
        all_metrics[key] = metrics

        # ── Save per-key artifacts ──
        model_path = snap_dir / f"dnn_key{key}_model.pt"
        scaler_path = snap_dir / f"dnn_key{key}_scaler.pkl"
        pred_path = snap_dir / f"dnn_key{key}_oos_predictions.parquet"
        hist_path = snap_dir / f"dnn_key{key}_training_history.json"

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_names": feature_names,
            "snapshot": snapshot,
            "key": key,
            "best_val_loss": float(min(history["val_loss"])),
            "best_val_ic": float(history["val_ic"][int(np.argmin(history["val_loss"]))]),
            "epochs_trained": len(history["val_loss"]),
        }, model_path)
        joblib.dump(scaler, scaler_path)
        preds.to_frame().to_parquet(pred_path)

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"    Saved → {model_path.name}, {scaler_path.name}, "
                     f"{pred_path.name}, {hist_path.name}")

        # Free memory
        del X_is, y_is, train_loader, val_loader, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Ensemble ──
    log_section(logger, f"Ensemble  —  Snapshot {snapshot}")

    pred_df = pd.DataFrame(all_predictions)
    pred_ensemble = pred_df.mean(axis=1)
    pred_ensemble.name = "prediction"

    dates = pred_ensemble.index.get_level_values("date")
    logger.info(f"    Ensemble: {len(pred_ensemble):,} samples, "
                f"{dates.nunique()} dates ({dates.min()} → {dates.max()})")

    ensemble_path = snap_dir / "dnn_ensemble_oos.parquet"
    pred_ensemble.to_frame().to_parquet(ensemble_path)

    # ── Simple CSV for easy reuse ──
    csv_path = snap_dir / "oos_predictions.csv"
    csv_df = pred_ensemble.reset_index()
    csv_df.columns = ["date", "code", "prediction"]
    csv_df = csv_df.sort_values(["date", "code"])
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"    CSV saved → {csv_path.name}  ({len(csv_df):,} rows)")

    # ── LMT API eval ──
    lmt_results = run_lmt_api_evaluation(
        pred_ensemble, logger, config, label=f"Snapshot {snapshot}",
        output_dir=snap_dir,
    )

    # ── Per-snapshot report ──
    snapshot_report = {
        "snapshot": snapshot,
        "cutoff_date": cutoff,
        "oos_end_date": oos_end,
        "oos_date_range": [int(dates.min()), int(dates.max())],
        "oos_n_dates": int(dates.nunique()),
        "oos_n_samples": len(pred_ensemble),
        "metrics_by_key": all_metrics,
        "lmt_api": lmt_results,
        "artifacts": {
            "directory": str(snap_dir),
            "files": sorted(p.name for p in snap_dir.iterdir()),
        },
        "timestamp": datetime.now().isoformat(),
    }

    report_path = snap_dir / "snapshot_report.json"
    with open(report_path, "w") as f:
        json.dump(snapshot_report, f, indent=2, default=str)

    logger.info(f"    Report saved → {report_path.name}")
    logger.info(f"  ✅ Snapshot {snapshot} complete  ({len(list(snap_dir.iterdir()))} files)")

    return {"report": snapshot_report, "pred_ensemble": pred_ensemble}


# =============================================================================
# Main Pipeline
# =============================================================================

def main(config_path: str, device: str = None, snapshots_override: List[str] = None,
         universe_override: str = None):
    config = load_config(config_path)

    if universe_override is not None:
        config["universe_file"] = universe_override

    script_dir = Path(__file__).resolve().parent

    # Resolve data_dir
    data_dir = Path(config.get("data_dir", "data/"))
    if not data_dir.is_absolute():
        config["data_dir"] = str(script_dir / data_dir)

    # Create timestamped run directory
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = script_dir / config.get("output", {}).get("output_dir", "outputs_dnn")
    run_dir = output_base / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["output_dir"] = str(run_dir)

    # Latest run marker
    latest_marker = output_base / "latest_run.txt"
    latest_marker.write_text(str(run_dir))

    log_file = run_dir / "training.log"
    logger = setup_logging(log_file)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    snapshots = snapshots_override or config.get("snapshots", ["20181228", "20191231", "20201231"])
    mp = config.get("model", {})
    tp = config.get("training", {})

    # Save config snapshot
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ─────────────────────────────────────────────────────────────────────
    #  Header
    # ─────────────────────────────────────────────────────────────────────
    log_banner(logger, "DNN  —  ROLLING PER-SNAPSHOT PIPELINE (v2)")
    log_kv(logger, "Config", config_path)
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Data directory", config["data_dir"])
    log_kv(logger, "Device", device)
    log_kv(logger, "Snapshots", " → ".join(snapshots))
    log_kv(logger, "Factor keys", ", ".join(config.get("factor_keys", ["0","1","2"])))
    log_kv(logger, "Architecture", f"{mp.get('hidden_sizes')}  dropout={mp.get('dropout')}")
    log_kv(logger, "Batch size", tp.get("batch_size"))
    log_kv(logger, "Learning rate", tp.get("learning_rate"))
    log_kv(logger, "Max epochs", tp.get("epochs"))
    log_kv(logger, "Early stop", f"patience={tp.get('early_stopping_patience')}")
    log_kv(logger, "Val ratio", tp.get("val_ratio"))
    log_kv(logger, "Random seed", tp.get("random_seed"))
    log_kv(logger, "Started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not LMT_API_AVAILABLE:
        logger.warning("  ⚠ lmt_data_api not installed — API evaluation will be skipped")

    # Load universe filter
    universe_path = config.get("universe_file", "")
    universe = load_universe(universe_path, logger) if universe_path else None
    if universe:
        log_kv(logger, "Universe", f"{len(universe)} instruments from {universe_path}")
    else:
        log_kv(logger, "Universe", "all instruments (no filter)")

    pipeline_start = datetime.now()

    # ─────────────────────────────────────────────────────────────────────
    #  Phase 1: Rolling per-snapshot training
    # ─────────────────────────────────────────────────────────────────────
    log_banner(logger, "PHASE 1  │  ROLLING PER-SNAPSHOT TRAINING")

    all_reports = []
    all_oos_preds = []

    for i, snapshot in enumerate(snapshots, 1):
        logger.info(f"\n  ▶ Snapshot {i}/{len(snapshots)}: {snapshot}")
        result = process_snapshot(snapshot, config, run_dir, device, logger, universe)
        all_reports.append(result["report"])
        all_oos_preds.append(result["pred_ensemble"])

    # ─────────────────────────────────────────────────────────────────────
    #  Phase 2: Aggregate all OOS predictions
    # ─────────────────────────────────────────────────────────────────────
    log_banner(logger, "PHASE 2  │  AGGREGATE OOS PREDICTIONS")

    combined_oos = pd.concat(all_oos_preds).sort_index()
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="last")]
    combined_oos.name = "prediction"

    dates = combined_oos.index.get_level_values("date")
    log_kv(logger, "Total samples", f"{len(combined_oos):,}")
    log_kv(logger, "Total dates", dates.nunique())
    log_kv(logger, "Date range", f"{dates.min()} → {dates.max()}")

    agg_pred_path = run_dir / "dnn_ensemble_all_oos.parquet"
    combined_oos.to_frame().to_parquet(agg_pred_path)
    logger.info(f"    Saved → {agg_pred_path.name}")

    # ── Aggregate CSV ──
    agg_csv_path = run_dir / "oos_predictions_all.csv"
    agg_csv = combined_oos.reset_index()
    agg_csv.columns = ["date", "code", "prediction"]
    agg_csv = agg_csv.sort_values(["date", "code"])
    agg_csv.to_csv(agg_csv_path, index=False)
    logger.info(f"    CSV saved → {agg_csv_path.name}  ({len(agg_csv):,} rows)")

    aggregate_lmt = run_lmt_api_evaluation(
        combined_oos, logger, config, label="AGGREGATE",
        output_dir=run_dir,
    )

    # ─────────────────────────────────────────────────────────────────────
    #  Phase 3: Report
    # ─────────────────────────────────────────────────────────────────────
    log_banner(logger, "PHASE 3  │  SUMMARY")

    rolling_report = {
        "pipeline": "DNN Rolling Per-Snapshot (v2)",
        "run_directory": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": config,
        "snapshots_processed": snapshots,
        "per_snapshot": all_reports,
        "aggregate": {
            "n_samples": len(combined_oos),
            "n_dates": int(dates.nunique()),
            "date_range": [int(dates.min()), int(dates.max())],
            "prediction_file": str(agg_pred_path),
            "lmt_api": aggregate_lmt,
        },
    }

    report_path = run_dir / "rolling_report.json"
    with open(report_path, "w") as f:
        json.dump(rolling_report, f, indent=2, default=str)

    # ── Pretty summary table ──
    logger.info("")
    hdr = (f"  {'Snapshot':>10s}  │  {'OOS Range':>21s}  │  "
           f"{'Days':>5s}  │  {'Samples':>9s}  │  "
           f"{'IC(k0)':>7s}  {'IC(k1)':>7s}  {'IC(k2)':>7s}")
    logger.info(hdr)
    logger.info("  " + "─" * len(hdr.strip()))

    for rpt in all_reports:
        snap = rpt["snapshot"]
        dr = rpt.get("oos_date_range", ["?", "?"])
        nd = rpt.get("oos_n_dates", "?")
        ns = rpt.get("oos_n_samples", "?")
        keys = config.get("factor_keys", ["0", "1", "2"])
        ics = []
        for k in keys:
            m = rpt.get("metrics_by_key", {}).get(k, {})
            ics.append(f"{m.get('ic', 0):+.4f}")
        logger.info(
            f"  {snap:>10s}  │  {dr[0]} → {dr[1]}  │  "
            f"{nd:>5}  │  {ns:>9,}  │  {'  '.join(ics)}"
        )

    logger.info("")
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Report", report_path.name)
    log_kv(logger, "All-OOS predictions", agg_pred_path.name)
    log_kv(logger, "Training log", log_file.name)

    # ── Final ──
    pipeline_end = datetime.now()
    duration = pipeline_end - pipeline_start

    log_banner(logger, "PIPELINE COMPLETE")
    log_kv(logger, "Duration", str(duration).split(".")[0])
    log_kv(logger, "Artifacts", f"{len(list(run_dir.rglob('*')))} files in {run_dir}")
    logger.info("")

    # Tree view
    logger.info("  Output structure:")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(run_dir)
            size_kb = p.stat().st_size / 1024
            if size_kb > 1024:
                size_str = f"{size_kb/1024:.1f} MB"
            else:
                size_str = f"{size_kb:.0f} KB"
            logger.info(f"    {str(rel):<55s} {size_str:>10s}")

    return rolling_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNN rolling per-snapshot training & evaluation (v2)"
    )
    parser.add_argument("--config", type=str, default="configs/dnn.yaml",
                        help="Config YAML file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, cuda:0, cuda:1, …")
    parser.add_argument("--snapshots", nargs="+", default=None,
                        help="Override snapshot list")
    parser.add_argument("--universe", type=str, default=None,
                        help="Path to universe.txt (one code per line)")
    args = parser.parse_args()
    main(args.config, args.device, args.snapshots, args.universe)
