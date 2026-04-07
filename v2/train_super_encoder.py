"""
SuperEncoder — Rolling Per-Snapshot Training & Evaluation (v2)

Combines autoencoder factor compression with cross-sectional attention
on the latent space.  Each stock's factors are compressed to a latent
representation, then all stocks on the same date attend to each other
in that compact space before per-stock prediction.

Reconstruction loss (from the pre-attention latent) regularises the
encoder to preserve factor structure.

⚠ Requires per-date batching with variable stock counts (padded).
  Uses a custom training loop — not the standard tabular pipeline.

Usage:
    python train_super_encoder.py --config configs/super_encoder.yaml
    python train_super_encoder.py --config configs/super_encoder.yaml --device cuda:0
    python train_super_encoder.py --config configs/super_encoder.yaml --universe ../universe.txt
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from common import (
    setup_logging, log_banner, log_kv, log_section,
    load_config, load_universe, load_snapshot_data,
    EarlyStopping, count_parameters,
    run_lmt_api_evaluation, LMT_API_AVAILABLE,
    make_parser, BASE_DEFAULT_CONFIG,
)
from models import SuperEncoder
import yaml


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "encoder_sizes": [512, 256],
        "latent_dim": 64,
        "predictor_sizes": [128],
        "dropout": 0.3,
        "recon_weight": 0.1,
        "use_residual": True,
        "ic_loss_weight": 1.0,
        "cs_n_heads": 4,
        "cs_n_layers": 2,
        "cs_dim_feedforward": 256,
        "cs_dropout": None,
    },
    "training": {
        "epochs": 100,
        "learning_rate": 0.00005,
        "weight_decay": 0.01,
        "warmup_epochs": 10,
        "early_stopping_patience": 15,
        "early_stopping_metric": "icir",
        "val_ratio": 0.15,
        "random_seed": 42,
    },
    "output": {"output_dir": "outputs_super_encoder"},
}


# =============================================================================
# Cross-Sectional Dataset  (groups by date for cross-stock attention)
# =============================================================================

class CrossSectionalDataset(Dataset):
    """Groups (date, instrument) data by date for cross-stock attention.

    Each __getitem__ returns one date's worth of stocks (padded).
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, max_stocks: int = None):
        self.dates = sorted(X.index.get_level_values("date").unique())

        self.date_data = {}
        max_n = 0
        for d in self.dates:
            mask = X.index.get_level_values("date") == d
            xd = X[mask].values.astype(np.float32)
            yd = y[mask].values.astype(np.float32)
            self.date_data[d] = (xd, yd)
            max_n = max(max_n, xd.shape[0])

        self.max_stocks = max_stocks or max_n

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        d = self.dates[idx]
        x, y = self.date_data[d]
        n = x.shape[0]

        n_factors = x.shape[1]
        x_pad = np.zeros((self.max_stocks, n_factors), dtype=np.float32)
        y_pad = np.zeros(self.max_stocks, dtype=np.float32)
        mask = np.zeros(self.max_stocks, dtype=np.bool_)

        x_pad[:n] = x
        y_pad[:n] = y
        mask[:n] = True

        return (
            torch.from_numpy(x_pad),
            torch.from_numpy(y_pad),
            torch.from_numpy(mask),
            n,
        )


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(
    X: pd.DataFrame, y: pd.Series, config: dict, logger,
) -> Tuple[DataLoader, DataLoader, StandardScaler, int]:
    """Temporal split → scale → CrossSectionalDataset → DataLoaders."""
    tp = config.get("training", {})
    val_ratio = tp.get("val_ratio", 0.15)
    seed = tp.get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"    Filling {nan_count:,} NaN values with column median")
        X = X.fillna(X.median()).fillna(0)

    # Temporal split with gap buffer
    dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)
    val_start = int(n_dates * (1 - val_ratio))
    label_period = config.get("evaluation", {}).get("label_period", 10)
    train_end = max(0, val_start - label_period)
    train_dates = dates[:train_end]
    val_dates = dates[val_start:]

    train_mask = X.index.get_level_values("date").isin(train_dates)
    val_mask = X.index.get_level_values("date").isin(val_dates)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    logger.info(f"    Train: {len(X_train):>8,} samples  ({len(train_dates)} dates)")
    logger.info(f"    Val:   {len(X_val):>8,} samples  ({len(val_dates)} dates)")
    logger.info(f"    Gap:   {label_period} days dropped between train/val")

    # Scale (fit on train)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train.values),
        index=X_train.index, columns=X_train.columns,
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val.values),
        index=X_val.index, columns=X_val.columns,
    )

    # Compute max_stocks across both splits
    max_stocks_train = X_train_s.groupby(level="date").size().max()
    max_stocks_val = X_val_s.groupby(level="date").size().max()
    max_stocks = max(max_stocks_train, max_stocks_val)

    train_ds = CrossSectionalDataset(X_train_s, y_train, max_stocks)
    val_ds = CrossSectionalDataset(X_val_s, y_val, max_stocks)

    # batch_size = number of *dates* per batch (not samples)
    batch_size = tp.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler, max_stocks


# =============================================================================
# Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger,
) -> Dict[str, list]:
    """Training loop for SuperEncoder with IC-aware loss and ICIR-based early stopping."""
    tp = config.get("training", {})
    epochs = tp.get("epochs", 100)
    lr = tp.get("learning_rate", 0.00005)
    wd = tp.get("weight_decay", 0.01)
    patience = tp.get("early_stopping_patience", 15)
    warmup_epochs = tp.get("warmup_epochs", 10)
    es_metric = tp.get("early_stopping_metric", "icir")  # "icir" or "val_loss"

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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

    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_icir": []}

    logger.info(f"    Early stopping on: {es_metric}")
    logger.info(f"    {'Epoch':>6s}  │  {'Train Loss':>10s}  │  "
                f"{'Val Loss':>10s}  │  {'Val IC':>8s}  │  {'Val ICIR':>8s}  │  {'LR':>10s}")
    logger.info(f"    {'─'*6}──┼──{'─'*10}──┼──{'─'*10}──┼──{'─'*8}──┼──{'─'*8}──┼──{'─'*10}")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x_b, y_b, mask_b, ns in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(x_b, y_b, mask_b, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * mask_b.sum().item()
            train_n += mask_b.sum().item()
        train_loss = train_loss_sum / max(train_n, 1)

        # ── Validate (per-date IC tracking) ──
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        date_preds = {}   # date_idx → (preds, targets)
        with torch.no_grad():
            for x_b, y_b, mask_b, ns in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                mask_b = mask_b.to(device)
                loss = model.compute_loss(x_b, y_b, mask_b, criterion)
                val_loss_sum += loss.item() * mask_b.sum().item()
                val_n += mask_b.sum().item()
                pred = model(x_b, mask_b)
                # Collect per-date predictions for ICIR
                for b in range(x_b.size(0)):
                    valid = mask_b[b]
                    if valid.sum() < 20:
                        continue
                    p = pred[b][valid].cpu().numpy()
                    t = y_b[b][valid].cpu().numpy()
                    # Use a unique key per date within the epoch
                    d_key = len(date_preds)
                    date_preds[d_key] = (p, t)
        val_loss = val_loss_sum / max(val_n, 1)

        # Compute per-date Spearman ICs → ICIR
        date_ics = []
        for (p, t) in date_preds.values():
            ic = float(stats.spearmanr(t, p)[0])
            if not np.isnan(ic):
                date_ics.append(ic)
        date_ics = np.array(date_ics) if date_ics else np.array([0.0])
        val_ic = float(date_ics.mean())
        val_icir = float(date_ics.mean() / (date_ics.std() + 1e-8)) if len(date_ics) >= 2 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)
        history["val_icir"].append(val_icir)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"    {epoch+1:4d}/{epochs}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  {val_icir:+7.4f}  │  {cur_lr:10.2e}"
            )

        # Early stopping: use -ICIR (lower=better) or val_loss
        es_score = -val_icir if es_metric == "icir" else val_loss
        if early_stopping(es_score, model):
            logger.info(
                f"    {epoch+1:4d}/{epochs}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  {val_icir:+7.4f}  │  early stop"
            )
            early_stopping.load_best_model(model)
            if es_metric == "icir":
                logger.info(f"    ⤷ Restored best (val_icir={-early_stopping.best_score:.4f})")
            else:
                logger.info(f"    ⤷ Restored best (val_loss={early_stopping.best_score:.6f})")
            break

    if not early_stopping.early_stop:
        early_stopping.load_best_model(model)

    if es_metric == "icir":
        best_epoch = int(np.argmax(history["val_icir"])) + 1
    else:
        best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_ic = history["val_ic"][best_epoch - 1]
    best_icir = history["val_icir"][best_epoch - 1]
    gate_val = torch.sigmoid(model.cs_gate).item()
    logger.info(f"    Best epoch: {best_epoch}  │  "
                f"val_ic {best_ic:+.4f}  │  val_icir {best_icir:+.4f}  │  "
                f"cs_gate {gate_val:.4f}")
    return history


# =============================================================================
# OOS Prediction
# =============================================================================

def predict_oos(
    model: nn.Module,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    max_stocks: int,
    device: torch.device,
    logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """OOS prediction with cross-sectional batching."""
    X_clean = X_oos.fillna(X_oos.median()).fillna(0)
    X_scaled = pd.DataFrame(
        scaler.transform(X_clean.values),
        index=X_clean.index, columns=X_clean.columns,
    )

    model.eval()
    all_preds = []
    all_indices = []

    oos_max = X_scaled.groupby(level="date").size().max()
    max_stocks_oos = max(max_stocks, oos_max)

    dates = sorted(X_scaled.index.get_level_values("date").unique())
    for d in dates:
        date_mask = X_scaled.index.get_level_values("date") == d
        x = X_scaled[date_mask].values.astype(np.float32)
        n = x.shape[0]

        # Pad to (1, max_stocks_oos, n_factors)
        x_pad = np.zeros((1, max_stocks_oos, x.shape[1]), dtype=np.float32)
        x_pad[0, :n] = x
        m_pad = np.zeros((1, max_stocks_oos), dtype=np.bool_)
        m_pad[0, :n] = True

        with torch.no_grad():
            pred = model(
                torch.from_numpy(x_pad).to(device),
                torch.from_numpy(m_pad).to(device),
            )
        preds_np = pred[0, :n].cpu().numpy()
        all_preds.extend(preds_np)
        all_indices.extend(X_scaled[date_mask].index.tolist())

    pred_s = pd.Series(
        all_preds,
        index=pd.MultiIndex.from_tuples(all_indices, names=X_oos.index.names),
        name="prediction",
    )

    preds_arr = pred_s.values
    y_arr = y_oos.loc[pred_s.index].values

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_arr, preds_arr))),
        "mae": float(mean_absolute_error(y_arr, preds_arr)),
        "r2": float(r2_score(y_arr, preds_arr)),
        "ic": float(stats.spearmanr(y_arr, preds_arr)[0]),
        "n_samples": len(preds_arr),
        "n_dates": int(pred_s.index.get_level_values("date").nunique()),
    }
    logger.info(f"    OOS metrics  │  IC {metrics['ic']:+.4f}  │  "
                f"R² {metrics['r2']:.6f}  │  RMSE {metrics['rmse']:.6f}  │  "
                f"MAE {metrics['mae']:.6f}")
    return pred_s, metrics


# =============================================================================
# Process Snapshot
# =============================================================================

def process_snapshot(
    snapshot: str,
    config: dict,
    run_dir: Path,
    device: torch.device,
    logger,
    universe: Optional[set] = None,
) -> Dict[str, Any]:
    data_dir = Path(config.get("data_dir", "data/"))
    factor_keys = config.get("factor_keys", ["0", "1", "2"])
    cutoff = int(snapshot)
    oos_end = config.get("snapshot_oos_end", {}).get(snapshot, cutoff + 10000)

    snap_dir = run_dir / f"snapshot_{snapshot}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    log_banner(logger, f"SNAPSHOT {snapshot}  │  IS ≤ {cutoff}  │  OOS ({cutoff+1}, {oos_end}]")

    all_predictions = {}
    all_metrics = {}

    for key in factor_keys:
        log_section(logger, f"Key {key}  —  Snapshot {snapshot}")

        X_is, y_is = load_snapshot_data(data_dir, snapshot, key, "is", logger, universe)
        feature_names = list(X_is.columns)

        train_loader, val_loader, scaler, max_stocks = prepare_data(
            X_is, y_is, config, logger,
        )

        mc = config.get("model", {})
        model = SuperEncoder(
            input_size=len(feature_names),
            encoder_sizes=mc.get("encoder_sizes", [512, 256]),
            latent_dim=mc.get("latent_dim", 64),
            predictor_sizes=mc.get("predictor_sizes", [128]),
            dropout=mc.get("dropout", 0.3),
            recon_weight=mc.get("recon_weight", 0.1),
            use_residual=mc.get("use_residual", True),
            ic_loss_weight=mc.get("ic_loss_weight", 0.0),
            cs_n_heads=mc.get("cs_n_heads", 4),
            cs_n_layers=mc.get("cs_n_layers", 2),
            cs_dim_feedforward=mc.get("cs_dim_feedforward", 256),
            cs_dropout=mc.get("cs_dropout", None),
        ).to(device)
        logger.info(f"    Model: {count_parameters(model):,} params  │  "
                    f"max_stocks={max_stocks}  │  device={device}")

        history = train_model(model, train_loader, val_loader, config, device, logger)

        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, "oos", logger, universe)
        preds, metrics = predict_oos(
            model, X_oos, y_oos, scaler, max_stocks, device, logger,
        )

        all_predictions[key] = preds
        all_metrics[key] = metrics

        # Save artifacts
        model_path = snap_dir / f"super_key{key}_model.pt"
        scaler_path = snap_dir / f"super_key{key}_scaler.pkl"
        pred_path = snap_dir / f"super_key{key}_oos_predictions.parquet"
        hist_path = snap_dir / f"super_key{key}_training_history.json"

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": mc,
            "feature_names": feature_names,
            "max_stocks": max_stocks,
            "snapshot": snapshot, "key": key,
            "best_val_loss": float(min(history["val_loss"])),
            "epochs_trained": len(history["val_loss"]),
        }, model_path)
        joblib.dump(scaler, scaler_path)
        preds.to_frame().to_parquet(pred_path)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"    Saved → {model_path.name}, {scaler_path.name}, "
                    f"{pred_path.name}, {hist_path.name}")

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

    ensemble_path = snap_dir / "super_ensemble_oos.parquet"
    pred_ensemble.to_frame().to_parquet(ensemble_path)

    csv_path = snap_dir / "oos_predictions.csv"
    csv_df = pred_ensemble.reset_index()
    csv_df.columns = ["date", "code", "prediction"]
    csv_df = csv_df.sort_values(["date", "code"])
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"    CSV saved → {csv_path.name}  ({len(csv_df):,} rows)")

    lmt_results = run_lmt_api_evaluation(
        pred_ensemble, logger, config,
        label=f"Snapshot {snapshot}", output_dir=snap_dir,
    )

    snapshot_report = {
        "snapshot": snapshot, "cutoff_date": cutoff, "oos_end_date": oos_end,
        "oos_date_range": [int(dates.min()), int(dates.max())],
        "oos_n_dates": int(dates.nunique()),
        "oos_n_samples": len(pred_ensemble),
        "metrics_by_key": all_metrics,
        "lmt_api": lmt_results,
        "artifacts": {"directory": str(snap_dir),
                      "files": sorted(p.name for p in snap_dir.iterdir())},
        "timestamp": datetime.now().isoformat(),
    }
    report_path = snap_dir / "snapshot_report.json"
    with open(report_path, "w") as f:
        json.dump(snapshot_report, f, indent=2, default=str)

    logger.info(f"    Report saved → {report_path.name}")
    logger.info(f"  ✅ Snapshot {snapshot} complete")

    return {"report": snapshot_report, "pred_ensemble": pred_ensemble}


# =============================================================================
# Main
# =============================================================================

def main(config_path: str, device: str = None,
         snapshots_override=None, universe_override=None):

    from common import _deep_copy_dict, _deep_merge
    config = load_config(config_path, DEFAULT_CONFIG)

    if universe_override is not None:
        config["universe_file"] = universe_override

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(config.get("data_dir", "data/"))
    if not data_dir.is_absolute():
        config["data_dir"] = str(script_dir / data_dir)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = script_dir / config.get("output", {}).get("output_dir", "outputs_super_encoder")
    run_dir = output_base / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["output_dir"] = str(run_dir)

    (output_base / "latest_run.txt").write_text(str(run_dir))

    log_file = run_dir / "training.log"
    logger = setup_logging(log_file, name="SuperEncoder")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    snapshots = snapshots_override or config.get("snapshots", [])

    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log_banner(logger, "SuperEncoder  —  ROLLING PER-SNAPSHOT (v2)")
    log_kv(logger, "Config", config_path)
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Data directory", config["data_dir"])
    log_kv(logger, "Device", device)
    log_kv(logger, "Snapshots", " → ".join(snapshots))
    log_kv(logger, "Model config", json.dumps(config.get("model", {}), default=str))
    log_kv(logger, "Started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not LMT_API_AVAILABLE:
        logger.warning("  ⚠ lmt_data_api not installed — API evaluation will be skipped")

    universe_path = config.get("universe_file", "")
    universe = load_universe(universe_path, logger) if universe_path else None
    if universe:
        log_kv(logger, "Universe", f"{len(universe)} instruments")
    else:
        log_kv(logger, "Universe", "all instruments (no filter)")

    pipeline_start = datetime.now()

    # Phase 1
    log_banner(logger, "PHASE 1  │  ROLLING PER-SNAPSHOT TRAINING")
    all_reports, all_oos_preds = [], []
    for i, snapshot in enumerate(snapshots, 1):
        logger.info(f"\n  ▶ Snapshot {i}/{len(snapshots)}: {snapshot}")
        result = process_snapshot(snapshot, config, run_dir, device, logger, universe)
        all_reports.append(result["report"])
        all_oos_preds.append(result["pred_ensemble"])

    # Phase 2
    log_banner(logger, "PHASE 2  │  AGGREGATE OOS PREDICTIONS")
    combined_oos = pd.concat(all_oos_preds).sort_index()
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="last")]
    combined_oos.name = "prediction"
    dates = combined_oos.index.get_level_values("date")
    log_kv(logger, "Total samples", f"{len(combined_oos):,}")
    log_kv(logger, "Date range", f"{dates.min()} → {dates.max()}")

    agg_pred_path = run_dir / "super_ensemble_all_oos.parquet"
    combined_oos.to_frame().to_parquet(agg_pred_path)

    agg_csv = run_dir / "oos_predictions_all.csv"
    csv_df = combined_oos.reset_index()
    csv_df.columns = ["date", "code", "prediction"]
    csv_df.sort_values(["date", "code"]).to_csv(agg_csv, index=False)
    logger.info(f"    CSV saved → {agg_csv.name}  ({len(csv_df):,} rows)")

    aggregate_lmt = run_lmt_api_evaluation(
        combined_oos, logger, config, label="AGGREGATE", output_dir=run_dir,
    )

    # Phase 3
    log_banner(logger, "PHASE 3  │  SUMMARY")
    rolling_report = {
        "pipeline": "SuperEncoder Rolling (v2)",
        "run_directory": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": config,
        "per_snapshot": all_reports,
        "aggregate": {
            "n_samples": len(combined_oos),
            "n_dates": int(dates.nunique()),
            "date_range": [int(dates.min()), int(dates.max())],
            "lmt_api": aggregate_lmt,
        },
    }
    report_path = run_dir / "rolling_report.json"
    with open(report_path, "w") as f:
        json.dump(rolling_report, f, indent=2, default=str)

    # Summary table
    factor_keys = config.get("factor_keys", ["0", "1", "2"])
    ic_headers = "  ".join(f"{'IC(k'+k+')':>7s}" for k in factor_keys)
    hdr = (f"  {'Snapshot':>10s}  │  {'OOS Range':>21s}  │  "
           f"{'Days':>5s}  │  {'Samples':>9s}  │  {ic_headers}")
    logger.info("")
    logger.info(hdr)
    logger.info("  " + "─" * len(hdr.strip()))
    for rpt in all_reports:
        snap = rpt["snapshot"]
        dr = rpt.get("oos_date_range", ["?", "?"])
        nd = rpt.get("oos_n_dates", "?")
        ns = rpt.get("oos_n_samples", "?")
        ics = [f"{rpt.get('metrics_by_key',{}).get(k,{}).get('ic',0):+.4f}" for k in factor_keys]
        logger.info(f"  {snap:>10s}  │  {dr[0]} → {dr[1]}  │  "
                    f"{nd:>5}  │  {ns:>9,}  │  {'  '.join(ics)}")

    duration = datetime.now() - pipeline_start
    log_banner(logger, "PIPELINE COMPLETE")
    log_kv(logger, "Duration", str(duration).split(".")[0])
    log_kv(logger, "Run directory", run_dir)

    return rolling_report


if __name__ == "__main__":
    args = make_parser(
        "SuperEncoder rolling per-snapshot training (v2)",
        "configs/super_encoder.yaml",
    ).parse_args()

    main(args.config, args.device, args.snapshots, args.universe)
