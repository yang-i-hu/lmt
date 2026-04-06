"""
Temporal Transformer — Rolling Per-Snapshot Training & Evaluation (v2)

Uses a sliding window of historical factor values per stock with causal
self-attention to model temporal dynamics: momentum decay, volatility
clustering, regime shifts.

⚠ Requires sliding-window data preparation.  The last ``window_size``
  dates of IS data are used as history context for the first OOS dates.

Usage:
    python train_temporal.py --config configs/temporal.yaml
    python train_temporal.py --config configs/temporal.yaml --device cuda:2
    python train_temporal.py --config configs/temporal.yaml --universe ../universe.txt
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
    make_parser,
)
from models import TemporalTransformer
import yaml


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.3,
        "window_size": 20,
        "pool": "last",
    },
    "training": {
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.00005,
        "weight_decay": 0.01,
        "warmup_epochs": 10,
        "early_stopping_patience": 15,
        "val_ratio": 0.15,
        "random_seed": 42,
    },
    "output": {"output_dir": "outputs_temporal"},
}


# =============================================================================
# Temporal Dataset
# =============================================================================

class TemporalWindowDataset(Dataset):
    """Creates (window, target) pairs from panel data.

    For each instrument with enough history, yields sliding windows of
    shape ``(window_size, n_factors)`` and the label at the last date.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int = 20,
    ):
        self.window_size = window_size
        self.windows = []   # list of (np.array (W, F), float target)
        self.indices = []   # (date, instrument) for each sample

        instruments = X.index.get_level_values("instrument").unique()
        for inst in instruments:
            mask = X.index.get_level_values("instrument") == inst
            x_inst = X[mask].sort_index(level="date")
            y_inst = y[mask].sort_index(level="date")

            if len(x_inst) <= window_size:
                continue

            vals = x_inst.values.astype(np.float32)   # (T, F)
            labels = y_inst.values.astype(np.float32)  # (T,)
            date_vals = x_inst.index.get_level_values("date").values

            for t in range(window_size, len(vals)):
                w = vals[t - window_size : t]   # (W, F)
                self.windows.append((w, labels[t]))
                self.indices.append((date_vals[t], inst))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w, target = self.windows[idx]
        return torch.from_numpy(w), torch.tensor(target, dtype=torch.float32)


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_temporal_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    logger,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Temporal split → scale → TemporalWindowDataset → DataLoaders."""
    tp = config.get("training", {})
    mc = config.get("model", {})
    val_ratio = tp.get("val_ratio", 0.15)
    batch_size = tp.get("batch_size", 2048)
    window_size = mc.get("window_size", 20)
    seed = tp.get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"    Filling {nan_count:,} NaN values with column median")
        X = X.fillna(X.median()).fillna(0)

    # Temporal split — need enough val dates for at least one window
    dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)
    val_start = int(n_dates * (1 - val_ratio))
    # Gap buffer: skip label_period days between train and val to
    # prevent forward-return overlap (data leakage)
    label_period = config.get("evaluation", {}).get("label_period", 10)
    train_end = max(0, val_start - label_period)
    # Ensure validation has history: push train/val boundary back by window_size
    # so val instruments can have windows from the val portion
    history_start = max(0, val_start - window_size)

    train_dates = dates[:train_end]
    val_dates = dates[history_start:]  # includes overlap for history context

    train_mask = X.index.get_level_values("date").isin(train_dates)
    val_overlap_mask = X.index.get_level_values("date").isin(val_dates)

    X_train, y_train = X[train_mask], y[train_mask]
    # Val includes overlap dates for window history but targets are only val dates
    X_val_full, y_val_full = X[val_overlap_mask], y[val_overlap_mask]

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train.values),
        index=X_train.index, columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_full.values),
        index=X_val_full.index, columns=X_val_full.columns,
    )

    train_ds = TemporalWindowDataset(X_train_scaled, y_train, window_size)
    val_ds = TemporalWindowDataset(X_val_scaled, y_val_full, window_size)

    logger.info(f"    Train: {len(train_ds):>8,} windows  "
                f"({len(train_dates)} dates, W={window_size})")
    logger.info(f"    Val:   {len(val_ds):>8,} windows  "
                f"(with {window_size} overlap dates for history)")
    logger.info(f"    Gap:   {label_period} days dropped between train/val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


# =============================================================================
# Training
# =============================================================================

def train_temporal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger,
) -> Dict[str, list]:
    tp = config.get("training", {})
    epochs = tp.get("epochs", 100)
    lr = tp.get("learning_rate", 0.00005)
    wd = tp.get("weight_decay", 0.01)
    patience = tp.get("early_stopping_patience", 15)
    warmup_epochs = tp.get("warmup_epochs", 10)

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

    logger.info(f"    {'Epoch':>6s}  │  {'Train Loss':>10s}  │  "
                f"{'Val Loss':>10s}  │  {'Val IC':>8s}  │  {'LR':>10s}")
    logger.info(f"    {'─'*6}──┼──{'─'*10}──┼──{'─'*10}──┼──{'─'*8}──┼──{'─'*10}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                val_loss += criterion(pred, y_b).item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())
        val_loss /= max(len(val_loader), 1)

        val_ic = float(stats.spearmanr(all_targets, all_preds)[0]) if len(all_preds) > 2 else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"    {epoch+1:4d}/{epochs}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  {cur_lr:10.2e}"
            )

        if early_stopping(val_loss, model):
            logger.info(
                f"    {epoch+1:4d}/{epochs}  │  {train_loss:10.6f}  │  "
                f"{val_loss:10.6f}  │  {val_ic:+7.4f}  │  early stop"
            )
            early_stopping.load_best_model(model)
            logger.info(f"    ⤷ Restored best (val_loss={early_stopping.best_score:.6f})")
            break

    if not early_stopping.early_stop:
        early_stopping.load_best_model(model)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_ic = history["val_ic"][best_epoch - 1]
    logger.info(f"    Best epoch: {best_epoch}  │  "
                f"val_loss {early_stopping.best_score:.6f}  │  val_ic {best_ic:+.4f}")
    return history


# =============================================================================
# OOS Prediction
# =============================================================================

def predict_temporal_oos(
    model: nn.Module,
    X_is: pd.DataFrame,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    window_size: int,
    device: torch.device,
    logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Predict OOS with temporal windows.

    Uses the tail of IS data as history context for the first OOS dates.
    """
    # Scale
    X_is_clean = X_is.fillna(X_is.median()).fillna(0)
    X_oos_clean = X_oos.fillna(X_oos.median()).fillna(0)

    X_is_s = pd.DataFrame(
        scaler.transform(X_is_clean.values),
        index=X_is_clean.index, columns=X_is_clean.columns,
    )
    X_oos_s = pd.DataFrame(
        scaler.transform(X_oos_clean.values),
        index=X_oos_clean.index, columns=X_oos_clean.columns,
    )

    # Combine last window_size IS dates + all OOS dates for windowing
    is_dates = X_is_s.index.get_level_values("date").unique().sort_values()
    if len(is_dates) > window_size:
        tail_dates = is_dates[-window_size:]
        is_tail_mask = X_is_s.index.get_level_values("date").isin(tail_dates)
        X_combined = pd.concat([X_is_s[is_tail_mask], X_oos_s])
        y_combined = pd.concat([
            pd.Series(np.nan, index=X_is_s[is_tail_mask].index),
            y_oos,
        ])
    else:
        X_combined = pd.concat([X_is_s, X_oos_s])
        y_combined = pd.concat([
            pd.Series(np.nan, index=X_is_s.index),
            y_oos,
        ])

    # Build windows for OOS dates only
    oos_dates_set = set(X_oos_s.index.get_level_values("date").unique())
    instruments = X_combined.index.get_level_values("instrument").unique()

    all_preds = {}
    model.eval()

    for inst in instruments:
        mask = X_combined.index.get_level_values("instrument") == inst
        x_inst = X_combined[mask].sort_index(level="date")

        if len(x_inst) <= window_size:
            continue

        vals = x_inst.values.astype(np.float32)
        inst_dates = x_inst.index.get_level_values("date").values

        # Find indices that correspond to OOS dates
        for t in range(window_size, len(vals)):
            d = inst_dates[t]
            if d not in oos_dates_set:
                continue

            w = vals[t - window_size : t]  # (W, F)
            w_t = torch.from_numpy(w).unsqueeze(0).to(device)  # (1, W, F)

            with torch.no_grad():
                pred = model(w_t).item()

            all_preds[(d, inst)] = pred

    if not all_preds:
        logger.warning("    No OOS predictions generated (insufficient history)")
        return pd.Series(dtype=float), {}

    pred_s = pd.Series(all_preds, name="prediction")
    pred_s.index = pd.MultiIndex.from_tuples(pred_s.index, names=["date", "instrument"])
    pred_s = pred_s.sort_index()

    # Align with y_oos
    common_idx = pred_s.index.intersection(y_oos.index)
    pred_aligned = pred_s.loc[common_idx]
    y_aligned = y_oos.loc[common_idx]

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_aligned, pred_aligned))),
        "mae": float(mean_absolute_error(y_aligned, pred_aligned)),
        "r2": float(r2_score(y_aligned, pred_aligned)),
        "ic": float(stats.spearmanr(y_aligned, pred_aligned)[0]),
        "n_samples": len(pred_aligned),
        "n_dates": int(pred_aligned.index.get_level_values("date").nunique()),
        "n_skipped": len(y_oos) - len(pred_aligned),
    }
    logger.info(f"    OOS metrics  │  IC {metrics['ic']:+.4f}  │  "
                f"R² {metrics['r2']:.6f}  │  RMSE {metrics['rmse']:.6f}  │  "
                f"MAE {metrics['mae']:.6f}  │  "
                f"({metrics['n_skipped']} samples skipped — insufficient history)")
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
    mc = config.get("model", {})
    window_size = mc.get("window_size", 20)
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

        train_loader, val_loader, scaler = prepare_temporal_data(
            X_is, y_is, config, logger,
        )

        model = TemporalTransformer(
            n_factors=len(feature_names),
            d_model=mc.get("d_model", 128),
            n_heads=mc.get("n_heads", 4),
            n_layers=mc.get("n_layers", 2),
            dim_feedforward=mc.get("dim_feedforward", 512),
            dropout=mc.get("dropout", 0.3),
            window_size=window_size,
            pool=mc.get("pool", "last"),
        ).to(device)
        logger.info(f"    Model: {count_parameters(model):,} params  │  "
                    f"window={window_size}  │  device={device}")

        history = train_temporal_model(model, train_loader, val_loader, config, device, logger)

        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, "oos", logger, universe)
        preds, metrics = predict_temporal_oos(
            model, X_is, X_oos, y_oos, scaler, window_size, device, logger,
        )

        all_predictions[key] = preds
        all_metrics[key] = metrics

        # Save artifacts
        model_path = snap_dir / f"temporal_key{key}_model.pt"
        scaler_path = snap_dir / f"temporal_key{key}_scaler.pkl"
        pred_path = snap_dir / f"temporal_key{key}_oos_predictions.parquet"
        hist_path = snap_dir / f"temporal_key{key}_training_history.json"

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": mc,
            "feature_names": feature_names,
            "window_size": window_size,
            "snapshot": snapshot, "key": key,
            "best_val_loss": float(min(history["val_loss"])),
            "epochs_trained": len(history["val_loss"]),
        }, model_path)
        joblib.dump(scaler, scaler_path)
        if len(preds) > 0:
            preds.to_frame().to_parquet(pred_path)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"    Saved → {model_path.name}, {scaler_path.name}, "
                    f"{pred_path.name}, {hist_path.name}")

        del train_loader, val_loader, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Ensemble ──
    log_section(logger, f"Ensemble  —  Snapshot {snapshot}")

    # Align predictions across keys
    valid_preds = {k: v for k, v in all_predictions.items() if len(v) > 0}
    if valid_preds:
        pred_df = pd.DataFrame(valid_preds)
        pred_ensemble = pred_df.mean(axis=1)
    else:
        pred_ensemble = pd.Series(dtype=float)
    pred_ensemble.name = "prediction"

    if len(pred_ensemble) > 0:
        dates = pred_ensemble.index.get_level_values("date")
        logger.info(f"    Ensemble: {len(pred_ensemble):,} samples, "
                    f"{dates.nunique()} dates ({dates.min()} → {dates.max()})")

        ensemble_path = snap_dir / "temporal_ensemble_oos.parquet"
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
    else:
        lmt_results = {"status": "skipped", "reason": "no predictions"}
        logger.warning("    No predictions — skipping LMT evaluation")
        dates = pd.Index([0])

    snapshot_report = {
        "snapshot": snapshot, "cutoff_date": cutoff, "oos_end_date": oos_end,
        "oos_date_range": [int(dates.min()), int(dates.max())] if len(pred_ensemble) > 0 else [],
        "oos_n_dates": int(dates.nunique()) if len(pred_ensemble) > 0 else 0,
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

    config = load_config(config_path, DEFAULT_CONFIG)

    if universe_override is not None:
        config["universe_file"] = universe_override

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(config.get("data_dir", "data/"))
    if not data_dir.is_absolute():
        config["data_dir"] = str(script_dir / data_dir)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = script_dir / config.get("output", {}).get("output_dir", "outputs_temporal")
    run_dir = output_base / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["output_dir"] = str(run_dir)

    (output_base / "latest_run.txt").write_text(str(run_dir))

    log_file = run_dir / "training.log"
    logger = setup_logging(log_file, name="Temporal")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    snapshots = snapshots_override or config.get("snapshots", [])

    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log_banner(logger, "Temporal Transformer  —  ROLLING PER-SNAPSHOT (v2)")
    log_kv(logger, "Config", config_path)
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Data directory", config["data_dir"])
    log_kv(logger, "Device", device)
    log_kv(logger, "Snapshots", " → ".join(snapshots))
    mc = config.get("model", {})
    log_kv(logger, "Window size", mc.get("window_size", 20))
    log_kv(logger, "Model config", json.dumps(mc, default=str))
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
        if len(result["pred_ensemble"]) > 0:
            all_oos_preds.append(result["pred_ensemble"])

    # Phase 2
    log_banner(logger, "PHASE 2  │  AGGREGATE OOS PREDICTIONS")
    if all_oos_preds:
        combined_oos = pd.concat(all_oos_preds).sort_index()
        combined_oos = combined_oos[~combined_oos.index.duplicated(keep="last")]
        combined_oos.name = "prediction"
        dates = combined_oos.index.get_level_values("date")
        log_kv(logger, "Total samples", f"{len(combined_oos):,}")
        log_kv(logger, "Date range", f"{dates.min()} → {dates.max()}")

        agg_pred_path = run_dir / "temporal_ensemble_all_oos.parquet"
        combined_oos.to_frame().to_parquet(agg_pred_path)

        agg_csv = run_dir / "oos_predictions_all.csv"
        csv_df = combined_oos.reset_index()
        csv_df.columns = ["date", "code", "prediction"]
        csv_df.sort_values(["date", "code"]).to_csv(agg_csv, index=False)
        logger.info(f"    CSV saved → {agg_csv.name}  ({len(csv_df):,} rows)")

        aggregate_lmt = run_lmt_api_evaluation(
            combined_oos, logger, config, label="AGGREGATE", output_dir=run_dir,
        )
    else:
        logger.warning("    No OOS predictions across any snapshot")
        aggregate_lmt = {"status": "skipped", "reason": "no predictions"}
        combined_oos = pd.Series(dtype=float)
        dates = pd.Index([0])

    # Phase 3
    log_banner(logger, "PHASE 3  │  SUMMARY")
    rolling_report = {
        "pipeline": "Temporal Transformer Rolling (v2)",
        "run_directory": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": config,
        "per_snapshot": all_reports,
        "aggregate": {
            "n_samples": len(combined_oos),
            "n_dates": int(dates.nunique()) if len(combined_oos) > 0 else 0,
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
        ns = rpt.get("oos_n_samples", 0)
        ics = [f"{rpt.get('metrics_by_key',{}).get(k,{}).get('ic',0):+.4f}" for k in factor_keys]
        logger.info(f"  {snap:>10s}  │  {dr[0] if dr else '?'} → {dr[1] if dr else '?'}  │  "
                    f"{nd:>5}  │  {ns:>9,}  │  {'  '.join(ics)}")

    duration = datetime.now() - pipeline_start
    log_banner(logger, "PIPELINE COMPLETE")
    log_kv(logger, "Duration", str(duration).split(".")[0])
    log_kv(logger, "Run directory", run_dir)

    return rolling_report


if __name__ == "__main__":
    args = make_parser(
        "Temporal Transformer rolling per-snapshot training (v2)",
        "configs/temporal.yaml",
    ).parse_args()

    main(args.config, args.device, args.snapshots, args.universe)
