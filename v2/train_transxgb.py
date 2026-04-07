"""
TransXGB — Rolling Per-Snapshot Training & Evaluation (v2)

Two-stage stacked model:
  Stage 1: Transformer backbone (self-attention over factors) trained via
           standard tabular pipeline (MSE, AdamW, cosine LR, early stop).
  Stage 2: XGBoost regressor fitted on Transformer residuals using original
           scaled features + Transformer latent embeddings.

Final prediction = Transformer pred + XGBoost residual pred.

Usage:
    python train_transxgb.py --config configs/transxgb.yaml
    python train_transxgb.py --config configs/transxgb.yaml --device cuda:0
    python train_transxgb.py --config configs/transxgb.yaml --snapshots 20181228
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

import xgboost as xgb

from common import (
    setup_logging, log_banner, log_kv, log_section,
    load_config, load_universe, load_snapshot_data,
    EarlyStopping, count_parameters,
    prepare_tabular_data, train_tabular_model,
    run_lmt_api_evaluation, LMT_API_AVAILABLE,
    make_parser, BASE_DEFAULT_CONFIG,
)
from models import TransXGBTransformer


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.3,
        "pool": "cls",
        "gradient_checkpointing": True,
        "xgb_features": "both",
        "xgb_params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 10,
            "early_stopping_rounds": 50,
            "tree_method": "hist",
            "random_state": 42,
        },
    },
    "training": {
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.00005,
        "weight_decay": 0.01,
        "warmup_epochs": 10,
        "early_stopping_patience": 15,
    },
    "output": {"output_dir": "outputs_transxgb"},
}


# =============================================================================
# Helpers
# =============================================================================

def prepare_xgb_arrays(
    X_is: pd.DataFrame,
    y_is: pd.Series,
    scaler: StandardScaler,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-split IS data with same temporal logic as prepare_tabular_data().

    Returns scaled numpy arrays: (X_train, y_train, X_val, y_val).
    Uses the already-fitted scaler from Stage 1.
    """
    tp = config.get("training", {})
    val_ratio = tp.get("val_ratio", 0.15)

    X = X_is.copy()
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        X = X.fillna(X.median()).fillna(0)

    dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)
    val_start = int(n_dates * (1 - val_ratio))
    label_period = config.get("evaluation", {}).get("label_period", 10)
    train_end = max(0, val_start - label_period)
    train_dates = dates[:train_end]
    val_dates = dates[val_start:]

    train_mask = X.index.get_level_values("date").isin(train_dates)
    val_mask = X.index.get_level_values("date").isin(val_dates)

    X_train, y_train = X[train_mask], y_is[train_mask]
    X_val, y_val = X[val_mask], y_is[val_mask]

    X_train_s = scaler.transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)

    logger.info(f"    XGB arrays: train {X_train_s.shape[0]:,} / val {X_val_s.shape[0]:,}")
    return X_train_s, y_train.values, X_val_s, y_val.values


def extract_features_and_predictions(
    model: TransXGBTransformer,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch inference: return (predictions, latents) as numpy arrays."""
    model.eval()
    all_preds, all_latents = [], []
    n = X_scaled.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_t = torch.FloatTensor(X_scaled[start:end]).to(device)
            pred, latent = model.forward_with_latent(x_t)
            all_preds.append(pred.cpu().numpy())
            all_latents.append(latent.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_latents)


def build_xgb_features(
    X_scaled: np.ndarray,
    latent: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Construct XGBoost input features based on mode."""
    if mode == "original":
        return X_scaled
    elif mode == "latent":
        return latent
    else:  # "both"
        return np.concatenate([X_scaled, latent], axis=1)


def train_xgb_stage(
    model: TransXGBTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Stage 2: compute residuals, construct features, train XGBoost."""
    mc = config.get("model", {})
    xgb_params = mc.get("xgb_params", {})
    xgb_feat_mode = mc.get("xgb_features", "both")

    log_section(logger, "Stage 2 — XGBoost residual training")

    # Compute transformer predictions + latent on train & val
    trans_pred_train, latent_train = extract_features_and_predictions(
        model, X_train, device,
    )
    trans_pred_val, latent_val = extract_features_and_predictions(
        model, X_val, device,
    )

    # Residuals
    residuals_train = y_train - trans_pred_train
    residuals_val = y_val - trans_pred_val

    logger.info(f"    Transformer train IC: "
                f"{stats.spearmanr(y_train, trans_pred_train)[0]:+.4f}")
    logger.info(f"    Transformer val   IC: "
                f"{stats.spearmanr(y_val, trans_pred_val)[0]:+.4f}")
    logger.info(f"    Residual std  (train): {residuals_train.std():.6f}")
    logger.info(f"    Residual std  (val):   {residuals_val.std():.6f}")

    # Construct XGBoost features
    X_xgb_train = build_xgb_features(X_train, latent_train, xgb_feat_mode)
    X_xgb_val = build_xgb_features(X_val, latent_val, xgb_feat_mode)

    logger.info(f"    XGB features mode: {xgb_feat_mode}  →  {X_xgb_train.shape[1]} cols")

    # Build and train XGBoost
    # Copy params so we don't mutate config; early_stopping_rounds goes to .fit()
    fit_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    early_stopping_rounds = xgb_params.get("early_stopping_rounds", 50)

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
        **fit_params,
    )

    xgb_model.fit(
        X_xgb_train, residuals_train,
        eval_set=[(X_xgb_val, residuals_val)],
        verbose=False,
    )

    best_iter = xgb_model.best_iteration
    best_score = xgb_model.best_score

    # Combined predictions on validation set
    xgb_residual_val = xgb_model.predict(X_xgb_val)
    combined_val = trans_pred_val + xgb_residual_val
    combined_ic = float(stats.spearmanr(y_val, combined_val)[0])

    logger.info(f"    XGB best iteration: {best_iter}  │  best val RMSE: {best_score:.6f}")
    logger.info(f"    Combined val IC:    {combined_ic:+.4f}")

    xgb_meta = {
        "best_iteration": int(best_iter),
        "best_score": float(best_score),
        "n_features": X_xgb_train.shape[1],
        "xgb_features_mode": xgb_feat_mode,
        "residual_std_train": float(residuals_train.std()),
        "residual_std_val": float(residuals_val.std()),
        "transformer_val_ic": float(stats.spearmanr(y_val, trans_pred_val)[0]),
        "combined_val_ic": combined_ic,
    }

    return xgb_model, xgb_meta


def predict_oos_transxgb(
    transformer: TransXGBTransformer,
    xgb_model: xgb.XGBRegressor,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """OOS prediction: transformer_pred + xgb_residual_pred."""
    mc = config.get("model", {})
    xgb_feat_mode = mc.get("xgb_features", "both")

    X_clean = X_oos.fillna(X_oos.median()).fillna(0)
    X_scaled = scaler.transform(X_clean.values)

    # Transformer prediction + latent
    trans_pred, latent = extract_features_and_predictions(
        transformer, X_scaled, device,
    )

    # XGBoost residual prediction
    X_xgb = build_xgb_features(X_scaled, latent, xgb_feat_mode)
    xgb_residual = xgb_model.predict(X_xgb)

    # Combined
    final_pred = trans_pred + xgb_residual

    pred_s = pd.Series(final_pred, index=X_oos.index, name="prediction")

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_oos, final_pred))),
        "mae": float(mean_absolute_error(y_oos, final_pred)),
        "r2": float(r2_score(y_oos, final_pred)),
        "ic": float(stats.spearmanr(y_oos, final_pred)[0]),
        "ic_transformer_only": float(stats.spearmanr(y_oos, trans_pred)[0]),
        "n_samples": len(final_pred),
        "n_dates": int(X_oos.index.get_level_values("date").nunique()),
    }

    logger.info(f"    OOS metrics (TransXGB)  │  IC {metrics['ic']:+.4f}  │  "
                f"R² {metrics['r2']:.6f}  │  RMSE {metrics['rmse']:.6f}")
    logger.info(f"    OOS metrics (Trans only) │  IC {metrics['ic_transformer_only']:+.4f}")

    return pred_s, metrics


# =============================================================================
# Per-Snapshot Pipeline
# =============================================================================

def process_snapshot(
    snapshot: str,
    config: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    universe: Optional[set],
) -> Dict[str, Any]:
    """Process one snapshot: Stage 1 (Transformer) → Stage 2 (XGBoost) → OOS."""
    data_dir = Path(config.get("data_dir", "data/"))
    factor_keys = config.get("factor_keys", ["0", "1", "2"])
    mc = config.get("model", {})
    cutoff = int(snapshot)
    oos_end = config.get("snapshot_oos_end", {}).get(snapshot, cutoff + 10000)

    snap_dir = run_dir / f"snapshot_{snapshot}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    log_banner(logger, f"SNAPSHOT {snapshot}  │  IS ≤ {cutoff}  │  OOS ({cutoff+1}, {oos_end}]")

    all_predictions = {}
    all_metrics = {}

    for key in factor_keys:
        log_section(logger, f"Key {key}  —  Snapshot {snapshot}")

        # ── Load IS data ──
        X_is, y_is = load_snapshot_data(data_dir, snapshot, key, "is", logger, universe)
        feature_names = list(X_is.columns)

        # ── Stage 1: Transformer ──
        log_section(logger, "Stage 1 — Transformer training")

        train_loader, val_loader, scaler = prepare_tabular_data(
            X_is, y_is, config, logger,
        )

        model = TransXGBTransformer(
            input_size=len(feature_names),
            d_model=mc.get("d_model", 64),
            n_heads=mc.get("n_heads", 4),
            n_layers=mc.get("n_layers", 2),
            dim_feedforward=mc.get("dim_feedforward", 128),
            dropout=mc.get("dropout", 0.3),
            pool=mc.get("pool", "cls"),
        ).to(device)

        if mc.get("gradient_checkpointing", False):
            model.enable_gradient_checkpointing()

        n_params = count_parameters(model)
        logger.info(f"    Transformer: {n_params:,} params  │  "
                    f"d_model={mc.get('d_model', 64)}  heads={mc.get('n_heads', 4)}  "
                    f"layers={mc.get('n_layers', 2)}  │  device={device}")

        history = train_tabular_model(model, train_loader, val_loader, config, device, logger)

        # ── Stage 2: XGBoost on residuals ──
        X_train_s, y_train_np, X_val_s, y_val_np = prepare_xgb_arrays(
            X_is, y_is, scaler, config, logger,
        )

        xgb_model, xgb_meta = train_xgb_stage(
            model, X_train_s, y_train_np, X_val_s, y_val_np,
            config, device, logger,
        )

        # ── OOS prediction ──
        log_section(logger, f"OOS Prediction — Key {key}")
        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, "oos", logger, universe)
        preds, metrics = predict_oos_transxgb(
            model, xgb_model, X_oos, y_oos, scaler, config, device, logger,
        )

        all_predictions[key] = preds
        all_metrics[key] = metrics

        # ── Save artifacts ──
        prefix = "transxgb"

        # Transformer
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": mc,
            "feature_names": feature_names,
            "snapshot": snapshot,
            "key": key,
            "best_val_loss": float(min(history["val_loss"])),
            "best_val_ic": float(history["val_ic"][int(np.argmin(history["val_loss"]))]),
            "epochs_trained": len(history["val_loss"]),
        }, snap_dir / f"{prefix}_key{key}_transformer.pt")

        # XGBoost
        xgb_path = snap_dir / f"{prefix}_key{key}_xgb.json"
        xgb_model.save_model(str(xgb_path))

        # Scaler
        joblib.dump(scaler, snap_dir / f"{prefix}_key{key}_scaler.pkl")

        # Predictions
        preds.to_frame().to_parquet(snap_dir / f"{prefix}_key{key}_oos_predictions.parquet")

        # Training history + XGBoost metadata
        combined_history = {
            "transformer": history,
            "xgboost": xgb_meta,
        }
        with open(snap_dir / f"{prefix}_key{key}_training_history.json", "w") as f:
            json.dump(combined_history, f, indent=2)

        logger.info(f"    Saved artifacts for key {key}")

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

    pred_ensemble.to_frame().to_parquet(snap_dir / "transxgb_ensemble_oos.parquet")

    csv_path = snap_dir / "oos_predictions.csv"
    csv_df = pred_ensemble.reset_index()
    csv_df.columns = ["date", "code", "prediction"]
    csv_df = csv_df.sort_values(["date", "code"])
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"    CSV saved → {csv_path.name}  ({len(csv_df):,} rows)")

    # LMT API evaluation
    lmt_results = run_lmt_api_evaluation(
        pred_ensemble, logger, config, label=f"Snapshot {snapshot}",
        output_dir=snap_dir,
    )

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
    logger.info(f"  Snapshot {snapshot} complete  ({len(list(snap_dir.iterdir()))} files)")

    return {"report": snapshot_report, "pred_ensemble": pred_ensemble}


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    config_path: str,
    device: str = None,
    snapshots_override: List[str] = None,
    universe_override: str = None,
):
    """End-to-end TransXGB rolling per-snapshot pipeline."""
    config = load_config(config_path, DEFAULT_CONFIG)

    if universe_override is not None:
        config["universe_file"] = universe_override

    script_dir = Path(__file__).resolve().parent

    # Resolve data_dir
    data_dir = Path(config.get("data_dir", "data/"))
    if not data_dir.is_absolute():
        config["data_dir"] = str(script_dir / data_dir)

    # Timestamped run directory
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = script_dir / config.get("output", {}).get("output_dir", "outputs_transxgb")
    run_dir = output_base / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["output_dir"] = str(run_dir)

    latest_marker = output_base / "latest_run.txt"
    latest_marker.write_text(str(run_dir))

    log_file = run_dir / "training.log"
    logger = setup_logging(log_file, name="TransXGB")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    snapshots = snapshots_override or config.get("snapshots", ["20181228", "20191231", "20201231"])
    tp = config.get("training", {})
    mc = config.get("model", {})

    # Save config
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ── Header ──
    log_banner(logger, "TransXGB  —  ROLLING PER-SNAPSHOT PIPELINE (v2)")
    log_kv(logger, "Config", config_path)
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Data directory", config["data_dir"])
    log_kv(logger, "Device", device)
    log_kv(logger, "Snapshots", " → ".join(snapshots))
    log_kv(logger, "Factor keys", ", ".join(config.get("factor_keys", ["0", "1", "2"])))
    log_kv(logger, "Transformer", f"d={mc.get('d_model', 64)} h={mc.get('n_heads', 4)} "
                                   f"L={mc.get('n_layers', 2)} ff={mc.get('dim_feedforward', 128)}")
    log_kv(logger, "XGB features", mc.get("xgb_features", "both"))
    log_kv(logger, "XGB estimators", mc.get("xgb_params", {}).get("n_estimators", 500))
    log_kv(logger, "Batch size", tp.get("batch_size"))
    log_kv(logger, "Learning rate", tp.get("learning_rate"))
    log_kv(logger, "Max epochs", tp.get("epochs"))
    log_kv(logger, "Early stop", f"patience={tp.get('early_stopping_patience')}")
    log_kv(logger, "Started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not LMT_API_AVAILABLE:
        logger.warning("  lmt_data_api not installed — API evaluation will be skipped")

    # Universe
    universe_path = config.get("universe_file", "")
    universe = load_universe(universe_path, logger) if universe_path else None
    if universe:
        log_kv(logger, "Universe", f"{len(universe)} instruments from {universe_path}")
    else:
        log_kv(logger, "Universe", "all instruments (no filter)")

    pipeline_start = datetime.now()

    # ── Phase 1: Per-snapshot training ──
    log_banner(logger, "PHASE 1  │  ROLLING PER-SNAPSHOT TRAINING")

    all_reports = []
    all_oos_preds = []

    for i, snapshot in enumerate(snapshots, 1):
        logger.info(f"\n  ▶ Snapshot {i}/{len(snapshots)}: {snapshot}")
        result = process_snapshot(snapshot, config, run_dir, device, logger, universe)
        all_reports.append(result["report"])
        all_oos_preds.append(result["pred_ensemble"])

    # ── Phase 2: Aggregate ──
    log_banner(logger, "PHASE 2  │  AGGREGATE OOS PREDICTIONS")

    combined_oos = pd.concat(all_oos_preds).sort_index()
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="last")]
    combined_oos.name = "prediction"

    dates = combined_oos.index.get_level_values("date")
    log_kv(logger, "Total samples", f"{len(combined_oos):,}")
    log_kv(logger, "Total dates", dates.nunique())
    log_kv(logger, "Date range", f"{dates.min()} → {dates.max()}")

    agg_pred_path = run_dir / "transxgb_ensemble_all_oos.parquet"
    combined_oos.to_frame().to_parquet(agg_pred_path)

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

    # ── Phase 3: Summary ──
    log_banner(logger, "PHASE 3  │  SUMMARY")

    rolling_report = {
        "pipeline": "TransXGB Rolling Per-Snapshot (v2)",
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

    # Summary table
    factor_keys = config.get("factor_keys", ["0", "1", "2"])
    ic_headers = "  ".join(f"{'IC(k'+k+')':>7s}" for k in factor_keys)
    ic_trans_headers = "  ".join(f"{'TrIC(k'+k+')':>9s}" for k in factor_keys)
    hdr = (f"  {'Snapshot':>10s}  │  {'OOS Range':>21s}  │  "
           f"{'Days':>5s}  │  {ic_headers}  │  {ic_trans_headers}")
    logger.info(hdr)
    logger.info("  " + "─" * len(hdr.strip()))

    for rpt in all_reports:
        snap = rpt["snapshot"]
        dr = rpt.get("oos_date_range", ["?", "?"])
        nd = rpt.get("oos_n_dates", "?")
        ics, tr_ics = [], []
        for k in factor_keys:
            m = rpt.get("metrics_by_key", {}).get(k, {})
            ics.append(f"{m.get('ic', 0):+.4f}")
            tr_ics.append(f"{m.get('ic_transformer_only', 0):+.4f}")
        logger.info(
            f"  {snap:>10s}  │  {dr[0]} → {dr[1]}  │  "
            f"{nd:>5}  │  {'  '.join(ics)}  │  {'  '.join(tr_ics)}"
        )

    logger.info("")
    log_kv(logger, "Run directory", run_dir)
    log_kv(logger, "Report", report_path.name)
    log_kv(logger, "All-OOS predictions", agg_pred_path.name)
    log_kv(logger, "Training log", log_file.name)

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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = make_parser(
        "TransXGB rolling per-snapshot training (v2)",
        "configs/transxgb.yaml",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        device=args.device,
        snapshots_override=args.snapshots,
        universe_override=args.universe,
    )
