"""
ElasticNet Factor Reweight — Rolling Per-Snapshot Training & Evaluation (v2)

Same rolling per-snapshot pipeline as train_dnn.py but using ElasticNet
as the model, with coefficient analysis.

  For each snapshot folder independently:
    1. Load IS data → train 3 ElasticNet models (keys 0, 1, 2)
    2. Analyze coefficients (sparsity, top features)
    3. Load OOS data → predict → compute ensemble
    4. Run LMT API evaluation on that snapshot's OOS
    5. Save per-snapshot artifacts

  After all snapshots:
    6. Concatenate all OOS predictions chronologically
    7. Run aggregate LMT API evaluation
    8. Generate rolling backtest report

⚠️ Data from different snapshot folders is NEVER merged for training.
   Each snapshot is a self-contained training + evaluation unit.

Usage:
    python train_elasticnet.py --config config_elasticnet.yaml
    python train_elasticnet.py --config config_elasticnet.yaml --alpha 0.01 --l1_ratio 0.7
    python train_elasticnet.py --config config_elasticnet.yaml --snapshots 20181228 20191231
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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

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
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "max_iter": 2000,
        "tol": 0.0001,
        "fit_intercept": True
    },
    "training": {
        "random_seed": 42
    },
    "evaluation": {
        "label_period": 10,
        "alpha": 1
    },
    "output": {
        "output_dir": "outputs_elasticnet"
    }
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path = None, name: str = "ElasticNetRolling") -> logging.Logger:
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
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Prepare IS data: fill NaN, scale features."""
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with median")
        X = X.fillna(X.median()).fillna(0)

    feature_names = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, y.values, scaler, feature_names


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    logger: logging.Logger
) -> ElasticNet:
    """Train an ElasticNet model."""
    model_params = config.get('model', {})
    alpha = model_params.get('alpha', 0.01)
    l1_ratio = model_params.get('l1_ratio', 0.5)
    max_iter = model_params.get('max_iter', 2000)
    tol = model_params.get('tol', 0.0001)
    fit_intercept = model_params.get('fit_intercept', True)
    random_seed = config.get('training', {}).get('random_seed', 42)

    logger.info(f"  ElasticNet(alpha={alpha}, l1_ratio={l1_ratio}, max_iter={max_iter})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            random_state=random_seed
        )
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

    n_nonzero = int(np.sum(model.coef_ != 0))
    n_features = len(model.coef_)
    sparsity = 1 - (n_nonzero / max(1, n_features))

    logger.info(f"  Trained in {training_time:.2f}s — "
                f"Non-zero: {n_nonzero}/{n_features} (sparsity: {sparsity:.1%})")

    return model


def analyze_coefficients(
    model: ElasticNet,
    feature_names: List[str],
    logger: logging.Logger,
    top_n: int = 20
) -> Dict[str, Any]:
    """Analyze model coefficients."""
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    n_nonzero = int((coef_df['coefficient'] != 0).sum())
    n_positive = int((coef_df['coefficient'] > 0).sum())
    n_negative = int((coef_df['coefficient'] < 0).sum())

    logger.info(f"  Coefficients: {n_nonzero} non-zero ({n_positive}+, {n_negative}-)")

    if top_n > 0:
        logger.info(f"  Top {min(top_n, n_nonzero)} features:")
        for _, row in coef_df.head(top_n).iterrows():
            if row['coefficient'] != 0:
                logger.info(f"    {str(row['feature'])[:25]:25s}: {row['coefficient']:+.6f}")

    return {
        'n_total': len(coef_df),
        'n_nonzero': n_nonzero,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'sparsity': float(1 - n_nonzero / max(1, len(coef_df))),
        'intercept': float(model.intercept_),
        'top_features': coef_df.head(top_n)[['feature', 'coefficient']].to_dict('records')
    }


# =============================================================================
# Prediction & Evaluation
# =============================================================================

def predict_oos(
    model: ElasticNet,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    logger: logging.Logger
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Generate OOS predictions and compute basic metrics."""
    X_oos_clean = X_oos.fillna(X_oos.median()).fillna(0)
    X_scaled = scaler.transform(X_oos_clean.values)
    predictions = model.predict(X_scaled)

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
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a single snapshot: train models, predict OOS, evaluate.

    Returns a dict with predictions, metrics, and LMT results for this snapshot.
    """
    data_dir = Path(config.get('data_dir', 'data/'))
    output_base = Path(config.get('output', {}).get('output_dir', 'outputs_elasticnet'))
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
    all_coef_analyses = {}

    for key in factor_keys:
        logger.info(f"\n--- Key {key} ---")

        # Load IS data
        X_is, y_is = load_snapshot_data(data_dir, snapshot, key, 'is', logger)

        # Prepare data
        X_train, y_train, scaler, feature_names = prepare_data(X_is, y_is, logger)
        logger.info(f"  Training samples: {len(X_train)}")

        # Train
        model = train_elasticnet(X_train, y_train, config, logger)

        # Analyze coefficients
        coef_analysis = analyze_coefficients(model, feature_names, logger)
        all_coef_analyses[key] = coef_analysis

        # Load OOS data
        X_oos, y_oos = load_snapshot_data(data_dir, snapshot, key, 'oos', logger)

        # Predict OOS
        predictions, metrics = predict_oos(model, X_oos, y_oos, scaler, logger)

        all_predictions[key] = predictions
        all_metrics[key] = metrics

        # Save model, scaler, coefficients
        joblib.dump(model, snapshot_output / f'model_key{key}.pkl')
        joblib.dump(scaler, snapshot_output / f'scaler_key{key}.pkl')

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        coef_df.to_csv(snapshot_output / f'coefficients_key{key}.csv', index=False)

        # Free memory
        del X_is, y_is

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
        'coefficient_analyses': all_coef_analyses,
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

def main(
    config_path: str,
    override_alpha: float = None,
    override_l1_ratio: float = None,
    snapshots_override: List[str] = None
):
    config = load_config(config_path)

    if override_alpha is not None:
        config.setdefault('model', {})['alpha'] = override_alpha
    if override_l1_ratio is not None:
        config.setdefault('model', {})['l1_ratio'] = override_l1_ratio

    config_dir = Path(config_path).resolve().parent if Path(config_path).exists() else Path(".")
    output_dir = config_dir / config.get('output', {}).get('output_dir', 'outputs_elasticnet')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve relative data_dir
    data_dir = Path(config.get('data_dir', 'data/'))
    if not data_dir.is_absolute():
        config['data_dir'] = str(config_dir / data_dir)

    # Resolve relative output_dir in config
    config['output']['output_dir'] = str(output_dir)

    log_file = output_dir / 'training.log'
    logger = setup_logging(log_file)

    snapshots = snapshots_override or config.get('snapshots', ["20181228", "20191231", "20201231"])

    logger.info("=" * 70)
    logger.info("ELASTICNET ROLLING PER-SNAPSHOT TRAINING PIPELINE (v2)")
    logger.info("=" * 70)
    logger.info(f"Config:    {config_path}")
    logger.info(f"Snapshots: {snapshots}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    model_params = config.get('model', {})
    logger.info(f"Model:     ElasticNet(alpha={model_params.get('alpha')}, "
                f"l1_ratio={model_params.get('l1_ratio')})")

    # =========================================================================
    # Phase 1: Rolling per-snapshot training
    # =========================================================================

    all_snapshot_results = []
    all_oos_predictions = []

    for snapshot in snapshots:
        result = process_snapshot(snapshot, config, logger)
        all_snapshot_results.append(result['report'])
        all_oos_predictions.append(result['pred_ensemble'])

    # =========================================================================
    # Phase 2: Aggregate all OOS predictions
    # =========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("ROLLING AGGREGATION — ALL SNAPSHOTS")
    logger.info("=" * 70)

    combined_oos = pd.concat(all_oos_predictions).sort_index()
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
        'pipeline': 'ElasticNet Rolling Per-Snapshot (v2)',
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
    print(f"Model: ElasticNet(alpha={model_params.get('alpha')}, l1_ratio={model_params.get('l1_ratio')})")

    for report in all_snapshot_results:
        snapshot = report['snapshot']
        oos_range = report.get('oos_date_range', ['?', '?'])
        print(f"\n📊 Snapshot {snapshot} (OOS: {oos_range[0]} → {oos_range[1]})")
        for key in config.get('factor_keys', ['0', '1', '2']):
            if key in report.get('metrics_by_key', {}):
                m = report['metrics_by_key'][key]
                c = report.get('coefficient_analyses', {}).get(key, {})
                sparsity = c.get('sparsity', 0)
                print(f"   Key {key}: IC={m['ic']:.4f}  R²={m['r2']:.6f}  "
                      f"RMSE={m['rmse']:.6f}  Sparsity={sparsity:.1%}")

    print(f"\n📈 Aggregate OOS: {len(combined_oos)} samples, {dates.nunique()} dates")
    print(f"   Date range: {dates.min()} → {dates.max()}")
    print("=" * 70)

    return rolling_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ElasticNet rolling per-snapshot training & evaluation (v2)"
    )
    parser.add_argument("--config", type=str, default="config_elasticnet.yaml",
                        help="Config YAML file")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha")
    parser.add_argument("--l1_ratio", type=float, default=None,
                        help="Override l1_ratio")
    parser.add_argument("--snapshots", nargs="+", default=None,
                        help="Override snapshot list")
    args = parser.parse_args()
    main(args.config, args.alpha, args.l1_ratio, args.snapshots)
