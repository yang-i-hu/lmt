"""
ElasticNet Factor Reweight Model Training - Full IS/OOS Pipeline

This script trains an ElasticNet model on FULL In-Sample (IS) data and evaluates on 
FULL Out-of-Sample (OOS) data, following the multi-snapshot workflow.

Key Differences from train_elasticnet.py:
1. Trains on ALL IS data (no train/test split within IS)
2. Uses separate OOS parquet files for evaluation
3. Evaluates ALL 3 keys (0, 1, 2) and computes ensemble
4. Runs LMT API evaluation on full OOS period

Workflow:
1. Load IS data (train_X.parquet from data/combined/)
2. Train ElasticNet model on 100% IS
3. Load OOS data (test_oos_X.parquet from data/combined/)
4. Generate predictions and ensemble
5. Evaluate with LMT API on full OOS period

Usage:
    python train_elasticnet_full.py --config config_elasticnet_full.yaml
    python train_elasticnet_full.py --config config_elasticnet_full.yaml --alpha 0.01
    python train_elasticnet_full.py --config config_elasticnet_full.yaml --snapshot 20201231
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
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "data_dir": "data/combined",
    "snapshot": "20201231",
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
        "output_dir": "outputs_elasticnet_full"
    }
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger("ElasticNetFullTrainer")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
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
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, pd.Index]:
    """Prepare data for training."""
    
    # Handle NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with median")
        X = X.fillna(X.median()).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    return X_scaled, y.values, scaler, X.index


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    logger: logging.Logger
) -> ElasticNet:
    """Train ElasticNet model."""
    
    model_params = config.get('model', {})
    alpha = model_params.get('alpha', 0.01)
    l1_ratio = model_params.get('l1_ratio', 0.5)
    max_iter = model_params.get('max_iter', 2000)
    tol = model_params.get('tol', 0.0001)
    fit_intercept = model_params.get('fit_intercept', True)
    random_seed = config.get('training', {}).get('random_seed', 42)
    
    logger.info("Model parameters:")
    logger.info(f"  alpha: {alpha}")
    logger.info(f"  l1_ratio: {l1_ratio}")
    logger.info(f"  max_iter: {max_iter}")
    
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
        
        logger.info("Training model...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Model summary
    n_nonzero = int(np.sum(model.coef_ != 0))
    n_features = len(model.coef_)
    sparsity = 1 - (n_nonzero / max(1, n_features))
    
    logger.info("Model summary:")
    logger.info(f"  Non-zero coefficients: {n_nonzero} / {n_features}")
    logger.info(f"  Sparsity: {sparsity:.2%}")
    logger.info(f"  Intercept: {model.intercept_:.6f}")
    
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_on_oos(
    model: ElasticNet,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    scaler: StandardScaler,
    logger: logging.Logger
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Generate predictions on OOS data and compute metrics."""
    
    # Handle NaN
    X_oos_clean = X_oos.fillna(X_oos.median()).fillna(0)
    
    # Scale and predict
    X_scaled = scaler.transform(X_oos_clean.values)
    predictions = model.predict(X_scaled)
    
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
    
    logger.info("Coefficient statistics:")
    logger.info(f"  Total features: {len(coef_df)}")
    logger.info(f"  Non-zero: {n_nonzero}")
    logger.info(f"  Positive: {n_positive}")
    logger.info(f"  Negative: {n_negative}")
    
    logger.info(f"\nTop {top_n} features by absolute coefficient:")
    for _, row in coef_df.head(top_n).iterrows():
        logger.info(f"  {str(row['feature'])[:30]:30s}: {row['coefficient']:+.6f}")
    
    return {
        'n_total': len(coef_df),
        'n_nonzero': n_nonzero,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'sparsity': float(1 - n_nonzero / max(1, len(coef_df))),
        'top_features': coef_df.head(top_n)[['feature', 'coefficient']].to_dict('records')
    }


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

def main(
    config_path: str,
    override_alpha: float = None,
    override_l1_ratio: float = None,
    snapshot: str = None
):
    """
    Run full IS training and OOS evaluation for all 3 keys.
    """
    # Load config
    config = load_config(config_path)
    
    if override_alpha is not None:
        config.setdefault('model', {})['alpha'] = override_alpha
    if override_l1_ratio is not None:
        config.setdefault('model', {})['l1_ratio'] = override_l1_ratio
    if snapshot:
        config['snapshot'] = snapshot
    
    # Setup output directory
    config_dir = Path(config_path).resolve().parent if Path(config_path).exists() else Path(".")
    output_dir = config_dir / config.get('output', {}).get('output_dir', 'outputs_elasticnet_full')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'training.log'
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("ELASTICNET FULL IS/OOS TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Snapshot: {config.get('snapshot', 'latest')}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    data_dir = Path(config.get('data_dir', 'data/combined'))
    
    # Train and evaluate each key
    all_predictions = {}
    all_metrics = {}
    models = {}
    scalers = {}
    coef_analyses = {}
    
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
            X_train, y_train, scaler, train_index = prepare_data(X_is, y_is, logger)
            
            logger.info(f"Training samples: {len(X_train)}")
            
            # Train
            logger.info("\nStep 3: Training model...")
            model = train_elasticnet(X_train, y_train, config, logger)
            
            # Coefficient analysis
            logger.info("\nStep 4: Analyzing coefficients...")
            coef_analysis = analyze_coefficients(model, feature_names, logger)
            
            # Load OOS data
            logger.info("\nStep 5: Loading OOS data...")
            X_oos, y_oos = load_oos_data(data_dir, key, logger)
            
            # Evaluate on OOS
            logger.info("\nStep 6: Evaluating on OOS...")
            predictions, metrics = evaluate_on_oos(model, X_oos, y_oos, scaler, logger)
            
            all_predictions[key] = predictions
            all_metrics[key] = metrics
            models[key] = model
            scalers[key] = scaler
            coef_analyses[key] = coef_analysis
            
            # Save individual model
            model_path = output_dir / f'model_key{key}.pkl'
            joblib.dump(model, model_path)
            
            scaler_path = output_dir / f'scaler_key{key}.pkl'
            joblib.dump(scaler, scaler_path)
            
            # Save coefficients
            coef_path = output_dir / f'coefficients_key{key}.csv'
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            coef_df.to_csv(coef_path, index=False)
            
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
        'coefficient_analyses': coef_analyses,
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
    model_params = config.get('model', {})
    print(f"Model: ElasticNet (alpha={model_params.get('alpha')}, l1_ratio={model_params.get('l1_ratio')})")
    
    for key in ['0', '1', '2']:
        if key in all_metrics:
            m = all_metrics[key]
            c = coef_analyses[key]
            print(f"\nKey {key}:")
            print(f"  Sparsity: {c['sparsity']:.2%}")
            print(f"  OOS IC: {m['ic']:.6f}")
            print(f"  OOS R²: {m['r2']:.6f}")
            print(f"  OOS RMSE: {m['rmse']:.6f}")
    print("=" * 60)
    
    return models, scalers, pred_ensemble, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ElasticNet on full IS data and evaluate on full OOS"
    )
    parser.add_argument("--config", type=str, default="config_elasticnet_full.yaml")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha")
    parser.add_argument("--l1_ratio", type=float, default=None, help="Override l1_ratio")
    parser.add_argument("--snapshot", type=str, default=None, help="Snapshot to use")
    
    args = parser.parse_args()
    main(args.config, args.alpha, args.l1_ratio, args.snapshot)
