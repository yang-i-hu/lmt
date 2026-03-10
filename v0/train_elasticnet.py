"""
ElasticNet Factor Reweight Model Training Pipeline

Complete pipeline for training an ElasticNet model on factor data:
1. Load aligned factor and label data
2. Preprocess (handle NaN, scale features)
3. Time-series train/test split
4. Train ElasticNet model
5. Evaluate and document results
6. Save model and artifacts

Key fixes vs original:
- Correct import: from modeling.dataloader import FactorDataLoader
- Resolve relative YAML paths relative to the YAML file location (NOT CWD)
- Safer logging handler setup (avoid duplicate handlers)
- Output directory defaults to config-relative unless absolute

Recommended run (from project root):
    python -m training.train_elasticnet --config training/config_enet_test.yaml

(You may still run as a script, but module execution is the robust option.)
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Correct import for your repo layout
from modeling.dataloader import FactorDataLoader

# LMT evaluation API
from lmt_data_api.api import DataApi
api = DataApi()


# =============================================================================
# Setup Logging
# =============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ElasticNetTrainer")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if main() is called multiple times (e.g., notebooks)
    if logger.handlers:
        return logger

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
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Tuple[Dict[str, Any], Path]:
    """
    Load YAML configuration file and return (config_dict, config_dir).

    Critical: config_dir is used to resolve any relative paths in the YAML.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config, cfg_path.parent


def _resolve_path_maybe_relative(p: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve a path relative to base_dir if it is not absolute."""
    if p is None:
        return None
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_data(
    config: Dict[str, Any],
    config_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load factor and label data using the FactorDataLoader.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)

    # Resolve data_dir relative to YAML location (config_dir)
    data_dir = _resolve_path_maybe_relative(config.get("data_dir", "raw_data/"), config_dir)
    if data_dir is None:
        raise ValueError("config.data_dir is required (or set a default).")

    # Universe file (optional)
    universe_file = _resolve_path_maybe_relative(config.get("universe_file"), config_dir)

    loader = FactorDataLoader(
        data_dir=data_dir,
        factor_key=config.get("factor_key", "0"),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        universe_file=universe_file,
        universe_list=config.get("universe_list"),
        aligned_only=config.get("aligned_only", True),
        drop_na_labels=config.get("drop_na_labels", True),
    )

    logger.info("DataLoader configuration:")
    logger.info(f"  - Config dir:     {config_dir}")
    logger.info(f"  - Data directory: {data_dir}")
    logger.info(f"  - Factor key:     {config.get('factor_key', '0')}")
    logger.info(f"  - Date range:     {config.get('start_date')} to {config.get('end_date')}")
    if universe_file:
        logger.info(f"  - Universe file:  {universe_file}")
    else:
        logger.info("  - Universe:       All instruments (no universe_file)")

    X, y = loader.load()

    logger.info("Data loaded:")
    logger.info(f"  - X shape: {X.shape}")
    logger.info(f"  - y shape: {y.shape}")
    logger.info(f"  - Memory usage: {X.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Data summary
    dates = X.index.get_level_values("date").unique().sort_values()
    instruments = X.index.get_level_values("instrument").unique()

    logger.info(f"  - Unique dates: {len(dates)}")
    logger.info(f"  - Date range: {dates.min()} to {dates.max()}")
    logger.info(f"  - Unique instruments: {len(instruments)}")

    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.Index, pd.Index]:
    """
    Preprocess data: handle NaN, time-series split, scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, train_index, test_index
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 60)

    train_params = config.get("training", {})
    train_ratio = float(train_params.get("train_ratio", 0.8))
    random_seed = int(train_params.get("random_seed", 42))

    np.random.seed(random_seed)

    # Handle NaN in features
    nan_count_before = int(X.isna().sum().sum())
    if nan_count_before > 0:
        logger.info(f"NaN values in features before fillna: {nan_count_before}")
        X = X.fillna(X.median(numeric_only=True))
        nan_count_after = int(X.isna().sum().sum())
        logger.info(f"NaN values after fillna(median): {nan_count_after}")

        # If still have NaN (entire column was NaN), fill with 0
        if nan_count_after > 0:
            X = X.fillna(0)
            logger.info("Filled remaining NaN with 0")
    else:
        logger.info("No NaN values in features")

    # Time-series split (respect temporal order)
    dates = X.index.get_level_values("date").unique().sort_values()
    if len(dates) < 2:
        raise ValueError(f"Not enough unique dates for a time split: {len(dates)}")

    split_idx = max(1, int(len(dates) * train_ratio))
    split_idx = min(split_idx, len(dates) - 1)

    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train_mask = X.index.get_level_values("date").isin(train_dates)
    test_mask = X.index.get_level_values("date").isin(test_dates)

    X_train_df = X.loc[train_mask]
    X_test_df = X.loc[test_mask]
    y_train_sr = y.loc[train_mask]
    y_test_sr = y.loc[test_mask]

    train_index = X_train_df.index
    test_index = X_test_df.index

    logger.info(f"Time-series split (ratio={train_ratio}):")
    logger.info(f"  - Train dates: {train_dates.min()} to {train_dates.max()} ({len(train_dates)} days)")
    logger.info(f"  - Test dates: {test_dates.min()} to {test_dates.max()} ({len(test_dates)} days)")
    logger.info(f"  - Train samples: {len(X_train_df)}")
    logger.info(f"  - Test samples: {len(X_test_df)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    logger.info("Features scaled with StandardScaler")

    return (
        X_train,
        X_test,
        y_train_sr.values,
        y_test_sr.values,
        scaler,
        train_index,
        test_index,
    )


# =============================================================================
# Model Training
# =============================================================================

def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    logger: logging.Logger
) -> ElasticNet:
    """
    Train ElasticNet model.

    Returns
    -------
    model : ElasticNet
        Trained model
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training ElasticNet Model")
    logger.info("=" * 60)

    model_params = config.get("model", {})

    alpha = float(model_params.get("alpha", 0.01))
    l1_ratio = float(model_params.get("l1_ratio", 0.5))
    max_iter = int(model_params.get("max_iter", 1000))
    tol = float(model_params.get("tol", 0.0001))
    fit_intercept = bool(model_params.get("fit_intercept", True))

    logger.info("Model parameters:")
    logger.info(f"  - alpha: {alpha}")
    logger.info(f"  - l1_ratio: {l1_ratio}")
    logger.info(f"  - max_iter: {max_iter}")
    logger.info(f"  - tol: {tol}")
    logger.info(f"  - fit_intercept: {fit_intercept}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            random_state=int(config.get("training", {}).get("random_seed", 42)),
        )

        logger.info("Training model...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Model summary
    n_nonzero = int(np.sum(model.coef_ != 0))
    n_features = int(len(model.coef_))
    sparsity = 1 - (n_nonzero / max(1, n_features))

    logger.info("Model summary:")
    logger.info(f"  - Non-zero coefficients: {n_nonzero} / {n_features}")
    logger.info(f"  - Sparsity: {sparsity:.2%}")
    logger.info(f"  - Intercept: {model.intercept_:.6f}")

    return model


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: ElasticNet,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_index: pd.Index,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluate model on train and test sets.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics including lmt_data_api evaluation
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating Model")
    logger.info("=" * 60)

    metrics: Dict[str, Any] = {}

    for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = model.predict(X)

        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))

        # IC (Spearman)
        ic, ic_pvalue = stats.spearmanr(y, y_pred)

        # Pearson
        pearson_r, pearson_pvalue = stats.pearsonr(y, y_pred)

        metrics[split_name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "ic": float(ic) if ic is not None else float("nan"),
            "ic_pvalue": float(ic_pvalue) if ic_pvalue is not None else float("nan"),
            "pearson_r": float(pearson_r) if pearson_r is not None else float("nan"),
            "pearson_pvalue": float(pearson_pvalue) if pearson_pvalue is not None else float("nan"),
            "n_samples": int(len(y)),
        }

        logger.info(f"\n{split_name.upper()} Metrics:")
        logger.info(f"  - RMSE:     {rmse:.6f}")
        logger.info(f"  - MAE:      {mae:.6f}")
        logger.info(f"  - R²:       {r2:.6f}")
        logger.info(f"  - IC:       {metrics[split_name]['ic']:.6f} (p={metrics[split_name]['ic_pvalue']:.2e})")
        logger.info(f"  - Pearson:  {metrics[split_name]['pearson_r']:.6f} (p={metrics[split_name]['pearson_pvalue']:.2e})")
        logger.info(f"  - Samples:  {len(y)}")

    # =========================================================================
    # LMT Data API Evaluation (test set only)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4b: LMT API Evaluation (Test Set)")
    logger.info("=" * 60)

    try:
        # Create pred_esem from test predictions
        y_test_pred = model.predict(X_test)
        
        # Build pred_esem with required format
        pred_esem = pd.Series(
            data=y_test_pred.astype("float64"),
            index=test_index,
            name="factor"
        )
        
        # Rename index levels: (date, instrument) -> (date, code)
        pred_esem.index = pred_esem.index.rename(["date", "code"])
        
        # Remove duplicates if any
        pred_esem = pred_esem[~pred_esem.index.duplicated(keep="last")]
        
        logger.info(f"pred_esem shape: {pred_esem.shape}")
        logger.info(f"pred_esem index names: {pred_esem.index.names}")
        logger.info(f"pred_esem date range: {pred_esem.index.get_level_values('date').min()} to {pred_esem.index.get_level_values('date').max()}")
        
        # Check if we have enough dates for label_period=10
        unique_dates = pred_esem.index.get_level_values('date').unique()
        n_dates = len(unique_dates)
        logger.info(f"pred_esem unique dates: {n_dates}")
        
        if n_dates < 10:
            logger.warning(f"Test set has only {n_dates} dates, need at least 10 for label_period=10")
            logger.warning("Skipping LMT API evaluation due to insufficient test data")
            metrics["lmt_api"] = {"error": f"Insufficient test dates: {n_dates} < 10"}
        else:
            # Debug: print first few rows of pred_esem
            logger.debug(f"\npred_esem head:\n{pred_esem.head(10)}")
            logger.debug(f"pred_esem dtypes: {pred_esem.dtype}")
            
            # 1) Group return related metrics (positional args as per API guide)
            logger.info("Calling api.da_eva_group_return...")
            try:
                group_re, group_ir, group_hs = api.da_eva_group_return(
                    pred_esem,
                    "factor",
                    alpha=1,
                    label_period=10
                )
            except Exception as api_err:
                logger.warning(f"api.da_eva_group_return failed: {api_err}")
                import traceback
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                raise
            
            # Debug group_re
            logger.debug(f"\n--- group_re debug ---")
            logger.debug(f"  type: {type(group_re)}")
            logger.debug(f"  value: {group_re}")
            if group_re is not None:
                if hasattr(group_re, 'shape'):
                    logger.debug(f"  shape: {group_re.shape}")
                if hasattr(group_re, 'columns'):
                    logger.debug(f"  columns: {list(group_re.columns)}")
                if hasattr(group_re, 'index'):
                    logger.debug(f"  index: {group_re.index}")
                if isinstance(group_re, pd.DataFrame):
                    logger.debug(f"  DataFrame:\n{group_re}")
                elif isinstance(group_re, pd.Series):
                    logger.debug(f"  Series:\n{group_re}")
            
            # Debug group_ir
            logger.debug(f"\n--- group_ir debug ---")
            logger.debug(f"  type: {type(group_ir)}")
            logger.debug(f"  value: {group_ir}")
            if group_ir is not None:
                if hasattr(group_ir, 'shape'):
                    logger.debug(f"  shape: {group_ir.shape}")
                if hasattr(group_ir, 'columns'):
                    logger.debug(f"  columns: {list(group_ir.columns)}")
                if isinstance(group_ir, pd.DataFrame):
                    logger.debug(f"  DataFrame:\n{group_ir}")
                elif isinstance(group_ir, pd.Series):
                    logger.debug(f"  Series:\n{group_ir}")
            
            # Debug group_hs
            logger.debug(f"\n--- group_hs debug ---")
            logger.debug(f"  type: {type(group_hs)}")
            logger.debug(f"  value: {group_hs}")
            if group_hs is not None:
                if hasattr(group_hs, 'shape'):
                    logger.debug(f"  shape: {group_hs.shape}")
                if hasattr(group_hs, 'columns'):
                    logger.debug(f"  columns: {list(group_hs.columns)}")
                if isinstance(group_hs, pd.DataFrame):
                    logger.debug(f"  DataFrame:\n{group_hs}")
                elif isinstance(group_hs, pd.Series):
                    logger.debug(f"  Series:\n{group_hs}")
            
            # 2) IC time series (positional args as per API guide)
            logger.info("Calling api.da_eva_ic...")
            try:
                ic_df = api.da_eva_ic(
                    pred_esem,
                    "factor",
                    10
                )
            except Exception as api_err:
                logger.warning(f"api.da_eva_ic failed: {api_err}")
                import traceback
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                raise
            
            # Debug ic_df
            logger.debug(f"\n--- ic_df debug ---")
            logger.debug(f"  type: {type(ic_df)}")
            logger.debug(f"  value: {ic_df}")
            if ic_df is not None:
                if hasattr(ic_df, 'shape'):
                    logger.debug(f"  shape: {ic_df.shape}")
                if hasattr(ic_df, 'columns'):
                    logger.debug(f"  columns: {list(ic_df.columns)}")
                if hasattr(ic_df, 'index'):
                    logger.debug(f"  index: {ic_df.index}")
                if isinstance(ic_df, pd.DataFrame):
                    logger.debug(f"  DataFrame:\n{ic_df}")
                elif isinstance(ic_df, pd.Series):
                    logger.debug(f"  Series:\n{ic_df}")
            
            # Store raw API metrics
            metrics["lmt_api"] = {
                "ic": str(ic_df),
                "group_return": str(group_re),
                "group_ir": str(group_ir),
                "group_hs": str(group_hs),
            }
            
            # Only try to build stats_all if we have valid DataFrames
            logger.debug("Attempting to build stats_all...")
            
            # Check if we can build the table
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
                    # Check column availability
                    logger.debug(f"Checking columns for concat...")
                    if hasattr(group_re, 'columns'):
                        logger.debug(f"  group_re has columns: {list(group_re.columns)}")
                    if hasattr(group_ir, 'columns'):
                        logger.debug(f"  group_ir has columns: {list(group_ir.columns)}")
                    if hasattr(group_hs, 'columns'):
                        logger.debug(f"  group_hs has columns: {list(group_hs.columns)}")
                    
                    # 3) Build unified stats table (as per API guide)
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
                    import traceback
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
            else:
                logger.warning("Cannot build stats_all - some API returns are None or empty")
                    
    except Exception as e:
        logger.warning(f"LMT API evaluation failed: {str(e)}")
        logger.warning("Continuing without API evaluation...")
        metrics["lmt_api"] = {"error": str(e)}

    return metrics


def analyze_coefficients(
    model: ElasticNet,
    feature_names: list,
    logger: logging.Logger,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyze model coefficients.

    Returns
    -------
    coef_analysis : dict
        Coefficient analysis results
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Coefficient Analysis")
    logger.info("=" * 60)

    coef_df = (
        pd.DataFrame(
            {"feature": feature_names, "coefficient": model.coef_, "abs_coefficient": np.abs(model.coef_)}
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    n_nonzero = int((coef_df["coefficient"] != 0).sum())
    n_positive = int((coef_df["coefficient"] > 0).sum())
    n_negative = int((coef_df["coefficient"] < 0).sum())

    logger.info("Coefficient statistics:")
    logger.info(f"  - Total features: {len(coef_df)}")
    logger.info(f"  - Non-zero: {n_nonzero}")
    logger.info(f"  - Positive: {n_positive}")
    logger.info(f"  - Negative: {n_negative}")
    logger.info(f"  - Zero: {len(coef_df) - n_nonzero}")

    logger.info(f"\nTop {top_n} features by absolute coefficient:")
    top_features = coef_df.head(top_n)
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature'][:30]:30s}: {row['coefficient']:+.6f}")

    analysis = {
        "n_total": int(len(coef_df)),
        "n_nonzero": int(n_nonzero),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "sparsity": float(1 - n_nonzero / max(1, len(coef_df))),
        "top_features": coef_df.head(top_n)[["feature", "coefficient"]].to_dict("records"),
        "coefficient_stats": {
            "mean": float(coef_df["coefficient"].mean()),
            "std": float(coef_df["coefficient"].std()),
            "min": float(coef_df["coefficient"].min()),
            "max": float(coef_df["coefficient"].max()),
            "median": float(coef_df["coefficient"].median()),
        },
    }

    return analysis


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    model: ElasticNet,
    scaler: StandardScaler,
    metrics: Dict[str, Any],
    coef_analysis: Dict[str, Any],
    config: Dict[str, Any],
    feature_names: list,
    config_dir: Path,
    logger: logging.Logger
) -> Dict[str, str]:
    """
    Save model, scaler, and results.

    Output dir behavior:
    - If output.output_dir is absolute: use as-is
    - Else: resolve relative to config_dir (same as other paths)
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Saving Results")
    logger.info("=" * 60)

    output_config = config.get("output", {})

    out_dir_raw = output_config.get("output_dir", "outputs/")
    output_dir = _resolve_path_maybe_relative(out_dir_raw, config_dir) or (config_dir / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / output_config.get("model_file", "elasticnet_model.pkl")
    scaler_file = output_dir / output_config.get("scaler_file", "scaler.pkl")
    results_file = output_dir / output_config.get("results_file", "training_results.json")

    # Save model/scaler
    joblib.dump(model, model_file)
    logger.info(f"Model saved to: {model_file}")

    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to: {scaler_file}")

    # Feature names
    feature_file = output_dir / "feature_names.json"
    with open(feature_file, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Feature names saved to: {feature_file}")

    # Coefficients
    coef_file = output_dir / "coefficients.csv"
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns=["abs_coef"])
    )
    coef_df.to_csv(coef_file, index=False)
    logger.info(f"Coefficients saved to: {coef_file}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_dir": str(config.get("data_dir")),
            "factor_key": config.get("factor_key"),
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "model_params": config.get("model", {}),
            "training_params": config.get("training", {}),
        },
        "data": {"n_features": len(feature_names)},
        "metrics": metrics,
        "coefficient_analysis": coef_analysis,
        "files": {
            "model": str(model_file),
            "scaler": str(scaler_file),
            "features": str(feature_file),
            "coefficients": str(coef_file),
        },
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    return {
        "model": str(model_file),
        "scaler": str(scaler_file),
        "features": str(feature_file),
        "coefficients": str(coef_file),
        "results": str(results_file),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main(config_path: str, override_alpha: float = None, override_l1_ratio: float = None):
    """
    Run the full ElasticNet training pipeline.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    override_alpha : float, optional
        Override alpha parameter from config
    override_l1_ratio : float, optional
        Override l1_ratio parameter from config
    """
    # Load config + config_dir
    config, config_dir = load_config(config_path)

    # Apply overrides
    if override_alpha is not None:
        config.setdefault("model", {})["alpha"] = override_alpha
    if override_l1_ratio is not None:
        config.setdefault("model", {})["l1_ratio"] = override_l1_ratio

    # Output/log paths
    # Extract config filename without extension for output directory name
    config_filename = Path(config_path).stem  # e.g., "config_enet_test" from "config_enet_test.yaml"
    output_dir_name = f"output_{config_filename}"
    output_dir = config_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("ElasticNet Factor Reweight Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config file: {Path(config_path).expanduser().resolve()}")
    logger.info(f"Config dir:  {config_dir}")
    logger.info(f"CWD:         {Path.cwd()}")
    logger.info(f"Timestamp:   {datetime.now().isoformat()}")

    try:
        # Step 1: Load data
        X, y = load_data(config, config_dir, logger)
        feature_names = list(X.columns)

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, scaler, train_idx, test_idx = preprocess_data(
            X, y, config, logger
        )

        # Step 3: Train model
        model = train_elasticnet(X_train, y_train, config, logger)

        # Step 4: Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, test_idx, logger)

        # Step 5: Coefficient analysis
        coef_analysis = analyze_coefficients(model, feature_names, logger)

        # Step 6: Save results
        saved_files = save_results(
            model, scaler, metrics, coef_analysis, config, feature_names, config_dir, logger
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Output files:")
        for name, path in saved_files.items():
            logger.info(f"  - {name}: {path}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: ElasticNet (alpha={config['model']['alpha']}, l1_ratio={config['model']['l1_ratio']})")
        print(f"Features: {len(feature_names)}")
        print(f"Sparsity: {coef_analysis['sparsity']:.2%}")
        print("\nTrain Metrics:")
        print(f"  RMSE: {metrics['train']['rmse']:.6f}")
        print(f"  R²:   {metrics['train']['r2']:.6f}")
        print(f"  IC:   {metrics['train']['ic']:.6f}")
        print("\nTest Metrics:")
        print(f"  RMSE: {metrics['test']['rmse']:.6f}")
        print(f"  R²:   {metrics['test']['r2']:.6f}")
        print(f"  IC:   {metrics['test']['ic']:.6f}")
        print("=" * 60)

        return model, scaler, metrics

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ElasticNet Factor Reweight Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config
    python -m training.train_elasticnet --config training/config_enet_test.yaml

    # Override alpha parameter
    python -m training.train_elasticnet --config training/config_enet_test.yaml --alpha 0.05

    # Override both alpha and l1_ratio
    python -m training.train_elasticnet --config training/config_enet_test.yaml --alpha 0.1 --l1_ratio 0.7
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="training/config_elasticnet_full.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha (regularization strength)",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=None,
        help="Override l1_ratio (0=Ridge, 1=Lasso, 0.5=balanced)",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        override_alpha=args.alpha,
        override_l1_ratio=args.l1_ratio,
    )
