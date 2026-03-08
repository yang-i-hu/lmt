"""
DataLoader for Factor Reweight Model Training

Loads aligned factor and label data with configurable:
- Date range (start_date to end_date)
- Instrument universe (from universe.txt)
- Factor key selection (0, 1, 2)
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Union


class FactorDataLoader:
    """
    DataLoader for factor and label data.
    
    Usage:
        # From config file
        loader = FactorDataLoader.from_config('config.yaml')
        X, y = loader.load()
        
        # Direct initialization
        loader = FactorDataLoader(
            data_dir='raw_data/',
            factor_key='0',
            start_date=20180102,
            end_date=20180201,
            universe_file='universe.txt'
        )
        X, y = loader.load()
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        factor_key: str = "0",
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        universe_file: Optional[Union[str, Path]] = None,
        universe_list: Optional[List[str]] = None,
        aligned_only: bool = True,
        drop_na_labels: bool = True,
    ):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing the parquet files
        factor_key : str
            Factor matrix key: '0', '1', or '2'
        start_date : int, optional
            Start date (inclusive), format YYYYMMDD
        end_date : int, optional
            End date (inclusive), format YYYYMMDD
        universe_file : str or Path, optional
            Path to universe.txt file with one instrument per line
        universe_list : list of str, optional
            List of instrument codes (alternative to universe_file)
        aligned_only : bool
            If True, load from factors_X_aligned.parquet (with labels)
            If False, load from factors_X.parquet (without labels)
        drop_na_labels : bool
            If True, drop rows with NaN labels
        """
        self.data_dir = Path(data_dir)
        self.factor_key = str(factor_key)
        self.start_date = start_date
        self.end_date = end_date
        self.universe_file = Path(universe_file) if universe_file else None
        self.universe_list = universe_list
        self.aligned_only = aligned_only
        self.drop_na_labels = drop_na_labels
        
        # Load universe
        self._universe = self._load_universe()
        
        # Validate
        self._validate()
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'FactorDataLoader':
        """
        Create DataLoader from a YAML config file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to the YAML config file
            
        Returns
        -------
        FactorDataLoader
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve paths relative to config file location
        config_dir = config_path.parent
        
        data_dir = config.get('data_dir', 'data/')
        if not Path(data_dir).is_absolute():
            data_dir = config_dir / data_dir
        
        universe_file = config.get('universe_file')
        if universe_file and not Path(universe_file).is_absolute():
            universe_file = config_dir / universe_file
        
        return cls(
            data_dir=data_dir,
            factor_key=config.get('factor_key', '0'),
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            universe_file=universe_file,
            universe_list=config.get('universe_list'),
            aligned_only=config.get('aligned_only', True),
            drop_na_labels=config.get('drop_na_labels', True),
        )
    
    def _validate(self):
        """Validate configuration."""
        if self.factor_key not in ['0', '1', '2']:
            raise ValueError(f"factor_key must be '0', '1', or '2', got '{self.factor_key}'")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check data file exists
        data_file = self._get_data_file_path()
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
    
    def _get_data_file_path(self) -> Path:
        """Get the path to the data file."""
        if self.aligned_only:
            return self.data_dir / f"factors_{self.factor_key}_aligned.parquet"
        else:
            return self.data_dir / f"factors_{self.factor_key}.parquet"
    
    def _load_universe(self) -> Optional[set]:
        """Load instrument universe from file or list."""
        if self.universe_file and self.universe_file.exists():
            with open(self.universe_file, 'r') as f:
                instruments = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(instruments)} instruments from {self.universe_file}")
            return set(instruments)
        elif self.universe_list:
            return set(self.universe_list)
        return None
    
    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        dates = df.index.get_level_values('date')
        
        mask = pd.Series(True, index=df.index)
        
        if self.start_date is not None:
            mask &= dates >= self.start_date
        
        if self.end_date is not None:
            mask &= dates <= self.end_date
        
        return df[mask]
    
    def _filter_by_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by instrument universe."""
        if self._universe is None:
            return df
        
        instruments = df.index.get_level_values('instrument')
        mask = instruments.isin(self._universe)
        
        return df[mask]
    
    def load(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load factor data and labels.
        
        Returns
        -------
        X : pd.DataFrame
            Factor values with MultiIndex (date, instrument)
        y : pd.Series or None
            Label values (None if aligned_only=False)
        """
        data_file = self._get_data_file_path()
        
        print(f"Loading data from: {data_file}")
        df = pd.read_parquet(data_file)
        
        original_shape = df.shape
        print(f"Original shape: {original_shape}")
        
        # Filter by date
        if self.start_date is not None or self.end_date is not None:
            df = self._filter_by_date(df)
            print(f"After date filter ({self.start_date} to {self.end_date}): {df.shape}")
        
        # Filter by universe
        if self._universe is not None:
            df = self._filter_by_universe(df)
            print(f"After universe filter ({len(self._universe)} instruments): {df.shape}")
        
        # Separate features and labels
        if self.aligned_only and 'labelValue' in df.columns:
            label_cols = ['labelValue', 'endDate']
            feature_cols = [c for c in df.columns if c not in label_cols]
            
            X = df[feature_cols]
            y = df['labelValue']
            
            # Drop NaN labels if requested
            if self.drop_na_labels:
                valid_mask = y.notna()
                X = X[valid_mask]
                y = y[valid_mask]
                print(f"After dropping NaN labels: {X.shape}")
            
            return X, y
        else:
            return df, None
    
    def load_batches(self, batch_size: int = 100000) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load data in batches (memory efficient for large datasets).
        
        Note: This loads the full filtered dataset but yields in batches.
        For true streaming, consider using dask or vaex.
        
        Parameters
        ----------
        batch_size : int
            Number of rows per batch
            
        Yields
        ------
        X_batch : pd.DataFrame
        y_batch : pd.Series or None
        """
        X, y = self.load()
        
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X))
            
            X_batch = X.iloc[start:end]
            y_batch = y.iloc[start:end] if y is not None else None
            
            yield X_batch, y_batch
    
    def get_info(self) -> dict:
        """
        Get information about the loaded data without loading it fully.
        
        Returns
        -------
        dict
            Dictionary with data info
        """
        data_file = self._get_data_file_path()
        
        # Read just the metadata
        df = pd.read_parquet(data_file)
        
        dates = df.index.get_level_values('date').unique().sort_values()
        instruments = df.index.get_level_values('instrument').unique()
        
        info = {
            'data_file': str(data_file),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_dates': len(dates),
            'date_range': (int(dates.min()), int(dates.max())),
            'unique_instruments': len(instruments),
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        if 'labelValue' in df.columns:
            info['label_nan_count'] = int(df['labelValue'].isna().sum())
            info['label_valid_count'] = int(df['labelValue'].notna().sum())
        
        return info
    
    def __repr__(self) -> str:
        return (
            f"FactorDataLoader(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  factor_key='{self.factor_key}',\n"
            f"  start_date={self.start_date},\n"
            f"  end_date={self.end_date},\n"
            f"  universe={len(self._universe) if self._universe else 'all'} instruments,\n"
            f"  aligned_only={self.aligned_only},\n"
            f"  drop_na_labels={self.drop_na_labels}\n"
            f")"
        )


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load factor data")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--info", action="store_true", help="Print data info only")
    args = parser.parse_args()
    
    loader = FactorDataLoader.from_config(args.config)
    print(loader)
    
    if args.info:
        info = loader.get_info()
        print("\nData Info:")
        for k, v in info.items():
            if k != 'columns':
                print(f"  {k}: {v}")
    else:
        X, y = loader.load()
        print(f"\nLoaded:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape if y is not None else None}")
        print(f"  X memory: {X.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
