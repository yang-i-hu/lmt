"""
Export Multi-Snapshot Factor and Label Data for Training & OOS Evaluation

This script processes multiple snapshot directories (20181228, 20191231, 20201231)
and exports aligned data for each:

Workflow per snapshot (cutoff date Di):
1. Load factors from 1128_weight_factors/{Di}/weakFactors.h5
2. Load labels from label10.h5 with endDate <= Di (no future leakage)
3. Align factors and labels by (date, instrument)
4. Split into:
   - IS (In-Sample): dates <= Di → for training
   - OOS (Out-of-Sample): dates > Di → for evaluation
5. Save to parquet files

Output Structure:
    data/
    ├── 20181228/                    # Snapshot cutoff 2018-12-28
    │   ├── factors_0_is.parquet     # Key 0, IS data (train)
    │   ├── factors_0_oos.parquet    # Key 0, OOS data (test for 2019)
    │   ├── factors_1_is.parquet
    │   ├── factors_1_oos.parquet
    │   ├── factors_2_is.parquet
    │   ├── factors_2_oos.parquet
    │   └── metadata.json
    ├── 20191231/                    # Snapshot cutoff 2019-12-31
    │   └── ...                      # OOS = 2020
    ├── 20201231/                    # Snapshot cutoff 2020-12-31
    │   └── ...                      # OOS = 2021
    └── combined/                    # Combined for convenience
        ├── train_all_is.parquet     # All IS data from latest snapshot (20201231)
        └── test_all_oos.parquet     # Combined OOS: 2019 + 2020 + 2021

Usage:
    python export_multi_snapshot.py
    python export_multi_snapshot.py --snapshots 20181228 20191231 20201231
    python export_multi_snapshot.py --output-dir ./processed_data
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import List, Dict, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths - adjust to your environment
BASE_FACTOR_DIR = Path("../1128_weight_factors")
LABEL_FILE = Path("../label10.h5")
OUTPUT_DIR = Path("data")

# Snapshots to process (folder names = cutoff dates)
SNAPSHOTS = ["20181228", "20191231", "20201231"]

# OOS boundaries for each snapshot
# Each snapshot's OOS ends at next snapshot's cutoff (or end of available data)
SNAPSHOT_OOS_END = {
    "20181228": 20191231,  # OOS: 2019-01-01 to 2019-12-31
    "20191231": 20201231,  # OOS: 2020-01-01 to 2020-12-31
    "20201231": 20211231,  # OOS: 2021-01-01 to 2021-12-31
}

FACTOR_KEYS = ["0", "1", "2"]
CHUNK_SIZE = 100000
LABEL_CHUNK_SIZE = 500000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_factor_slice(h5_file: Path, key: str = "0", start: int = 0, stop: int = 1000) -> pd.DataFrame:
    """
    Safely load a slice of factor data using h5py.
    Returns a pandas DataFrame with proper MultiIndex.
    
    IMPORTANT: Do NOT use pd.read_hdf for this file (fixed format issues)
    """
    with h5py.File(h5_file, "r") as f:
        grp = f[key]
        
        # Load factor values slice
        values = grp["block0_values"][start:stop, :]
        
        # Load columns (factor IDs)
        columns = grp["axis0"][:]
        
        # Load index components
        level0 = grp["axis1_level0"][:]  # unique dates
        level1 = grp["axis1_level1"][:]  # unique instruments
        label0 = grp["axis1_label0"][start:stop]
        label1 = grp["axis1_label1"][start:stop]
        
        # Decode bytes if needed
        if level1.dtype.kind == 'S':
            level1 = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in level1])
    
    # Reconstruct index
    dates = level0[label0]
    instruments = level1[label1]
    index = pd.MultiIndex.from_arrays([dates, instruments], names=["date", "instrument"])
    
    # Create DataFrame
    df = pd.DataFrame(values, index=index, columns=columns)
    return df


def get_factor_shape(h5_file: Path, key: str = "0") -> Tuple[int, int]:
    """Get shape of factor matrix without loading data."""
    with h5py.File(h5_file, "r") as f:
        shape = f[key]["block0_values"].shape
    return shape


def find_label_date_position(label_file: Path, target_date: int, total_rows: int) -> int:
    """Binary search to find approximate row where labelDate >= target_date."""
    low, high = 0, total_rows
    
    while low < high:
        mid = (low + high) // 2
        sample = pd.read_hdf(label_file, key="Data", start=mid, stop=mid+1)
        sample = sample.reset_index()
        mid_date = sample['labelDate'].iloc[0]
        
        if mid_date < target_date:
            low = mid + 1
        else:
            high = mid
    
    return low


def load_labels_for_snapshot(
    label_file: Path,
    cutoff_date: int,
    start_date: int = None,
    label_start_row: int = 0
) -> pd.DataFrame:
    """
    Load labels where endDate <= cutoff_date (no future leakage).
    
    Parameters
    ----------
    label_file : Path
        Path to Label10.h5
    cutoff_date : int
        Snapshot cutoff date (endDate must be <= this)
    start_date : int, optional
        Minimum labelDate to include
    label_start_row : int
        Row offset to start reading from
        
    Returns
    -------
    pd.DataFrame
        Labels with columns: labelDate, code, labelValue, endDate
    """
    print(f"  Loading labels with endDate <= {cutoff_date}...")
    
    with pd.HDFStore(label_file, mode='r') as store:
        total_rows = store.get_storer('Data').nrows
    
    all_labels = []
    rows_loaded = 0
    rows_kept = 0
    
    for start in range(label_start_row, total_rows, LABEL_CHUNK_SIZE):
        labels = pd.read_hdf(label_file, key="Data", start=start, stop=start + LABEL_CHUNK_SIZE)
        labels = labels.reset_index()
        rows_loaded += len(labels)
        
        # Filter: endDate <= cutoff (critical for no future leakage)
        labels = labels[labels['endDate'] <= cutoff_date]
        
        # Optional: filter by start_date
        if start_date is not None:
            labels = labels[labels['labelDate'] >= start_date]
        
        if len(labels) > 0:
            all_labels.append(labels)
            rows_kept += len(labels)
        
        if start % 2000000 == 0:
            print(f"    Scanned {start:,} rows, kept {rows_kept:,} labels...")
    
    print(f"  ✅ Labels loaded: {rows_kept:,} rows (scanned {rows_loaded:,})")
    
    if all_labels:
        return pd.concat(all_labels, ignore_index=True)
    return pd.DataFrame()


def export_snapshot(
    snapshot_name: str,
    factor_file: Path,
    labels_indexed: pd.DataFrame,
    output_dir: Path,
    cutoff_date: int,
    oos_end_date: int
) -> Dict:
    """
    Export IS and OOS data for one snapshot.
    
    Parameters
    ----------
    snapshot_name : str
        Snapshot identifier (e.g., "20181228")
    factor_file : Path
        Path to weakFactors.h5
    labels_indexed : pd.DataFrame
        Labels with MultiIndex (date, instrument)
    output_dir : Path
        Output directory for this snapshot
    cutoff_date : int
        IS/OOS split date (IS <= cutoff, OOS > cutoff)
    oos_end_date : int
        End date for OOS period
        
    Returns
    -------
    Dict
        Metadata about exported data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "snapshot": snapshot_name,
        "cutoff_date": cutoff_date,
        "oos_end_date": oos_end_date,
        "keys": {}
    }
    
    for key in FACTOR_KEYS:
        print(f"\n  Processing Factor Key: /{key}")
        print(f"  " + "-" * 50)
        
        n_rows, n_cols = get_factor_shape(factor_file, key)
        print(f"    Total rows: {n_rows:,}, columns: {n_cols}")
        
        is_chunks = []
        oos_chunks = []
        
        for start in range(0, n_rows, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, n_rows)
            
            # Load factor slice
            factors = load_factor_slice(factor_file, key=key, start=start, stop=stop)
            
            # Align with labels
            common_idx = factors.index.intersection(labels_indexed.index)
            if len(common_idx) == 0:
                continue
                
            aligned = factors.loc[common_idx].copy()
            aligned['labelValue'] = labels_indexed.loc[common_idx, 'labelValue']
            aligned['endDate'] = labels_indexed.loc[common_idx, 'endDate']
            
            # Split IS / OOS by date
            dates = aligned.index.get_level_values('date')
            
            is_mask = dates <= cutoff_date
            oos_mask = (dates > cutoff_date) & (dates <= oos_end_date)
            
            if is_mask.sum() > 0:
                is_chunks.append(aligned[is_mask])
            if oos_mask.sum() > 0:
                oos_chunks.append(aligned[oos_mask])
            
            if start % 500000 == 0:
                print(f"    Processed {start:,}/{n_rows:,} rows...")
        
        # Save IS data
        if is_chunks:
            is_df = pd.concat(is_chunks)
            is_path = output_dir / f"factors_{key}_is.parquet"
            is_df.to_parquet(is_path)
            
            is_dates = is_df.index.get_level_values('date')
            metadata["keys"][key] = metadata["keys"].get(key, {})
            metadata["keys"][key]["is"] = {
                "path": str(is_path),
                "shape": list(is_df.shape),
                "date_range": [int(is_dates.min()), int(is_dates.max())],
                "n_dates": len(is_dates.unique()),
                "n_instruments": len(is_df.index.get_level_values('instrument').unique()),
                "file_size_mb": is_path.stat().st_size / 1024 / 1024
            }
            print(f"    ✅ IS saved: {is_df.shape} ({is_dates.min()} to {is_dates.max()})")
        else:
            print(f"    ⚠️ No IS data for key {key}")
        
        # Save OOS data
        if oos_chunks:
            oos_df = pd.concat(oos_chunks)
            oos_path = output_dir / f"factors_{key}_oos.parquet"
            oos_df.to_parquet(oos_path)
            
            oos_dates = oos_df.index.get_level_values('date')
            metadata["keys"][key]["oos"] = {
                "path": str(oos_path),
                "shape": list(oos_df.shape),
                "date_range": [int(oos_dates.min()), int(oos_dates.max())],
                "n_dates": len(oos_dates.unique()),
                "n_instruments": len(oos_df.index.get_level_values('instrument').unique()),
                "file_size_mb": oos_path.stat().st_size / 1024 / 1024
            }
            print(f"    ✅ OOS saved: {oos_df.shape} ({oos_dates.min()} to {oos_dates.max()})")
        else:
            print(f"    ⚠️ No OOS data for key {key}")
        
        # Free memory
        del is_chunks, oos_chunks
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  📋 Metadata saved: {metadata_path}")
    
    return metadata


def create_combined_datasets(output_dir: Path, snapshots: List[str]) -> Dict:
    """
    Create combined datasets for convenience:
    - train_all.parquet: All IS data from latest snapshot (for full training)
    - test_all_oos.parquet: Combined OOS from all snapshots (for full evaluation)
    
    Parameters
    ----------
    output_dir : Path
        Base output directory
    snapshots : List[str]
        List of snapshot names in chronological order
        
    Returns
    -------
    Dict
        Combined dataset metadata
    """
    print("\n" + "=" * 60)
    print("Creating Combined Datasets")
    print("=" * 60)
    
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {"train": {}, "test_oos": {}}
    
    # Latest snapshot for training (has most IS data)
    latest_snapshot = snapshots[-1]
    print(f"\n1. Training data from latest snapshot: {latest_snapshot}")
    
    for key in FACTOR_KEYS:
        is_file = output_dir / latest_snapshot / f"factors_{key}_is.parquet"
        if is_file.exists():
            # Just copy/link to combined for clarity
            df = pd.read_parquet(is_file)
            out_path = combined_dir / f"train_{key}.parquet"
            df.to_parquet(out_path)
            
            dates = df.index.get_level_values('date')
            metadata["train"][key] = {
                "path": str(out_path),
                "shape": list(df.shape),
                "date_range": [int(dates.min()), int(dates.max())],
                "source_snapshot": latest_snapshot
            }
            print(f"   ✅ train_{key}.parquet: {df.shape}")
    
    # Combined OOS from all snapshots (rolling evaluation)
    print(f"\n2. Combined OOS test data from all snapshots")
    
    for key in FACTOR_KEYS:
        oos_dfs = []
        for snapshot in snapshots:
            oos_file = output_dir / snapshot / f"factors_{key}_oos.parquet"
            if oos_file.exists():
                df = pd.read_parquet(oos_file)
                df['source_snapshot'] = snapshot  # Track which model to use
                oos_dfs.append(df)
                print(f"   - {snapshot}/{key}: {len(df):,} rows")
        
        if oos_dfs:
            combined_oos = pd.concat(oos_dfs)
            # Remove duplicates (if any overlap)
            combined_oos = combined_oos[~combined_oos.index.duplicated(keep='last')]
            
            out_path = combined_dir / f"test_oos_{key}.parquet"
            combined_oos.to_parquet(out_path)
            
            dates = combined_oos.index.get_level_values('date')
            metadata["test_oos"][key] = {
                "path": str(out_path),
                "shape": list(combined_oos.shape),
                "date_range": [int(dates.min()), int(dates.max())],
                "source_snapshots": snapshots
            }
            print(f"   ✅ test_oos_{key}.parquet: {combined_oos.shape}")
    
    # Save combined metadata
    meta_path = combined_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main(
    snapshots: List[str] = None,
    base_factor_dir: Path = None,
    label_file: Path = None,
    output_dir: Path = None
):
    """
    Main export pipeline.
    
    Parameters
    ----------
    snapshots : List[str]
        Snapshot names to process (default: SNAPSHOTS)
    base_factor_dir : Path
        Base directory containing snapshot folders
    label_file : Path
        Path to Label10.h5
    output_dir : Path
        Output directory
    """
    snapshots = snapshots or SNAPSHOTS
    base_factor_dir = base_factor_dir or BASE_FACTOR_DIR
    label_file = label_file or LABEL_FILE
    output_dir = output_dir or OUTPUT_DIR
    
    start_time = datetime.now()
    
    print("=" * 60)
    print("MULTI-SNAPSHOT FACTOR EXPORT PIPELINE")
    print("=" * 60)
    print(f"\nStarted at: {start_time}")
    print(f"\nConfiguration:")
    print(f"  Factor base dir: {base_factor_dir.absolute()}")
    print(f"  Label file:      {label_file.absolute()}")
    print(f"  Output dir:      {output_dir.absolute()}")
    print(f"  Snapshots:       {snapshots}")
    
    # Validate inputs
    if not label_file.exists():
        print(f"\n❌ Error: Label file not found: {label_file}")
        return
    
    for snapshot in snapshots:
        factor_file = base_factor_dir / snapshot / "weakFactors.h5"
        if not factor_file.exists():
            print(f"\n❌ Error: Factor file not found: {factor_file}")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metadata = {}
    
    # Process each snapshot
    for snapshot in snapshots:
        cutoff_date = int(snapshot)
        oos_end_date = SNAPSHOT_OOS_END.get(snapshot, 20211231)
        
        print("\n" + "=" * 60)
        print(f"SNAPSHOT: {snapshot}")
        print(f"  Cutoff (IS end): {cutoff_date}")
        print(f"  OOS period: {cutoff_date + 1} to {oos_end_date}")
        print("=" * 60)
        
        factor_file = base_factor_dir / snapshot / "weakFactors.h5"
        snapshot_output_dir = output_dir / snapshot
        
        # Load labels for this snapshot (endDate <= cutoff)
        print("\nStep 1: Loading labels...")
        labels = load_labels_for_snapshot(
            label_file,
            cutoff_date=oos_end_date,  # Allow labels up to OOS end
            start_date=20180101  # Start from 2018
        )
        
        if labels.empty:
            print(f"  ❌ No labels found for snapshot {snapshot}")
            continue
        
        # Index labels
        labels_indexed = labels.set_index(['labelDate', 'code'])
        labels_indexed.index.names = ['date', 'instrument']
        
        print(f"  Labels shape: {labels_indexed.shape}")
        print(f"  Date range: {labels_indexed.index.get_level_values('date').min()} to "
              f"{labels_indexed.index.get_level_values('date').max()}")
        
        # Export IS/OOS for this snapshot
        print("\nStep 2: Exporting factor data...")
        metadata = export_snapshot(
            snapshot_name=snapshot,
            factor_file=factor_file,
            labels_indexed=labels_indexed,
            output_dir=snapshot_output_dir,
            cutoff_date=cutoff_date,
            oos_end_date=oos_end_date
        )
        
        all_metadata[snapshot] = metadata
        
        # Free memory
        del labels, labels_indexed
    
    # Create combined datasets
    combined_metadata = create_combined_datasets(output_dir, snapshots)
    all_metadata["combined"] = combined_metadata
    
    # Save global metadata
    global_meta_path = output_dir / "all_snapshots_metadata.json"
    with open(global_meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    
    print(f"\n📁 Output Directory: {output_dir.absolute()}")
    print(f"\nSnapshot Summary:")
    for snapshot in snapshots:
        cutoff = int(snapshot)
        oos_end = SNAPSHOT_OOS_END.get(snapshot, 20211231)
        print(f"  {snapshot}/")
        print(f"    IS:  dates <= {cutoff}")
        print(f"    OOS: {cutoff + 1} to {oos_end}")
    
    print(f"\nCombined datasets:")
    print(f"  combined/train_{{0,1,2}}.parquet   - Full IS from {snapshots[-1]}")
    print(f"  combined/test_oos_{{0,1,2}}.parquet - All OOS periods")
    
    print(f"\n⏱️  Duration: {duration}")
    print(f"✅ Finished at: {end_time}")
    
    print("""
📋 Usage Guide:

1. For training on full IS data (latest snapshot):
   df = pd.read_parquet('data/combined/train_0.parquet')
   X = df.drop(['labelValue', 'endDate'], axis=1)
   y = df['labelValue']

2. For OOS evaluation (all periods):
   df = pd.read_parquet('data/combined/test_oos_0.parquet')
   
3. For snapshot-specific evaluation:
   # Train on 20201231 IS, evaluate on 20201231 OOS (2021)
   train = pd.read_parquet('data/20201231/factors_0_is.parquet')
   test = pd.read_parquet('data/20201231/factors_0_oos.parquet')
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export multi-snapshot factor data")
    parser.add_argument(
        "--snapshots", 
        nargs="+", 
        default=SNAPSHOTS,
        help="Snapshot names to process"
    )
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=BASE_FACTOR_DIR,
        help="Base directory containing snapshot folders"
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=LABEL_FILE,
        help="Path to Label10.h5"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    main(
        snapshots=args.snapshots,
        base_factor_dir=args.factor_dir,
        label_file=args.label_file,
        output_dir=args.output_dir
    )
