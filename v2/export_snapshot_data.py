"""
Export Per-Snapshot Factor and Label Data (v2 — No Cross-Snapshot Merging)

This script processes each snapshot directory independently and exports
aligned factor+label data split into IS (In-Sample) and OOS (Out-of-Sample).

⚠️ KEY DESIGN PRINCIPLE:
   Data from different snapshot folders MUST NOT be merged.
   Each snapshot is a self-contained unit for training and evaluation.

Data Structures (confirmed via inspect_data.py):
────────────────────────────────────────────────
  weakFactors.h5 (per key /0, /1, /2):
    axis0:          int64   shape (1128,)    → factor IDs 0..1127
    axis1_level0:   int64   shape (933,)     → trading dates (20160104...)
    axis1_level1:   |S6     shape (3646,)    → instrument codes (b'000001'...)
    axis1_label0:   int16   shape (N,)       → date index codes
    axis1_label1:   int16   shape (N,)       → instrument index codes
    block0_values:  float32 shape (N, 1128)  → factor values

  Label10.h5 (PyTables TABLE format):
    Index:    endDate    (int64)    → forward-looking end date
    Columns:  code       (object)  → instrument code ('000001'...)
              labelValue (float64) → target label
              labelDate  (int64)   → label date (aligns with factor date)

Workflow per snapshot (cutoff date Di):
1. Load factors from {factor_dir}/{Di}/weakFactors.h5
2. Load labels from Label10.h5 with endDate filtering
3. Align factors and labels by (date, instrument)
4. Split into:
   - IS (In-Sample): dates <= Di  →  for training
   - OOS (Out-of-Sample): Di < dates <= OOS_end  →  for evaluation
5. Save to parquet files under data/{Di}/

Output Structure:
    data/
    ├── 20181228/
    │   ├── factors_0_is.parquet
    │   ├── factors_0_oos.parquet
    │   ├── factors_1_is.parquet
    │   ├── factors_1_oos.parquet
    │   ├── factors_2_is.parquet
    │   ├── factors_2_oos.parquet
    │   └── metadata.json
    ├── 20191231/
    │   └── ...
    └── 20201231/
        └── ...

Usage:
    python export_snapshot_data.py
    python export_snapshot_data.py --snapshots 20181228 20191231 20201231
    python export_snapshot_data.py --output-dir ./processed_data
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

BASE_FACTOR_DIR = Path("../1128_weight_factors")
LABEL_FILE = Path("../Label10.h5")
OUTPUT_DIR = Path("data")

SNAPSHOTS = ["20181228", "20191231", "20201231"]

SNAPSHOT_OOS_END = {
    "20181228": 20191231,
    "20191231": 20201231,
    "20201231": 20211231,
}

FACTOR_KEYS = ["0", "1", "2"]
CHUNK_SIZE = 100000
LABEL_CHUNK_SIZE = 500000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_factor_slice(
    h5_file: Path, key: str = "0", start: int = 0, stop: int = 1000
) -> pd.DataFrame:
    """
    Load a slice of factor data from HDF5 using h5py (NOT pd.read_hdf).

    Returns a DataFrame with:
      - MultiIndex: (date: int64, instrument: str)
      - Columns: str names '0', '1', ..., '1127'
      - Values: float32
    """
    with h5py.File(h5_file, "r") as f:
        grp = f[key]

        # Factor values: float32 (N, 1128)
        values = grp["block0_values"][start:stop, :]

        # Column labels: int64 (1128,) → MUST convert to str for Parquet
        columns_raw = grp["axis0"][:]
        columns = [str(int(c)) for c in columns_raw]

        # Date level: int64 (933,) → trading dates like 20160104
        level0 = grp["axis1_level0"][:]

        # Instrument level: bytes |S6 (3646,) → MUST decode to str
        level1_raw = grp["axis1_level1"][:]
        level1 = np.array([
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in level1_raw
        ])

        # Index codes: int16 → map to actual date/instrument values
        label0 = grp["axis1_label0"][start:stop]
        label1 = grp["axis1_label1"][start:stop]

    dates = level0[label0]        # int64 dates
    instruments = level1[label1]  # str instruments

    index = pd.MultiIndex.from_arrays(
        [dates, instruments], names=["date", "instrument"]
    )
    df = pd.DataFrame(values, index=index, columns=columns)
    return df


def get_factor_shape(h5_file: Path, key: str = "0") -> Tuple[int, int]:
    """Get the shape of the factor matrix without loading it."""
    with h5py.File(h5_file, "r") as f:
        shape = f[key]["block0_values"].shape
    return shape


def load_labels_for_snapshot(
    label_file: Path,
    cutoff_date: int,
    oos_end_date: int,
    start_date: int = 20180101,
) -> pd.DataFrame:
    """
    Load labels valid for a snapshot (endDate <= oos_end_date).

    Label10.h5 layout (confirmed):
      Index:    endDate    (int64)
      Columns:  code       (object/str)  → instrument code
                labelValue (float64)     → target
                labelDate  (int64)       → aligns with factor date

    Returns DataFrame indexed by (date=labelDate, instrument=code)
    with columns ['endDate', 'labelValue'].
    """
    print(f"  Loading labels (endDate <= {oos_end_date})...")

    with pd.HDFStore(label_file, mode="r") as store:
        total_rows = store.get_storer("Data").nrows

    all_labels = []
    rows_loaded = 0
    rows_kept = 0

    for start in range(0, total_rows, LABEL_CHUNK_SIZE):
        chunk = pd.read_hdf(label_file, key="Data", start=start, stop=start + LABEL_CHUNK_SIZE)
        # After read: Index=endDate(int64), Columns=[code(obj), labelValue(f64), labelDate(i64)]
        chunk = chunk.reset_index()
        # Now: Columns=[endDate(i64), code(obj), labelValue(f64), labelDate(i64)]
        rows_loaded += len(chunk)

        # Filter by date range
        chunk = chunk[chunk["endDate"] <= oos_end_date]
        if start_date is not None:
            chunk = chunk[chunk["labelDate"] >= start_date]

        if len(chunk) > 0:
            all_labels.append(chunk)
            rows_kept += len(chunk)

        if start % 2000000 == 0:
            print(f"    Scanned {start:,} rows, kept {rows_kept:,} labels...")

    print(f"  ✅ Labels loaded: {rows_kept:,} rows (scanned {rows_loaded:,})")

    if not all_labels:
        return pd.DataFrame()

    labels = pd.concat(all_labels, ignore_index=True)

    # Set index to (labelDate, code) → renamed to (date, instrument)
    # This aligns with factor DataFrame index: (date: int64, instrument: str)
    labels = labels.set_index(["labelDate", "code"])
    labels.index.names = ["date", "instrument"]

    # Remove duplicate index entries (keep first)
    if labels.index.duplicated().any():
        n_dup = labels.index.duplicated().sum()
        print(f"  ⚠️ Removing {n_dup} duplicate label entries")
        labels = labels[~labels.index.duplicated(keep="first")]

    return labels  # columns: ['endDate', 'labelValue']


def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save DataFrame to Parquet with guaranteed string column names."""
    # Force ALL column names to Python str — Parquet requires this
    df.columns = pd.Index([str(c) for c in df.columns])
    print(f"    Saving {label}: shape={df.shape}, cols_dtype={df.columns.dtype}, "
          f"all_str={all(isinstance(c, str) for c in df.columns)}")
    df.to_parquet(path)


def export_snapshot(
    snapshot_name: str,
    factor_file: Path,
    labels_indexed: pd.DataFrame,
    output_dir: Path,
    cutoff_date: int,
    oos_end_date: int,
) -> Dict:
    """Export IS and OOS data for a single snapshot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "snapshot": snapshot_name,
        "cutoff_date": cutoff_date,
        "oos_end_date": oos_end_date,
        "export_timestamp": datetime.now().isoformat(),
        "keys": {},
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
            factors = load_factor_slice(factor_file, key=key, start=start, stop=stop)

            # Align with labels by (date, instrument) intersection
            common_idx = factors.index.intersection(labels_indexed.index)
            if len(common_idx) == 0:
                continue

            aligned = factors.loc[common_idx].copy()

            # Use .values to avoid index alignment issues during assignment
            aligned["labelValue"] = labels_indexed.loc[common_idx, "labelValue"].values
            aligned["endDate"] = labels_indexed.loc[common_idx, "endDate"].values

            # Split IS / OOS by date
            dates = aligned.index.get_level_values("date")
            is_mask = dates <= cutoff_date
            oos_mask = (dates > cutoff_date) & (dates <= oos_end_date)

            if is_mask.sum() > 0:
                is_chunks.append(aligned[is_mask])
            if oos_mask.sum() > 0:
                oos_chunks.append(aligned[oos_mask])

            if start % 500000 == 0:
                print(f"    Processed {start:,}/{n_rows:,} rows...")

        # Save IS
        if is_chunks:
            is_df = pd.concat(is_chunks)
            is_path = output_dir / f"factors_{key}_is.parquet"
            _save_parquet(is_df, is_path, f"IS key={key}")
            is_dates = is_df.index.get_level_values("date")
            metadata["keys"][key] = metadata["keys"].get(key, {})
            metadata["keys"][key]["is"] = {
                "path": str(is_path),
                "shape": list(is_df.shape),
                "date_range": [int(is_dates.min()), int(is_dates.max())],
                "n_dates": int(is_dates.nunique()),
                "n_instruments": int(is_df.index.get_level_values("instrument").nunique()),
                "file_size_mb": round(is_path.stat().st_size / 1024 / 1024, 2),
            }
            print(f"    ✅ IS saved: {is_df.shape} ({is_dates.min()} to {is_dates.max()})")
        else:
            print(f"    ⚠️ No IS data for key {key}")

        # Save OOS
        if oos_chunks:
            oos_df = pd.concat(oos_chunks)
            oos_path = output_dir / f"factors_{key}_oos.parquet"
            _save_parquet(oos_df, oos_path, f"OOS key={key}")
            oos_dates = oos_df.index.get_level_values("date")
            metadata["keys"][key]["oos"] = {
                "path": str(oos_path),
                "shape": list(oos_df.shape),
                "date_range": [int(oos_dates.min()), int(oos_dates.max())],
                "n_dates": int(oos_dates.nunique()),
                "n_instruments": int(oos_df.index.get_level_values("instrument").nunique()),
                "file_size_mb": round(oos_path.stat().st_size / 1024 / 1024, 2),
            }
            print(f"    ✅ OOS saved: {oos_df.shape} ({oos_dates.min()} to {oos_dates.max()})")
        else:
            print(f"    ⚠️ No OOS data for key {key}")

        del is_chunks, oos_chunks

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  📋 Metadata saved: {metadata_path}")

    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main(
    snapshots: List[str] = None,
    base_factor_dir: Path = None,
    label_file: Path = None,
    output_dir: Path = None,
):
    snapshots = snapshots or SNAPSHOTS
    base_factor_dir = base_factor_dir or BASE_FACTOR_DIR
    label_file = label_file or LABEL_FILE
    output_dir = output_dir or OUTPUT_DIR

    start_time = datetime.now()

    print("=" * 60)
    print("PER-SNAPSHOT FACTOR EXPORT PIPELINE (v2)")
    print("=" * 60)
    print(f"\nStarted at: {start_time}")
    print(f"\nConfiguration:")
    print(f"  Factor base dir: {base_factor_dir.absolute()}")
    print(f"  Label file:      {label_file.absolute()}")
    print(f"  Output dir:      {output_dir.absolute()}")
    print(f"  Snapshots:       {snapshots}")
    print(f"\n⚠️  No cross-snapshot merging — each snapshot is self-contained.")

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

    for snapshot in snapshots:
        cutoff_date = int(snapshot)
        oos_end_date = SNAPSHOT_OOS_END.get(snapshot, cutoff_date + 10000)

        print("\n" + "=" * 60)
        print(f"SNAPSHOT: {snapshot}")
        print(f"  Cutoff (IS end): {cutoff_date}")
        print(f"  OOS period: {cutoff_date + 1} to {oos_end_date}")
        print("=" * 60)

        factor_file = base_factor_dir / snapshot / "weakFactors.h5"
        snapshot_output_dir = output_dir / snapshot

        # Step 1: Load labels
        print("\nStep 1: Loading labels...")
        labels_indexed = load_labels_for_snapshot(
            label_file,
            cutoff_date=cutoff_date,
            oos_end_date=oos_end_date,
            start_date=20180101,
        )

        if labels_indexed.empty:
            print(f"  ❌ No labels found for snapshot {snapshot}")
            continue

        print(f"  Labels indexed shape: {labels_indexed.shape}")
        print(f"  Labels columns: {list(labels_indexed.columns)}")
        print(f"  Labels index dtypes: "
              f"{[labels_indexed.index.get_level_values(i).dtype for i in range(labels_indexed.index.nlevels)]}")
        lbl_dates = labels_indexed.index.get_level_values("date")
        print(f"  Date range: {lbl_dates.min()} to {lbl_dates.max()}")

        # Step 2: Export factor data
        print("\nStep 2: Exporting factor data...")
        metadata = export_snapshot(
            snapshot_name=snapshot,
            factor_file=factor_file,
            labels_indexed=labels_indexed,
            output_dir=snapshot_output_dir,
            cutoff_date=cutoff_date,
            oos_end_date=oos_end_date,
        )

        all_metadata[snapshot] = metadata
        del labels_indexed

    # Save global metadata
    global_meta_path = output_dir / "all_snapshots_metadata.json"
    with open(global_meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\n📁 Output Directory: {output_dir.absolute()}")
    print(f"\nSnapshot Summary:")
    for snapshot in snapshots:
        cutoff = int(snapshot)
        oos_end = SNAPSHOT_OOS_END.get(snapshot, cutoff + 10000)
        print(f"  {snapshot}/")
        print(f"    IS:  dates <= {cutoff}")
        print(f"    OOS: {cutoff + 1} to {oos_end}")

    print(f"\n⏱️  Duration: {duration}")
    print(f"✅ Finished at: {end_time}")
    print("""
📋 Usage (per-snapshot training):

  For each snapshot independently:
    is_df = pd.read_parquet('data/{snapshot}/factors_0_is.parquet')
    X_train = is_df.drop(['labelValue', 'endDate'], axis=1)
    y_train = is_df['labelValue']

    oos_df = pd.read_parquet('data/{snapshot}/factors_0_oos.parquet')
    X_test = oos_df.drop(['labelValue', 'endDate'], axis=1)
    y_test = oos_df['labelValue']

  ⚠️ Do NOT merge data across snapshot folders!
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-snapshot factor data (v2)")
    parser.add_argument(
        "--snapshots", nargs="+", default=SNAPSHOTS,
        help="Snapshot names to process",
    )
    parser.add_argument(
        "--factor-dir", type=Path, default=BASE_FACTOR_DIR,
        help="Base directory containing snapshot folders",
    )
    parser.add_argument(
        "--label-file", type=Path, default=LABEL_FILE,
        help="Path to Label10.h5",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()
    main(
        snapshots=args.snapshots,
        base_factor_dir=args.factor_dir,
        label_file=args.label_file,
        output_dir=args.output_dir,
    )
